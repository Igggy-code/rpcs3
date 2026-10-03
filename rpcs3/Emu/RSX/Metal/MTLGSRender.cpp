#include "stdafx.h"
#include "MTLGSRender.h"
#include "MTLPresenter.h"

#include "Emu/IdManager.h"
#include "Emu/Memory/vm.h"
#include "Emu/RSX/gcm_enums.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/system_config.h"

u64 MTLGSRender::get_cycles()
{
	return thread_ctrl::get_cycles(static_cast<named_thread<MTLGSRender>&>(*this));
}

MTLGSRender::MTLGSRender(utils::serial* ar) noexcept : GSRender(ar)
{
}

MTLGSRender::~MTLGSRender()
{
	// Normally released in on_exit() on the RSX thread
	mtl::destroy_presenter(std::exchange(m_presenter, nullptr));
}

void MTLGSRender::on_init_thread()
{
	GSRender::on_init_thread();

	if (!m_frame)
	{
		rsx_log.warning("Metal: no game window, presenting is disabled");
		return;
	}

	std::string error;
	m_presenter = mtl::create_presenter(m_frame->handle(), g_cfg.video.vsync != vsync_mode::off, error);

	if (!m_presenter)
	{
		rsx_log.fatal("Metal: failed to initialize: %s", error);
		return;
	}

	rsx_log.notice("Metal: using device '%s' (phase 1: display buffer presentation only, draws are skipped)", mtl::get_device_name(m_presenter));
}

void MTLGSRender::on_exit()
{
	mtl::destroy_presenter(std::exchange(m_presenter, nullptr));
	GSRender::on_exit();
}

void MTLGSRender::end()
{
	// No draw support yet: consume the draw like the Null renderer
	execute_nop_draw();
	rsx::thread::end();
}

void MTLGSRender::flip(const rsx::display_flip_info_t& info)
{
	if (m_presenter && m_frame && !info.skip_frame)
	{
		if (m_vsync_mode != g_cfg.video.vsync)
		{
			m_vsync_mode = g_cfg.video.vsync;
			mtl::set_vsync(m_presenter, m_vsync_mode != vsync_mode::off);
		}

		const auto& avconfig = g_fxo->get<rsx::avconf>();

		mtl::present_params params{};

		if (info.buffer < display_buffers_count)
		{
			u32 width = display_buffers[info.buffer].width;
			u32 height = display_buffers[info.buffer].height;
			u32 pitch = display_buffers[info.buffer].pitch;
			u32 format = CELL_GCM_TEXTURE_A8R8G8B8;

			if (!width)
			{
				width = avconfig.resolution_x;
				height = avconfig.resolution_y;
			}

			if (avconfig.state)
			{
				format = avconfig.get_compatible_gcm_format();
				if (!pitch) pitch = width * avconfig.get_bpp();

				const size2u frame_size = avconfig.video_frame_size();
				width = std::min(width, frame_size.width);
				height = std::min(height, frame_size.height);
			}
			else if (!pitch)
			{
				pitch = width * 4;
			}

			const u32 address = rsx::get_address(display_buffers[info.buffer].offset, CELL_GCM_LOCATION_LOCAL);

			// Only 32-bit XRGB is handled for now (FP16 display buffers are rare)
			if (format == CELL_GCM_TEXTURE_A8R8G8B8 && width && height && pitch >= width * 4 &&
				vm::check_addr(address, vm::page_readable, pitch * height))
			{
				params.pixels = vm::base(address);
				params.width = width;
				params.height = height;
				params.pitch = pitch;
			}
		}

		params.output_width = m_frame->client_width();
		params.output_height = m_frame->client_height();

		if (params.pixels)
		{
			const areau area = avconfig.aspect_convert_region({ params.width, params.height }, { params.output_width, params.output_height });
			params.viewport_x = area.x1;
			params.viewport_y = area.y1;
			params.viewport_width = area.width();
			params.viewport_height = area.height();
		}

		mtl::present(m_presenter, params);
	}

	if (m_frame)
	{
		m_frame->flip(m_context);
	}

	rsx::thread::flip(info);
}
