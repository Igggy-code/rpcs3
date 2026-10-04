#include "stdafx.h"
#include "MTLGSRender.h"
#include "MTLPresenter.h"

#include "Emu/IdManager.h"
#include "Emu/Memory/vm.h"
#include "Emu/RSX/gcm_enums.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/Utils/color_utils.hpp"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/system_config.h"
#include "Emu/RSX/Program/SPIRVCommon.h"

u64 MTLGSRender::get_cycles()
{
	return thread_ctrl::get_cycles(static_cast<named_thread<MTLGSRender>&>(*this));
}

MTLGSRender::MTLGSRender(utils::serial* ar) noexcept : GSRender(ar)
{
	// Apple GPUs interpolate varyings correctly; the manual barycentric path relies on
	// pervertex inputs, which MSL does not support.
	backend_config.supports_normalized_barycentrics = true;

	// Metal always uses the first vertex as the provoking vertex; flat shading falls back to smooth
	backend_config.supports_last_provoking_vertex = false;
}

MTLGSRender::~MTLGSRender()
{
	// Normally released in on_exit() on the RSX thread
	mtl::destroy_presenter(std::exchange(m_presenter, nullptr));
}

void MTLGSRender::on_init_thread()
{
	GSRender::on_init_thread();

	std::string device_name, error;

	if (!mtl::init_device(device_name, error))
	{
		rsx_log.fatal("Metal: failed to initialize the device: %s", error);
		return;
	}

	m_device_ready = true;

	// Sampler descriptors are consulted by the program analysis; without a texture cache they stay neutral
	for (auto& sampler : fs_sampler_state) sampler = std::make_unique<mtl::null_sampled_image>();
	for (auto& sampler : vs_sampler_state) sampler = std::make_unique<mtl::null_sampled_image>();

	if (mtl::shader_translation_available())
	{
		spirv::initialize_compiler_context();
		m_prog_buffer = std::make_unique<MTLProgramBuffer>();
	}
	else
	{
		rsx_log.error("Metal: built without SPIRV-Cross, shaders cannot be translated");
	}

	if (!m_frame)
	{
		rsx_log.warning("Metal: no game window, presenting is disabled");
		return;
	}

	m_presenter = mtl::create_presenter(m_frame->handle(), g_cfg.video.vsync != vsync_mode::off, error);

	if (!m_presenter)
	{
		rsx_log.fatal("Metal: failed to initialize presentation: %s", error);
		return;
	}

	rsx_log.notice("Metal: using device '%s' (phase 3b: draws with placeholder textures)", device_name);
}

void MTLGSRender::on_exit()
{
	if (m_device_ready)
	{
		if (m_prog_buffer)
		{
			m_prog_buffer.reset();
			spirv::finalize_compiler_context();
		}

		m_rtts.destroy();
		mtl::shutdown_draw_resources();
		mtl::destroy_presenter(std::exchange(m_presenter, nullptr));
		mtl::shutdown_device();
		m_device_ready = false;
	}

	GSRender::on_exit();
}

void MTLGSRender::init_buffers(rsx::framebuffer_creation_context context)
{
	m_graphics_state.clear(
		rsx::rtt_config_dirty |
		rsx::rtt_config_contested |
		rsx::rtt_config_valid |
		rsx::rtt_cache_state_dirty |
		rsx::pipeline_config_dirty);

	get_framebuffer_layout(context, m_framebuffer_layout);

	if (!m_graphics_state.test(rsx::rtt_config_valid))
	{
		return;
	}

	mtl::command_context cmd;
	m_rtts.prepare_render_target(cmd,
		m_framebuffer_layout.color_format, m_framebuffer_layout.depth_format,
		m_framebuffer_layout.width, m_framebuffer_layout.height,
		m_framebuffer_layout.target, m_framebuffer_layout.aa_mode, m_framebuffer_layout.raster_type,
		m_framebuffer_layout.color_addresses, m_framebuffer_layout.zeta_address,
		m_framebuffer_layout.actual_color_pitch, m_framebuffer_layout.actual_zeta_pitch,
		resolution_scaling_config);

	const u8 color_bpp = get_format_block_size_in_bytes(m_framebuffer_layout.color_format);
	const auto samples = get_format_sample_count(m_framebuffer_layout.aa_mode);

	for (int i = 0; i < rsx::limits::color_buffers_count; ++i)
	{
		if (std::get<0>(m_rtts.m_bound_render_targets[i]))
		{
			m_surface_info[i].address = m_framebuffer_layout.color_addresses[i];
			m_surface_info[i].pitch = m_framebuffer_layout.actual_color_pitch[i];
			m_surface_info[i].width = m_framebuffer_layout.width;
			m_surface_info[i].height = m_framebuffer_layout.height;
			m_surface_info[i].color_format = m_framebuffer_layout.color_format;
			m_surface_info[i].bpp = color_bpp;
			m_surface_info[i].samples = samples;
		}
		else
		{
			m_surface_info[i] = {};
		}
	}

	if (std::get<0>(m_rtts.m_bound_depth_stencil))
	{
		m_depth_surface_info.address = m_framebuffer_layout.zeta_address;
		m_depth_surface_info.pitch = m_framebuffer_layout.actual_zeta_pitch;
		m_depth_surface_info.width = m_framebuffer_layout.width;
		m_depth_surface_info.height = m_framebuffer_layout.height;
		m_depth_surface_info.depth_format = m_framebuffer_layout.depth_format;
		m_depth_surface_info.bpp = get_format_block_size_in_bytes(m_framebuffer_layout.depth_format);
		m_depth_surface_info.samples = samples;
	}
	else
	{
		m_depth_surface_info = {};
	}

	// There is no framebuffer object to build in Metal: attachments are bound per render pass
	m_graphics_state.set(rsx::rtt_config_valid);
}

void MTLGSRender::clear_surface(u32 arg)
{
	if (skip_current_frame || !m_device_ready) return;

	// If stencil write mask is disabled, remove clear_stencil bit
	if (!rsx::method_registers.stencil_mask()) arg &= ~RSX_GCM_CLEAR_STENCIL_BIT;

	// Ignore invalid clear flags
	if ((arg & RSX_GCM_CLEAR_ANY_MASK) == 0) return;

	u8 ctx = rsx::framebuffer_creation_context::context_draw;
	if (arg & RSX_GCM_CLEAR_COLOR_RGBA_MASK) ctx |= rsx::framebuffer_creation_context::context_clear_color;
	if (arg & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK) ctx |= rsx::framebuffer_creation_context::context_clear_depth;

	init_buffers(static_cast<rsx::framebuffer_creation_context>(ctx));

	if (!m_graphics_state.test(rsx::rtt_config_valid)) return;

	bool update_color = false, update_z = false;
	const rsx::surface_depth_format2 surface_depth_format = rsx::method_registers.surface_depth_fmt();

	// TODO: partial (scissored) and masked clears are applied to the whole surface for now
	if (auto ds = std::get<1>(m_rtts.m_bound_depth_stencil); ds && (arg & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK))
	{
		bool clear_depth = false, clear_stencil = false;
		float depth = 1.f;
		u8 stencil = 0xff;

		if (arg & RSX_GCM_CLEAR_DEPTH_BIT)
		{
			const u32 max_depth_value = get_max_depth_value(surface_depth_format);
			const u32 clear_depth_value = rsx::method_registers.z_clear_value(is_depth_stencil_format(surface_depth_format));
			depth = f32(clear_depth_value) / max_depth_value;
			clear_depth = true;
		}

		if (is_depth_stencil_format(surface_depth_format) && (arg & RSX_GCM_CLEAR_STENCIL_BIT))
		{
			stencil = static_cast<u8>(rsx::method_registers.stencil_clear_value());
			clear_stencil = true;
		}

		if (clear_depth || clear_stencil)
		{
			ds->state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
			mtl::clear_depth_stencil(*ds, clear_depth, depth, clear_stencil, stencil);
			update_z = true;
		}
	}

	if (auto colormask = (arg & 0xf0))
	{
		u8 clear_a = rsx::method_registers.clear_color_a();
		u8 clear_r = rsx::method_registers.clear_color_r();
		u8 clear_g = rsx::method_registers.clear_color_g();
		u8 clear_b = rsx::method_registers.clear_color_b();

		switch (rsx::method_registers.surface_color())
		{
		case rsx::surface_color_format::x32:
		case rsx::surface_color_format::w16z16y16x16:
		case rsx::surface_color_format::w32z32y32x32:
			// Nop
			colormask = 0;
			break;
		case rsx::surface_color_format::b8:
			rsx::get_b8_clear_color(clear_r, clear_g, clear_b, clear_a);
			colormask = rsx::get_b8_clearmask(colormask);
			break;
		case rsx::surface_color_format::g8b8:
			rsx::get_g8b8_clear_color(clear_r, clear_g, clear_b, clear_a);
			colormask = rsx::get_g8b8_r8g8_clearmask(colormask);
			break;
		case rsx::surface_color_format::r5g6b5:
			rsx::get_rgb565_clear_color(clear_r, clear_g, clear_b, clear_a);
			break;
		case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			rsx::get_a1rgb555_clear_color(clear_r, clear_g, clear_b, clear_a, 255);
			break;
		case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
			rsx::get_a1rgb555_clear_color(clear_r, clear_g, clear_b, clear_a, 0);
			break;
		case rsx::surface_color_format::a8b8g8r8:
		case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
		case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
			rsx::get_abgr8_clear_color(clear_r, clear_g, clear_b, clear_a);
			colormask = rsx::get_abgr8_clearmask(colormask);
			break;
		default:
			break;
		}

		if (colormask)
		{
			// Metal formats already match the RSX channel order (BGRA8 for ARGB8, RGBA8 for ABGR8),
			// so the logical clear color is used as-is
			const float rgba[4] = { clear_r / 255.f, clear_g / 255.f, clear_b / 255.f, clear_a / 255.f };

			for (const auto& index : m_rtts.m_bound_render_target_ids)
			{
				auto rtt = m_rtts.m_bound_render_targets[index].second;
				rtt->state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
				mtl::clear_color(*rtt, rgba, colormask >> 4);
			}

			update_color = true;
		}
	}

	if (update_color || update_z)
	{
		m_rtts.on_write({ update_color, update_color, update_color, update_color }, update_z);
	}
}

mtl::render_target* MTLGSRender::get_present_surface(u32 address, u32 width, u32 height, u32 pitch, u32& out_width, u32& out_height)
{
	mtl::command_context cmd;
	const auto overlap_info = m_rtts.get_merged_texture_memory_region(cmd, address, width, height, pitch, 4, rsx::surface_access::transfer_read);

	if (overlap_info.empty())
	{
		return nullptr;
	}

	const auto& section = overlap_info.back();
	auto surface = mtl::as_rtt(section.surface);

	if (section.base_address != address || surface->is_depth_surface())
	{
		// TODO: insets / letterboxed surfaces (see GLGSRender::get_present_source)
		return nullptr;
	}

	const auto surface_width = surface->get_surface_width<rsx::surface_metrics::samples>();
	const auto surface_height = surface->get_surface_height<rsx::surface_metrics::samples>();

	if (surface_width < width || surface_height < height)
	{
		return nullptr;
	}

	std::tie(out_width, out_height) = rsx::apply_resolution_scale<true>(resolution_scaling_config, std::min(surface_width, width), std::min(surface_height, height));
	return surface;
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
		u32 width = 0, height = 0;

		if (info.buffer < display_buffers_count)
		{
			width = display_buffers[info.buffer].width;
			height = display_buffers[info.buffer].height;
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
			if (format == CELL_GCM_TEXTURE_A8R8G8B8 && width && height && pitch >= width * 4)
			{
				// Prefer the surface the RSX rendered into; fall back to memory (e.g. software-rendered output)
				if (auto surface = get_present_surface(address, width, height, pitch, params.surface_width, params.surface_height))
				{
					params.surface = surface;
				}
				else if (vm::check_addr(address, vm::page_readable, pitch * height))
				{
					params.pixels = vm::base(address);
					params.width = width;
					params.height = height;
					params.pitch = pitch;
				}
			}
		}

		params.output_width = m_frame->client_width();
		params.output_height = m_frame->client_height();

		if (params.surface || params.pixels)
		{
			const areau area = avconfig.aspect_convert_region({ width, height }, { params.output_width, params.output_height });
			params.viewport_x = area.x1;
			params.viewport_y = area.y1;
			params.viewport_width = area.width();
			params.viewport_height = area.height();
		}

		mtl::present(m_presenter, params);
	}
	else if (m_device_ready)
	{
		mtl::flush();
	}

	if (m_device_ready)
	{
		mtl::command_context cmd;
		m_rtts.trim(cmd);

		if (const u32 total = mtl::g_programs_ok + mtl::g_programs_failed; total != m_reported_programs)
		{
			m_reported_programs = total;
			rsx_log.notice("Metal: shader programs translated: %u ok, %u failed", mtl::g_programs_ok.load(), mtl::g_programs_failed.load());
		}

		if (m_draws_submitted && !m_draws_reported)
		{
			m_draws_reported = m_draws_submitted;
			rsx_log.notice("Metal: first frame with draws (%u draw calls so far)", m_draws_submitted);
		}
	}

	if (m_frame)
	{
		m_frame->flip(m_context);
	}

	rsx::thread::flip(info);
}
