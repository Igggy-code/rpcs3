#include "stdafx.h"
#include "MTLGSRender.h"
#include "MTLDebugDump.h"
#include "MTLPresenter.h"

#include "Emu/IdManager.h"
#include "Emu/Memory/vm.h"
#include "Emu/Memory/vm_locking.h"
#include "Emu/RSX/gcm_enums.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/RSX/Utils/color_utils.hpp"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/system_config.h"
#include "Emu/RSX/Program/SPIRVCommon.h"
#include "Emu/RSX/Host/MM.h"

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
	if (zcull_ctrl.get() == static_cast<::rsx::reports::ZCULL_control*>(this))
	{
		zcull_ctrl.release();
	}

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

	// Real ZCULL pixel counts (the default controller reports every query as visible with a bogus count)
	zcull_ctrl.reset(static_cast<::rsx::reports::ZCULL_control*>(this));

	// Sampler descriptors are consulted by the program analysis; they are filled by the texture cache
	for (auto& sampler : fs_sampler_state) sampler = std::make_unique<mtl::texture_cache::sampled_image_descriptor>();
	for (auto& sampler : vs_sampler_state) sampler = std::make_unique<mtl::texture_cache::sampled_image_descriptor>();

	m_texture_cache.initialize();

	mtl::set_log_handler([](const char* message) { rsx_log.error("%s", message); });

	// Build the transfer shaders now rather than in the middle of a frame
	mtl::prepare_texture_ops();

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

	rsx_log.notice("Metal: using device '%s' (phase 4: draws with textures)", device_name);
}

void MTLGSRender::on_exit()
{
	if (zcull_ctrl.get() == static_cast<::rsx::reports::ZCULL_control*>(this))
	{
		// Queries live in device resources that are released below
		for (auto& query : m_occlusion_map)
		{
			mtl::occlusion_release(query);
		}

		m_active_query = nullptr;
		zcull_ctrl.release();
		zcull_ctrl = std::make_unique<::rsx::reports::ZCULL_control>();
	}

	if (m_device_ready)
	{
		if (m_prog_buffer)
		{
			mtl::shutdown_pipeline_compiler();
			m_prog_buffer.reset();
			spirv::finalize_compiler_context();
		}

		m_texture_cache.destroy();
		m_rtts.destroy();
		mtl::shutdown_texture_ops();
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

	if (m_dump)
	{
		dump_on_framebuffer_change();
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
		if (m_surface_info[i].pitch && g_cfg.video.write_color_buffers)
		{
			const utils::address_range32 surface_range = m_surface_info[i].get_memory_range();
			m_texture_cache.set_memory_read_flags(surface_range, rsx::memory_read_flags::flush_once);
			m_texture_cache.flush_if_cache_miss_likely(cmd, surface_range);
		}

		if (std::get<0>(m_rtts.m_bound_render_targets[i]))
		{
			m_surface_info[i].address = m_framebuffer_layout.color_addresses[i];
			m_surface_info[i].pitch = m_framebuffer_layout.actual_color_pitch[i];
			m_surface_info[i].width = m_framebuffer_layout.width;
			m_surface_info[i].height = m_framebuffer_layout.height;
			m_surface_info[i].color_format = m_framebuffer_layout.color_format;
			m_surface_info[i].bpp = color_bpp;
			m_surface_info[i].samples = samples;
			m_texture_cache.notify_surface_changed(m_surface_info[i].get_memory_range(m_framebuffer_layout.aa_factors));
		}
		else
		{
			m_surface_info[i] = {};
		}
	}

	if (m_depth_surface_info.pitch && g_cfg.video.write_depth_buffer)
	{
		const utils::address_range32 surface_range = m_depth_surface_info.get_memory_range();
		m_texture_cache.set_memory_read_flags(surface_range, rsx::memory_read_flags::flush_once);
		m_texture_cache.flush_if_cache_miss_likely(cmd, surface_range);
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
		m_texture_cache.notify_surface_changed(m_depth_surface_info.get_memory_range(m_framebuffer_layout.aa_factors));
	}
	else
	{
		m_depth_surface_info = {};
	}

	// There is no framebuffer object to build in Metal: attachments are bound per render pass
	m_graphics_state.set(rsx::rtt_config_valid);

	m_texture_cache.clear_ro_tex_invalidate_intr();

	if (!m_rtts.superseded_surfaces.empty())
	{
		for (auto& surface : m_rtts.superseded_surfaces)
		{
			m_texture_cache.discard_framebuffer_memory_region(cmd, surface->get_memory_range());
		}

		m_rtts.superseded_surfaces.clear();
	}

	if (!m_rtts.orphaned_surfaces.empty())
	{
		for (auto& [base_addr, surface] : m_rtts.orphaned_surfaces)
		{
			const bool lock = surface->is_depth_surface() ? !!g_cfg.video.write_depth_buffer : !!g_cfg.video.write_color_buffers;

			if (!lock || !surface->is_locked())
			{
				m_texture_cache.commit_framebuffer_memory_region(cmd, surface->get_memory_range());
				continue;
			}

			m_texture_cache.lock_memory_region(
				cmd, surface, surface->get_memory_range(), false,
				surface->get_surface_width<rsx::surface_metrics::pixels>(), surface->get_surface_height<rsx::surface_metrics::pixels>(), surface->get_rsx_pitch(),
				static_cast<const mtl::render_target*>(surface));
		}

		m_rtts.orphaned_surfaces.clear();
	}

	// Protect the bound surfaces so that guest reads of their memory flush the GPU contents
	for (u8 i = 0; i < rsx::limits::color_buffers_count; ++i)
	{
		if (!m_surface_info[i].address || !m_surface_info[i].pitch) continue;

		const auto surface_range = m_surface_info[i].get_memory_range();
		if (g_cfg.video.write_color_buffers)
		{
			const auto surface = m_rtts.m_bound_render_targets[i].second;
			m_texture_cache.lock_memory_region(
				cmd, surface, surface_range, true,
				m_surface_info[i].width, m_surface_info[i].height, m_surface_info[i].pitch,
				static_cast<const mtl::render_target*>(surface));
		}
		else
		{
			m_texture_cache.commit_framebuffer_memory_region(cmd, surface_range);
		}
	}

	if (m_depth_surface_info.address && m_depth_surface_info.pitch)
	{
		const auto surface_range = m_depth_surface_info.get_memory_range();
		if (g_cfg.video.write_depth_buffer)
		{
			const auto surface = m_rtts.m_bound_depth_stencil.second;
			m_texture_cache.lock_memory_region(
				cmd, surface, surface_range, true,
				m_depth_surface_info.width, m_depth_surface_info.height, m_depth_surface_info.pitch,
				static_cast<const mtl::render_target*>(surface));
		}
		else
		{
			m_texture_cache.commit_framebuffer_memory_region(cmd, surface_range);
		}
	}

	if (m_texture_cache.get_ro_tex_invalidate_intr())
	{
		// Invalidate cached sampler state
		m_samplers_dirty.store(true);
	}
}

bool MTLGSRender::on_access_violation(u32 address, bool is_writing)
{
	rsx::mm_flush(address);

	const bool can_flush = is_current_thread();
	const rsx::invalidation_cause cause = is_writing
		? (can_flush ? rsx::invalidation_cause::write : rsx::invalidation_cause::deferred_write)
		: (can_flush ? rsx::invalidation_cause::read : rsx::invalidation_cause::deferred_read);

	mtl::command_context cmd;
	mtl::stall_probe probe(can_flush ? "access violation (RSX thread flush)" : "access violation");
	auto result = m_texture_cache.invalidate_address(cmd, address, cause);

	if (result.invalidate_samplers)
	{
		std::lock_guard lock(m_sampler_mutex);
		m_samplers_dirty.store(true);
	}

	if (mtl::debug::g_frame_dump_active)
	{
		rsx_log.notice("Metal dump: access violation at 0x%x (%s), handled %d, flushable %d, sections %u",
			address, is_writing ? "write" : "read", result.violation_handled, result.num_flushable, ::size32(result.sections_to_flush));
	}

	if (!result.violation_handled)
	{
		return zcull_ctrl->on_access_violation(address);
	}

	if (result.num_flushable > 0)
	{
		// GPU readbacks are recorded on the RSX thread
		auto& task = post_flush_request(address, result);

		m_eng_interrupt_mask |= rsx::backend_interrupt;
		vm::temporary_unlock();
		task.producer_wait();
	}

	return true;
}

void MTLGSRender::on_invalidate_memory_range(const utils::address_range32& range, rsx::invalidation_cause cause)
{
	mtl::command_context cmd;
	auto data = m_texture_cache.invalidate_range(cmd, range, cause);
	AUDIT(data.empty());

	if (cause == rsx::invalidation_cause::unmap && data.violation_handled)
	{
		m_texture_cache.purge_unreleased_sections();
		{
			std::lock_guard lock(m_sampler_mutex);
			m_samplers_dirty.store(true);
		}
	}
}

void MTLGSRender::on_semaphore_acquire_wait()
{
	if (!m_work_queue.empty() ||
		(async_flip_requested & flip_request::emu_requested))
	{
		do_local_task(rsx::FIFO::state::lock_wait);
	}
}

void MTLGSRender::do_local_task(rsx::FIFO::state state)
{
	if (!m_work_queue.empty())
	{
		std::lock_guard lock(m_queue_guard);

		m_work_queue.remove_if([](auto& q) { return q.received.load(); });

		for (auto& q : m_work_queue)
		{
			if (q.processed.load()) continue;

			mtl::command_context cmd;
			mtl::stall_probe probe("flush request");
			q.result = m_texture_cache.flush_all(cmd, q.section_data);
			q.processed = true;
		}
	}
	else if (!in_begin_end && state != rsx::FIFO::state::lock_wait)
	{
		if (m_graphics_state & rsx::pipeline_state::framebuffer_reads_dirty)
		{
			// Re-engages locks; only safe when nobody waits in the access violation handler
			m_texture_cache.do_update();
			m_graphics_state.clear(rsx::pipeline_state::framebuffer_reads_dirty);
		}
	}

	rsx::thread::do_local_task(state);
}

mtl::work_item& MTLGSRender::post_flush_request(u32 address, mtl::texture_cache::thrashed_set& flush_data)
{
	std::lock_guard lock(m_queue_guard);

	auto& result = m_work_queue.emplace_back();
	result.address_to_flush = address;
	result.section_data = std::move(flush_data);
	return result;
}

bool MTLGSRender::scaled_image_from_memory(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate)
{
	mtl::stall_probe probe("NV3089 blit");

	if (!m_device_ready)
	{
		return false;
	}

	mtl::command_context cmd;
	const bool handled = m_texture_cache.blit(cmd, src, dst, interpolate, m_rtts);

	if (mtl::debug::g_frame_dump_active)
	{
		const auto text = fmt::format("blit src 0x%x fmt %d off %u,%u %ux%u pitch %u bpp %u -> dst 0x%x fmt %d off %u,%u %ux%u clip %u,%u %ux%u pitch %u bpp %u scale %.3f,%.3f swz %d: %s",
			src.rsx_address, static_cast<int>(src.format), src.offset_x, src.offset_y, src.width, src.height, src.pitch, src.bpp,
			dst.rsx_address, static_cast<int>(dst.format), dst.offset_x, dst.offset_y, dst.width, dst.height, dst.clip_x, dst.clip_y, dst.clip_width, dst.clip_height,
			dst.pitch, dst.bpp, dst.scale_x, dst.scale_y, dst.swizzled, handled ? "GPU" : "CPU fallback");
		rsx_log.notice("Metal dump: %s", text);

		if (m_dump)
		{
			m_dump->write("  " + text + "\n");
		}
	}

	if (handled)
	{
		m_samplers_dirty.store(true);
		return true;
	}

	return false;
}

void MTLGSRender::clear_surface(u32 arg)
{
	mtl::stall_probe probe("clear");

	if (skip_current_frame || !m_device_ready) return;

	// If stencil write mask is disabled, remove clear_stencil bit
	if (!rsx::method_registers.stencil_mask()) arg &= ~RSX_GCM_CLEAR_STENCIL_BIT;

	// Ignore invalid clear flags
	if ((arg & RSX_GCM_CLEAR_ANY_MASK) == 0) return;

	u8 ctx = rsx::framebuffer_creation_context::context_draw;
	if (arg & RSX_GCM_CLEAR_COLOR_RGBA_MASK) ctx |= rsx::framebuffer_creation_context::context_clear_color;
	if (arg & RSX_GCM_CLEAR_DEPTH_STENCIL_MASK) ctx |= rsx::framebuffer_creation_context::context_clear_depth;

	init_buffers(static_cast<rsx::framebuffer_creation_context>(ctx));

	if (m_dump)
	{
		dump_clear(arg);
	}

	if (!m_graphics_state.test(rsx::rtt_config_valid)) return;

	bool update_color = false, update_z = false;
	const rsx::surface_depth_format2 surface_depth_format = rsx::method_registers.surface_depth_fmt();

	// Clear rectangle: the (unclipped) scissor, limited to the framebuffer. Games clear sub-regions of
	// atlases (e.g. one shadow cascade at a time), so partial clears must not touch the rest.
	if (areau scissor; get_scissor(scissor, false))
	{
		m_scissor = scissor;
	}

	const auto [fb_width, fb_height] = rsx::apply_resolution_scale<true>(resolution_scaling_config, m_framebuffer_layout.width, m_framebuffer_layout.height);
	const u32 clear_x = std::min<u32>(m_scissor.x1, fb_width);
	const u32 clear_y = std::min<u32>(m_scissor.y1, fb_height);
	const u32 clear_w = std::min<u32>(m_scissor.x2, fb_width) - clear_x;
	const u32 clear_h = std::min<u32>(m_scissor.y2, fb_height) - clear_y;
	// A load-action clear wipes the whole texture: only allowed when the rectangle covers it
	// (surfaces can be larger than the current framebuffer region)
	const auto covers = [&](const mtl::texture* tex)
	{
		return clear_x == 0 && clear_y == 0 && clear_w >= tex->width() && clear_h >= tex->height();
	};

	if (!clear_w || !clear_h)
	{
		return;
	}

	mtl::clear_desc region{};
	region.x = clear_x;
	region.y = clear_y;
	region.width = clear_w;
	region.height = clear_h;

	// Same attachments as the draws, so partial clears are recorded in the open render pass
	{
		u32 slot = 0;
		for (const auto& index : m_rtts.m_bound_render_target_ids)
		{
			if (auto rtt = m_rtts.m_bound_render_targets[index].second; rtt && slot < 4) region.color[slot++] = rtt;
		}

		region.depth = std::get<1>(m_rtts.m_bound_depth_stencil);
	}

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
			const u8 stencil_mask = static_cast<u8>(rsx::method_registers.stencil_mask());
			const bool stencil_complete = !clear_stencil || stencil_mask == 0xff || !has_stencil(ds->format());

			if (covers(ds) && stencil_complete)
			{
				ds->state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
				mtl::clear_depth_stencil(*ds, clear_depth, depth, clear_stencil, stencil);
			}
			else
			{
				// Partial clear: keep the rest of the surface (resolve inherited contents first)
				mtl::command_context cmd;
				ds->write_barrier(cmd);

				mtl::clear_desc dsc = region;
				dsc.depth = ds;
				dsc.clear_depth = clear_depth;
				dsc.depth_value = depth;
				dsc.clear_stencil = clear_stencil;
				dsc.stencil_value = stencil;
				dsc.stencil_write_mask = stencil_mask;
				mtl::clear_region(dsc);
			}

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

			bool full_frame = colormask == 0xf0;
			for (const auto& index : m_rtts.m_bound_render_target_ids)
			{
				if (auto rtt = m_rtts.m_bound_render_targets[index].second; rtt && !covers(rtt)) full_frame = false;
			}

			if (full_frame)
			{
				for (const auto& index : m_rtts.m_bound_render_target_ids)
				{
					auto rtt = m_rtts.m_bound_render_targets[index].second;
					rtt->state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
					mtl::clear_color(*rtt, rgba, colormask >> 4);
				}
			}
			else
			{
				// Partial or channel-masked clear of all bound targets in one draw
				mtl::command_context cmd;
				mtl::clear_desc cc = region;

				for (const auto& index : m_rtts.m_bound_render_target_ids)
				{
					if (auto rtt = m_rtts.m_bound_render_targets[index].second) rtt->write_barrier(cmd);
				}

				std::copy(rgba, rgba + 4, cc.rgba);
				cc.color_mask =
					((colormask & RSX_GCM_CLEAR_RED_BIT) ? 8u : 0u) |
					((colormask & RSX_GCM_CLEAR_GREEN_BIT) ? 4u : 0u) |
					((colormask & RSX_GCM_CLEAR_BLUE_BIT) ? 2u : 0u) |
					((colormask & RSX_GCM_CLEAR_ALPHA_BIT) ? 1u : 0u);
				mtl::clear_region(cc);
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
	mtl::stall_probe probe("flip", 50'000);

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

		if (m_dump)
		{
			dump_on_flip(const_cast<mtl::render_target*>(static_cast<const mtl::render_target*>(params.surface)));
		}

		mtl::present(m_presenter, params);
	}
	else if (m_device_ready)
	{
		mtl::flush();
	}

	if (m_device_ready && !m_dump)
	{
		// Starts a frame dump when requested (see MTLDebugDump.h)
		dump_on_flip(nullptr);
	}

	if (m_device_ready)
	{
		mtl::command_context cmd;
		m_rtts.trim(cmd);
		m_texture_cache.on_frame_end();

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

// ---------------------------------------------------------------------------------------------
// Occlusion queries
// ---------------------------------------------------------------------------------------------

void MTLGSRender::begin_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	query->result = 0;
	m_active_query = query;
}

void MTLGSRender::end_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	if (m_active_query == query)
	{
		m_active_query = nullptr;
	}
}

bool MTLGSRender::check_occlusion_query_status(rsx::reports::occlusion_query_info* query)
{
	if (!query->num_draws)
	{
		return true;
	}

	const auto& data = m_occlusion_map[query->driver_handle];
	if (data.slots.empty())
	{
		return true;
	}

	if (mtl::occlusion_pending_submit(data))
	{
		// Still being recorded; the zcull sync hint submits it when the result is needed
		return false;
	}

	return mtl::occlusion_ready(data);
}

void MTLGSRender::get_occlusion_query_result(rsx::reports::occlusion_query_info* query)
{
	auto& data = m_occlusion_map[query->driver_handle];

	if (query->num_draws)
	{
		u64 samples = 0;

		if (!data.slots.empty())
		{
			mtl::stall_probe probe("occlusion query read", 5000);
			samples = mtl::occlusion_result(data, !!g_cfg.video.precise_zpass_count);
		}

		if (data.skipped)
		{
			// Draws skipped while their pipeline compiled count as visible: reporting them as occluded
			// makes games hide objects (and conditional rendering drop draws)
			samples += 0xffff;
		}

		query->result += static_cast<u32>(std::min<u64>(samples, 0xffffffffu));
	}

	mtl::occlusion_release(data);
}

void MTLGSRender::sync_hint(rsx::FIFO::interrupt_hint hint, rsx::reports::sync_hint_payload_t payload)
{
	rsx::thread::sync_hint(hint, payload);

	if (hint != rsx::FIFO::interrupt_hint::zcull_sync || !payload.query || !m_device_ready)
	{
		return;
	}

	// The result is needed soon: submit the pending work now so the GPU finishes it before the read
	if (mtl::occlusion_pending_submit(m_occlusion_map[payload.query->driver_handle]))
	{
		mtl::stall_probe probe("zcull submit", 5000);
		mtl::flush();
	}
}

void MTLGSRender::discard_occlusion_query(rsx::reports::occlusion_query_info* query)
{
	if (m_active_query == query)
	{
		m_active_query = nullptr;
	}

	mtl::occlusion_release(m_occlusion_map[query->driver_handle]);
}
