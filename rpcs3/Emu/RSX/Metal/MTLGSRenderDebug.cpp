#include "stdafx.h"
#include "MTLGSRender.h"
#include "MTLDebugDump.h"

#include "Emu/RSX/rsx_methods.h"
#include "Utilities/File.h"
#include "Emu/Memory/vm.h"

#include <cmath>

#include <chrono>

// Frame dump (see MTLDebugDump.h). Everything here runs on the RSX thread.

namespace
{
	constexpr u32 max_texture_dumps = 400;

	std::string request_path()
	{
		return fs::get_cache_dir() + "metal_dump_request";
	}
}

void MTLGSRender::dump_on_flip(mtl::render_target* presented)
{
	if (m_dump)
	{
		// Last pass of the frame and the presented image
		dump_bound_surfaces("end of frame");

		// Surfaces in main memory are usually exchanged with the CPU/SPUs (post-processing inputs and outputs)
		fs::create_path(m_dump->dir + "/memory_surfaces");
		m_dump->write("\n== surfaces in main memory at the end of the frame\n");
		m_rtts.for_each_surface([&](u32 address, mtl::render_target* surface)
		{
			if (address >= 0xc0000000)
			{
				return;
			}

			const std::string name = fmt::format("memory_surfaces/0x%08x_%s_%ux%u_pitch%u", address,
				mtl::debug::format_name(surface->format()), surface->width(), surface->height(), surface->get_rsx_pitch());
			const auto result = mtl::debug::dump_texture(*surface, m_dump->dir + "/" + name, true);
			m_dump->images++;
			m_dump->write(fmt::format("  %s: %s\n", name, result));

			// What the CPU/SPUs see in guest memory: as big-endian ARGB8 and as big-endian FP16 RGBA
			const u32 pitch = surface->get_rsx_pitch();
			const u32 w = surface->get_surface_width<rsx::surface_metrics::pixels>();
			const u32 h = surface->get_surface_height<rsx::surface_metrics::pixels>();
			if (surface->get_bpp() == 4 && pitch >= w * 4 && vm::check_addr(address, vm::page_readable, pitch * h))
			{
				const u8* mem = vm::get_super_ptr<const u8>(address);
				std::vector<u8> argb(static_cast<usz>(w) * h * 3), fp16(static_cast<usz>(w / 2) * h * 3);

				const auto half = [](const u8* p)
				{
					const u16 v = static_cast<u16>((p[0] << 8) | p[1]);
					const u32 e = (v >> 10) & 0x1f, m = v & 0x3ff;
					const f32 f = e == 0 ? m / 16777216.f : std::ldexp(1.f + m / 1024.f, static_cast<int>(e) - 15);
					return static_cast<u8>(std::clamp((v & 0x8000) ? 0.f : f, 0.f, 1.f) * 255.f);
				};

				for (u32 y = 0; y < h; ++y)
				{
					const u8* row = mem + static_cast<usz>(y) * pitch;
					for (u32 x = 0; x < w; ++x)
					{
						const u8* p = row + x * 4;
						u8* o = &argb[(static_cast<usz>(y) * w + x) * 3];
						o[0] = p[1]; o[1] = p[2]; o[2] = p[3];
					}

					for (u32 x = 0; x < w / 2; ++x)
					{
						const u8* p = row + x * 8;
						u8* o = &fp16[(static_cast<usz>(y) * (w / 2) + x) * 3];
						o[0] = half(p); o[1] = half(p + 2); o[2] = half(p + 4);
					}
				}

				mtl::debug::write_png(m_dump->dir + "/" + name + "_guest_argb8.png", w, h, 3, argb.data());
				mtl::debug::write_png(m_dump->dir + "/" + name + "_guest_fp16.png", w / 2, h, 3, fp16.data());
			}
		});

		if (presented)
		{
			const auto result = mtl::debug::dump_texture(*presented, m_dump->dir + "/presented", false);
			m_dump->write(fmt::format("\n== flip: presented surface 0x%x %ux%u %s: %s\n",
				presented->base_addr, presented->width(), presented->height(), mtl::debug::format_name(presented->format()), result));
		}
		else
		{
			m_dump->write("\n== flip: presented from guest memory (no surface)\n");
		}

		mtl::debug::g_frame_dump_active = false;
		rsx_log.success("Metal: frame dump written to %s (%u passes, %u draws, %u images)", m_dump->dir, m_dump->pass, m_dump->draw, m_dump->images);
		m_dump.reset();
		return;
	}

	if (!fs::is_file(request_path()))
	{
		return;
	}

	fs::remove_file(request_path());

	const auto stamp = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
	auto dump = std::make_unique<frame_dump>();
	dump->dir = fs::get_cache_dir() + fmt::format("metal_dump/%d", stamp);

	if (!fs::create_path(dump->dir + "/textures") || !fs::create_path(dump->dir + "/passes") || !fs::create_path(dump->dir + "/programs") ||
		!dump->log.open(dump->dir + "/frame.txt", fs::rewrite))
	{
		rsx_log.error("Metal: cannot create the frame dump directory %s", dump->dir);
		return;
	}

	dump->write(fmt::format("Metal frame dump. Read Color=%d Write Color=%d Read Depth=%d Write Depth=%d\n",
		!!g_cfg.video.read_color_buffers, !!g_cfg.video.write_color_buffers, !!g_cfg.video.read_depth_buffer, !!g_cfg.video.write_depth_buffer));

	rsx_log.success("Metal: dumping the next frame to %s", dump->dir);
	mtl::debug::g_frame_dump_active = true;
	m_dump = std::move(dump);
}

void MTLGSRender::dump_bound_surfaces(const char* reason)
{
	if (!m_dump)
	{
		return;
	}

	const u32 pass = m_dump->pass++;
	m_dump->write(fmt::format("\n== pass %u result (%s)\n", pass, reason));

	const auto dump_one = [&](mtl::render_target* surface, const std::string& slot)
	{
		if (!surface)
		{
			return;
		}

		const std::string name = fmt::format("passes/p%03u_%s_0x%08x_%s_%ux%u", pass, slot, surface->base_addr,
			mtl::debug::format_name(surface->format()), surface->width(), surface->height());

		const auto result = mtl::debug::dump_texture(*surface, m_dump->dir + "/" + name, true);
		m_dump->images++;
		m_dump->write(fmt::format("  %s: %s\n", name, result));
	};

	for (u32 i = 0; i < 4; ++i)
	{
		dump_one(m_rtts.m_bound_render_targets[i].second, fmt::format("c%u", i));
	}

	dump_one(m_rtts.m_bound_depth_stencil.second, "z");
}

void MTLGSRender::dump_on_framebuffer_change()
{
	if (!m_dump)
	{
		return;
	}

	const auto& l = m_framebuffer_layout;
	const std::array<u32, 6> key =
	{
		l.color_addresses[0], l.color_addresses[1], l.color_addresses[2], l.color_addresses[3], l.zeta_address,
		static_cast<u32>(l.width) | (static_cast<u32>(l.height) << 16)
	};

	if (key == m_dump->pass_key)
	{
		return;
	}

	if (m_dump->draw)
	{
		dump_bound_surfaces("render targets changed");
	}

	m_dump->pass_key = key;
	m_dump->write(fmt::format("\n== new render targets: color [0x%x 0x%x 0x%x 0x%x] fmt %d, zeta 0x%x fmt %d, %ux%u, target %d\n",
		l.color_addresses[0], l.color_addresses[1], l.color_addresses[2], l.color_addresses[3], static_cast<int>(l.color_format),
		l.zeta_address, static_cast<int>(l.depth_format), l.width, l.height, static_cast<int>(l.target)));
}

void MTLGSRender::dump_clear(u32 arg)
{
	if (m_dump)
	{
		const auto& regs = rsx::method_registers;
		m_dump->write(fmt::format("  clear arg=0x%x color argb=(%u %u %u %u)\n", arg, regs.clear_color_a(), regs.clear_color_r(), regs.clear_color_g(), regs.clear_color_b()));
	}
}

void MTLGSRender::dump_draw(const std::vector<mtl::resource_binding>& fs_textures)
{
	if (!m_dump)
	{
		return;
	}

	auto& d = *m_dump;
	const u32 draw = d.draw++;
	const auto& regs = rsx::method_registers;
	const auto& p = m_pipeline_properties;

	d.write(fmt::format("draw %u: vp %u fp %u prim %d, blend en=0x%x factors=0x%08x ops=0x%x masks=0x%x, depth test=%d write=%d func=%d, alpha test=%d\n",
		draw, m_vertex_prog->id, m_fragment_prog->id, static_cast<int>(regs.current_draw_clause.primitive),
		p.blend_enable, p.blend_factors, p.blend_ops, p.write_masks,
		regs.depth_test_enabled(), regs.depth_write_enabled(), static_cast<int>(regs.depth_func()), regs.alpha_test_enabled()));

	for (const auto& [program, is_vertex] : { std::pair{ &m_vertex_prog->result, true }, std::pair{ &m_fragment_prog->result, false } })
	{
		const u32 key = (program->id << 1) | (is_vertex ? 1 : 0);
		if (d.dumped_programs.insert(key).second)
		{
			fs::write_file(d.dir + fmt::format("/programs/%s%u.glsl", is_vertex ? "vp" : "fp", program->id), fs::rewrite, program->glsl);
		}
	}

	usz binding = 0;
	for (const auto& res : m_fragment_prog->result.resources)
	{
		if (res.kind != mtl::resource_kind::texture)
		{
			continue;
		}

		const auto& b = fs_textures[binding++];

		u32 unit = 0;
		if (std::sscanf(res.name.c_str(), "tex%u", &unit) != 1 || unit >= rsx::limits::fragment_textures_count)
		{
			continue;
		}

		const auto& tex = regs.fragment_textures[unit];
		const auto state = static_cast<mtl::texture_cache::sampled_image_descriptor*>(fs_sampler_state[unit].get());

		std::string line = fmt::format("    %s (dim %d depth %d): rsx 0x%08x fmt 0x%02x %ux%u mips %u, class %d ctx %d remap 0x%x -> ",
			res.name, static_cast<int>(res.dimension), res.depth,
			rsx::get_address(tex.offset(), tex.location()), tex.format() & ~(CELL_GCM_TEXTURE_UN | CELL_GCM_TEXTURE_LN),
			tex.width(), tex.height(), tex.get_exact_mipmap_count(),
			state ? static_cast<int>(state->format_class) : -1, state ? static_cast<int>(state->upload_context) : -1, tex.remap());

		if (b.source != mtl::binding_source::texture)
		{
			line += "DUMMY";
		}
		else if (state && state->image_handle)
		{
			auto image = state->image_handle->image();
			line += fmt::format("image %s %ux%u", mtl::debug::format_name(image->format()), image->width(), image->height());

			if (d.images < max_texture_dumps && d.dumped_textures.insert(image).second)
			{
				const std::string name = fmt::format("textures/d%04u_t%u_0x%08x_%s_%ux%u", draw, unit,
					rsx::get_address(tex.offset(), tex.location()), mtl::debug::format_name(image->format()), image->width(), image->height());
				const auto result = mtl::debug::dump_texture(*image, d.dir + "/" + name, true);
				d.images++;
				line += fmt::format(" [%s: %s]", name, result);
			}
		}
		else
		{
			line += "temporary subresource";
		}

		d.write(line + "\n");
	}
}
