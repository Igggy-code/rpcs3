#include "stdafx.h"
#include "MTLRenderTargets.h"

namespace mtl
{
	pixel_format surface_color_format_to_mtl(rsx::surface_color_format format)
	{
		switch (format)
		{
		case rsx::surface_color_format::a8r8g8b8:
		case rsx::surface_color_format::x8r8g8b8_z8r8g8b8:
		case rsx::surface_color_format::x8r8g8b8_o8r8g8b8:
			return pixel_format::bgra8;
		case rsx::surface_color_format::a8b8g8r8:
		case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
		case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
			return pixel_format::rgba8;
		case rsx::surface_color_format::r5g6b5:
			return pixel_format::b5g6r5;
		case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
		case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
			return pixel_format::bgr5a1;
		case rsx::surface_color_format::b8:
			return pixel_format::r8;
		case rsx::surface_color_format::g8b8:
			return pixel_format::rg8;
		case rsx::surface_color_format::w16z16y16x16:
			return pixel_format::rgba16f;
		case rsx::surface_color_format::w32z32y32x32:
			return pixel_format::rgba32f;
		case rsx::surface_color_format::x32:
			return pixel_format::r32f;
		}

		fmt::throw_exception("Unknown surface color format 0x%x", static_cast<u32>(format));
	}

	pixel_format surface_depth_format_to_mtl(rsx::surface_depth_format2 format)
	{
		switch (format)
		{
		case rsx::surface_depth_format2::z16_uint:
			return pixel_format::depth16;
		case rsx::surface_depth_format2::z16_float:
			return pixel_format::depth32f;
		case rsx::surface_depth_format2::z24s8_uint:
		case rsx::surface_depth_format2::z24s8_float:
			return pixel_format::depth32f_stencil8;
		}

		fmt::throw_exception("Unknown surface depth format 0x%x", static_cast<u32>(format));
	}

	void render_target::initialize_memory(command_context& /*cmd*/, rsx::surface_access /*access*/)
	{
		// TODO: load from guest memory when Read Color/Depth Buffers is enabled (needs format conversion uploads)
		if (is_depth_surface())
		{
			clear_depth_stencil(*this, true, 1.f, has_stencil(format()), 0xff);
		}
		else
		{
			const float zero[4] = {};
			clear_color(*this, zero, 0xf);
		}

		state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
		msaa_flags = rsx::surface_state_flags::ready;
	}

	void render_target::memory_barrier(command_context& cmd, rsx::surface_access access)
	{
		if (access == rsx::surface_access::gpu_reference)
		{
			// Metal resources are always resident
			return;
		}

		const bool read_access = access.is_read();

		if (old_contents.empty())
		{
			// No memory to inherit
			if (dirty() && (read_access || state_flags & rsx::surface_state_flags::erase_bkgnd))
			{
				initialize_memory(cmd, access);
				on_write();
			}

			return;
		}

		const unsigned first = prepare_rw_barrier_for_transfer(this);
		u64 newest_tag = 0;

		for (auto i = first; i < old_contents.size(); ++i)
		{
			auto& section = old_contents[i];
			auto src_texture = as_rtt(section.source);

			src_texture->memory_barrier(cmd, rsx::surface_access::transfer_read);

			section.init_transfer(this);
			const auto src_area = section.src_rect();
			const auto dst_area = section.dst_rect();

			if (dst_area.x1 == 0 && dst_area.y1 == 0 &&
				unsigned(dst_area.x2) == width() && unsigned(dst_area.y2) == height())
			{
				// Full overwrite, nothing to initialize
				state_flags &= ~rsx::surface_state_flags::erase_bkgnd;
				msaa_flags = rsx::surface_state_flags::ready;
				stencil_init_flags = src_texture->stencil_init_flags;
			}
			else if (state_flags & rsx::surface_state_flags::erase_bkgnd)
			{
				initialize_memory(cmd, rsx::surface_access::memory_write);
			}

			// TODO: scaled and typeless (format-changing) transfers need a draw-based blit
			if (src_texture->format() != format() || src_area.width() != dst_area.width() || src_area.height() != dst_area.height())
			{
				rsx_log.todo("Metal: unsupported surface transfer (fmt %d -> %d, %dx%d -> %dx%d)",
					static_cast<int>(src_texture->format()), static_cast<int>(format()),
					src_area.width(), src_area.height(), dst_area.width(), dst_area.height());
			}
			else
			{
				copy_region(*src_texture, *this, src_area.x1, src_area.y1, dst_area.x1, dst_area.y1, src_area.width(), src_area.height());
			}

			newest_tag = src_texture->last_use_tag;
		}

		if (!newest_tag) [[unlikely]]
		{
			// Underlying memory has been modified and we could not find valid data to fill it
			clear_rw_barrier();

			state_flags |= rsx::surface_state_flags::erase_bkgnd;
			initialize_memory(cmd, access);
		}

		on_write_copy(newest_tag);
	}
}
