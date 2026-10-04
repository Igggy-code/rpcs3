#include "stdafx.h"
#include "MTLRenderTargets.h"
#include "MTLTextureOps.h"

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

	std::array<swizzle, 4> surface_color_component_layout(rsx::surface_color_format format)
	{
		using enum swizzle;

		switch (format)
		{
		case rsx::surface_color_format::x1r5g5b5_o1r5g5b5:
		case rsx::surface_color_format::x8r8g8b8_o8r8g8b8:
		case rsx::surface_color_format::x8b8g8r8_o8b8g8r8:
			return { one, red, green, blue };
		case rsx::surface_color_format::x1r5g5b5_z1r5g5b5:
		case rsx::surface_color_format::x8r8g8b8_z8r8g8b8:
		case rsx::surface_color_format::x8b8g8r8_z8b8g8r8:
			return { zero, red, green, blue };
		case rsx::surface_color_format::b8:
			return { one, red, red, red };
		case rsx::surface_color_format::g8b8:
			return { green, red, green, red };
		case rsx::surface_color_format::x32:
			return { red, red, red, red };
		default:
			return { alpha, red, green, blue };
		}
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

	void render_target::transfer_contents(render_target& src, const areai& src_area, const areai& dst_area)
	{
		const bool same_size = src_area.width() == dst_area.width() && src_area.height() == dst_area.height();

		if (src.format() == format())
		{
			if (same_size)
			{
				copy_region(src, *this, src_area.x1, src_area.y1, dst_area.x1, dst_area.y1, src_area.width(), src_area.height());
				return;
			}

			const blit_rect src_rect{ src_area.x1, src_area.y1, src_area.width(), src_area.height() };
			const blit_rect dst_rect{ dst_area.x1, dst_area.y1, dst_area.width(), dst_area.height() };
			if (blit_texture(src, 0, 0, src_rect, *this, 0, 0, dst_rect, !is_depth()))
			{
				return;
			}
		}
		else
		{
			// Same memory viewed with another format: reinterpret the guest bytes. Rows keep their byte width.
			const u32 src_bpp = get_guest_texel_size(src.format());
			const u32 dst_bpp = get_guest_texel_size(format());
			const u32 row_bytes = static_cast<u32>(src_area.width()) * src_bpp;

			const typeless_options options
			{
				.src_depth_float = src.format_class() == rsx::RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32,
				.dst_depth_float = format_class() == rsx::RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32,
			};

			if (src_area.height() == dst_area.height() && row_bytes % dst_bpp == 0 &&
				copy_typeless(src, src_area.x1, src_area.y1, src_area.width(), src_area.height(),
					*this, dst_area.x1, dst_area.y1, row_bytes / dst_bpp, options))
			{
				return;
			}
		}

		rsx_log.todo("Metal: unsupported surface transfer (fmt %d -> %d, %dx%d -> %dx%d)",
			static_cast<int>(src.format()), static_cast<int>(format()),
			src_area.width(), src_area.height(), dst_area.width(), dst_area.height());
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

			transfer_contents(*src_texture, src_area, dst_area);

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
