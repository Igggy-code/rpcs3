#include "stdafx.h"
#include "MTLTextureCache.h"

#include "Emu/RSX/Common/BufferUtils.h"
#include "Emu/RSX/Utils/image_utils.hpp"

#include "util/asm.hpp"

namespace mtl
{
	namespace
	{
		// Bytes per pixel of a surface/texture as stored in guest memory
		u32 get_guest_bpp(const texture* tex, u32 gcm_format)
		{
			switch (tex->format())
			{
			case pixel_format::depth32f_stencil8: return 4; // D24S8
			case pixel_format::depth32f: return 2;          // Z16 float
			default: break;
			}

			if (gcm_format)
			{
				return rsx::get_format_block_size_in_bytes(gcm_format);
			}

			return tex->block_size();
		}

		u16 float_to_half(f32 value)
		{
			const u32 bits = std::bit_cast<u32>(value);
			const u32 sign = (bits >> 16) & 0x8000;
			const s32 exponent = static_cast<s32>((bits >> 23) & 0xff) - 127 + 15;
			const u32 mantissa = bits & 0x7fffff;

			if (exponent <= 0)
			{
				return static_cast<u16>(sign);
			}

			if (exponent >= 31)
			{
				return static_cast<u16>(sign | 0x7c00);
			}

			return static_cast<u16>(sign | (exponent << 10) | (mantissa >> 13));
		}

		u64 encode_properties(pixel_format format, rsx::texture_dimension_extended type, u16 width, u16 height, u16 depth, u8 mipmaps)
		{
			return static_cast<u64>(width) |
				(static_cast<u64>(height) << 16) |
				((static_cast<u64>(depth) & 0x7ff) << 32) |
				((static_cast<u64>(mipmaps) & 0x1f) << 43) |
				((static_cast<u64>(type) & 0x7) << 48) |
				((static_cast<u64>(format) & 0xff) << 51);
		}

		u32 gcm_format_from_surface(const render_target* surface)
		{
			switch (surface->format())
			{
			case pixel_format::bgra8: return CELL_GCM_TEXTURE_A8R8G8B8;
			case pixel_format::rgba8: return CELL_GCM_TEXTURE_A8R8G8B8;
			case pixel_format::b5g6r5: return CELL_GCM_TEXTURE_R5G6B5;
			case pixel_format::bgr5a1: return CELL_GCM_TEXTURE_A1R5G5B5;
			case pixel_format::r8: return CELL_GCM_TEXTURE_B8;
			case pixel_format::rg8: return CELL_GCM_TEXTURE_G8B8;
			case pixel_format::rgba16f: return CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT;
			case pixel_format::rgba32f: return CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT;
			case pixel_format::r32f: return CELL_GCM_TEXTURE_X32_FLOAT;
			case pixel_format::depth16: return CELL_GCM_TEXTURE_DEPTH16;
			case pixel_format::depth32f: return CELL_GCM_TEXTURE_DEPTH16_FLOAT;
			case pixel_format::depth32f_stencil8:
				return surface->get_surface_depth_format() == rsx::surface_depth_format2::z24s8_float ? CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT : CELL_GCM_TEXTURE_DEPTH24_D8;
			default:
				return CELL_GCM_TEXTURE_A8R8G8B8;
			}
		}

		// Reads a region of 'src' into guest (big-endian) layout
		void read_guest_pixels(texture& src, u32 x, u32 y, u32 width, u32 height, u8* out, u32 out_pitch, u32 guest_bpp, bool is_float)
		{
			const u32 texels = width * height;

			switch (src.format())
			{
			case pixel_format::depth32f_stencil8:
			{
				std::vector<f32> depth_data(texels);
				std::vector<u8> stencil_data(texels);

				download_texture(src, 0, 0, x, y, width, height, image_aspect::depth, depth_data.data(), width * 4);
				download_texture(src, 0, 0, x, y, width, height, image_aspect::stencil, stencil_data.data(), width);

				for (u32 row = 0; row < height; ++row)
				{
					auto dst = reinterpret_cast<be_t<u32>*>(out + static_cast<usz>(row) * out_pitch);

					for (u32 col = 0; col < width; ++col)
					{
						const u32 index = row * width + col;
						const f32 d = std::clamp(depth_data[index], 0.f, 1.f);
						const u32 d24 = is_float ? (std::bit_cast<u32>(d) >> 7) & 0xffffff : static_cast<u32>(d * 16777215.f);
						dst[col] = (d24 << 8) | stencil_data[index];
					}
				}
				break;
			}

			case pixel_format::depth32f:
			{
				std::vector<f32> depth_data(texels);
				download_texture(src, 0, 0, x, y, width, height, image_aspect::depth, depth_data.data(), width * 4);

				for (u32 row = 0; row < height; ++row)
				{
					auto dst = reinterpret_cast<be_t<u16>*>(out + static_cast<usz>(row) * out_pitch);

					for (u32 col = 0; col < width; ++col)
					{
						dst[col] = float_to_half(depth_data[row * width + col]);
					}
				}
				break;
			}

			default:
			{
				download_texture(src, 0, 0, x, y, width, height, image_aspect::color, out, out_pitch);

				// Guest memory is big-endian
				u32 swap_size = 0;
				switch (src.format())
				{
				case pixel_format::bgra8:
				case pixel_format::rgba8:
				case pixel_format::rgba32f:
				case pixel_format::r32f:
					swap_size = 4;
					break;
				case pixel_format::b5g6r5:
				case pixel_format::bgr5a1:
				case pixel_format::a1bgr5:
				case pixel_format::abgr4:
				case pixel_format::rgba16f:
				case pixel_format::r16:
				case pixel_format::rg16:
				case pixel_format::rg16f:
				case pixel_format::depth16:
					swap_size = 2;
					break;
				default:
					break;
				}

				const u32 row_bytes = width * guest_bpp;
				for (u32 row = 0; row < height && swap_size; ++row)
				{
					u8* data = out + static_cast<usz>(row) * out_pitch;

					if (swap_size == 4)
					{
						copy_data_swap_u32(reinterpret_cast<u32*>(data), reinterpret_cast<const u32*>(data), row_bytes / 4);
					}
					else
					{
						auto words = reinterpret_cast<u16*>(data);
						for (u32 i = 0; i < row_bytes / 2; ++i)
						{
							words[i] = std::byteswap(words[i]);
						}
					}
				}
				break;
			}
			}
		}

		u32 get_layer_for(const texture* dst, u32 dst_z)
		{
			return dst->type() == texture_type::tex_cube ? dst_z : 0u;
		}

		u32 get_z_for(const texture* dst, u32 dst_z)
		{
			return dst->type() == texture_type::tex_3d ? dst_z : 0u;
		}
	}

	// ---------------------------------------------------------------------------------------------
	// blitter
	// ---------------------------------------------------------------------------------------------

	void blitter::scale_image(command_context&, texture* src, texture* dst, areai src_rect, areai dst_rect,
		bool linear_interpolation, const rsx::typeless_xfer& xfer_info)
	{
		std::unique_ptr<texture> typeless_src;
		std::unique_ptr<texture> typeless_dst;
		texture* real_src = src;
		texture* real_dst = dst;

		const bool flipped = xfer_info.flip_horizontal || xfer_info.flip_vertical ||
			src_rect.x2 < src_rect.x1 || src_rect.y2 < src_rect.y1 || dst_rect.x2 < dst_rect.x1 || dst_rect.y2 < dst_rect.y1;

		// Pass-through transfer
		if (!flipped && src_rect.height() == dst_rect.height())
		{
			auto src_w = src_rect.width();
			auto dst_w = dst_rect.width();

			if (xfer_info.src_is_typeless) src_w = static_cast<int>(src_w * xfer_info.src_scaling_hint);
			if (xfer_info.dst_is_typeless) dst_w = static_cast<int>(dst_w * xfer_info.dst_scaling_hint);

			if (src_w == dst_w)
			{
				if (xfer_info.src_is_typeless || xfer_info.dst_is_typeless || src->format() != dst->format())
				{
					if (copy_typeless(*src, src_rect.x1, src_rect.y1, src_rect.width(), src_rect.height(),
						*dst, dst_rect.x1, dst_rect.y1, dst_rect.width()))
					{
						return;
					}
				}
				else
				{
					const region3d region{ static_cast<u32>(src_rect.x1), static_cast<u32>(src_rect.y1), 0, static_cast<u32>(src_rect.width()), static_cast<u32>(src_rect.height()), 1 };
					copy_texture(*src, 0, 0, region, *dst, 0, 0, static_cast<u32>(dst_rect.x1), static_cast<u32>(dst_rect.y1), 0);
					return;
				}
			}
		}

		if (xfer_info.src_is_typeless)
		{
			const auto format = get_compatible_sampler_format(xfer_info.src_gcm_format);
			if (format != src->format())
			{
				const u32 internal_width = static_cast<u32>(src->width() * xfer_info.src_scaling_hint);
				typeless_src = std::make_unique<texture>(internal_width, src->height(), format, usage_sampled | usage_render_target);

				if (copy_typeless(*src, 0, 0, src->width(), src->height(), *typeless_src, 0, 0, internal_width))
				{
					real_src = typeless_src.get();
					src_rect.x1 = static_cast<int>(src_rect.x1 * xfer_info.src_scaling_hint);
					src_rect.x2 = static_cast<int>(src_rect.x2 * xfer_info.src_scaling_hint);
				}
			}
		}

		if (xfer_info.dst_is_typeless)
		{
			const auto format = get_compatible_sampler_format(xfer_info.dst_gcm_format);
			if (format != dst->format())
			{
				const u32 internal_width = static_cast<u32>(dst->width() * xfer_info.dst_scaling_hint);
				typeless_dst = std::make_unique<texture>(internal_width, dst->height(), format, usage_sampled | usage_render_target);

				if (copy_typeless(*dst, 0, 0, dst->width(), dst->height(), *typeless_dst, 0, 0, internal_width))
				{
					real_dst = typeless_dst.get();
					dst_rect.x1 = static_cast<int>(dst_rect.x1 * xfer_info.dst_scaling_hint);
					dst_rect.x2 = static_cast<int>(dst_rect.x2 * xfer_info.dst_scaling_hint);
				}
			}
		}

		if (xfer_info.flip_horizontal)
		{
			std::swap(src_rect.x1, src_rect.x2);
		}

		if (xfer_info.flip_vertical)
		{
			std::swap(src_rect.y1, src_rect.y2);
		}

		const blit_rect src_blit{ src_rect.x1, src_rect.y1, src_rect.x2 - src_rect.x1, src_rect.y2 - src_rect.y1 };
		const blit_rect dst_blit{ dst_rect.x1, dst_rect.y1, dst_rect.x2 - dst_rect.x1, dst_rect.y2 - dst_rect.y1 };

		if (!blit_texture(*real_src, 0, 0, src_blit, *real_dst, 0, 0, dst_blit, linear_interpolation && !real_dst->is_depth()))
		{
			rsx_log.todo("Metal: unsupported scaled blit (fmt %d -> %d)", static_cast<int>(real_src->format()), static_cast<int>(real_dst->format()));
		}

		if (typeless_dst)
		{
			// Transfer contents from the typeless destination back to the original
			copy_typeless(*typeless_dst, 0, 0, typeless_dst->width(), typeless_dst->height(), *dst, 0, 0, dst->width());
		}
	}

	// ---------------------------------------------------------------------------------------------
	// cached_texture_section
	// ---------------------------------------------------------------------------------------------

	void cached_texture_section::create(u16 w, u16 h, u16 depth, u16 mipmaps, texture* image, u32 rsx_pitch, bool managed)
	{
		auto new_texture = static_cast<viewable_image*>(image);
		ensure(!exists() || !is_managed() || vram_texture == new_texture);

		if (vram_texture && vram_texture != new_texture && !managed_texture && get_protection() == utils::protection::no)
		{
			// In-place image swap, still locked. Likely a color buffer that got rebound as depth buffer or vice-versa.
			as_rtt(vram_texture)->on_swap_out();

			if (!managed)
			{
				as_rtt(image)->on_swap_in(is_locked());
			}
		}

		vram_texture = new_texture;

		if (managed)
		{
			managed_texture.reset(vram_texture);
		}
		else
		{
			ensure(!managed_texture);
		}

		if (auto rtt = dynamic_cast<render_target*>(image))
		{
			swizzled = (rtt->raster_type != rsx::surface_raster_type::linear);
		}

		flushed = false;
		synchronized = false;
		sync_timestamp = 0ull;

		ensure(rsx_pitch);

		this->rsx_pitch = rsx_pitch;
		this->width = w;
		this->height = h;
		this->real_pitch = 0;
		this->depth = depth;
		this->mipmaps = mipmaps;

		baseclass::on_section_resources_created();
	}

	void cached_texture_section::create(u16 w, u16 h, u16 depth, u16 mipmaps, texture* image, u32 rsx_pitch, bool managed, const render_target* surface)
	{
		gcm_format = gcm_format_from_surface(surface);
		create(w, h, depth, mipmaps, image, rsx_pitch, managed);
	}

	void cached_texture_section::copy_texture(command_context& cmd, bool miss)
	{
		ensure(exists());

		if (!miss) [[likely]]
		{
			baseclass::on_speculative_flush();
		}
		else
		{
			baseclass::on_miss();
		}

		texture* target_texture = vram_texture;
		u32 transfer_width = width;
		u32 transfer_height = height;
		u32 transfer_x = 0, transfer_y = 0;

		std::unique_ptr<texture> scaled;

		if (context == rsx::texture_upload_context::framebuffer_storage)
		{
			auto surface = as_rtt(vram_texture);
			surface->memory_barrier(cmd, rsx::surface_access::transfer_read);
			transfer_width *= surface->samples_x;
			transfer_height *= surface->samples_y;
		}

		if (vram_texture->width() != transfer_width || vram_texture->height() != transfer_height)
		{
			// Resolution scaling: bring the image back to native size first
			scaled = std::make_unique<texture>(transfer_width, transfer_height, vram_texture->format(), usage_sampled | usage_render_target);

			const blit_rect src_rect{ 0, 0, static_cast<int>(vram_texture->width()), static_cast<int>(vram_texture->height()) };
			const blit_rect dst_rect{ 0, 0, static_cast<int>(transfer_width), static_cast<int>(transfer_height) };

			if (blit_texture(*vram_texture, 0, 0, src_rect, *scaled, 0, 0, dst_rect, !vram_texture->is_depth()))
			{
				target_texture = scaled.get();
			}
		}

		const u32 guest_bpp = get_guest_bpp(vram_texture, gcm_format);

		const auto valid_range = get_confirmed_range();
		if (const auto section_range = get_section_range(); section_range != valid_range)
		{
			if (const auto offset = (valid_range.start - get_section_base()))
			{
				transfer_y = offset / rsx_pitch;
				transfer_x = (offset % rsx_pitch) / guest_bpp;

				ensure(transfer_width >= transfer_x);
				ensure(transfer_height >= transfer_y);
				transfer_width -= transfer_x;
				transfer_height -= transfer_y;
			}

			if (const auto tail = (section_range.end - valid_range.end))
			{
				const auto row_count = tail / rsx_pitch;

				ensure(transfer_height >= row_count);
				transfer_height -= row_count;
			}
		}

		transfer_width = std::min(transfer_width, target_texture->width() - std::min(transfer_x, target_texture->width()));
		transfer_height = std::min(transfer_height, target_texture->height() - std::min(transfer_y, target_texture->height()));

		real_pitch = std::max(1u, width * (context == rsx::texture_upload_context::framebuffer_storage ? as_rtt(vram_texture)->samples_x : 1u)) * guest_bpp;
		m_flush_buffer.assign(static_cast<usz>(real_pitch) * std::max(transfer_height + transfer_y, 1u), 0);

		if (transfer_width && transfer_height)
		{
			u8* out = m_flush_buffer.data() + static_cast<usz>(transfer_y) * real_pitch + static_cast<usz>(transfer_x) * guest_bpp;
			read_guest_pixels(*target_texture, transfer_x, transfer_y, transfer_width, transfer_height, out, real_pitch, guest_bpp, gcm_format == CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT);
		}

		synchronized = true;
		sync_timestamp = rsx::get_shared_tag();
	}

	void cached_texture_section::dma_transfer(command_context&, texture* src, const areai& src_area, const utils::address_range32& valid_range, u32 pitch)
	{
		const u32 guest_bpp = get_guest_bpp(src, context == rsx::texture_upload_context::dma ? 0 : gcm_format);

		rsx_pitch = pitch;
		real_pitch = pitch;

		u32 offset = 0;
		if (valid_range.valid())
		{
			const u32 section_base = get_section_base();
			ensure(valid_range.start >= section_base);
			offset = valid_range.start - section_base;
		}

		const u32 x = static_cast<u32>(std::max(src_area.x1, 0));
		const u32 y = static_cast<u32>(std::max(src_area.y1, 0));
		const u32 w = std::min(static_cast<u32>(std::max(src_area.width(), 0)), src->width() - std::min(x, src->width()));
		const u32 h = std::min(static_cast<u32>(std::max(src_area.height(), 0)), src->height() - std::min(y, src->height()));

		m_flush_buffer.assign(std::max<usz>(get_section_size(), static_cast<usz>(offset) + static_cast<usz>(h) * pitch), 0);

		if (w && h)
		{
			read_guest_pixels(*src, x, y, std::min(w, pitch / guest_bpp), h, m_flush_buffer.data() + offset, pitch, guest_bpp, gcm_format == CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT);
		}

		synchronized = true;
		sync_timestamp = rsx::get_shared_tag();
	}

	void* cached_texture_section::map_synchronized(u32 offset, u32 size)
	{
		ensure(synchronized);
		ensure(static_cast<usz>(offset) + size <= m_flush_buffer.size());
		return m_flush_buffer.data() + offset;
	}

	void cached_texture_section::finish_flush()
	{
		if (!is_swizzled())
		{
			return;
		}

		// CPU readback of swizzled data: convert the linear copy back to the swizzled layout
		const auto valid_range = get_confirmed_range_delta();
		void* dst = get_ptr(get_section_base() + valid_range.first);

		std::vector<u8> tmp_data(static_cast<usz>(rsx_pitch) * height);
		std::memcpy(tmp_data.data(), dst, tmp_data.size());

		switch (get_guest_bpp(vram_texture, gcm_format))
		{
		case 4:
			rsx::convert_linear_swizzle<u32, false>(tmp_data.data(), dst, width, height, rsx_pitch);
			break;
		case 2:
			rsx::convert_linear_swizzle<u16, false>(tmp_data.data(), dst, width, height, rsx_pitch);
			break;
		default:
			rsx_log.error("Metal: unexpected swizzled texture readback (gcm format 0x%x)", gcm_format);
			break;
		}
	}

	void cached_texture_section::destroy()
	{
		if (!is_locked() && vram_texture == nullptr && !managed_texture && m_flush_buffer.empty())
		{
			// Already destroyed
			return;
		}

		m_flush_buffer = {};
		managed_texture.reset();
		vram_texture = nullptr;

		baseclass::on_section_resources_destroyed();
	}

	void cached_texture_section::sync_surface_memory(const rsx::simple_array<cached_texture_section*>& surfaces)
	{
		auto rtt = as_rtt(vram_texture);
		rtt->sync_tag();

		for (auto& surface : surfaces)
		{
			rtt->inherit_surface_contents(as_rtt(surface->vram_texture));
		}
	}

	// ---------------------------------------------------------------------------------------------
	// texture_cache
	// ---------------------------------------------------------------------------------------------

	void texture_cache::clear()
	{
		baseclass::clear();
		clear_temporary_subresources();
	}

	void texture_cache::clear_temporary_subresources()
	{
		m_temporary_surfaces.clear();
	}

	std::array<swizzle, 4> texture_cache::get_component_mapping_for(u32 gcm_format, rsx::component_order flags) const
	{
		switch (gcm_format)
		{
		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			return { swizzle::red, swizzle::red, swizzle::red, swizzle::red };
		default:
			break;
		}

		switch (flags)
		{
		case rsx::component_order::default_:
			return get_component_mapping(gcm_format);
		case rsx::component_order::native:
			return { swizzle::alpha, swizzle::red, swizzle::green, swizzle::blue };
		case rsx::component_order::swapped_native:
			return { swizzle::blue, swizzle::alpha, swizzle::red, swizzle::green };
		default:
			break;
		}

		fmt::throw_exception("Unknown texture create flags");
	}

	texture* texture_cache::get_template_from_collection_impl(const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const
	{
		if (sections_to_transfer.size() == 1) [[likely]]
		{
			return sections_to_transfer.front().src;
		}

		texture* result = nullptr;
		for (const auto& section : sections_to_transfer)
		{
			if (!section.src)
			{
				continue;
			}

			if (!result)
			{
				result = section.src;
			}
			else if (result->native_component_layout() != section.src->native_component_layout())
			{
				// Mixing channel layouts needs compute assistance
				return nullptr;
			}
		}

		return result;
	}

	texture_view* texture_cache::create_temporary_subresource_impl(command_context& cmd, texture* src, pixel_format format, rsx::texture_dimension_extended dst_type, u32 gcm_format,
		u16 width, u16 height, u16 depth, u8 mipmaps, const rsx::texture_channel_remap_t& remap, const copy_region_descriptor* copy)
	{
		if (format == pixel_format::invalid)
		{
			format = get_compatible_sampler_format(gcm_format);
		}

		temporary_image_t* dst = nullptr;
		const auto match_key = encode_properties(format, dst_type, width, height, depth, mipmaps);

		// Search image cache
		for (auto& e : m_temporary_surfaces)
		{
			if (e->has_refs())
			{
				continue;
			}

			if (e->properties_encoding == match_key)
			{
				dst = e.get();
				break;
			}
		}

		if (!dst)
		{
			texture_desc desc{};
			desc.type = get_texture_type(dst_type);
			desc.width = width;
			desc.height = height;
			desc.depth = depth;
			desc.levels = std::max<u8>(mipmaps, 1);
			desc.format = format;
			desc.usage = is_compressed_format(format) ? usage_sampled : (usage_sampled | usage_render_target);

			auto data = std::make_unique<temporary_image_t>(desc, rsx::classify_format(gcm_format));
			dst = data.get();
			dst->properties_encoding = match_key;
			dst->set_label(fmt::format("[Temp View] fmt=0x%x", gcm_format));
			m_temporary_surfaces.emplace_back(std::move(data));
		}

		dst->add_ref();

		if (copy)
		{
			rsx::simple_array<copy_region_descriptor> region = { *copy };
			copy_transfer_regions_impl(cmd, dst, region);
		}

		if (!src || src->format() != format)
		{
			// Apply the base component map onto the new texture if a data cast has been done
			dst->set_native_component_layout(get_component_mapping_for(gcm_format, rsx::component_order::default_));
		}
		else
		{
			dst->set_native_component_layout(src->native_component_layout());
		}

		return dst->get_view(remap);
	}

	void texture_cache::initialize_subresource_from_memory(command_context&, texture* dst, const deferred_subresource& desc, rsx::texture_dimension_extended type) const
	{
		const auto subresources_layout = rsx::get_subresources_layout(desc, type);
		mtl::upload_texture(dst, desc.gcm_format, desc.swizzled, subresources_layout);
	}

	void texture_cache::copy_transfer_regions_impl(command_context&, texture* dst_image, const rsx::simple_array<copy_region_descriptor>& sources) const
	{
		const u32 dst_bpp = get_guest_texel_size(dst_image->format());
		std::unique_ptr<texture> tmp;

		const auto is_depth_float = [](const texture* tex)
		{
			const auto image = dynamic_cast<const viewable_image*>(tex);
			return image && image->format_class() == rsx::RSX_FORMAT_CLASS_DEPTH24_FLOAT_X8_PACK32;
		};

		for (const auto& slice : sources)
		{
			if (!slice.src)
			{
				continue;
			}

			texture* src_image = slice.src;
			u32 src_x = slice.src_x;
			u32 src_y = slice.src_y;
			u32 src_w = slice.src_w;
			u32 src_h = slice.src_h;

			const u32 src_bpp = get_guest_texel_size(slice.src->format());

			if (slice.xform == rsx::surface_transform::coordinate_transform)
			{
				// Dimensions were given in 'dst' space. Work out the real source coordinates
				src_x = (src_x * dst_bpp) / src_bpp;
				src_w = utils::aligned_div<u32>(src_w * dst_bpp, src_bpp);
			}

			if (auto surface = dynamic_cast<render_target*>(slice.src))
			{
				surface->transform_samples_to_pixels(src_x, src_w, src_y, src_h);
			}

			const u32 dst_layer = get_layer_for(dst_image, slice.dst_z);
			const u32 dst_z = get_z_for(dst_image, slice.dst_z);

			if (slice.src->format() != dst_image->format())
			{
				const typeless_options options
				{
					.src_depth_float = is_depth_float(slice.src),
					.dst_depth_float = is_depth_float(dst_image),
				};

				const u32 src_w2 = (src_w * src_bpp) / dst_bpp;
				const u32 src_x2 = (src_x * src_bpp) / dst_bpp;

				if (src_w2 == slice.dst_w && src_h == slice.dst_h && slice.level == 0 && dst_layer == 0 && dst_z == 0 &&
					dst_image->type() == texture_type::tex_2d)
				{
					// Bit-cast straight into the destination
					if (!copy_typeless(*slice.src, src_x, src_y, src_w, src_h, *dst_image, slice.dst_x, slice.dst_y, slice.dst_w, options))
					{
						rsx_log.todo("Metal: unsupported typeless region copy (fmt %d -> %d)", static_cast<int>(slice.src->format()), static_cast<int>(dst_image->format()));
					}
					continue;
				}

				const u32 convert_w = (slice.src->width() * src_bpp) / dst_bpp;
				if (!tmp || tmp->width() < convert_w || tmp->height() < slice.src->height() || tmp->format() != dst_image->format())
				{
					tmp = std::make_unique<texture>(convert_w, slice.src->height(), dst_image->format(), usage_sampled | usage_render_target);
				}

				if (!copy_typeless(*slice.src, src_x, src_y, src_w, src_h, *tmp, src_x2, src_y, src_w2, options))
				{
					rsx_log.todo("Metal: unsupported typeless region copy (fmt %d -> %d)", static_cast<int>(slice.src->format()), static_cast<int>(dst_image->format()));
					continue;
				}

				src_image = tmp.get();
				src_x = src_x2;
				src_w = src_w2;
			}

			if (src_w == slice.dst_w && src_h == slice.dst_h)
			{
				const region3d region{ src_x, src_y, 0, src_w, src_h, 1 };
				copy_texture(*src_image, 0, 0, region, *dst_image, slice.level, dst_layer, slice.dst_x, slice.dst_y, dst_z);
			}
			else
			{
				const blit_rect src_rect{ static_cast<int>(src_x), static_cast<int>(src_y), static_cast<int>(src_w), static_cast<int>(src_h) };
				const blit_rect dst_rect{ slice.dst_x, slice.dst_y, slice.dst_w, slice.dst_h };

				if (!blit_texture(*src_image, 0, 0, src_rect, *dst_image, slice.level, dst_layer, dst_rect, false))
				{
					rsx_log.todo("Metal: unsupported scaled region copy (fmt %d -> %d)", static_cast<int>(src_image->format()), static_cast<int>(dst_image->format()));
				}
			}
		}
	}

	texture_view* texture_cache::create_temporary_subresource_view(command_context& cmd, const deferred_subresource& desc)
	{
		ensure(desc.sections_to_copy.size() == 1);
		const auto& section = desc.sections_to_copy.front();
		return create_temporary_subresource_impl(cmd, section.src, pixel_format::invalid,
			rsx::texture_dimension_extended::texture_dimension_2d, desc.gcm_format,
			desc.width, desc.height, 1, 1, desc.remap, &section);
	}

	texture_view* texture_cache::generate_cubemap_from_images(command_context& cmd, const deferred_subresource& desc)
	{
		auto _template = get_template_from_collection_impl(desc.sections_to_copy);
		auto result = create_temporary_subresource_impl(cmd, _template, pixel_format::invalid, rsx::texture_dimension_extended::texture_dimension_cubemap,
			desc.gcm_format, desc.width, desc.height, 1, desc.exact_mip_count(), desc.remap);

		if (desc.force_bg_load)
		{
			initialize_subresource_from_memory(cmd, result->image(), desc, rsx::texture_dimension_extended::texture_dimension_cubemap);
		}

		copy_transfer_regions_impl(cmd, result->image(), desc.sections_to_copy);
		return result;
	}

	texture_view* texture_cache::generate_3d_from_2d_images(command_context& cmd, const deferred_subresource& desc)
	{
		auto _template = get_template_from_collection_impl(desc.sections_to_copy);
		auto result = create_temporary_subresource_impl(cmd, _template, pixel_format::invalid, rsx::texture_dimension_extended::texture_dimension_3d,
			desc.gcm_format, desc.width, desc.height, desc.depth, desc.exact_mip_count(), desc.remap);

		if (desc.force_bg_load)
		{
			initialize_subresource_from_memory(cmd, result->image(), desc, rsx::texture_dimension_extended::texture_dimension_3d);
		}

		copy_transfer_regions_impl(cmd, result->image(), desc.sections_to_copy);
		return result;
	}

	texture_view* texture_cache::generate_atlas_from_images(command_context& cmd, const deferred_subresource& desc)
	{
		auto _template = get_template_from_collection_impl(desc.sections_to_copy);
		auto result = create_temporary_subresource_impl(cmd, _template, pixel_format::invalid, rsx::texture_dimension_extended::texture_dimension_2d,
			desc.gcm_format, desc.width, desc.height, 1, 1, desc.remap);

		if (desc.force_bg_load)
		{
			initialize_subresource_from_memory(cmd, result->image(), desc, rsx::texture_dimension_extended::texture_dimension_2d);
		}

		copy_transfer_regions_impl(cmd, result->image(), desc.sections_to_copy);
		return result;
	}

	texture_view* texture_cache::generate_2d_mipmaps_from_images(command_context& cmd, const deferred_subresource& desc)
	{
		const auto mipmaps = ::narrow<u8>(desc.sections_to_copy.size());
		auto _template = get_template_from_collection_impl(desc.sections_to_copy);
		auto result = create_temporary_subresource_impl(cmd, _template, pixel_format::invalid, rsx::texture_dimension_extended::texture_dimension_2d,
			desc.gcm_format, desc.width, desc.height, 1, mipmaps, desc.remap);

		if (desc.force_bg_load)
		{
			initialize_subresource_from_memory(cmd, result->image(), desc, rsx::texture_dimension_extended::texture_dimension_2d);
		}

		copy_transfer_regions_impl(cmd, result->image(), desc.sections_to_copy);
		return result;
	}

	void texture_cache::release_temporary_subresource(texture_view* view)
	{
		for (auto& e : m_temporary_surfaces)
		{
			if (e.get() == view->image())
			{
				e->release();
				return;
			}
		}
	}

	void texture_cache::update_image_contents(command_context& cmd, texture_view* dst, const deferred_subresource& desc)
	{
		copy_transfer_regions_impl(cmd, dst->image(), desc.sections_to_copy);
	}

	cached_texture_section* texture_cache::create_new_texture(command_context& /*cmd*/, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch,
		u32 gcm_format, rsx::texture_upload_context context, rsx::texture_dimension_extended type, bool swizzled, rsx::component_order swizzle_flags, rsx::flags32_t /*flags*/)
	{
		const rsx::image_section_attributes_t search_desc = { .gcm_format = gcm_format, .width = width, .height = height, .depth = depth, .mipmaps = mipmaps };
		const bool allow_dirty = (context != rsx::texture_upload_context::framebuffer_storage);
		auto& cached = *find_cached_texture(rsx_range, search_desc, true, true, allow_dirty);
		ensure(!cached.is_locked());

		viewable_image* image = nullptr;
		if (cached.exists())
		{
			// Try and reuse this image data. It is very likely to match our needs
			image = cached.get_raw_texture();
			if (!image || cached.get_image_type() != type)
			{
				// Type mismatch, discard
				cached.destroy();
				image = nullptr;
			}
			else
			{
				ensure(cached.is_managed());
				cached.set_dimensions(width, height, depth, pitch);

				// Clear the image before use if it is not going to be uploaded wholly from CPU
				if (context != rsx::texture_upload_context::shader_read)
				{
					if (image->is_depth())
					{
						clear_depth_stencil(*image, true, 1.f, has_stencil(image->format()), 0);
					}
					else
					{
						const float zero[4] = {};
						clear_color(*image, zero, 0xf);
					}
				}
			}
		}

		if (!image)
		{
			ensure(!cached.exists());
			image = create_texture(gcm_format, width, height, depth, mipmaps, type).release();

			// Prepare section
			cached.reset(rsx_range);
			cached.set_image_type(type);
			cached.set_gcm_format(gcm_format);
			cached.create(width, height, depth, mipmaps, image, pitch, true);
		}

		cached.set_view_flags(swizzle_flags);
		cached.set_context(context);
		cached.set_swizzled(swizzled);
		cached.set_dirty(false);

		image->set_native_component_layout(get_component_mapping_for(gcm_format, swizzle_flags));

		if (context != rsx::texture_upload_context::blit_engine_dst)
		{
			AUDIT(cached.get_memory_read_flags() != rsx::memory_read_flags::flush_always);
			read_only_range = cached.get_min_max(read_only_range, rsx::section_bounds::locked_range);
			cached.protect(utils::protection::ro);
		}
		else
		{
			switch (gcm_format)
			{
			case CELL_GCM_TEXTURE_A8R8G8B8:
			case CELL_GCM_TEXTURE_R5G6B5:
			case CELL_GCM_TEXTURE_DEPTH24_D8:
			case CELL_GCM_TEXTURE_DEPTH16:
				break;
			default:
				fmt::throw_exception("Unexpected gcm format 0x%X", gcm_format);
			}

			// NOTE: Protection is handled by the caller
			no_access_range = cached.get_min_max(no_access_range, rsx::section_bounds::locked_range);
		}

		update_cache_tag();
		return &cached;
	}

	cached_texture_section* texture_cache::create_nul_section(command_context&, const utils::address_range32& rsx_range, const rsx::image_section_attributes_t& attrs,
		const rsx::GCM_tile_reference&, bool)
	{
		auto& cached = *find_cached_texture(rsx_range, { .gcm_format = RSX_GCM_FORMAT_IGNORED }, true, false, false);
		ensure(!cached.is_locked());

		// Prepare section
		cached.reset(rsx_range);
		cached.create_dma_only(attrs.width, attrs.height, attrs.pitch);
		cached.set_dirty(false);

		no_access_range = cached.get_min_max(no_access_range, rsx::section_bounds::locked_range);
		update_cache_tag();
		return &cached;
	}

	cached_texture_section* texture_cache::upload_image_from_cpu(command_context& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch, u32 gcm_format,
		rsx::texture_upload_context context, const std::vector<rsx::subresource_layout>& subresource_layout, rsx::texture_dimension_extended type, bool input_swizzled)
	{
		auto section = create_new_texture(cmd, rsx_range, width, height, depth, mipmaps, pitch, gcm_format, context, type, input_swizzled,
			rsx::component_order::default_, 0);

		mtl::upload_texture(section->get_raw_texture(), gcm_format, input_swizzled, subresource_layout);

		section->get_raw_texture()->set_label(fmt::format("Raw Texture @0x%x", rsx_range.start));
		section->last_write_tag = rsx::get_shared_tag();
		return section;
	}

	void texture_cache::set_component_order(cached_texture_section& section, u32 gcm_format, rsx::component_order flags)
	{
		if (flags == section.get_view_flags())
		{
			return;
		}

		auto image = section.get_raw_texture();
		ensure(image);
		image->set_native_component_layout(get_component_mapping_for(gcm_format, flags));

		section.set_view_flags(flags);
	}

	bool texture_cache::render_target_format_is_compatible(texture* tex, u32 gcm_format)
	{
		const auto format = tex->format();

		switch (gcm_format)
		{
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT:
			return format == pixel_format::rgba16f;
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT:
			return format == pixel_format::rgba32f;
		case CELL_GCM_TEXTURE_X32_FLOAT:
			return format == pixel_format::r32f;
		case CELL_GCM_TEXTURE_R5G6B5:
			return format == pixel_format::b5g6r5;
		case CELL_GCM_TEXTURE_A8R8G8B8:
		case CELL_GCM_TEXTURE_D8R8G8B8:
			// Depth data read as ARGB8 is reconstructed in the shader from the depth and stencil views
			return format == pixel_format::bgra8 || format == pixel_format::depth32f_stencil8;
		case CELL_GCM_TEXTURE_B8:
			return format == pixel_format::r8;
		case CELL_GCM_TEXTURE_G8B8:
			return format == pixel_format::rg8;
		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
			return format == pixel_format::depth32f_stencil8;
		case CELL_GCM_TEXTURE_X16:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			return format == pixel_format::depth16 || format == pixel_format::depth32f;
		default:
			return false;
		}
	}

	bool texture_cache::is_depth_texture(u32 rsx_address, u32 rsx_size)
	{
		reader_lock lock(m_cache_mutex);

		auto& block = m_storage.block_for(rsx_address);

		if (block.get_locked_count() == 0)
		{
			return false;
		}

		for (auto& tex : block)
		{
			if (tex.is_dirty())
			{
				continue;
			}

			if (!tex.overlaps(rsx_address, rsx::section_bounds::full_range))
			{
				continue;
			}

			if ((rsx_address + rsx_size - tex.get_section_base()) <= tex.get_section_size())
			{
				return tex.is_depth_texture();
			}
		}

		return false;
	}

	void texture_cache::on_frame_end()
	{
		trim_sections();

		if (m_storage.m_unreleased_texture_objects >= m_max_zombie_objects)
		{
			purge_unreleased_sections();
		}

		if (m_temporary_surfaces.size() > max_cached_image_pool_size)
		{
			m_temporary_surfaces.resize(max_cached_image_pool_size / 2);
		}

		baseclass::on_frame_end();
	}

	bool texture_cache::blit(command_context& cmd, const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool linear_interpolate, mtl_render_targets& rtts)
	{
		blitter helper;
		auto result = upload_scaled_image(src, dst, linear_interpolate, cmd, rtts, helper);

		if (result.succeeded)
		{
			if (result.dst_range.valid())
			{
				flush_if_cache_miss_likely(cmd, result.dst_range);
			}

			return true;
		}

		return false;
	}
}
