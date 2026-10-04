#include "stdafx.h"
#include "MTLTexture.h"
#include "MTLDraw.h"

#include "Emu/RSX/gcm_enums.h"
#include "Emu/RSX/Common/io_buffer.h"
#include "Emu/RSX/Utils/color_utils.hpp"

#include <cstring>

namespace mtl
{
	viewable_image::viewable_image(const texture_desc& desc, rsx::format_class format_class)
		: texture(desc), m_format_class(format_class)
	{
	}

	viewable_image::viewable_image(u32 width, u32 height, pixel_format format, u32 usage, rsx::format_class format_class)
		: texture(width, height, format, usage), m_format_class(format_class)
	{
	}

	texture_view* viewable_image::get_view(const rsx::texture_channel_remap_t& remap, image_aspect aspect)
	{
		if (is_depth() || aspect == image_aspect::stencil)
		{
			// Swizzles do not apply to depth/stencil sampling
			return texture::get_view(identity_swizzle, aspect, remap.encoded);
		}

		return texture::get_view(apply_swizzle_remap(native_component_layout(), remap), aspect, remap.encoded);
	}

	texture_view* viewable_image::get_raw_view()
	{
		return texture::get_view(identity_swizzle, image_aspect::color, 0xFFFFFFFFu);
	}

	pixel_format get_compatible_sampler_format(u32 gcm_format)
	{
		switch (gcm_format)
		{
		case CELL_GCM_TEXTURE_R5G6B5: return pixel_format::b5g6r5;
		case CELL_GCM_TEXTURE_R6G5B5: return pixel_format::b5g6r5; // Expanded to 565 by the uploader
		case CELL_GCM_TEXTURE_R5G5B5A1: return pixel_format::a1bgr5;
		case CELL_GCM_TEXTURE_D1R5G5B5: return pixel_format::bgr5a1;
		case CELL_GCM_TEXTURE_A1R5G5B5: return pixel_format::bgr5a1;
		case CELL_GCM_TEXTURE_A4R4G4B4: return pixel_format::abgr4;
		case CELL_GCM_TEXTURE_B8: return pixel_format::r8;
		case CELL_GCM_TEXTURE_A8R8G8B8: return pixel_format::bgra8;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1: return pixel_format::bc1;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23: return pixel_format::bc2;
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45: return pixel_format::bc3;
		case CELL_GCM_TEXTURE_G8B8: return pixel_format::rg8;
		case CELL_GCM_TEXTURE_DEPTH24_D8: return pixel_format::depth32f_stencil8;
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT: return pixel_format::depth32f_stencil8;
		case CELL_GCM_TEXTURE_DEPTH16: return pixel_format::depth16;
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT: return pixel_format::depth32f;
		case CELL_GCM_TEXTURE_X16: return pixel_format::r16;
		case CELL_GCM_TEXTURE_Y16_X16: return pixel_format::rg16;
		case CELL_GCM_TEXTURE_Y16_X16_FLOAT: return pixel_format::rg16f;
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT: return pixel_format::rgba16f;
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT: return pixel_format::rgba32f;
		case CELL_GCM_TEXTURE_X32_FLOAT: return pixel_format::r32f;
		case CELL_GCM_TEXTURE_D8R8G8B8: return pixel_format::bgra8;
		case CELL_GCM_TEXTURE_COMPRESSED_HILO8: return pixel_format::rg8;
		case CELL_GCM_TEXTURE_COMPRESSED_HILO_S8: return pixel_format::rg8_snorm;
		case CELL_GCM_TEXTURE_COMPRESSED_B8R8_G8R8: return pixel_format::bgra8;
		case CELL_GCM_TEXTURE_COMPRESSED_R8B8_R8G8: return pixel_format::bgra8;
		default:
			break;
		}

		fmt::throw_exception("Invalid or unsupported sampler format for texture format (0x%x)", gcm_format);
	}

	std::array<swizzle, 4> get_component_mapping(u32 gcm_format)
	{
		using enum swizzle;

		// ARGB order, same as vk::get_component_mapping
		switch (gcm_format)
		{
		case CELL_GCM_TEXTURE_A1R5G5B5:
		case CELL_GCM_TEXTURE_R5G5B5A1:
		case CELL_GCM_TEXTURE_R6G5B5:
		case CELL_GCM_TEXTURE_R5G6B5:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT1:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT23:
		case CELL_GCM_TEXTURE_COMPRESSED_DXT45:
		case CELL_GCM_TEXTURE_W16_Z16_Y16_X16_FLOAT:
		case CELL_GCM_TEXTURE_W32_Z32_Y32_X32_FLOAT:
		case CELL_GCM_TEXTURE_COMPRESSED_B8R8_G8R8:
		case CELL_GCM_TEXTURE_COMPRESSED_R8B8_R8G8:
		case CELL_GCM_TEXTURE_A8R8G8B8:
			return { alpha, red, green, blue };

		case CELL_GCM_TEXTURE_DEPTH24_D8:
		case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
		case CELL_GCM_TEXTURE_DEPTH16:
		case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
		case CELL_GCM_TEXTURE_X32_FLOAT:
			return { red, red, red, red };

		case CELL_GCM_TEXTURE_A4R4G4B4:
			return { red, green, blue, alpha };

		case CELL_GCM_TEXTURE_G8B8:
		case CELL_GCM_TEXTURE_Y16_X16:
			return { green, red, green, red };

		case CELL_GCM_TEXTURE_B8:
			return { one, red, red, red };

		case CELL_GCM_TEXTURE_X16:
			return { red, one, red, one };

		case CELL_GCM_TEXTURE_Y16_X16_FLOAT:
		case CELL_GCM_TEXTURE_COMPRESSED_HILO8:
		case CELL_GCM_TEXTURE_COMPRESSED_HILO_S8:
			return { red, green, red, green };

		case CELL_GCM_TEXTURE_D8R8G8B8:
		case CELL_GCM_TEXTURE_D1R5G5B5:
			return { one, red, green, blue };

		default:
			break;
		}

		fmt::throw_exception("Invalid or unsupported component mapping for texture format (0x%x)", gcm_format);
	}

	swizzle_rgba apply_swizzle_remap(const std::array<swizzle, 4>& native_layout, const rsx::texture_channel_remap_t& remap)
	{
		std::array<swizzle, 4> argb = native_layout;

		if (remap.encoded != RSX_TEXTURE_REMAP_IDENTITY)
		{
			argb = remap.remap<swizzle>(native_layout, swizzle::zero, swizzle::one);
		}

		return { argb[1], argb[2], argb[3], argb[0] };
	}

	texture_type get_texture_type(rsx::texture_dimension_extended type)
	{
		switch (type)
		{
		case rsx::texture_dimension_extended::texture_dimension_1d: return texture_type::tex_1d;
		case rsx::texture_dimension_extended::texture_dimension_2d: return texture_type::tex_2d;
		case rsx::texture_dimension_extended::texture_dimension_3d: return texture_type::tex_3d;
		case rsx::texture_dimension_extended::texture_dimension_cubemap: return texture_type::tex_cube;
		default: break;
		}

		return texture_type::tex_2d;
	}

	std::unique_ptr<viewable_image> create_texture(u32 gcm_format, u16 width, u16 height, u16 depth, u16 mipmaps, rsx::texture_dimension_extended type)
	{
		texture_desc desc{};
		desc.type = get_texture_type(type);
		desc.width = std::max<u16>(width, 1);
		desc.height = std::max<u16>(height, 1);
		desc.depth = std::max<u16>(depth, 1);
		desc.levels = std::max<u16>(mipmaps, 1);
		desc.format = get_compatible_sampler_format(gcm_format);
		desc.usage = usage_sampled | usage_render_target;

		if (is_compressed_format(desc.format))
		{
			// Block-compressed textures cannot be render targets
			desc.usage = usage_sampled;
		}

		auto result = std::make_unique<viewable_image>(desc, rsx::classify_format(gcm_format));
		result->set_native_component_layout(get_component_mapping(gcm_format));
		return result;
	}

	namespace
	{
		f32 half_to_float(u16 h)
		{
			const u32 sign = (h & 0x8000u) << 16;
			u32 exponent = (h >> 10) & 0x1f;
			u32 mantissa = h & 0x3ff;

			u32 bits;
			if (exponent == 0)
			{
				if (mantissa == 0)
				{
					bits = sign;
				}
				else
				{
					// Denormal: normalize
					exponent = 127 - 15 + 1;
					while (!(mantissa & 0x400))
					{
						mantissa <<= 1;
						exponent--;
					}
					mantissa &= 0x3ff;
					bits = sign | (exponent << 23) | (mantissa << 13);
				}
			}
			else if (exponent == 0x1f)
			{
				bits = sign | 0x7f800000u | (mantissa << 13);
			}
			else
			{
				bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
			}

			return std::bit_cast<f32>(bits);
		}

		// Bytes per texel/block of the data produced by upload_texture_subresource
		u32 get_decoded_block_size(u32 gcm_format, pixel_format host_format)
		{
			switch (gcm_format)
			{
			case CELL_GCM_TEXTURE_DEPTH24_D8:
			case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
				return 4;
			case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
				return 2;
			default:
				return get_format_block_size(host_format);
			}
		}
	}

	void upload_texture(texture* dst, u32 gcm_format, bool is_swizzled, const std::vector<rsx::subresource_layout>& subresources_layout)
	{
		if (!dst || !dst->valid())
		{
			return;
		}

		const pixel_format host_format = dst->format();
		const bool compressed = is_compressed_format(host_format);
		const u32 block_size = get_decoded_block_size(gcm_format, host_format);

		for (const rsx::subresource_layout& layout : subresources_layout)
		{
			if (layout.level >= dst->levels())
			{
				continue;
			}

			const u32 units_x = compressed ? layout.width_in_block : layout.width_in_texel;
			const u32 rows = compressed ? layout.height_in_block : layout.height_in_texel;
			const u32 depth = std::max<u32>(layout.depth, 1);
			const u32 row_pitch = units_x * block_size;
			const u32 image_size = row_pitch * rows;
			const u32 total_size = image_size * depth;

			if (!total_size)
			{
				continue;
			}

			auto staging = ring_alloc(total_size + 16, 256);
			if (!staging.ptr)
			{
				rsx_log.error("Metal: texture upload of %u bytes does not fit the data ring", total_size);
				continue;
			}

			rsx::io_buffer io_buf(static_cast<void*>(staging.ptr), total_size);

			rsx::texture_uploader_capabilities caps{};
			caps.supports_dxt = true;
			caps.alignment = 1; // Tightly packed rows

			upload_texture_subresource(io_buf, layout, gcm_format, is_swizzled, caps);

			// The subresource may be larger than the mip level when the guest uses odd sizes
			const u32 mip_w = std::max(1u, dst->width() >> layout.level);
			const u32 mip_h = std::max(1u, dst->height() >> layout.level);
			const u32 mip_d = dst->type() == texture_type::tex_3d ? std::max(1u, dst->depth() >> layout.level) : 1u;

			region3d region{};
			region.width = std::min<u32>(layout.width_in_texel, mip_w);
			region.height = std::min<u32>(layout.height_in_texel, mip_h);
			region.depth = std::min(depth, mip_d);

			if (!region.width || !region.height)
			{
				continue;
			}

			switch (gcm_format)
			{
			case CELL_GCM_TEXTURE_DEPTH24_D8:
			case CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT:
			{
				// D24S8 words -> D32F plane + S8 plane
				const u32 texels = units_x * rows * depth;
				auto depth_plane = ring_alloc(texels * 4, 256);
				auto stencil_plane = ring_alloc(texels, 256);

				if (!depth_plane.ptr || !stencil_plane.ptr)
				{
					break;
				}

				// The ring may have moved to a new segment; the staging data is still valid (segments are reused only when idle)
				const u32* src = reinterpret_cast<const u32*>(staging.ptr);
				f32* out_depth = reinterpret_cast<f32*>(depth_plane.ptr);
				u8* out_stencil = stencil_plane.ptr;
				const bool is_float = gcm_format == CELL_GCM_TEXTURE_DEPTH24_D8_FLOAT;

				for (u32 i = 0; i < texels; ++i)
				{
					const u32 value = src[i];
					const u32 d24 = value >> 8;
					out_depth[i] = is_float ? std::bit_cast<f32>(d24 << 7) : static_cast<f32>(d24) / 16777215.f;
					out_stencil[i] = static_cast<u8>(value & 0xff);
				}

				upload_texture_data(*dst, layout.level, layout.layer, region, depth_plane.offset, units_x * 4, units_x * 4 * rows, image_aspect::depth);
				upload_texture_data(*dst, layout.level, layout.layer, region, stencil_plane.offset, units_x, units_x * rows, image_aspect::stencil);
				break;
			}

			case CELL_GCM_TEXTURE_DEPTH16_FLOAT:
			{
				const u32 texels = units_x * rows * depth;
				auto depth_plane = ring_alloc(texels * 4, 256);

				if (!depth_plane.ptr)
				{
					break;
				}

				const u16* src = reinterpret_cast<const u16*>(staging.ptr);
				f32* out = reinterpret_cast<f32*>(depth_plane.ptr);

				for (u32 i = 0; i < texels; ++i)
				{
					out[i] = half_to_float(src[i]);
				}

				upload_texture_data(*dst, layout.level, layout.layer, region, depth_plane.offset, units_x * 4, units_x * 4 * rows, image_aspect::depth);
				break;
			}

			default:
				upload_texture_data(*dst, layout.level, layout.layer, region, staging.offset, row_pitch, image_size, image_aspect::color);
				break;
			}
		}
	}

	namespace
	{
		sampler_address wrap_mode(rsx::texture_wrap_mode wrap)
		{
			switch (wrap)
			{
			case rsx::texture_wrap_mode::wrap: return sampler_address::repeat;
			case rsx::texture_wrap_mode::mirror: return sampler_address::mirror_repeat;
			case rsx::texture_wrap_mode::clamp_to_edge: return sampler_address::clamp_to_edge;
			case rsx::texture_wrap_mode::border: return sampler_address::clamp_to_border;
			case rsx::texture_wrap_mode::clamp: return sampler_address::clamp_to_edge;
			case rsx::texture_wrap_mode::mirror_once_clamp_to_edge: return sampler_address::mirror_clamp_to_edge;
			case rsx::texture_wrap_mode::mirror_once_border: return sampler_address::mirror_clamp_to_edge;
			case rsx::texture_wrap_mode::mirror_once_clamp: return sampler_address::mirror_clamp_to_edge;
			default: return sampler_address::repeat;
			}
		}

		u8 max_aniso(rsx::texture_max_anisotropy aniso)
		{
			switch (aniso)
			{
			case rsx::texture_max_anisotropy::x1: return 1;
			case rsx::texture_max_anisotropy::x2: return 2;
			case rsx::texture_max_anisotropy::x4: return 4;
			case rsx::texture_max_anisotropy::x6: return 6;
			case rsx::texture_max_anisotropy::x8: return 8;
			case rsx::texture_max_anisotropy::x10: return 10;
			case rsx::texture_max_anisotropy::x12: return 12;
			case rsx::texture_max_anisotropy::x16: return 16;
			default: return 1;
			}
		}

		sampler_border border_color(u32 encoded)
		{
			// Metal only has three border colors; pick the closest one
			const color4f color = rsx::decode_border_color(encoded);

			if (color.a < 0.5f)
			{
				return sampler_border::transparent_black;
			}

			return (color.r + color.g + color.b) > 1.5f ? sampler_border::opaque_white : sampler_border::opaque_black;
		}

		// The stored compare function is reversed with respect to the shader comparison
		u8 shadow_compare_func(rsx::comparison_function func)
		{
			switch (func)
			{
			case rsx::comparison_function::never: return static_cast<u8>(compare_func::never);
			case rsx::comparison_function::greater: return static_cast<u8>(compare_func::less);
			case rsx::comparison_function::less: return static_cast<u8>(compare_func::greater);
			case rsx::comparison_function::less_or_equal: return static_cast<u8>(compare_func::greater_equal);
			case rsx::comparison_function::greater_or_equal: return static_cast<u8>(compare_func::less_equal);
			case rsx::comparison_function::equal: return static_cast<u8>(compare_func::equal);
			case rsx::comparison_function::not_equal: return static_cast<u8>(compare_func::not_equal);
			case rsx::comparison_function::always: return static_cast<u8>(compare_func::always);
			default: return static_cast<u8>(compare_func::always);
			}
		}

		void set_min_filter(sampler_desc& desc, rsx::texture_minify_filter filter, bool allow_mipmaps)
		{
			switch (filter)
			{
			case rsx::texture_minify_filter::nearest:
				desc.min_filter = sampler_filter::nearest;
				desc.mip_filter = sampler_mip_filter::none;
				break;
			case rsx::texture_minify_filter::linear:
				desc.min_filter = sampler_filter::linear;
				desc.mip_filter = sampler_mip_filter::none;
				break;
			case rsx::texture_minify_filter::nearest_nearest:
				desc.min_filter = sampler_filter::nearest;
				desc.mip_filter = sampler_mip_filter::nearest;
				break;
			case rsx::texture_minify_filter::linear_nearest:
				desc.min_filter = sampler_filter::linear;
				desc.mip_filter = sampler_mip_filter::nearest;
				break;
			case rsx::texture_minify_filter::nearest_linear:
				desc.min_filter = sampler_filter::nearest;
				desc.mip_filter = sampler_mip_filter::linear;
				break;
			case rsx::texture_minify_filter::linear_linear:
			case rsx::texture_minify_filter::convolution_min:
			default:
				desc.min_filter = sampler_filter::linear;
				desc.mip_filter = sampler_mip_filter::linear;
				break;
			}

			if (!allow_mipmaps)
			{
				desc.mip_filter = sampler_mip_filter::none;
			}
		}
	}

	template <>
	sampler_desc make_sampler_desc(const rsx::fragment_texture& tex, const rsx::sampled_image_descriptor_base* sampled_image, u32 mipmap_count)
	{
		sampler_desc desc{};
		desc.address_s = wrap_mode(tex.wrap_s());
		desc.address_t = wrap_mode(tex.wrap_t());
		desc.address_r = wrap_mode(tex.wrap_r());

		if (rsx::is_border_clamped_texture(tex))
		{
			const bool sext = sampled_image && (sampled_image->format_ex.texel_remap_control & rsx::texture_control_bits::SEXT_MASK) != 0;
			desc.border = border_color(tex.border_color(sext));
		}

		const bool allow_mipmaps = mipmap_count > 1;
		set_min_filter(desc, tex.min_filter(), allow_mipmaps);
		desc.mag_filter = tex.mag_filter() == rsx::texture_magnify_filter::nearest ? sampler_filter::nearest : sampler_filter::linear;

		if (allow_mipmaps)
		{
			desc.lod_min = tex.min_lod();
			desc.lod_max = tex.max_lod();
		}
		else
		{
			desc.lod_min = 0.f;
			desc.lod_max = 1000.f;
		}

		desc.max_anisotropy = max_aniso(tex.max_aniso());
		if (desc.max_anisotropy > 1 && (desc.min_filter != sampler_filter::linear || desc.mag_filter != sampler_filter::linear))
		{
			// Metal only allows anisotropy with linear filtering
			desc.max_anisotropy = 1;
		}

		const u32 texture_format = tex.format() & ~(CELL_GCM_TEXTURE_UN | CELL_GCM_TEXTURE_LN);
		if (texture_format >= CELL_GCM_TEXTURE_DEPTH24_D8 && texture_format <= CELL_GCM_TEXTURE_DEPTH16_FLOAT)
		{
			desc.compare_enable = 1;
			desc.compare_func = shadow_compare_func(tex.zfunc());
		}

		if (sampled_image)
		{
			// Depth data reinterpreted as color and SNORM data must not be filtered
			const u32 format = sampled_image->format_ex.format();
			const bool is_depth_reconstructed = sampled_image->format_class != rsx::classify_format(format) &&
				(format == CELL_GCM_TEXTURE_A8R8G8B8 || format == CELL_GCM_TEXTURE_D8R8G8B8);
			const bool is_snorm = (sampled_image->format_ex.texel_remap_control & rsx::texture_control_bits::SEXT_MASK) != 0;

			if (is_depth_reconstructed || is_snorm)
			{
				desc.min_filter = sampler_filter::nearest;
				desc.mag_filter = sampler_filter::nearest;
				desc.max_anisotropy = 1;

				if (desc.mip_filter == sampler_mip_filter::linear)
				{
					desc.mip_filter = sampler_mip_filter::nearest;
				}
			}
		}

		return desc;
	}

	template <>
	sampler_desc make_sampler_desc(const rsx::vertex_texture& tex, const rsx::sampled_image_descriptor_base*, u32)
	{
		sampler_desc desc{};
		desc.address_s = wrap_mode(tex.wrap_s());
		desc.address_t = wrap_mode(tex.wrap_t());
		desc.address_r = wrap_mode(tex.wrap_r());

		if (rsx::is_border_clamped_texture(tex))
		{
			desc.border = border_color(tex.border_color());
		}

		desc.min_filter = sampler_filter::nearest;
		desc.mag_filter = sampler_filter::nearest;
		desc.mip_filter = sampler_mip_filter::nearest;
		desc.lod_min = tex.min_lod();
		desc.lod_max = tex.max_lod();
		return desc;
	}
}
