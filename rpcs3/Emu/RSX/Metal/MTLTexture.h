#pragma once

// RSX texture formats on Metal: format tables, channel mappings, texture creation and uploads.
// Modelled on the Vulkan backend (VKFormats.cpp, VKTexture.cpp); the Metal formats are the ones
// MoltenVK uses for the same Vulkan formats, so the channel mappings carry over unchanged.

#include "MTLDevice.h"
#include "MTLTextureOps.h"

#include "Emu/RSX/Common/TextureUtils.h"
#include "Emu/RSX/RSXTexture.h"

#include <vector>

namespace mtl
{
	// Texture with RSX-aware views (channel remaps) and a format class, used by the texture cache
	// and the surface store (render targets derive from it).
	class viewable_image : public texture
	{
	public:
		viewable_image(const texture_desc& desc, rsx::format_class format_class);
		viewable_image(u32 width, u32 height, pixel_format format, u32 usage, rsx::format_class format_class);

		rsx::format_class format_class() const { return m_format_class; }
		bool is_depth_image() const { return is_depth(); }

		// View applying an RSX channel remap on top of the native component layout
		texture_view* get_view(const rsx::texture_channel_remap_t& remap, image_aspect aspect = image_aspect::color);

		// Identity view of the stored channels (RGBA)
		texture_view* get_raw_view();

	private:
		rsx::format_class m_format_class = rsx::RSX_FORMAT_CLASS_COLOR;
	};

	// Host format used to sample an RSX texture format
	pixel_format get_compatible_sampler_format(u32 gcm_format);

	// Channel mapping in RSX ARGB order (which host channel provides A, R, G, B)
	std::array<swizzle, 4> get_component_mapping(u32 gcm_format);

	// Final view swizzle (RGBA order) for a native ARGB layout and an RSX remap
	swizzle_rgba apply_swizzle_remap(const std::array<swizzle, 4>& native_layout, const rsx::texture_channel_remap_t& remap);

	texture_type get_texture_type(rsx::texture_dimension_extended type);

	std::unique_ptr<viewable_image> create_texture(u32 gcm_format, u16 width, u16 height, u16 depth, u16 mipmaps, rsx::texture_dimension_extended type);

	// Decodes the guest subresources (byte swap, deswizzle, depth conversion) and uploads them
	void upload_texture(texture* dst, u32 gcm_format, bool is_swizzled, const std::vector<rsx::subresource_layout>& subresources_layout);

	// Sampler state for an RSX texture unit
	template <typename Texture>
	sampler_desc make_sampler_desc(const Texture& tex, const rsx::sampled_image_descriptor_base* sampled_image, u32 mipmap_count);
}
