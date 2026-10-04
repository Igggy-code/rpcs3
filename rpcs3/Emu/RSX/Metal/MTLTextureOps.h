#pragma once

// Texture transfer operations and samplers for the Metal backend (MTLTextureOps.mm).
// All operations are recorded into the device's pending command buffer, in submission order.

#include "MTLDevice.h"

namespace mtl
{
	struct region3d
	{
		u32 x = 0, y = 0, z = 0;
		u32 width = 1, height = 1, depth = 1;
	};

	// Signed rectangle; negative extents flip the image
	struct blit_rect
	{
		int x = 0, y = 0;
		int width = 0, height = 0;
	};

	// Copies from the per-draw data ring (ring_alloc) into a texture subresource.
	// For depth/stencil textures, 'aspect' selects which plane the buffer holds.
	void upload_texture_data(texture& dst, u32 level, u32 layer, const region3d& region,
		u32 ring_offset, u32 bytes_per_row, u32 bytes_per_image, image_aspect aspect = image_aspect::color);

	// Same-format copy between subresources
	void copy_texture(texture& src, u32 src_level, u32 src_layer, const region3d& src_region,
		texture& dst, u32 dst_level, u32 dst_layer, u32 dst_x, u32 dst_y, u32 dst_z);

	// Bit-cast copy between color formats of different texel sizes. The source rows (src_w texels) are
	// reinterpreted as dst texels; dst_w * dst_bpp must equal src_w * src_bpp. Returns false if unsupported.
	bool copy_typeless(texture& src, u32 src_x, u32 src_y, u32 src_w, u32 height,
		texture& dst, u32 dst_x, u32 dst_y, u32 dst_w);

	// Draw-based scaled copy (filtering, flips, format conversion within the same aspect)
	bool blit_texture(texture& src, u32 src_level, u32 src_layer, const blit_rect& src_rect,
		texture& dst, u32 dst_level, u32 dst_layer, const blit_rect& dst_rect, bool linear);

	// Synchronous readback of a 2D region into tightly packed CPU memory (submits and waits)
	bool download_texture(texture& src, u32 level, u32 layer, u32 x, u32 y, u32 width, u32 height,
		image_aspect aspect, void* dst, u32 dst_bytes_per_row);

	enum class sampler_filter : u8 { nearest = 0, linear = 1 };
	enum class sampler_mip_filter : u8 { none = 0, nearest = 1, linear = 2 };

	// Same numbering as MTLSamplerAddressMode
	enum class sampler_address : u8
	{
		clamp_to_edge = 0,
		mirror_clamp_to_edge = 1,
		repeat = 2,
		mirror_repeat = 3,
		clamp_to_zero = 4,
		clamp_to_border = 5,
	};

	// Same numbering as MTLSamplerBorderColor
	enum class sampler_border : u8
	{
		transparent_black = 0,
		opaque_black = 1,
		opaque_white = 2,
	};

	struct sampler_desc
	{
		sampler_filter min_filter = sampler_filter::linear;
		sampler_filter mag_filter = sampler_filter::linear;
		sampler_mip_filter mip_filter = sampler_mip_filter::none;
		sampler_address address_s = sampler_address::repeat;
		sampler_address address_t = sampler_address::repeat;
		sampler_address address_r = sampler_address::repeat;
		sampler_border border = sampler_border::transparent_black;
		u8 max_anisotropy = 1;
		u8 compare_enable = 0;
		u8 compare_func = 0; // compare_func (MTLDraw.h numbering)
		float lod_min = 0.f;
		float lod_max = 1000.f;

		bool operator==(const sampler_desc&) const = default;
	};

	// Cached id<MTLSamplerState>; owned by the device layer
	void* get_sampler(const sampler_desc& desc);

	// Releases blit pipelines, samplers and staging resources
	void shutdown_texture_ops();
}
