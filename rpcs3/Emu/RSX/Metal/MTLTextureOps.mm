// Texture transfers and samplers for the native Metal RSX backend. Objective-C++, ARC, no RPCS3 headers.

// Objective-C bridging casts are C-style by nature; the project builds with -Werror=old-style-cast
#pragma clang diagnostic ignored "-Wold-style-cast"

#import <Metal/Metal.h>

#include "MTLTextureOps.h"
#include "MTLDraw.h"
#include "MTLDeviceInternal.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace
{
	using mtl::u8;
	using mtl::u32;
	using mtl::u64;

	id<MTLTexture> as_mtl(const mtl::texture& tex)
	{
		return (__bridge id<MTLTexture>)tex.native();
	}

	MTLBlitOption blit_option(const mtl::texture& tex, mtl::image_aspect aspect)
	{
		if (!mtl::has_stencil(tex.format()))
		{
			return MTLBlitOptionNone;
		}

		return aspect == mtl::image_aspect::stencil ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionDepthFromDepthStencil;
	}

	// Cube faces are slices, 3D textures use z
	u32 slice_of(const mtl::texture& tex, u32 layer)
	{
		return tex.type() == mtl::texture_type::tex_cube ? layer : 0;
	}

	// -------------------------------------------------------------------------------------------
	// Scaled blits: fullscreen quad over the destination rectangle, sampling the source rectangle
	// -------------------------------------------------------------------------------------------

	constexpr const char* s_blit_shader = R"(
#include <metal_stdlib>
using namespace metal;

struct blit_params
{
	float4 src_rect; // Normalized x, y, w, h (negative extents flip)
	float lod;
};

struct vs_out
{
	float4 pos [[position]];
	float2 uv;
};

vertex vs_out blit_vs(uint vid [[vertex_id]], constant blit_params& params [[buffer(0)]])
{
	const float2 t = float2(vid & 1, (vid >> 1) & 1);
	vs_out o;
	o.pos = float4(t.x * 2.0 - 1.0, 1.0 - t.y * 2.0, 0.0, 1.0);
	o.uv = params.src_rect.xy + t * params.src_rect.zw;
	return o;
}

fragment float4 blit_color_fs(vs_out in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]],
	constant blit_params& params [[buffer(0)]])
{
	return tex.sample(smp, in.uv, level(params.lod));
}

struct depth_out
{
	float depth [[depth(any)]];
};

fragment depth_out blit_depth_fs(vs_out in [[stage_in]], depth2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]],
	constant blit_params& params [[buffer(0)]])
{
	depth_out o;
	o.depth = tex.sample(smp, in.uv, level(params.lod));
	return o;
}
)";

	struct blit_params
	{
		float src_rect[4];
		float lod;
		float pad[3];
	};

	std::mutex s_ops_lock;
	id<MTLLibrary> s_blit_library = nil;
	std::unordered_map<u32, id<MTLRenderPipelineState>> s_blit_pipelines;
	id<MTLSamplerState> s_blit_linear = nil;
	id<MTLSamplerState> s_blit_nearest = nil;
	id<MTLDepthStencilState> s_depth_write_always = nil;

	struct sampler_hash
	{
		size_t operator()(const mtl::sampler_desc& d) const
		{
			u64 h = 0;
			h |= static_cast<u64>(d.min_filter);
			h |= static_cast<u64>(d.mag_filter) << 2;
			h |= static_cast<u64>(d.mip_filter) << 4;
			h |= static_cast<u64>(d.address_s) << 8;
			h |= static_cast<u64>(d.address_t) << 12;
			h |= static_cast<u64>(d.address_r) << 16;
			h |= static_cast<u64>(d.border) << 20;
			h |= static_cast<u64>(d.max_anisotropy) << 24;
			h |= static_cast<u64>(d.compare_enable) << 32;
			h |= static_cast<u64>(d.compare_func) << 36;

			u32 lo = 0, hi = 0;
			std::memcpy(&lo, &d.lod_min, 4);
			std::memcpy(&hi, &d.lod_max, 4);
			return std::hash<u64>()(h ^ (static_cast<u64>(lo) * 0x9e3779b97f4a7c15ull) ^ (static_cast<u64>(hi) << 7));
		}
	};

	std::unordered_map<mtl::sampler_desc, id<MTLSamplerState>, sampler_hash> s_samplers;

	bool ensure_blit_library()
	{
		if (s_blit_library)
		{
			return true;
		}

		NSError* error = nil;
		MTLCompileOptions* options = [MTLCompileOptions new];
		options.languageVersion = MTLLanguageVersion2_4;
		s_blit_library = [mtl::internal::device() newLibraryWithSource:@(s_blit_shader) options:options error:&error];

		if (!s_blit_library)
		{
			return false;
		}

		MTLSamplerDescriptor* desc = [MTLSamplerDescriptor new];
		desc.minFilter = MTLSamplerMinMagFilterLinear;
		desc.magFilter = MTLSamplerMinMagFilterLinear;
		desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
		desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
		s_blit_linear = [mtl::internal::device() newSamplerStateWithDescriptor:desc];

		desc.minFilter = MTLSamplerMinMagFilterNearest;
		desc.magFilter = MTLSamplerMinMagFilterNearest;
		s_blit_nearest = [mtl::internal::device() newSamplerStateWithDescriptor:desc];

		MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
		ds.depthCompareFunction = MTLCompareFunctionAlways;
		ds.depthWriteEnabled = YES;
		s_depth_write_always = [mtl::internal::device() newDepthStencilStateWithDescriptor:ds];
		return true;
	}

	id<MTLRenderPipelineState> get_blit_pipeline(const mtl::texture& dst)
	{
		const MTLPixelFormat format = mtl::internal::to_mtl_format(dst.format());
		const u32 key = static_cast<u32>(format);

		if (auto found = s_blit_pipelines.find(key); found != s_blit_pipelines.end())
		{
			return found->second;
		}

		if (!ensure_blit_library())
		{
			return nil;
		}

		const bool depth = dst.is_depth();

		MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
		desc.label = @"RSX blit";
		desc.vertexFunction = [s_blit_library newFunctionWithName:@"blit_vs"];
		desc.fragmentFunction = [s_blit_library newFunctionWithName:depth ? @"blit_depth_fs" : @"blit_color_fs"];

		if (depth)
		{
			desc.depthAttachmentPixelFormat = format;

			if (mtl::has_stencil(dst.format()))
			{
				desc.stencilAttachmentPixelFormat = format;
			}
		}
		else
		{
			desc.colorAttachments[0].pixelFormat = format;
		}

		NSError* error = nil;
		id<MTLRenderPipelineState> pso = [mtl::internal::device() newRenderPipelineStateWithDescriptor:desc error:&error];
		s_blit_pipelines[key] = pso;
		return pso;
	}
}

namespace mtl
{
	void upload_texture_data(texture& dst, u32 level, u32 layer, const region3d& region,
		u32 ring_offset, u32 bytes_per_row, u32 bytes_per_image, image_aspect aspect)
	{
		if (!dst.valid() || level >= dst.levels())
		{
			return;
		}

		@autoreleasepool
		{
			internal::close_render_pass();

			id<MTLBlitCommandEncoder> blit = [internal::command_buffer() blitCommandEncoder];
			blit.label = @"RSX texture upload";

			// Compressed formats: Metal wants rows of blocks; 1D/2D textures take bytesPerImage = 0
			const NSUInteger image_stride = dst.type() == texture_type::tex_3d ? bytes_per_image : 0;

			[blit copyFromBuffer:internal::ring_buffer()
				sourceOffset:ring_offset
				sourceBytesPerRow:bytes_per_row
				sourceBytesPerImage:image_stride
				sourceSize:MTLSizeMake(region.width, region.height, region.depth)
				toTexture:as_mtl(dst)
				destinationSlice:slice_of(dst, layer)
				destinationLevel:level
				destinationOrigin:MTLOriginMake(region.x, region.y, region.z)
				options:blit_option(dst, aspect)];

			[blit endEncoding];
		}
	}

	void copy_texture(texture& src, u32 src_level, u32 src_layer, const region3d& src_region,
		texture& dst, u32 dst_level, u32 dst_layer, u32 dst_x, u32 dst_y, u32 dst_z)
	{
		if (!src.valid() || !dst.valid() || src_level >= src.levels() || dst_level >= dst.levels())
		{
			return;
		}

		// Clamp to both subresources
		const u32 src_w = std::max(1u, src.width() >> src_level), src_h = std::max(1u, src.height() >> src_level);
		const u32 dst_w = std::max(1u, dst.width() >> dst_level), dst_h = std::max(1u, dst.height() >> dst_level);

		if (src_region.x >= src_w || src_region.y >= src_h || dst_x >= dst_w || dst_y >= dst_h)
		{
			return;
		}

		const u32 width = std::min({ src_region.width, src_w - src_region.x, dst_w - dst_x });
		const u32 height = std::min({ src_region.height, src_h - src_region.y, dst_h - dst_y });

		if (!width || !height)
		{
			return;
		}

		@autoreleasepool
		{
			internal::close_render_pass();

			id<MTLBlitCommandEncoder> blit = [internal::command_buffer() blitCommandEncoder];
			[blit copyFromTexture:as_mtl(src)
				sourceSlice:slice_of(src, src_layer)
				sourceLevel:src_level
				sourceOrigin:MTLOriginMake(src_region.x, src_region.y, src_region.z)
				sourceSize:MTLSizeMake(width, height, std::max(1u, src_region.depth))
				toTexture:as_mtl(dst)
				destinationSlice:slice_of(dst, dst_layer)
				destinationLevel:dst_level
				destinationOrigin:MTLOriginMake(dst_x, dst_y, dst_z)];
			[blit endEncoding];
		}
	}

	bool copy_typeless(texture& src, u32 src_x, u32 src_y, u32 src_w, u32 height,
		texture& dst, u32 dst_x, u32 dst_y, u32 dst_w)
	{
		if (!src.valid() || !dst.valid() || src.is_depth() || dst.is_depth() ||
			is_compressed_format(src.format()) || is_compressed_format(dst.format()))
		{
			return false;
		}

		src_w = std::min(src_w, src.width() - std::min(src_x, src.width()));
		dst_w = std::min(dst_w, dst.width() - std::min(dst_x, dst.width()));
		height = std::min({ height, src.height() - std::min(src_y, src.height()), dst.height() - std::min(dst_y, dst.height()) });

		const u32 row_bytes = src_w * src.block_size();
		if (!src_w || !dst_w || !height || dst_w * dst.block_size() != row_bytes)
		{
			return false;
		}

		const auto staging = ring_alloc(row_bytes * height, 256);
		if (!staging.ptr)
		{
			return false;
		}

		@autoreleasepool
		{
			internal::close_render_pass();

			id<MTLBlitCommandEncoder> blit = [internal::command_buffer() blitCommandEncoder];
			blit.label = @"RSX typeless copy";

			[blit copyFromTexture:as_mtl(src)
				sourceSlice:0
				sourceLevel:0
				sourceOrigin:MTLOriginMake(src_x, src_y, 0)
				sourceSize:MTLSizeMake(src_w, height, 1)
				toBuffer:internal::ring_buffer()
				destinationOffset:staging.offset
				destinationBytesPerRow:row_bytes
				destinationBytesPerImage:0];

			[blit copyFromBuffer:internal::ring_buffer()
				sourceOffset:staging.offset
				sourceBytesPerRow:row_bytes
				sourceBytesPerImage:0
				sourceSize:MTLSizeMake(dst_w, height, 1)
				toTexture:as_mtl(dst)
				destinationSlice:0
				destinationLevel:0
				destinationOrigin:MTLOriginMake(dst_x, dst_y, 0)];

			[blit endEncoding];
		}

		return true;
	}

	bool blit_texture(texture& src, u32 src_level, u32 src_layer, const blit_rect& src_rect_in,
		texture& dst, u32 dst_level, u32 dst_layer, const blit_rect& dst_rect_in, bool linear)
	{
		if (!src.valid() || !dst.valid() || src.is_depth() != dst.is_depth() ||
			src.type() != texture_type::tex_2d || (dst.type() != texture_type::tex_2d && dst.type() != texture_type::tex_cube))
		{
			return false;
		}

		blit_rect src_rect = src_rect_in;
		blit_rect dst_rect = dst_rect_in;

		// Normalize the destination; flips move to the source rectangle
		if (dst_rect.width < 0)
		{
			dst_rect.x += dst_rect.width;
			dst_rect.width = -dst_rect.width;
			src_rect.x += src_rect.width;
			src_rect.width = -src_rect.width;
		}

		if (dst_rect.height < 0)
		{
			dst_rect.y += dst_rect.height;
			dst_rect.height = -dst_rect.height;
			src_rect.y += src_rect.height;
			src_rect.height = -src_rect.height;
		}

		if (!dst_rect.width || !dst_rect.height || !src_rect.width || !src_rect.height)
		{
			return false;
		}

		// Same size and orientation: plain copy
		if (src_rect.width == dst_rect.width && src_rect.height == dst_rect.height && src.format() == dst.format() &&
			src_rect.x >= 0 && src_rect.y >= 0 && dst_rect.x >= 0 && dst_rect.y >= 0)
		{
			region3d region{ static_cast<u32>(src_rect.x), static_cast<u32>(src_rect.y), 0, static_cast<u32>(src_rect.width), static_cast<u32>(src_rect.height), 1 };
			copy_texture(src, src_level, 0, region, dst, dst_level, dst_layer, static_cast<u32>(dst_rect.x), static_cast<u32>(dst_rect.y), 0);
			return true;
		}

		std::lock_guard lock(s_ops_lock);

		@autoreleasepool
		{
			id<MTLRenderPipelineState> pso = get_blit_pipeline(dst);
			if (!pso)
			{
				return false;
			}

			(void)src_layer;

			const float src_w = static_cast<float>(std::max(1u, src.width() >> src_level));
			const float src_h = static_cast<float>(std::max(1u, src.height() >> src_level));

			blit_params params{};
			params.src_rect[0] = src_rect.x / src_w;
			params.src_rect[1] = src_rect.y / src_h;
			params.src_rect[2] = src_rect.width / src_w;
			params.src_rect[3] = src_rect.height / src_h;
			params.lod = static_cast<float>(src_level);

			internal::close_render_pass();

			MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];

			if (dst.is_depth())
			{
				pass.depthAttachment.texture = as_mtl(dst);
				pass.depthAttachment.level = dst_level;
				pass.depthAttachment.slice = slice_of(dst, dst_layer);
				pass.depthAttachment.loadAction = MTLLoadActionLoad;
				pass.depthAttachment.storeAction = MTLStoreActionStore;

				if (has_stencil(dst.format()))
				{
					pass.stencilAttachment.texture = as_mtl(dst);
					pass.stencilAttachment.level = dst_level;
					pass.stencilAttachment.slice = slice_of(dst, dst_layer);
					pass.stencilAttachment.loadAction = MTLLoadActionLoad;
					pass.stencilAttachment.storeAction = MTLStoreActionStore;
				}
			}
			else
			{
				pass.colorAttachments[0].texture = as_mtl(dst);
				pass.colorAttachments[0].level = dst_level;
				pass.colorAttachments[0].slice = slice_of(dst, dst_layer);
				pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
				pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			}

			id<MTLRenderCommandEncoder> enc = [internal::command_buffer() renderCommandEncoderWithDescriptor:pass];
			enc.label = @"RSX scaled blit";

			const u32 dst_w = std::max(1u, dst.width() >> dst_level);
			const u32 dst_h = std::max(1u, dst.height() >> dst_level);

			const MTLViewport viewport = { static_cast<double>(dst_rect.x), static_cast<double>(dst_rect.y),
				static_cast<double>(dst_rect.width), static_cast<double>(dst_rect.height), 0.0, 1.0 };

			const u32 sx = static_cast<u32>(std::clamp(dst_rect.x, 0, static_cast<int>(dst_w) - 1));
			const u32 sy = static_cast<u32>(std::clamp(dst_rect.y, 0, static_cast<int>(dst_h) - 1));
			const u32 ex = static_cast<u32>(std::clamp(dst_rect.x + dst_rect.width, 1, static_cast<int>(dst_w)));
			const u32 ey = static_cast<u32>(std::clamp(dst_rect.y + dst_rect.height, 1, static_cast<int>(dst_h)));

			[enc setRenderPipelineState:pso];
			[enc setViewport:viewport];
			[enc setScissorRect:MTLScissorRect{ sx, sy, std::max(1u, ex - std::min(ex, sx)), std::max(1u, ey - std::min(ey, sy)) }];

			if (dst.is_depth())
			{
				[enc setDepthStencilState:s_depth_write_always];
			}

			[enc setVertexBytes:&params length:sizeof(params) atIndex:0];
			[enc setFragmentBytes:&params length:sizeof(params) atIndex:0];
			[enc setFragmentTexture:as_mtl(src) atIndex:0];
			[enc setFragmentSamplerState:(linear && !dst.is_depth()) ? s_blit_linear : s_blit_nearest atIndex:0];
			[enc drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
			[enc endEncoding];
		}

		return true;
	}

	bool download_texture(texture& src, u32 level, u32 layer, u32 x, u32 y, u32 width, u32 height,
		image_aspect aspect, void* dst, u32 dst_bytes_per_row)
	{
		if (!src.valid() || !width || !height || is_compressed_format(src.format()))
		{
			return false;
		}

		u32 bpp = src.block_size();
		if (has_stencil(src.format()))
		{
			bpp = aspect == image_aspect::stencil ? 1 : 4;
		}

		const u32 row_bytes = width * bpp;

		@autoreleasepool
		{
			id<MTLBuffer> staging = [internal::device() newBufferWithLength:row_bytes * height options:MTLResourceStorageModeShared];
			if (!staging)
			{
				return false;
			}

			internal::close_render_pass();

			id<MTLBlitCommandEncoder> blit = [internal::command_buffer() blitCommandEncoder];
			blit.label = @"RSX readback";

			[blit copyFromTexture:as_mtl(src)
				sourceSlice:slice_of(src, layer)
				sourceLevel:level
				sourceOrigin:MTLOriginMake(x, y, 0)
				sourceSize:MTLSizeMake(width, height, 1)
				toBuffer:staging
				destinationOffset:0
				destinationBytesPerRow:row_bytes
				destinationBytesPerImage:0
				options:blit_option(src, aspect)];

			[blit endEncoding];

			finish();

			const u8* data = static_cast<const u8*>(staging.contents);
			u8* out = static_cast<u8*>(dst);

			for (u32 row = 0; row < height; ++row)
			{
				std::memcpy(out + row * dst_bytes_per_row, data + row * row_bytes, std::min(row_bytes, dst_bytes_per_row));
			}
		}

		return true;
	}

	void* get_sampler(const sampler_desc& d)
	{
		std::lock_guard lock(s_ops_lock);

		if (auto found = s_samplers.find(d); found != s_samplers.end())
		{
			return (__bridge void*)found->second;
		}

		@autoreleasepool
		{
			MTLSamplerDescriptor* desc = [MTLSamplerDescriptor new];
			desc.minFilter = d.min_filter == sampler_filter::linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
			desc.magFilter = d.mag_filter == sampler_filter::linear ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;

			switch (d.mip_filter)
			{
			case sampler_mip_filter::none: desc.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
			case sampler_mip_filter::nearest: desc.mipFilter = MTLSamplerMipFilterNearest; break;
			case sampler_mip_filter::linear: desc.mipFilter = MTLSamplerMipFilterLinear; break;
			}

			desc.sAddressMode = static_cast<MTLSamplerAddressMode>(d.address_s);
			desc.tAddressMode = static_cast<MTLSamplerAddressMode>(d.address_t);
			desc.rAddressMode = static_cast<MTLSamplerAddressMode>(d.address_r);
			desc.borderColor = static_cast<MTLSamplerBorderColor>(d.border);
			desc.maxAnisotropy = std::clamp<NSUInteger>(d.max_anisotropy, 1, 16);
			desc.lodMinClamp = d.lod_min;
			desc.lodMaxClamp = std::max(d.lod_min, d.lod_max);
			desc.normalizedCoordinates = YES;

			if (d.compare_enable)
			{
				desc.compareFunction = static_cast<MTLCompareFunction>(d.compare_func);
			}

			id<MTLSamplerState> sampler = [internal::device() newSamplerStateWithDescriptor:desc];
			s_samplers[d] = sampler;
			return (__bridge void*)sampler;
		}
	}

	void shutdown_texture_ops()
	{
		std::lock_guard lock(s_ops_lock);

		s_samplers.clear();
		s_blit_pipelines.clear();
		s_blit_library = nil;
		s_blit_linear = nil;
		s_blit_nearest = nil;
		s_depth_write_always = nil;
	}
}

static_assert(static_cast<unsigned>(MTLSamplerAddressModeClampToBorderColor) == static_cast<unsigned>(mtl::sampler_address::clamp_to_border));
static_assert(static_cast<unsigned>(MTLSamplerAddressModeMirrorRepeat) == static_cast<unsigned>(mtl::sampler_address::mirror_repeat));
static_assert(static_cast<unsigned>(MTLSamplerBorderColorOpaqueWhite) == static_cast<unsigned>(mtl::sampler_border::opaque_white));
static_assert(static_cast<unsigned>(MTLTextureSwizzleAlpha) == static_cast<unsigned>(mtl::swizzle::alpha));
static_assert(static_cast<unsigned>(MTLTextureSwizzleZero) == static_cast<unsigned>(mtl::swizzle::zero));
