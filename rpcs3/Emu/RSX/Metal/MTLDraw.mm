// Draw submission for the native Metal RSX backend. Objective-C++, ARC, no RPCS3 headers.

// Objective-C bridging casts are C-style by nature; the project builds with -Werror=old-style-cast
#pragma clang diagnostic ignored "-Wold-style-cast"

#import <Metal/Metal.h>

#include "MTLDraw.h"
#include "MTLDeviceInternal.h"

#include <algorithm>
#include <string>
#include <unordered_map>

// The plain C++ enums in MTLDraw.h are cast directly to their Metal counterparts
static_assert(static_cast<unsigned>(MTLCompareFunctionNever) == static_cast<unsigned>(mtl::compare_func::never));
static_assert(static_cast<unsigned>(MTLCompareFunctionAlways) == static_cast<unsigned>(mtl::compare_func::always));
static_assert(static_cast<unsigned>(MTLCompareFunctionGreaterEqual) == static_cast<unsigned>(mtl::compare_func::greater_equal));
static_assert(static_cast<unsigned>(MTLStencilOperationDecrementWrap) == static_cast<unsigned>(mtl::stencil_op::decr_wrap));
static_assert(static_cast<unsigned>(MTLStencilOperationInvert) == static_cast<unsigned>(mtl::stencil_op::invert));
static_assert(static_cast<unsigned>(MTLBlendFactorSourceAlphaSaturated) == static_cast<unsigned>(mtl::blend_factor::src_alpha_saturated));
static_assert(static_cast<unsigned>(MTLBlendFactorBlendColor) == static_cast<unsigned>(mtl::blend_factor::constant_color));
static_assert(static_cast<unsigned>(MTLBlendFactorOneMinusBlendAlpha) == static_cast<unsigned>(mtl::blend_factor::one_minus_constant_alpha));
static_assert(static_cast<unsigned>(MTLBlendOperationMax) == static_cast<unsigned>(mtl::blend_op::max));
static_assert(static_cast<unsigned>(MTLCullModeBack) == static_cast<unsigned>(mtl::cull_mode::back));
static_assert(static_cast<unsigned>(MTLWindingCounterClockwise) == static_cast<unsigned>(mtl::winding::counter_clockwise));
static_assert(static_cast<unsigned>(MTLPrimitiveTypeTriangleStrip) == static_cast<unsigned>(mtl::primitive::triangle_strip));
static_assert(static_cast<unsigned>(MTLColorWriteMaskRed) == static_cast<unsigned>(mtl::write_r) && static_cast<unsigned>(MTLColorWriteMaskAlpha) == static_cast<unsigned>(mtl::write_a));

namespace
{
	using mtl::u8;
	using mtl::u32;

	// -------------------------------------------------------------------------------------------
	// Data ring: one shared buffer split in segments. A segment is reused only after the last
	// command buffer that referenced it has completed.
	// -------------------------------------------------------------------------------------------

	constexpr u32 ring_segment_count = 4;
	constexpr u32 ring_segment_size = 32u << 20;
	constexpr u32 ring_size = ring_segment_count * ring_segment_size;

	struct data_ring
	{
		id<MTLBuffer> buffer = nil;
		id<MTLTexture> texels = nil; // R8Uint view of the whole buffer for vertex pulling
		u8* base = nullptr;

		id<MTLCommandBuffer> segment_users[ring_segment_count] = {};
		u32 segment = 0;
		u32 head = 0;
	};

	data_ring s_ring;

	bool ensure_ring()
	{
		if (s_ring.buffer)
		{
			return true;
		}

		id<MTLDevice> device = mtl::internal::device();
		if (!device)
		{
			return false;
		}

		s_ring.buffer = [device newBufferWithLength:ring_size options:MTLResourceStorageModeShared];
		if (!s_ring.buffer)
		{
			return false;
		}

		s_ring.buffer.label = @"RSX data ring";
		s_ring.base = static_cast<u8*>(s_ring.buffer.contents);

		MTLTextureDescriptor* desc = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:MTLPixelFormatR8Uint
			width:ring_size resourceOptions:MTLResourceStorageModeShared usage:MTLTextureUsageShaderRead];
		s_ring.texels = [s_ring.buffer newTextureWithDescriptor:desc offset:0 bytesPerRow:ring_size];

		s_ring.segment = 0;
		s_ring.head = 0;
		return s_ring.texels != nil;
	}

	void wait_for_segment(u32 segment)
	{
		id<MTLCommandBuffer> cmd = s_ring.segment_users[segment];
		s_ring.segment_users[segment] = nil;

		if (!cmd)
		{
			return;
		}

		if (cmd.status == MTLCommandBufferStatusNotEnqueued)
		{
			// Still the pending command buffer: submit it before waiting
			mtl::flush();
		}

		[cmd waitUntilCompleted];
	}

	// -------------------------------------------------------------------------------------------
	// Placeholder textures and samplers
	// -------------------------------------------------------------------------------------------

	std::unordered_map<u32, id<MTLTexture>> s_dummy_textures;
	id<MTLSamplerState> s_sampler = nil;
	id<MTLSamplerState> s_compare_sampler = nil;
	std::unordered_map<std::string, id<MTLDepthStencilState>> s_depth_states;

	id<MTLTexture> get_dummy_texture(mtl::texture_dimension dim, bool depth, bool multisampled)
	{
		// Combinations Metal does not have (depth 1D/3D, multisampled non-2D) fall back to 2D
		if ((depth && (dim == mtl::texture_dimension::dim_1d || dim == mtl::texture_dimension::dim_3d)) ||
			(multisampled && dim != mtl::texture_dimension::dim_2d))
		{
			dim = mtl::texture_dimension::dim_2d;
		}

		const u32 key = static_cast<unsigned>(dim) | (depth ? 0x10u : 0u) | (multisampled ? 0x20u : 0u);

		if (auto found = s_dummy_textures.find(key); found != s_dummy_textures.end())
		{
			return found->second;
		}

		id<MTLDevice> device = mtl::internal::device();

		MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
		desc.width = 1;
		desc.height = 1;
		desc.depth = 1;
		desc.mipmapLevelCount = 1;
		desc.pixelFormat = depth ? MTLPixelFormatDepth32Float : MTLPixelFormatRGBA8Unorm;
		desc.usage = MTLTextureUsageShaderRead;

		switch (dim)
		{
		case mtl::texture_dimension::dim_1d: desc.textureType = MTLTextureType1D; break;
		case mtl::texture_dimension::dim_3d: desc.textureType = MTLTextureType3D; break;
		case mtl::texture_dimension::dim_cube: desc.textureType = MTLTextureTypeCube; break;
		case mtl::texture_dimension::dim_2d: desc.textureType = multisampled ? MTLTextureType2DMultisample : MTLTextureType2D; break;
		}

		if (multisampled)
		{
			desc.sampleCount = 4;
		}

		// Private textures are initialized with a clear pass, plain color ones are written directly
		const bool use_clear = depth || multisampled;
		desc.storageMode = use_clear ? MTLStorageModePrivate : MTLStorageModeShared;

		if (use_clear)
		{
			desc.usage |= MTLTextureUsageRenderTarget;
		}

		id<MTLTexture> tex = [device newTextureWithDescriptor:desc];

		if (!tex)
		{
			return nil;
		}

		tex.label = @"RSX placeholder texture";

		if (use_clear)
		{
			id<MTLCommandBuffer> cmd = [mtl::internal::queue() commandBuffer];
			const u32 slices = desc.textureType == MTLTextureTypeCube ? 6 : 1;

			for (u32 slice = 0; slice < slices; ++slice)
			{
				MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];

				if (depth)
				{
					pass.depthAttachment.texture = tex;
					pass.depthAttachment.slice = slice;
					pass.depthAttachment.loadAction = MTLLoadActionClear;
					pass.depthAttachment.storeAction = MTLStoreActionStore;
					pass.depthAttachment.clearDepth = 1.0;
				}
				else
				{
					pass.colorAttachments[0].texture = tex;
					pass.colorAttachments[0].slice = slice;
					pass.colorAttachments[0].loadAction = MTLLoadActionClear;
					pass.colorAttachments[0].storeAction = MTLStoreActionStore;
					pass.colorAttachments[0].clearColor = MTLClearColorMake(1.0, 1.0, 1.0, 1.0);
				}

				id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
				[enc endEncoding];
			}

			[cmd commit];
		}
		else
		{
			const u8 white[4] = { 0xff, 0xff, 0xff, 0xff };
			const u32 slices = desc.textureType == MTLTextureTypeCube ? 6 : 1;

			for (u32 slice = 0; slice < slices; ++slice)
			{
				const NSUInteger bytes_per_image = desc.textureType == MTLTextureType3D ? 4 : 0;
				[tex replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:slice withBytes:white bytesPerRow:4 bytesPerImage:bytes_per_image];
			}
		}

		s_dummy_textures[key] = tex;
		return tex;
	}

	id<MTLSamplerState> get_sampler(bool compare)
	{
		__strong id<MTLSamplerState>& sampler = compare ? s_compare_sampler : s_sampler;

		if (!sampler)
		{
			MTLSamplerDescriptor* desc = [MTLSamplerDescriptor new];
			desc.minFilter = MTLSamplerMinMagFilterLinear;
			desc.magFilter = MTLSamplerMinMagFilterLinear;
			desc.sAddressMode = MTLSamplerAddressModeRepeat;
			desc.tAddressMode = MTLSamplerAddressModeRepeat;
			desc.rAddressMode = MTLSamplerAddressModeRepeat;

			if (compare)
			{
				desc.compareFunction = MTLCompareFunctionLessEqual;
			}

			sampler = [mtl::internal::device() newSamplerStateWithDescriptor:desc];
		}

		return sampler;
	}

	MTLStencilDescriptor* make_stencil(const mtl::stencil_face& face)
	{
		MTLStencilDescriptor* desc = [MTLStencilDescriptor new];
		desc.stencilFailureOperation = static_cast<MTLStencilOperation>(face.fail);
		desc.depthFailureOperation = static_cast<MTLStencilOperation>(face.depth_fail);
		desc.depthStencilPassOperation = static_cast<MTLStencilOperation>(face.pass);
		desc.stencilCompareFunction = static_cast<MTLCompareFunction>(face.func);
		desc.readMask = face.read_mask;
		desc.writeMask = face.write_mask;
		return desc;
	}

	id<MTLDepthStencilState> get_depth_stencil_state(const mtl::depth_stencil_state& state)
	{
		// Reference values are dynamic state and not part of the key
		std::string key;
		key.reserve(16);
		key.push_back(static_cast<char>(state.depth_test));
		key.push_back(static_cast<char>(state.depth_write));
		key.push_back(static_cast<char>(state.depth_func));
		key.push_back(static_cast<char>(state.stencil_test));

		for (const mtl::stencil_face* face : { &state.front, &state.back })
		{
			key.push_back(static_cast<char>(face->fail));
			key.push_back(static_cast<char>(face->depth_fail));
			key.push_back(static_cast<char>(face->pass));
			key.push_back(static_cast<char>(face->func));
			key.push_back(static_cast<char>(face->read_mask));
			key.push_back(static_cast<char>(face->write_mask));
		}

		if (auto found = s_depth_states.find(key); found != s_depth_states.end())
		{
			return found->second;
		}

		MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];

		if (state.depth_test)
		{
			desc.depthCompareFunction = static_cast<MTLCompareFunction>(state.depth_func);
			desc.depthWriteEnabled = state.depth_write ? YES : NO;
		}
		else
		{
			desc.depthCompareFunction = MTLCompareFunctionAlways;
			desc.depthWriteEnabled = NO;
		}

		if (state.stencil_test)
		{
			desc.frontFaceStencil = make_stencil(state.front);
			desc.backFaceStencil = make_stencil(state.back);
		}

		id<MTLDepthStencilState> result = [mtl::internal::device() newDepthStencilStateWithDescriptor:desc];
		s_depth_states[key] = result;
		return result;
	}
}

namespace mtl::internal
{
	id<MTLBuffer> ring_buffer()
	{
		ensure_ring();
		return s_ring.buffer;
	}
}

namespace mtl
{
	ring_allocation ring_alloc(u32 size, u32 alignment)
	{
		@autoreleasepool
		{
			if (!size || size > ring_segment_size || !ensure_ring())
			{
				return {};
			}

			u32 offset = (s_ring.head + (alignment - 1)) & ~(alignment - 1);
			const u32 segment_end = (s_ring.segment + 1) * ring_segment_size;

			if (offset + size > segment_end)
			{
				// Move to the next segment once the GPU is done with it
				s_ring.segment = (s_ring.segment + 1) % ring_segment_count;
				wait_for_segment(s_ring.segment);
				offset = s_ring.segment * ring_segment_size;
			}

			s_ring.head = offset + size;
			s_ring.segment_users[s_ring.segment] = internal::command_buffer();

			return { s_ring.base + offset, offset };
		}
	}

	render_pipeline::~render_pipeline()
	{
		if (m_handle)
		{
			CFRelease(m_handle);
			m_handle = nullptr;
		}
	}

	bool render_pipeline::create(const shader_function& vs, const shader_function& fs, const render_pipeline_state& state, std::string& error)
	{
		@autoreleasepool
		{
			id<MTLDevice> device = internal::device();

			if (!device || !vs.valid() || !fs.valid())
			{
				error = "invalid device or shaders";
				return false;
			}

			MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
			desc.label = @"RSX pipeline";
			desc.vertexFunction = (__bridge id<MTLFunction>)vs.native();
			desc.fragmentFunction = (__bridge id<MTLFunction>)fs.native();
			desc.rasterSampleCount = 1;
			desc.alphaToCoverageEnabled = state.alpha_to_coverage ? YES : NO;

			for (u32 i = 0; i < 4; ++i)
			{
				const auto format = static_cast<pixel_format>(state.color_formats[i]);
				if (format == pixel_format::invalid)
				{
					continue;
				}

				MTLRenderPipelineColorAttachmentDescriptor* att = desc.colorAttachments[i];
				att.pixelFormat = internal::to_mtl_format(format);
				att.writeMask = static_cast<MTLColorWriteMask>((state.write_masks >> (i * 4)) & 0xf);

				if (state.blend_enable & (1u << i))
				{
					att.blendingEnabled = YES;
					att.sourceRGBBlendFactor = static_cast<MTLBlendFactor>(state.blend_factors & 0xff);
					att.destinationRGBBlendFactor = static_cast<MTLBlendFactor>((state.blend_factors >> 8) & 0xff);
					att.sourceAlphaBlendFactor = static_cast<MTLBlendFactor>((state.blend_factors >> 16) & 0xff);
					att.destinationAlphaBlendFactor = static_cast<MTLBlendFactor>((state.blend_factors >> 24) & 0xff);
					att.rgbBlendOperation = static_cast<MTLBlendOperation>(state.blend_ops & 0xff);
					att.alphaBlendOperation = static_cast<MTLBlendOperation>((state.blend_ops >> 8) & 0xff);
				}
			}

			const auto depth_format = static_cast<pixel_format>(state.depth_format);
			if (depth_format != pixel_format::invalid)
			{
				desc.depthAttachmentPixelFormat = internal::to_mtl_format(depth_format);

				if (has_stencil(depth_format))
				{
					desc.stencilAttachmentPixelFormat = desc.depthAttachmentPixelFormat;
				}
			}

			NSError* ns_error = nil;
			id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithDescriptor:desc error:&ns_error];

			if (!pso)
			{
				error = ns_error ? ns_error.localizedDescription.UTF8String : "unknown error";
				return false;
			}

			if (m_handle)
			{
				CFRelease(m_handle);
			}

			m_handle = (__bridge_retained void*)pso;
			return true;
		}
	}

	void draw(const draw_desc& desc)
	{
		if (!desc.pipeline || !desc.pipeline->valid() || desc.ranges.empty() || !ensure_ring())
		{
			return;
		}

		@autoreleasepool
		{
			__unsafe_unretained id<MTLTexture> color[4] = {};
			u32 width = ~0u, height = ~0u;

			for (u32 i = 0; i < 4; ++i)
			{
				if (desc.color[i] && desc.color[i]->valid())
				{
					color[i] = (__bridge id<MTLTexture>)desc.color[i]->native();
					width = std::min(width, desc.color[i]->width());
					height = std::min(height, desc.color[i]->height());
				}
			}

			id<MTLTexture> depth = nil;
			bool stencil = false;

			if (desc.depth_target && desc.depth_target->valid())
			{
				depth = (__bridge id<MTLTexture>)desc.depth_target->native();
				stencil = has_stencil(desc.depth_target->format());
				width = std::min(width, desc.depth_target->width());
				height = std::min(height, desc.depth_target->height());
			}

			if (width == ~0u || !width || !height)
			{
				return;
			}

			bool is_new = false;
			id<MTLRenderCommandEncoder> enc = internal::render_encoder(color, depth, stencil, is_new);

			if (!enc)
			{
				return;
			}

			[enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)desc.pipeline->native()];
			[enc setDepthStencilState:get_depth_stencil_state(desc.depth_stencil)];
			[enc setStencilFrontReferenceValue:desc.depth_stencil.front.reference backReferenceValue:desc.depth_stencil.back.reference];

			const MTLViewport viewport =
			{
				static_cast<double>(desc.viewport[0]), static_cast<double>(desc.viewport[1]),
				static_cast<double>(std::max(desc.viewport[2], 1.f)), static_cast<double>(std::max(desc.viewport[3], 1.f)),
				0.0, 1.0
			};
			[enc setViewport:viewport];

			// The scissor rectangle must lie inside the render pass
			const u32 sx = std::min(desc.scissor[0], width - 1);
			const u32 sy = std::min(desc.scissor[1], height - 1);
			const u32 sw = std::max(1u, std::min(desc.scissor[2], width - sx));
			const u32 sh = std::max(1u, std::min(desc.scissor[3], height - sy));
			[enc setScissorRect:MTLScissorRect{ sx, sy, sw, sh }];

			[enc setCullMode:static_cast<MTLCullMode>(desc.cull)];
			[enc setFrontFacingWinding:static_cast<MTLWinding>(desc.front_face)];
			[enc setDepthClipMode:desc.depth_clamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip];
			[enc setDepthBias:desc.depth_bias slopeScale:desc.depth_slope_scale clamp:0.f];
			[enc setBlendColorRed:desc.blend_color[0] green:desc.blend_color[1] blue:desc.blend_color[2] alpha:desc.blend_color[3]];

			for (const resource_binding& b : desc.bindings)
			{
				const bool vertex = b.stage == shader_stage::vertex;

				switch (b.source)
				{
				case binding_source::ring_buffer:
					if (vertex) [enc setVertexBuffer:s_ring.buffer offset:b.offset atIndex:b.index];
					else [enc setFragmentBuffer:s_ring.buffer offset:b.offset atIndex:b.index];
					break;

				case binding_source::ring_texels:
					if (vertex) [enc setVertexTexture:s_ring.texels atIndex:b.index];
					else [enc setFragmentTexture:s_ring.texels atIndex:b.index];
					break;

				case binding_source::texture:
				{
					id<MTLTexture> tex = (__bridge id<MTLTexture>)b.texture_handle;
					id<MTLSamplerState> sampler = b.sampler_handle ? (__bridge id<MTLSamplerState>)b.sampler_handle : get_sampler(b.depth);

					if (!tex)
					{
						tex = get_dummy_texture(b.dimension, b.depth, b.multisampled);
					}

					if (vertex)
					{
						[enc setVertexTexture:tex atIndex:b.index];
						[enc setVertexSamplerState:sampler atIndex:b.sampler];
					}
					else
					{
						[enc setFragmentTexture:tex atIndex:b.index];
						[enc setFragmentSamplerState:sampler atIndex:b.sampler];
					}
					break;
				}

				case binding_source::dummy_texture:
				{
					id<MTLTexture> tex = get_dummy_texture(b.dimension, b.depth, b.multisampled);
					id<MTLSamplerState> sampler = get_sampler(b.depth);

					if (vertex)
					{
						[enc setVertexTexture:tex atIndex:b.index];
						[enc setVertexSamplerState:sampler atIndex:b.sampler];
					}
					else
					{
						[enc setFragmentTexture:tex atIndex:b.index];
						[enc setFragmentSamplerState:sampler atIndex:b.sampler];
					}
					break;
				}
				}
			}

			const auto type = static_cast<MTLPrimitiveType>(desc.topology);

			if (desc.indexed)
			{
				const u32 index_size = desc.index_32bit ? 4 : 2;
				const MTLIndexType index_type = desc.index_32bit ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16;

				for (const draw_range& range : desc.ranges)
				{
					if (!range.count) continue;

					[enc drawIndexedPrimitives:type indexCount:range.count indexType:index_type
						indexBuffer:s_ring.buffer indexBufferOffset:desc.index_offset + range.first * index_size];
				}
			}
			else
			{
				for (const draw_range& range : desc.ranges)
				{
					if (!range.count) continue;

					[enc drawPrimitives:type vertexStart:range.first vertexCount:range.count];
				}
			}
		}
	}

	void shutdown_draw_resources()
	{
		@autoreleasepool
		{
			internal::close_render_pass();

			for (auto& user : s_ring.segment_users)
			{
				user = nil;
			}

			s_ring.texels = nil;
			s_ring.buffer = nil;
			s_ring.base = nullptr;
			s_ring.head = 0;
			s_ring.segment = 0;

			s_dummy_textures.clear();
			s_depth_states.clear();
			s_sampler = nil;
			s_compare_sampler = nil;
		}
	}
}
