// Metal device layer for the native Metal RSX backend. Objective-C++, ARC, no RPCS3 headers.

// Objective-C bridging casts are C-style by nature; the project builds with -Werror=old-style-cast
#pragma clang diagnostic ignored "-Wold-style-cast"

#import <Metal/Metal.h>

#include "MTLDevice.h"
#include "MTLDeviceInternal.h"
#include "MTLShaderCompiler.h"

#include <algorithm>
#include <mutex>

namespace
{
	// All RSX work happens on the RSX thread, but presentation and teardown can come from elsewhere
	std::recursive_mutex s_lock;
	id<MTLDevice> s_device = nil;
	id<MTLCommandQueue> s_queue = nil;
	id<MTLCommandBuffer> s_pending = nil;

	// Open render pass for draws and its attachments
	id<MTLRenderCommandEncoder> s_encoder = nil;
	id<MTLTexture> s_encoder_color[4] = {};
	id<MTLTexture> s_encoder_depth = nil;

	MTLPixelFormat to_mtl(mtl::pixel_format format)
	{
		switch (format)
		{
		case mtl::pixel_format::bgra8: return MTLPixelFormatBGRA8Unorm;
		case mtl::pixel_format::rgba8: return MTLPixelFormatRGBA8Unorm;
		case mtl::pixel_format::b5g6r5: return MTLPixelFormatB5G6R5Unorm;
		case mtl::pixel_format::bgr5a1: return MTLPixelFormatBGR5A1Unorm;
		case mtl::pixel_format::r8: return MTLPixelFormatR8Unorm;
		case mtl::pixel_format::rg8: return MTLPixelFormatRG8Unorm;
		case mtl::pixel_format::rgba16f: return MTLPixelFormatRGBA16Float;
		case mtl::pixel_format::rgba32f: return MTLPixelFormatRGBA32Float;
		case mtl::pixel_format::r32f: return MTLPixelFormatR32Float;
		case mtl::pixel_format::depth16: return MTLPixelFormatDepth16Unorm;
		case mtl::pixel_format::depth32f: return MTLPixelFormatDepth32Float;
		case mtl::pixel_format::depth32f_stencil8: return MTLPixelFormatDepth32Float_Stencil8;
		case mtl::pixel_format::a1bgr5: return MTLPixelFormatA1BGR5Unorm;
		case mtl::pixel_format::abgr4: return MTLPixelFormatABGR4Unorm;
		case mtl::pixel_format::rg8_snorm: return MTLPixelFormatRG8Snorm;
		case mtl::pixel_format::r16: return MTLPixelFormatR16Unorm;
		case mtl::pixel_format::rg16: return MTLPixelFormatRG16Unorm;
		case mtl::pixel_format::rg16f: return MTLPixelFormatRG16Float;
		case mtl::pixel_format::bc1: return MTLPixelFormatBC1_RGBA;
		case mtl::pixel_format::bc2: return MTLPixelFormatBC2_RGBA;
		case mtl::pixel_format::bc3: return MTLPixelFormatBC3_RGBA;
		case mtl::pixel_format::r8_uint: return MTLPixelFormatR8Uint;
		case mtl::pixel_format::r16_uint: return MTLPixelFormatR16Uint;
		case mtl::pixel_format::r32_uint: return MTLPixelFormatR32Uint;
		case mtl::pixel_format::rg32_uint: return MTLPixelFormatRG32Uint;
		case mtl::pixel_format::rgba32_uint: return MTLPixelFormatRGBA32Uint;
		case mtl::pixel_format::x32_stencil8: return MTLPixelFormatX32_Stencil8;
		case mtl::pixel_format::invalid: break;
		}

		return MTLPixelFormatInvalid;
	}

	id<MTLTexture> as_texture(const mtl::texture& tex)
	{
		return (__bridge id<MTLTexture>)tex.native();
	}
}

namespace mtl::internal
{
	MTLPixelFormat to_mtl_format(mtl::pixel_format format)
	{
		return to_mtl(format);
	}

	id<MTLDevice> device()
	{
		return s_device;
	}

	id<MTLCommandQueue> queue()
	{
		return s_queue;
	}

	id<MTLCommandBuffer> command_buffer()
	{
		std::lock_guard lock(s_lock);

		if (!s_pending)
		{
			s_pending = [s_queue commandBuffer];
			s_pending.label = @"RSX";
		}

		return s_pending;
	}

	void close_render_pass()
	{
		std::lock_guard lock(s_lock);

		if (s_encoder)
		{
			[s_encoder endEncoding];
			s_encoder = nil;
		}

		for (auto& tex : s_encoder_color) tex = nil;
		s_encoder_depth = nil;
	}

	id<MTLRenderCommandEncoder> render_encoder(__unsafe_unretained const id<MTLTexture>* color, id<MTLTexture> depth, bool has_stencil, bool& is_new)
	{
		std::lock_guard lock(s_lock);

		is_new = false;

		if (s_encoder)
		{
			bool same = s_encoder_depth == depth;
			for (int i = 0; i < 4 && same; ++i)
			{
				same = s_encoder_color[i] == color[i];
			}

			if (same)
			{
				return s_encoder;
			}

			close_render_pass();
		}

		MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];

		for (int i = 0; i < 4; ++i)
		{
			if (color[i])
			{
				pass.colorAttachments[i].texture = color[i];
				pass.colorAttachments[i].loadAction = MTLLoadActionLoad;
				pass.colorAttachments[i].storeAction = MTLStoreActionStore;
			}

			s_encoder_color[i] = color[i];
		}

		if (depth)
		{
			pass.depthAttachment.texture = depth;
			pass.depthAttachment.loadAction = MTLLoadActionLoad;
			pass.depthAttachment.storeAction = MTLStoreActionStore;

			if (has_stencil)
			{
				pass.stencilAttachment.texture = depth;
				pass.stencilAttachment.loadAction = MTLLoadActionLoad;
				pass.stencilAttachment.storeAction = MTLStoreActionStore;
			}
		}

		s_encoder_depth = depth;
		s_encoder = [command_buffer() renderCommandEncoderWithDescriptor:pass];
		s_encoder.label = @"RSX draws";
		is_new = true;
		return s_encoder;
	}

	id<MTLCommandBuffer> take_command_buffer()
	{
		std::lock_guard lock(s_lock);
		close_render_pass();
		id<MTLCommandBuffer> cmd = s_pending;
		s_pending = nil;
		return cmd;
	}
}

namespace mtl
{
	bool is_depth_format(pixel_format format)
	{
		return format == pixel_format::depth16 || format == pixel_format::depth32f || format == pixel_format::depth32f_stencil8;
	}

	bool has_stencil(pixel_format format)
	{
		return format == pixel_format::depth32f_stencil8;
	}

	bool init_device(std::string& device_name, std::string& error)
	{
		std::lock_guard lock(s_lock);

		if (s_device)
		{
			device_name = s_device.name.UTF8String;
			return true;
		}

		@autoreleasepool
		{
			s_device = MTLCreateSystemDefaultDevice();

			if (!s_device)
			{
				error = "MTLCreateSystemDefaultDevice failed";
				return false;
			}

			s_queue = [s_device newCommandQueue];
			s_queue.label = @"RPCS3 RSX";
			device_name = s_device.name.UTF8String;
			return true;
		}
	}

	void shutdown_device()
	{
		std::lock_guard lock(s_lock);

		@autoreleasepool
		{
			internal::close_render_pass();

			if (s_pending)
			{
				[s_pending commit];
				[s_pending waitUntilCompleted];
				s_pending = nil;
			}

			if (s_queue)
			{
				// Drain everything still in flight (e.g. presentation)
				id<MTLCommandBuffer> cmd = [s_queue commandBuffer];
				[cmd commit];
				[cmd waitUntilCompleted];
			}

			s_queue = nil;
			s_device = nil;
		}
	}

	static texture_desc make_2d_desc(u32 width, u32 height, pixel_format format, u32 usage)
	{
		texture_desc desc{};
		desc.type = texture_type::tex_2d;
		desc.width = width;
		desc.height = height;
		desc.format = format;
		desc.usage = usage;
		return desc;
	}

	texture::texture(u32 width, u32 height, pixel_format format, u32 usage)
		: texture(make_2d_desc(width, height, format, usage))
	{
	}

	texture::texture(const texture_desc& info)
		: m_desc(info)
	{
		@autoreleasepool
		{
			const MTLPixelFormat mtl_format = to_mtl(info.format);

			if (!s_device || mtl_format == MTLPixelFormatInvalid || !info.width || !info.height)
			{
				return;
			}

			m_desc.depth = std::max(1u, info.type == texture_type::tex_3d ? info.depth : 1u);
			m_desc.levels = std::max(1u, info.levels);

			MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
			desc.pixelFormat = mtl_format;
			desc.width = info.width;
			desc.height = info.type == texture_type::tex_1d ? 1 : info.height;
			desc.depth = m_desc.depth;
			desc.mipmapLevelCount = m_desc.levels;
			desc.storageMode = MTLStorageModePrivate;

			switch (info.type)
			{
			case texture_type::tex_1d: desc.textureType = MTLTextureType1D; break;
			case texture_type::tex_2d: desc.textureType = MTLTextureType2D; break;
			case texture_type::tex_3d: desc.textureType = MTLTextureType3D; break;
			case texture_type::tex_cube: desc.textureType = MTLTextureTypeCube; desc.height = info.width; break;
			}

			// Views are used for channel swizzles and depth/stencil aspects
			desc.usage = MTLTextureUsagePixelFormatView;
			if (info.usage & usage_sampled) desc.usage |= MTLTextureUsageShaderRead;
			if (info.usage & usage_render_target) desc.usage |= MTLTextureUsageRenderTarget;

			id<MTLTexture> tex = [s_device newTextureWithDescriptor:desc];
			m_handle = (__bridge_retained void*)tex;

			if (info.type == texture_type::tex_cube)
			{
				m_desc.height = info.width;
			}
		}
	}

	texture_view::~texture_view()
	{
		if (m_handle)
		{
			CFRelease(m_handle);
			m_handle = nullptr;
		}
	}

	void texture::set_native_component_layout(const std::array<swizzle, 4>& layout)
	{
		if (layout != m_native_layout)
		{
			m_native_layout = layout;
			m_views.clear();
		}
	}

	texture_view* texture::get_view(const swizzle_rgba& mapping, image_aspect aspect, u32 remap_encoding)
	{
		if (!m_handle)
		{
			return nullptr;
		}

		u64 key = (static_cast<u64>(remap_encoding) << 32) | (static_cast<u64>(aspect) << 16);
		for (int i = 0; i < 4; ++i)
		{
			key |= static_cast<u64>(mapping[i]) << (i * 4);
		}

		if (auto found = m_views.find(key); found != m_views.end())
		{
			return found->second.get();
		}

		@autoreleasepool
		{
			id<MTLTexture> base = (__bridge id<MTLTexture>)m_handle;
			MTLPixelFormat view_format = base.pixelFormat;

			if (aspect == image_aspect::stencil && has_stencil(m_desc.format))
			{
				view_format = MTLPixelFormatX32_Stencil8;
			}

			const MTLTextureSwizzleChannels channels = MTLTextureSwizzleChannelsMake(
				static_cast<MTLTextureSwizzle>(mapping[0]),
				static_cast<MTLTextureSwizzle>(mapping[1]),
				static_cast<MTLTextureSwizzle>(mapping[2]),
				static_cast<MTLTextureSwizzle>(mapping[3]));

			id<MTLTexture> view = [base newTextureViewWithPixelFormat:view_format
				textureType:base.textureType
				levels:NSMakeRange(0, base.mipmapLevelCount)
				slices:NSMakeRange(0, base.arrayLength * (base.textureType == MTLTextureTypeCube ? 6 : 1))
				swizzle:channels];

			if (!view)
			{
				return nullptr;
			}

			auto result = std::make_unique<texture_view>(this, (__bridge_retained void*)view, mapping, aspect, remap_encoding);
			auto ptr = result.get();
			m_views.emplace(key, std::move(result));
			return ptr;
		}
	}

	bool is_compressed_format(pixel_format format)
	{
		return format == pixel_format::bc1 || format == pixel_format::bc2 || format == pixel_format::bc3;
	}

	u32 get_format_block_size(pixel_format format)
	{
		switch (format)
		{
		case pixel_format::r8:
		case pixel_format::r8_uint:
			return 1;
		case pixel_format::b5g6r5:
		case pixel_format::bgr5a1:
		case pixel_format::a1bgr5:
		case pixel_format::abgr4:
		case pixel_format::rg8:
		case pixel_format::rg8_snorm:
		case pixel_format::r16:
		case pixel_format::r16_uint:
		case pixel_format::depth16:
			return 2;
		case pixel_format::bgra8:
		case pixel_format::rgba8:
		case pixel_format::rg16:
		case pixel_format::rg16f:
		case pixel_format::r32f:
		case pixel_format::r32_uint:
		case pixel_format::depth32f:
			return 4;
		case pixel_format::depth32f_stencil8:
		case pixel_format::x32_stencil8:
		case pixel_format::rgba16f:
		case pixel_format::rg32_uint:
		case pixel_format::bc1:
			return 8;
		case pixel_format::rgba32f:
		case pixel_format::rgba32_uint:
		case pixel_format::bc2:
		case pixel_format::bc3:
			return 16;
		case pixel_format::invalid:
			break;
		}

		return 4;
	}

	texture::~texture()
	{
		m_views.clear();

		if (m_handle)
		{
			// Balance __bridge_retained. Pending command buffers keep their own references to textures they use.
			CFRelease(m_handle);
			m_handle = nullptr;
		}
	}

	void texture::set_label(const std::string& label)
	{
		if (m_handle)
		{
			as_texture(*this).label = @(label.c_str());
		}
	}

	void clear_color(texture& dst, const float rgba[4], u32 write_mask)
	{
		if (!dst.valid() || dst.is_depth())
		{
			return;
		}

		@autoreleasepool
		{
			// TODO: masked and scissored clears need a draw-based clear; the load action clears the whole image
			(void)write_mask;

			internal::close_render_pass();

			MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
			pass.colorAttachments[0].texture = as_texture(dst);
			pass.colorAttachments[0].loadAction = MTLLoadActionClear;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			pass.colorAttachments[0].clearColor = MTLClearColorMake(rgba[0], rgba[1], rgba[2], rgba[3]);

			id<MTLRenderCommandEncoder> enc = [internal::command_buffer() renderCommandEncoderWithDescriptor:pass];
			enc.label = @"RSX clear color";
			[enc endEncoding];
		}
	}

	void clear_depth_stencil(texture& dst, bool clear_depth, float depth, bool clear_stencil, u8 stencil)
	{
		if (!dst.valid() || !dst.is_depth() || (!clear_depth && !clear_stencil))
		{
			return;
		}

		@autoreleasepool
		{
			internal::close_render_pass();

			MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
			pass.depthAttachment.texture = as_texture(dst);
			pass.depthAttachment.loadAction = clear_depth ? MTLLoadActionClear : MTLLoadActionLoad;
			pass.depthAttachment.storeAction = MTLStoreActionStore;
			pass.depthAttachment.clearDepth = depth;

			if (has_stencil(dst.format()))
			{
				pass.stencilAttachment.texture = as_texture(dst);
				pass.stencilAttachment.loadAction = clear_stencil ? MTLLoadActionClear : MTLLoadActionLoad;
				pass.stencilAttachment.storeAction = MTLStoreActionStore;
				pass.stencilAttachment.clearStencil = stencil;
			}

			id<MTLRenderCommandEncoder> enc = [internal::command_buffer() renderCommandEncoderWithDescriptor:pass];
			enc.label = @"RSX clear depth/stencil";
			[enc endEncoding];
		}
	}

	bool copy_region(texture& src, texture& dst, u32 src_x, u32 src_y, u32 dst_x, u32 dst_y, u32 width, u32 height)
	{
		if (!src.valid() || !dst.valid() || src.format() != dst.format())
		{
			return false;
		}

		// Clamp to both images
		if (src_x >= src.width() || src_y >= src.height() || dst_x >= dst.width() || dst_y >= dst.height())
		{
			return false;
		}

		width = std::min({width, src.width() - src_x, dst.width() - dst_x});
		height = std::min({height, src.height() - src_y, dst.height() - dst_y});

		if (!width || !height)
		{
			return false;
		}

		@autoreleasepool
		{
			internal::close_render_pass();

			id<MTLBlitCommandEncoder> blit = [internal::command_buffer() blitCommandEncoder];
			[blit copyFromTexture:as_texture(src) sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(src_x, src_y, 0) sourceSize:MTLSizeMake(width, height, 1)
				toTexture:as_texture(dst) destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(dst_x, dst_y, 0)];
			[blit endEncoding];
		}

		return true;
	}

	void finish()
	{
		@autoreleasepool
		{
			if (id<MTLCommandBuffer> cmd = internal::take_command_buffer())
			{
				[cmd commit];
				[cmd waitUntilCompleted];
			}
		}
	}

	void flush()
	{
		@autoreleasepool
		{
			if (id<MTLCommandBuffer> cmd = internal::take_command_buffer())
			{
				[cmd commit];
			}
		}
	}

	shader_function::~shader_function()
	{
		if (m_handle)
		{
			CFRelease(m_handle);
			m_handle = nullptr;
		}
	}

	bool shader_function::create(const std::string& msl, const std::string& entry, std::string& error)
	{
		if (!s_device)
		{
			error = "Metal device is not initialized";
			return false;
		}

		@autoreleasepool
		{
			MTLCompileOptions* options = [MTLCompileOptions new];
			options.languageVersion = MTLLanguageVersion2_4;

			// RSX programs move raw bits through floats (packed depth, UP4/PK4 conversions) and rely on IEEE
			// NaN/Inf/denormal handling; fast math may reassociate or drop those.
			if (@available(macOS 15.0, *))
			{
				options.mathMode = MTLMathModeSafe;
				options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
			}
			else
			{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
				options.fastMathEnabled = NO;
#pragma clang diagnostic pop
			}

			NSError* ns_error = nil;
			id<MTLLibrary> library = [s_device newLibraryWithSource:@(msl.c_str()) options:options error:&ns_error];

			if (!library)
			{
				error = ns_error ? ns_error.localizedDescription.UTF8String : "unknown error";
				return false;
			}

			id<MTLFunction> function = [library newFunctionWithName:@(entry.c_str())];

			if (!function)
			{
				error = "entry point '" + entry + "' not found";
				return false;
			}

			if (m_handle)
			{
				CFRelease(m_handle);
			}

			m_handle = (__bridge_retained void*)function;
			return true;
		}
	}
}
