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

	id<MTLCommandBuffer> take_command_buffer()
	{
		std::lock_guard lock(s_lock);
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

	texture::texture(u32 width, u32 height, pixel_format format, u32 usage)
		: m_width(width), m_height(height), m_format(format)
	{
		@autoreleasepool
		{
			const MTLPixelFormat mtl_format = to_mtl(format);

			if (!s_device || mtl_format == MTLPixelFormatInvalid || !width || !height)
			{
				return;
			}

			MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtl_format width:width height:height mipmapped:NO];
			desc.storageMode = MTLStorageModePrivate;
			desc.usage = MTLTextureUsageUnknown;

			if (usage & usage_sampled) desc.usage |= MTLTextureUsageShaderRead;
			if (usage & usage_render_target) desc.usage |= MTLTextureUsageRenderTarget;

			id<MTLTexture> tex = [s_device newTextureWithDescriptor:desc];
			m_handle = (__bridge_retained void*)tex;
		}
	}

	texture::~texture()
	{
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
			id<MTLBlitCommandEncoder> blit = [internal::command_buffer() blitCommandEncoder];
			[blit copyFromTexture:as_texture(src) sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(src_x, src_y, 0) sourceSize:MTLSizeMake(width, height, 1)
				toTexture:as_texture(dst) destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(dst_x, dst_y, 0)];
			[blit endEncoding];
		}

		return true;
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
