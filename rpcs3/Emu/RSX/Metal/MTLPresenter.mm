// Metal presentation for the native Metal RSX backend (MTLGSRender).
// Objective-C++, compiled with ARC. Includes no RPCS3 headers on purpose.

// Objective-C bridging casts are C-style by nature; the project builds with -Werror=old-style-cast
#pragma clang diagnostic ignored "-Wold-style-cast"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "MTLDevice.h"
#include "MTLDeviceInternal.h"
#include "MTLPresenter.h"

#include <algorithm>
#include <array>

namespace
{
	// Fullscreen triangle. 'uv_scale' selects the shown part of the source (surfaces can be larger than the display buffer).
	// present_fs_guest swizzles big-endian A8R8G8B8 memory (bytes A,R,G,B read as RGBA8) to RGB.
	constexpr const char* s_present_shader = R"(
#include <metal_stdlib>
using namespace metal;

struct vs_out
{
	float4 pos [[position]];
	float2 uv;
};

vertex vs_out present_vs(uint vid [[vertex_id]], constant float2& uv_scale [[buffer(0)]])
{
	const float2 uv = float2((vid << 1) & 2, vid & 2);
	vs_out o;
	o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
	o.uv = uv * uv_scale;
	return o;
}

fragment float4 present_fs_surface(vs_out in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]])
{
	return float4(tex.sample(smp, in.uv).rgb, 1.0);
}

fragment float4 present_fs_guest(vs_out in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]])
{
	const float4 c = tex.sample(smp, in.uv);
	return float4(c.g, c.b, c.a, 1.0);
}
)";
}

namespace mtl
{
	struct presenter
	{
		CAMetalLayer* layer = nil;
		id<MTLRenderPipelineState> surface_pipeline = nil;
		id<MTLRenderPipelineState> guest_pipeline = nil;
		id<MTLSamplerState> sampler = nil;

		// Ring of upload textures for guest memory so the CPU never overwrites one the GPU is still sampling
		std::array<id<MTLTexture>, 3> uploads{};
		std::uint32_t upload_index = 0;
		dispatch_semaphore_t frames_in_flight = nullptr;
	};

	static id<MTLTexture> get_upload_texture(presenter* p, std::uint32_t width, std::uint32_t height)
	{
		auto& tex = p->uploads[p->upload_index];
		p->upload_index = (p->upload_index + 1) % p->uploads.size();

		if (!tex || tex.width != width || tex.height != height)
		{
			MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
			desc.usage = MTLTextureUsageShaderRead;
			desc.storageMode = MTLStorageModeShared;
			tex = [internal::device() newTextureWithDescriptor:desc];
		}

		return tex;
	}

	static id<MTLRenderPipelineState> make_pipeline(id<MTLLibrary> library, NSString* fragment, MTLPixelFormat format, std::string& error)
	{
		MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
		desc.vertexFunction = [library newFunctionWithName:@"present_vs"];
		desc.fragmentFunction = [library newFunctionWithName:fragment];
		desc.colorAttachments[0].pixelFormat = format;

		NSError* ns_error = nil;
		id<MTLRenderPipelineState> pipeline = [internal::device() newRenderPipelineStateWithDescriptor:desc error:&ns_error];

		if (!pipeline)
		{
			error = std::string("Present pipeline creation failed: ") + (ns_error ? ns_error.localizedDescription.UTF8String : "unknown error");
		}

		return pipeline;
	}

	presenter* create_presenter(void* nsview, bool vsync, std::string& error)
	{
		@autoreleasepool
		{
			if (!nsview)
			{
				error = "No game window";
				return nullptr;
			}

			if (!internal::device())
			{
				error = "Metal device is not initialized";
				return nullptr;
			}

			NSView* view = (__bridge NSView*)nsview;
			CALayer* view_layer = view.layer;

			if (![view_layer isKindOfClass:[CAMetalLayer class]])
			{
				// gs_frame requests QSurface::MetalSurface for this renderer, so Qt should already have created a CAMetalLayer
				error = "The game window has no CAMetalLayer";
				return nullptr;
			}

			auto p = new presenter();
			p->frames_in_flight = dispatch_semaphore_create(p->uploads.size());

			p->layer = (CAMetalLayer*)view_layer;
			p->layer.device = internal::device();
			p->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
			p->layer.framebufferOnly = YES;
			p->layer.maximumDrawableCount = 3;
			p->layer.displaySyncEnabled = vsync ? YES : NO;

			NSError* ns_error = nil;
			id<MTLLibrary> library = [internal::device() newLibraryWithSource:@(s_present_shader) options:nil error:&ns_error];

			if (!library)
			{
				error = std::string("Present shader compilation failed: ") + (ns_error ? ns_error.localizedDescription.UTF8String : "unknown error");
				delete p;
				return nullptr;
			}

			p->surface_pipeline = make_pipeline(library, @"present_fs_surface", p->layer.pixelFormat, error);
			p->guest_pipeline = make_pipeline(library, @"present_fs_guest", p->layer.pixelFormat, error);

			if (!p->surface_pipeline || !p->guest_pipeline)
			{
				delete p;
				return nullptr;
			}

			MTLSamplerDescriptor* smp_desc = [MTLSamplerDescriptor new];
			smp_desc.minFilter = MTLSamplerMinMagFilterLinear;
			smp_desc.magFilter = MTLSamplerMinMagFilterLinear;
			smp_desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
			smp_desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
			p->sampler = [internal::device() newSamplerStateWithDescriptor:smp_desc];

			return p;
		}
	}

	void destroy_presenter(presenter* p)
	{
		if (!p)
		{
			return;
		}

		@autoreleasepool
		{
			// Wait for frames in flight before releasing their resources
			for (std::size_t i = 0; i < p->uploads.size(); ++i)
			{
				dispatch_semaphore_wait(p->frames_in_flight, DISPATCH_TIME_FOREVER);
			}

			for (std::size_t i = 0; i < p->uploads.size(); ++i)
			{
				dispatch_semaphore_signal(p->frames_in_flight);
			}

			delete p;
		}
	}

	void set_vsync(presenter* p, bool vsync)
	{
		p->layer.displaySyncEnabled = vsync ? YES : NO;
	}

	bool present(presenter* p, const present_params& params)
	{
		@autoreleasepool
		{
			// Everything recorded so far must reach the GPU before the frame that shows it
			id<MTLCommandBuffer> cmd = internal::command_buffer();
			internal::take_command_buffer();

			if (!params.output_width || !params.output_height)
			{
				[cmd commit];
				return false;
			}

			const CGSize size = CGSizeMake(params.output_width, params.output_height);

			if (!CGSizeEqualToSize(p->layer.drawableSize, size))
			{
				p->layer.drawableSize = size;
			}

			// Throttle the CPU to the number of upload textures. A GPU that stops completing work must not
			// block the RSX thread forever (the game would deadlock waiting for it): report and continue.
			if (dispatch_semaphore_wait(p->frames_in_flight, dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC)) != 0)
			{
				NSLog(@"RPCS3 Metal: presentation waited 2 s for the GPU");
				internal::report_gpu_stall("presentation");
			}

			id<MTLTexture> source = nil;
			id<MTLRenderPipelineState> pipeline = nil;
			float uv_scale[2] = { 1.f, 1.f };

			if (params.surface && params.surface->valid() && params.surface_width && params.surface_height)
			{
				source = (__bridge id<MTLTexture>)params.surface->native();
				pipeline = p->surface_pipeline;
				uv_scale[0] = std::min(1.f, static_cast<float>(params.surface_width) / source.width);
				uv_scale[1] = std::min(1.f, static_cast<float>(params.surface_height) / source.height);
			}
			else if (params.pixels && params.width && params.height && params.pitch >= params.width * 4)
			{
				source = get_upload_texture(p, params.width, params.height);
				[source replaceRegion:MTLRegionMake2D(0, 0, params.width, params.height) mipmapLevel:0 withBytes:params.pixels bytesPerRow:params.pitch];
				pipeline = p->guest_pipeline;
			}

			id<CAMetalDrawable> drawable = [p->layer nextDrawable];

			if (!drawable)
			{
				[cmd commit];
				dispatch_semaphore_signal(p->frames_in_flight);
				return false;
			}

			MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
			pass.colorAttachments[0].texture = drawable.texture;
			pass.colorAttachments[0].loadAction = MTLLoadActionClear;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

			id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
			enc.label = @"Present";

			if (source && params.viewport_width && params.viewport_height)
			{
				const MTLViewport viewport =
				{
					static_cast<double>(params.viewport_x), static_cast<double>(params.viewport_y),
					static_cast<double>(params.viewport_width), static_cast<double>(params.viewport_height),
					0.0, 1.0
				};

				[enc setViewport:viewport];
				[enc setRenderPipelineState:pipeline];
				[enc setVertexBytes:uv_scale length:sizeof(uv_scale) atIndex:0];
				[enc setFragmentTexture:source atIndex:0];
				[enc setFragmentSamplerState:p->sampler atIndex:0];
				[enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
			}

			[enc endEncoding];
			[cmd presentDrawable:drawable];

			dispatch_semaphore_t sema = p->frames_in_flight;
			[cmd addCompletedHandler:^(id<MTLCommandBuffer>)
			{
				dispatch_semaphore_signal(sema);
			}];

			[cmd commit];
			return true;
		}
	}
}
