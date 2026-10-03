// Metal presentation for the native Metal RSX backend (MTLGSRender).
// Objective-C++, compiled with ARC. Includes no RPCS3 headers on purpose.

// Objective-C bridging casts are C-style by nature; the project builds with -Werror=old-style-cast
#pragma clang diagnostic ignored "-Wold-style-cast"

#import <AppKit/AppKit.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include "MTLPresenter.h"

#include <array>

namespace
{
	// Fullscreen triangle; the fragment shader swizzles big-endian A8R8G8B8 (bytes A,R,G,B read as RGBA8) to RGB
	constexpr const char* s_present_shader = R"(
#include <metal_stdlib>
using namespace metal;

struct vs_out
{
	float4 pos [[position]];
	float2 uv;
};

vertex vs_out present_vs(uint vid [[vertex_id]])
{
	const float2 uv = float2((vid << 1) & 2, vid & 2);
	vs_out o;
	o.pos = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
	o.uv = uv;
	return o;
}

fragment float4 present_fs(vs_out in [[stage_in]], texture2d<float> tex [[texture(0)]], sampler smp [[sampler(0)]])
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
		id<MTLDevice> device = nil;
		id<MTLCommandQueue> queue = nil;
		CAMetalLayer* layer = nil;
		id<MTLRenderPipelineState> pipeline = nil;
		id<MTLSamplerState> sampler = nil;

		// Ring of upload textures so the CPU never overwrites one the GPU is still sampling
		std::array<id<MTLTexture>, 3> textures{};
		std::uint32_t texture_index = 0;
		dispatch_semaphore_t frames_in_flight = nullptr;

		std::string device_name;
	};

	static id<MTLTexture> get_upload_texture(presenter* p, std::uint32_t width, std::uint32_t height)
	{
		auto& tex = p->textures[p->texture_index];
		p->texture_index = (p->texture_index + 1) % p->textures.size();

		if (!tex || tex.width != width || tex.height != height)
		{
			MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:width height:height mipmapped:NO];
			desc.usage = MTLTextureUsageShaderRead;
			desc.storageMode = MTLStorageModeShared;
			tex = [p->device newTextureWithDescriptor:desc];
		}

		return tex;
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

			NSView* view = (__bridge NSView*)nsview;
			CALayer* view_layer = view.layer;

			if (![view_layer isKindOfClass:[CAMetalLayer class]])
			{
				// gs_frame requests QSurface::MetalSurface for this renderer, so Qt should already have created a CAMetalLayer
				error = "The game window has no CAMetalLayer";
				return nullptr;
			}

			auto p = new presenter();
			p->device = MTLCreateSystemDefaultDevice();

			if (!p->device)
			{
				error = "MTLCreateSystemDefaultDevice failed";
				delete p;
				return nullptr;
			}

			p->device_name = p->device.name.UTF8String;
			p->queue = [p->device newCommandQueue];
			p->frames_in_flight = dispatch_semaphore_create(p->textures.size());

			p->layer = (CAMetalLayer*)view_layer;
			p->layer.device = p->device;
			p->layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
			p->layer.framebufferOnly = YES;
			p->layer.maximumDrawableCount = 3;
			p->layer.displaySyncEnabled = vsync ? YES : NO;

			NSError* ns_error = nil;
			id<MTLLibrary> library = [p->device newLibraryWithSource:@(s_present_shader) options:nil error:&ns_error];

			if (!library)
			{
				error = std::string("Present shader compilation failed: ") + (ns_error ? ns_error.localizedDescription.UTF8String : "unknown error");
				delete p;
				return nullptr;
			}

			MTLRenderPipelineDescriptor* pipe_desc = [MTLRenderPipelineDescriptor new];
			pipe_desc.vertexFunction = [library newFunctionWithName:@"present_vs"];
			pipe_desc.fragmentFunction = [library newFunctionWithName:@"present_fs"];
			pipe_desc.colorAttachments[0].pixelFormat = p->layer.pixelFormat;

			p->pipeline = [p->device newRenderPipelineStateWithDescriptor:pipe_desc error:&ns_error];

			if (!p->pipeline)
			{
				error = std::string("Present pipeline creation failed: ") + (ns_error ? ns_error.localizedDescription.UTF8String : "unknown error");
				delete p;
				return nullptr;
			}

			MTLSamplerDescriptor* smp_desc = [MTLSamplerDescriptor new];
			smp_desc.minFilter = MTLSamplerMinMagFilterLinear;
			smp_desc.magFilter = MTLSamplerMinMagFilterLinear;
			smp_desc.sAddressMode = MTLSamplerAddressModeClampToEdge;
			smp_desc.tAddressMode = MTLSamplerAddressModeClampToEdge;
			p->sampler = [p->device newSamplerStateWithDescriptor:smp_desc];

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
			// Wait for all frames in flight before releasing their resources
			if (p->queue)
			{
				id<MTLCommandBuffer> cmd = [p->queue commandBuffer];
				[cmd commit];
				[cmd waitUntilCompleted];
			}

			delete p;
		}
	}

	const std::string& get_device_name(const presenter* p)
	{
		return p->device_name;
	}

	void set_vsync(presenter* p, bool vsync)
	{
		p->layer.displaySyncEnabled = vsync ? YES : NO;
	}

	bool present(presenter* p, const present_params& params)
	{
		@autoreleasepool
		{
			if (!params.output_width || !params.output_height)
			{
				return false;
			}

			const CGSize size = CGSizeMake(params.output_width, params.output_height);

			if (!CGSizeEqualToSize(p->layer.drawableSize, size))
			{
				p->layer.drawableSize = size;
			}

			// Throttle the CPU to the number of upload textures
			dispatch_semaphore_wait(p->frames_in_flight, DISPATCH_TIME_FOREVER);

			id<MTLTexture> tex = nil;

			if (params.pixels && params.width && params.height && params.pitch >= params.width * 4)
			{
				tex = get_upload_texture(p, params.width, params.height);
				[tex replaceRegion:MTLRegionMake2D(0, 0, params.width, params.height) mipmapLevel:0 withBytes:params.pixels bytesPerRow:params.pitch];
			}

			id<CAMetalDrawable> drawable = [p->layer nextDrawable];

			if (!drawable)
			{
				dispatch_semaphore_signal(p->frames_in_flight);
				return false;
			}

			MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
			pass.colorAttachments[0].texture = drawable.texture;
			pass.colorAttachments[0].loadAction = MTLLoadActionClear;
			pass.colorAttachments[0].storeAction = MTLStoreActionStore;
			pass.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

			id<MTLCommandBuffer> cmd = [p->queue commandBuffer];
			id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];

			if (tex && params.viewport_width && params.viewport_height)
			{
				const MTLViewport viewport =
				{
					static_cast<double>(params.viewport_x), static_cast<double>(params.viewport_y),
					static_cast<double>(params.viewport_width), static_cast<double>(params.viewport_height),
					0.0, 1.0
				};

				[enc setViewport:viewport];
				[enc setRenderPipelineState:p->pipeline];
				[enc setFragmentTexture:tex atIndex:0];
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
