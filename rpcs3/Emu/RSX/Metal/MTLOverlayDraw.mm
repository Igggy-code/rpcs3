// Native overlay drawing (RPCS3's own UI) for the Metal backend.
// The shaders are a direct port of Program/GLSLSnippets/OverlayRenderVS.glsl / OverlayRenderFS.glsl
// (Vulkan flavour: top-left window origin, RGBA image data), with the Y axis flipped for Metal's NDC.

#import <Metal/Metal.h>

#include "MTLOverlayDraw.h"
#include "MTLDeviceInternal.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace mtl::overlay
{
	namespace
	{
		const char* s_overlay_shader = R"(
#include <metal_stdlib>
using namespace metal;

struct vs_config_t
{
	float4 ui_scale;
	float4 albedo;
	float4 viewport;
	float4 clip_bounds;
	uint options;
	uint pad0, pad1, pad2;
};

struct fs_config_t
{
	uint options;
	float timestamp;
	float blur_intensity;
	float pad;
	float4 sdf_params;
	float4 sdf_origin;
	float4 sdf_border_color;
};

struct v2f
{
	float4 position [[position]];
	float2 tc0;
	float4 color;
	float4 clip_rect;
};

static float4 clip_to_ndc(float4 coord, constant vs_config_t& cfg, bool flip_vertically)
{
	float4 ret = (coord * cfg.ui_scale.zwzw) / cfg.ui_scale.xyxy;
	if (flip_vertically) ret.yw = 1.f - ret.yw;
	return ret;
}

static float4 ndc_to_window(float4 coord, constant vs_config_t& cfg)
{
	return fma(coord, cfg.viewport.xyxy, cfg.viewport.zwzw);
}

static float4 make_aabb(float4 coords)
{
	float4 result = coords;
	if (coords.x > coords.z) result.xz = coords.zx;
	if (coords.y > coords.w) result.yw = coords.wy;
	return result;
}

vertex v2f overlay_vs(uint vid [[vertex_id]],
	const device float4* vertices [[buffer(0)]],
	constant vs_config_t& cfg [[buffer(1)]])
{
	const float4 in_pos = vertices[vid];
	const bool no_vertex_snap = (cfg.options & 1u) != 0;
	const bool flip_vertically = (cfg.options & 2u) != 0;

	v2f out;
	out.tc0 = in_pos.zw;
	out.color = cfg.albedo;
	out.clip_rect = make_aabb(ndc_to_window(clip_to_ndc(cfg.clip_bounds, cfg, flip_vertically), cfg));

	float4 pos = float4(clip_to_ndc(in_pos, cfg, flip_vertically).xy, 0.5f, 1.f);
	if (!no_vertex_snap)
	{
		pos.xy = floor(fma(pos.xy, cfg.viewport.xy, float2(0.5f))) / cfg.viewport.xy;
	}

	pos = (pos + pos) - 1.f;
	pos.y = -pos.y; // UI space is Y-down, Metal NDC is Y-up
	out.position = pos;
	return out;
}

static float4 sdf_blend(float sd, float border_width, float4 inner_color, float4 border_color, float4 outer_color)
{
	const float fw = fwidth(sd);
	const float a = smoothstep(-border_width + fw, -border_width - fw, sd);
	const float b = smoothstep(fw, -fw, sd);
	float4 color = mix(outer_color, border_color, b);
	return mix(color, inner_color, a);
}

static float sdf_fn(uint sdf, float2 frag_coord, constant fs_config_t& cfg)
{
	const float2 p = floor(frag_coord) - cfg.sdf_origin.xy;
	const float2 hs = cfg.sdf_params.xy;
	const float r = cfg.sdf_params.z;
	float2 v;

	switch (sdf)
	{
	case 1:
		return (length(p / hs) - 1.f) * length(hs);
	case 2:
		v = abs(p) - hs;
		return length(max(v, 0.f)) + min(max(v.x, v.y), 0.f);
	case 3:
		v = abs(p) - (hs - r);
		return length(max(v, 0.f)) + min(max(v.x, v.y), 0.f) - r;
	default:
		return -1.f;
	}
}

static float4 blur_sample(texture2d<float> tex, sampler smp, float2 coord, float2 tex_offset)
{
	const float weights[9] = { 1.f, 2.f, 1.f, 2.f, 4.f, 2.f, 1.f, 2.f, 1.f };
	float4 blurred = 0.f;
	int n = 0;
	for (int y = -1; y <= 1; ++y)
	{
		for (int x = -1; x <= 1; ++x)
		{
			blurred += tex.sample(smp, coord + float2(x, y) * tex_offset) * weights[n++];
		}
	}
	return blurred / 16.f;
}

static float4 sample_image(texture2d<float> tex, sampler smp, float2 coord, float blur_strength)
{
	const float4 original = tex.sample(smp, coord);
	if (blur_strength == 0.f) return original;

	const float2 constraints = 1.f / float2(640.f, 360.f);
	const float2 res_offset = 1.f / float2(tex.get_width(), tex.get_height());
	const float2 tex_offset = max(res_offset, constraints);

	const float4 blur0 = blur_sample(tex, smp, coord + float2(-res_offset.x, 0.f), tex_offset);
	const float4 blur1 = blur_sample(tex, smp, coord + float2(res_offset.x, 0.f), tex_offset);
	const float4 blur2 = blur_sample(tex, smp, coord + float2(0.f, res_offset.y), tex_offset);
	return mix(original, (blur0 + blur1 + blur2) / 3.f, blur_strength);
}

fragment float4 overlay_fs(v2f in [[stage_in]],
	constant fs_config_t& cfg [[buffer(0)]],
	texture2d<float> fs0 [[texture(0)]],
	texture2d_array<float> fs1 [[texture(1)]],
	sampler smp [[sampler(0)]])
{
	const bool clip_fragments = (cfg.options & 1u) != 0;
	const bool pulse_glow = (cfg.options & 2u) != 0;
	const uint sampler_mode = (cfg.options >> 2) & 3u;
	const uint sdf = (cfg.options >> 4) & 3u;

	const float2 frag = in.position.xy;
	if (clip_fragments)
	{
		if (frag.x < in.clip_rect.x || frag.x > in.clip_rect.z || frag.y < in.clip_rect.y || frag.y > in.clip_rect.w)
		{
			discard_fragment();
		}
	}

	float4 diff_color = in.color;
	if (pulse_glow)
	{
		diff_color.a *= (sin(cfg.timestamp) + 1.f) * 0.5f;
	}

	if (sdf != 0)
	{
		const float d = sdf_fn(sdf, frag, cfg);
		diff_color = sdf_blend(d, cfg.sdf_params.w, diff_color, cfg.sdf_border_color, float4(0.f));
	}

	switch (sampler_mode)
	{
	default:
	case 0:
		return diff_color;
	case 1:
		return fs0.sample(smp, in.tc0).rrrr * diff_color;
	case 2:
		return fs1.sample(smp, float2(in.tc0.x, fract(in.tc0.y)), uint(trunc(in.tc0.y))).rrrr * diff_color;
	case 3:
		return sample_image(fs0, smp, in.tc0, cfg.blur_intensity) * diff_color;
	}

	return diff_color;
}
)";

		struct stored_texture
		{
			id<MTLTexture> texture = nil;
			u32 width = 0, height = 0, layers = 0;
			u32 owner = ~0u;
		};
	}

	struct renderer
	{
		id<MTLLibrary> library = nil;
		std::unordered_map<NSUInteger, id<MTLRenderPipelineState>> pipelines; // By color pixel format
		id<MTLSamplerState> sampler = nil;
		id<MTLTexture> dummy_2d = nil;
		id<MTLTexture> dummy_array = nil;
		std::unordered_map<u64, stored_texture> textures;
	};

	namespace
	{
		id<MTLRenderPipelineState> get_pipeline(renderer* r, MTLPixelFormat format)
		{
			if (auto found = r->pipelines.find(static_cast<NSUInteger>(format)); found != r->pipelines.end())
			{
				return found->second;
			}

			MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
			desc.label = @"RSX overlay";
			desc.vertexFunction = [r->library newFunctionWithName:@"overlay_vs"];
			desc.fragmentFunction = [r->library newFunctionWithName:@"overlay_fs"];
			desc.colorAttachments[0].pixelFormat = format;
			desc.colorAttachments[0].blendingEnabled = YES;
			desc.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
			desc.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
			desc.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
			desc.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;

			NSError* error = nil;
			id<MTLRenderPipelineState> pso = [internal::device() newRenderPipelineStateWithDescriptor:desc error:&error];
			if (!pso)
			{
				NSLog(@"RPCS3 Metal: overlay pipeline failed: %@", error);
			}

			r->pipelines[static_cast<NSUInteger>(format)] = pso;
			return pso;
		}

		id<MTLTexture> make_texture(u32 width, u32 height, u32 layers, bool font)
		{
			MTLTextureDescriptor* desc = [MTLTextureDescriptor new];
			desc.textureType = font ? MTLTextureType2DArray : MTLTextureType2D;
			desc.pixelFormat = font ? MTLPixelFormatR8Unorm : MTLPixelFormatRGBA8Unorm;
			desc.width = std::max(width, 1u);
			desc.height = std::max(height, 1u);
			desc.arrayLength = font ? std::max(layers, 1u) : 1;
			desc.usage = MTLTextureUsageShaderRead;
			desc.storageMode = MTLStorageModeShared;
			return [internal::device() newTextureWithDescriptor:desc];
		}
	}

	renderer* create_renderer(std::string& error)
	{
		@autoreleasepool
		{
			auto r = new renderer();

			NSError* ns_error = nil;
			MTLCompileOptions* options = [MTLCompileOptions new];
			r->library = [internal::device() newLibraryWithSource:@(s_overlay_shader) options:options error:&ns_error];

			if (!r->library)
			{
				error = ns_error ? ns_error.localizedDescription.UTF8String : "unknown error";
				delete r;
				return nullptr;
			}

			MTLSamplerDescriptor* smp = [MTLSamplerDescriptor new];
			smp.minFilter = MTLSamplerMinMagFilterLinear;
			smp.magFilter = MTLSamplerMinMagFilterLinear;
			smp.sAddressMode = MTLSamplerAddressModeClampToEdge;
			smp.tAddressMode = MTLSamplerAddressModeClampToEdge;
			r->sampler = [internal::device() newSamplerStateWithDescriptor:smp];

			r->dummy_2d = make_texture(1, 1, 1, false);
			r->dummy_array = make_texture(1, 1, 1, true);
			return r;
		}
	}

	void destroy_renderer(renderer* r)
	{
		@autoreleasepool
		{
			delete r;
		}
	}

	bool has_texture(renderer* r, u64 key, u32 width, u32 height, u32 layers)
	{
		const auto found = r->textures.find(key);
		return found != r->textures.end() && found->second.width == width && found->second.height == height && found->second.layers == layers;
	}

	void upload_texture(renderer* r, u64 key, u32 width, u32 height, u32 layers, bool font, const void* data, u32 owner)
	{
		@autoreleasepool
		{
			auto& entry = r->textures[key];

			if (!entry.texture || entry.width != width || entry.height != height || entry.layers != layers)
			{
				// The previous texture stays alive while command buffers that sampled it are in flight
				entry.texture = make_texture(width, height, layers, font);
				entry.width = width;
				entry.height = height;
				entry.layers = layers;
			}

			entry.owner = owner;

			if (!data || !width || !height)
			{
				return;
			}

			const u32 bpp = font ? 1 : 4;
			for (u32 layer = 0; layer < std::max(layers, 1u); ++layer)
			{
				const auto src = static_cast<const u8*>(data) + static_cast<size_t>(layer) * width * height * bpp;
				[entry.texture replaceRegion:MTLRegionMake2D(0, 0, width, height) mipmapLevel:0 slice:layer withBytes:src
					bytesPerRow:width * bpp bytesPerImage:static_cast<NSUInteger>(width) * height * bpp];
			}
		}
	}

	void remove_textures_of_owner(renderer* r, u32 owner)
	{
		std::erase_if(r->textures, [owner](const auto& entry) { return entry.second.owner == owner; });
	}

	void draw(renderer* r, void* encoder, const float viewport[4], const draw_cmd& cmd)
	{
		if (!cmd.vertices || !cmd.vertex_count || viewport[2] <= 0.f || viewport[3] <= 0.f)
		{
			return;
		}

		@autoreleasepool
		{
			id<MTLRenderCommandEncoder> enc = (__bridge id<MTLRenderCommandEncoder>)encoder;
			// Pipelines are created for the format of the attachment being drawn to (the drawable)
			id<MTLRenderPipelineState> pso = get_pipeline(r, MTLPixelFormatBGRA8Unorm);
			if (!pso)
			{
				return;
			}

			// Geometry: quads and fans become indexed triangle lists (Metal has neither)
			std::vector<u32> indices;
			MTLPrimitiveType type = MTLPrimitiveTypeTriangleStrip;

			switch (cmd.type)
			{
			case primitive::quad_list:
				type = MTLPrimitiveTypeTriangle;
				for (u32 base = 0; base + 3 < cmd.vertex_count; base += 4)
				{
					indices.insert(indices.end(), { base, base + 1, base + 2, base + 2, base + 1, base + 3 });
				}
				break;
			case primitive::triangle_fan:
				type = MTLPrimitiveTypeTriangle;
				for (u32 i = 1; i + 1 < cmd.vertex_count; ++i)
				{
					indices.insert(indices.end(), { 0u, i, i + 1 });
				}
				break;
			case primitive::triangle_strip:
				type = MTLPrimitiveTypeTriangleStrip;
				break;
			case primitive::line_list:
				type = MTLPrimitiveTypeLine;
				break;
			case primitive::line_strip:
				type = MTLPrimitiveTypeLineStrip;
				break;
			}

			const NSUInteger vertex_bytes = static_cast<NSUInteger>(cmd.vertex_count) * 16;

			[enc setRenderPipelineState:pso];
			[enc setViewport:MTLViewport{ viewport[0], viewport[1], viewport[2], viewport[3], 0.0, 1.0 }];

			if (vertex_bytes <= 4096)
			{
				[enc setVertexBytes:cmd.vertices length:vertex_bytes atIndex:0];
			}
			else
			{
				id<MTLBuffer> vb = [internal::device() newBufferWithBytes:cmd.vertices length:vertex_bytes options:MTLResourceStorageModeShared];
				[enc setVertexBuffer:vb offset:0 atIndex:0];
			}

			[enc setVertexBytes:&cmd.vs length:sizeof(cmd.vs) atIndex:1];
			[enc setFragmentBytes:&cmd.fs length:sizeof(cmd.fs) atIndex:0];
			[enc setFragmentSamplerState:r->sampler atIndex:0];

			id<MTLTexture> image = r->dummy_2d;
			id<MTLTexture> font = r->dummy_array;

			if (cmd.source != sampler_source::none)
			{
				if (auto found = r->textures.find(cmd.texture_key); found != r->textures.end() && found->second.texture)
				{
					if (cmd.source == sampler_source::font)
					{
						font = found->second.texture;
					}
					else
					{
						image = found->second.texture;
					}
				}
			}

			[enc setFragmentTexture:image atIndex:0];
			[enc setFragmentTexture:font atIndex:1];

			if (!indices.empty())
			{
				id<MTLBuffer> ib = [internal::device() newBufferWithBytes:indices.data() length:indices.size() * sizeof(u32) options:MTLResourceStorageModeShared];
				[enc drawIndexedPrimitives:type indexCount:indices.size() indexType:MTLIndexTypeUInt32 indexBuffer:ib indexBufferOffset:0];
			}
			else if (cmd.type == primitive::quad_list || cmd.type == primitive::triangle_fan)
			{
				return;
			}
			else
			{
				[enc drawPrimitives:type vertexStart:0 vertexCount:cmd.vertex_count];
			}
		}
	}
}
