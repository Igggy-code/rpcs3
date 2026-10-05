#pragma once

// Plain C++ interface to the native overlay drawing code (MTLOverlayDraw.mm): textures, shaders and
// draw calls for RPCS3's own UI (performance overlay, trophy popups, message dialogs...).
// MTLOverlays.cpp translates rsx::overlays objects into these calls.

#include "MTLDevice.h"

#include <string>

namespace mtl::overlay
{
	enum class primitive : u8
	{
		quad_list,      // 4 vertices per quad (drawn as separate triangle strips)
		triangle_strip,
		line_list,
		line_strip,
		triangle_fan,
	};

	enum class sampler_source : u8
	{
		none,
		image,  // RGBA8 2D texture
		font,   // R8 2D array (one layer per glyph page)
	};

	// Same layout as the uniform blocks of OverlayRenderVS/FS.glsl
	struct vertex_config
	{
		float ui_scale[4];
		float albedo[4];
		float viewport[4];
		float clip_bounds[4];
		u32 options;
		u32 pad[3];
	};

	struct fragment_config
	{
		u32 options;
		float timestamp;
		float blur_intensity;
		float pad;
		float sdf_params[4];
		float sdf_origin[4];
		float sdf_border_color[4];
	};

	struct draw_cmd
	{
		primitive type = primitive::quad_list;
		const float* vertices = nullptr; // vec4 per vertex: position.xy, texcoord.xy
		u32 vertex_count = 0;
		sampler_source source = sampler_source::none;
		u64 texture_key = 0;
		vertex_config vs{};
		fragment_config fs{};
	};

	struct renderer;

	renderer* create_renderer(std::string& error);
	void destroy_renderer(renderer* r);

	// Texture store. Keys are chosen by the caller; 'owner' groups temporary images for removal.
	bool has_texture(renderer* r, u64 key, u32 width, u32 height, u32 layers);
	void upload_texture(renderer* r, u64 key, u32 width, u32 height, u32 layers, bool font, const void* data, u32 owner);
	void remove_textures_of_owner(renderer* r, u32 owner);

	// Records one draw into a render encoder (an id<MTLRenderCommandEncoder> passed as void*) whose
	// color attachment is a BGRA8 drawable. 'viewport' is x, y, width, height in pixels.
	void draw(renderer* r, void* encoder, const float viewport[4], const draw_cmd& cmd);
}
