#pragma once

// Frame dump for debugging the Metal backend.
//
// Create the file "<cache dir>/metal_dump_request" while a game runs: the next full frame is dumped to
// "<cache dir>/metal_dump/<timestamp>/": a text log of every draw (render targets, programs, textures,
// blend/depth state) plus PNG images of each render pass result and of every sampled texture.

#include "MTLDevice.h"

#include <string>

namespace mtl::debug
{
	// Writes an image of level 0 / layer 0 of 'tex' (color as RGB, alpha as a separate gray image for
	// formats with alpha, depth as contrast-stretched gray). Returns a short description of what was written.
	std::string dump_texture(texture& tex, const std::string& path_without_extension, bool with_alpha);

	// Encodes tightly packed 8-bit gray (channels = 1) or RGB (channels = 3) data as PNG
	bool write_png(const std::string& path, u32 width, u32 height, u32 channels, const u8* data);

	const char* format_name(pixel_format format);
}
