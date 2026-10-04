#pragma once

// Objective-C++ only: shared state of the Metal device layer for the .mm files.

#import <Metal/Metal.h>

#include "MTLDevice.h"

namespace mtl::internal
{
	id<MTLDevice> device();
	id<MTLCommandQueue> queue();

	// Pending command buffer for resource operations (created on demand)
	id<MTLCommandBuffer> command_buffer();

	// Returns the pending command buffer (or nil) and forgets it; the caller commits it.
	// Closes the open render pass first.
	id<MTLCommandBuffer> take_command_buffer();

	MTLPixelFormat to_mtl_format(mtl::pixel_format format);

	// Render pass used by draws. It stays open across draws while the attachments do not change;
	// every other command (clears, copies, presentation, flushes) closes it first.
	id<MTLRenderCommandEncoder> render_encoder(__unsafe_unretained const id<MTLTexture>* color, id<MTLTexture> depth, bool has_stencil, bool& is_new);
	void close_render_pass();

	// Buffer behind mtl::ring_alloc (MTLDraw.mm)
	id<MTLBuffer> ring_buffer();
}
