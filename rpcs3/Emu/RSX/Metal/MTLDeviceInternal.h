#pragma once

// Objective-C++ only: shared state of the Metal device layer for the .mm files.

#import <Metal/Metal.h>

namespace mtl::internal
{
	id<MTLDevice> device();
	id<MTLCommandQueue> queue();

	// Pending command buffer for resource operations (created on demand)
	id<MTLCommandBuffer> command_buffer();

	// Returns the pending command buffer (or nil) and forgets it; the caller commits it
	id<MTLCommandBuffer> take_command_buffer();
}
