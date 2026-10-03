#pragma once

// Plain C++ interface to the Metal presentation code (MTLPresenter.mm).
// Kept free of Objective-C and RPCS3 headers so both sides compile independently.

#include <cstdint>
#include <string>

namespace mtl
{
	class texture;
	struct presenter;

	struct present_params
	{
		// Source, either an RSX surface rendered by the Metal backend...
		const texture* surface = nullptr;
		std::uint32_t surface_width = 0;  // Region of the surface to show (top-left origin)
		std::uint32_t surface_height = 0;

		// ...or the guest display buffer in memory (big-endian A8R8G8B8)
		const void* pixels = nullptr;
		std::uint32_t width = 0;
		std::uint32_t height = 0;
		std::uint32_t pitch = 0;

		std::uint32_t output_width = 0;  // Drawable size in pixels
		std::uint32_t output_height = 0;

		std::int32_t viewport_x = 0; // Letterboxed image area inside the drawable
		std::int32_t viewport_y = 0;
		std::uint32_t viewport_width = 0;
		std::uint32_t viewport_height = 0;
	};

	// nsview: the NSView* of the game window (display_handle_t on macOS). Requires mtl::init_device().
	presenter* create_presenter(void* nsview, bool vsync, std::string& error);
	void destroy_presenter(presenter* p);

	void set_vsync(presenter* p, bool vsync);

	// Submits all pending RSX work, then draws the source into the next drawable and presents it.
	// Returns false if no drawable was available.
	bool present(presenter* p, const present_params& params);
}
