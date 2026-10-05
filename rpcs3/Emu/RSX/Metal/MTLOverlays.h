#pragma once

#include "MTLOverlayDraw.h"

#include "util/types.hpp"
#include "Utilities/geometry.h"

namespace rsx::overlays
{
	struct overlay;
}

namespace mtl
{
	// Draws RPCS3's native overlays (rsx::overlays::display_manager views) for the Metal backend
	class ui_overlay_renderer
	{
	public:
		ui_overlay_renderer() = default;
		ui_overlay_renderer(const ui_overlay_renderer&) = delete;
		ui_overlay_renderer& operator=(const ui_overlay_renderer&) = delete;
		~ui_overlay_renderer();

		bool create();
		void destroy();

		// Draws 'ui' into 'encoder' (an id<MTLRenderCommandEncoder> passed as void*) inside 'viewport'
		void run(void* encoder, const areau& viewport, rsx::overlays::overlay& ui);

		// Forgets the temporary images of a disposed overlay
		void remove_temp_resources(u32 owner_uid);

	private:
		overlay::renderer* m_renderer = nullptr;
	};
}
