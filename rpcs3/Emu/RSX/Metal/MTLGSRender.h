#pragma once

#include "Emu/RSX/GSRender.h"

namespace mtl
{
	struct presenter;
}

// Native Metal RSX backend (macOS only, selected with Renderer: "Metal").
//
// Phase 1: the RSX command stream is processed like the Null renderer (draws are no-ops),
// and every flip presents the guest display buffer from RSX local memory through a CAMetalLayer.
// Later phases add surfaces, clears, draws and shaders; see the "Metal backend" project notes.
class MTLGSRender : public GSRender
{
public:
	u64 get_cycles() final;

	MTLGSRender(utils::serial* ar) noexcept;
	MTLGSRender() noexcept : MTLGSRender(nullptr) {}
	~MTLGSRender() override;

private:
	void on_init_thread() override;
	void on_exit() override;
	void end() override;
	void flip(const rsx::display_flip_info_t& info) override;

	mtl::presenter* m_presenter = nullptr;
};
