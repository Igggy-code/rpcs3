#pragma once

#include "Emu/RSX/GSRender.h"
#include "MTLRenderTargets.h"
#include "MTLProgram.h"

namespace mtl
{
	struct presenter;
}

// Native Metal RSX backend (macOS only, selected with Renderer: "Metal").
//
// Phase 3a: shaders are translated (RSX -> GLSL -> SPIR-V -> MSL) but draws are still skipped.
// Phase 2: RSX render targets live in an rsx::surface_store of Metal textures; clears are executed,
// draws are still skipped, and flips present the surface that backs the display buffer (falling back
// to the guest memory contents). See the "Metal backend" project notes for the roadmap.
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
	void clear_surface(u32 arg) override;
	void flip(const rsx::display_flip_info_t& info) override;

	// Binds the current RSX framebuffer configuration to surfaces (creates them when needed)
	void init_buffers(rsx::framebuffer_creation_context context);

	// Surface showing the display buffer at 'address', if the surface store has one
	mtl::render_target* get_present_surface(u32 address, u32 width, u32 height, u32 pitch, u32& out_width, u32& out_height);

	// Translates the current RSX programs to Metal (phase 3a: no draws yet, only shader translation)
	void load_program();

	mtl::presenter* m_presenter = nullptr;
	bool m_device_ready = false;
	mtl_render_targets m_rtts;

	std::unique_ptr<MTLProgramBuffer> m_prog_buffer;
	mtl::pipeline_props m_pipeline_properties{};
	u32 m_reported_programs = 0;
};
