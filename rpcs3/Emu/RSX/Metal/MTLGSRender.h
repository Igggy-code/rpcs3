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
// RSX render targets live in an rsx::surface_store of Metal textures; clears are executed and flips
// present the surface that backs the display buffer (falling back to the guest memory contents).
// Programs are translated RSX -> GLSL (Vulkan decompilers) -> SPIR-V -> MSL. Draws pull vertex data
// from texel buffers like the Vulkan backend; textures are placeholders until the texture cache exists.
// See the "Metal backend" project notes for the roadmap.
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

	// Draw path (MTLGSRenderDraw.cpp)
	struct vertex_upload_info;
	struct draw_env;

	// Finds or builds the pipeline for the current programs and fixed-function state
	bool load_program();
	void fill_pipeline_properties();
	// Uploads constants and per-draw state shared by all subdraws
	bool upload_draw_env(draw_env& env);
	bool upload_vertex_data(vertex_upload_info& info);
	void emit_draw(u32 sub_index, const draw_env& env);
	void fill_fixed_function_state(mtl::draw_desc& desc);

	mtl::presenter* m_presenter = nullptr;
	bool m_device_ready = false;
	mtl_render_targets m_rtts;

	std::unique_ptr<MTLProgramBuffer> m_prog_buffer;
	mtl::pipeline_props m_pipeline_properties{};
	u32 m_reported_programs = 0;

	mtl::pipeline* m_pipeline = nullptr;
	const MTLVertexProgram* m_vertex_prog = nullptr;
	const MTLFragmentProgram* m_fragment_prog = nullptr;

	rsx::vertex_input_layout m_vertex_layout;
	std::array<mtl::texture*, 4> m_color_attachments{};
	areau m_scissor{};
	u64 m_draws_submitted = 0;
	u64 m_draws_reported = 0;
};
