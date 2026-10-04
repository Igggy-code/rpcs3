#pragma once

#include "Emu/RSX/GSRender.h"
#include "MTLRenderTargets.h"
#include "MTLProgram.h"
#include "MTLTextureCache.h"

#include <list>

namespace mtl
{
	struct presenter;

	// Texture cache flush requested by another thread, executed on the RSX thread
	struct work_item
	{
		u32 address_to_flush = 0;
		mtl::texture_cache::thrashed_set section_data;

		atomic_t<bool> processed = false;
		volatile bool result = false;
		atomic_t<bool> received = false;

		void producer_wait()
		{
			while (!processed)
			{
				utils::pause();
			}

			received = true;
		}
	};
}

// Native Metal RSX backend (macOS only, selected with Renderer: "Metal").
//
// RSX render targets live in an rsx::surface_store of Metal textures; clears are executed and flips
// present the surface that backs the display buffer (falling back to the guest memory contents).
// Programs are translated RSX -> GLSL (Vulkan decompilers) -> SPIR-V -> MSL. Draws pull vertex data
// from texel buffers like the Vulkan backend; textures come from an rsx::texture_cache of Metal images.
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

	// Guest accesses to protected pages (texture cache sections and ZCULL report pages)
	bool on_access_violation(u32 address, bool is_writing) override;
	void on_invalidate_memory_range(const utils::address_range32& range, rsx::invalidation_cause cause) override;
	void on_semaphore_acquire_wait() override;
	void do_local_task(rsx::FIFO::state state) override;
	bool scaled_image_from_memory(const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool interpolate) override;

	mtl::work_item& post_flush_request(u32 address, mtl::texture_cache::thrashed_set& flush_data);

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

	// Textures (MTLGSRenderDraw.cpp)
	void load_texture_env();
	void bind_texture_env(std::vector<mtl::resource_binding>& vs_textures, std::vector<mtl::resource_binding>& fs_textures);

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

	mtl::texture_cache m_texture_cache;
	shared_mutex m_sampler_mutex;
	atomic_t<bool> m_samplers_dirty = { true };
	std::array<void*, rsx::limits::fragment_textures_count> m_fs_samplers{};
	std::array<void*, rsx::limits::vertex_textures_count> m_vs_samplers{};

	shared_mutex m_queue_guard;
	std::list<mtl::work_item> m_work_queue;
};
