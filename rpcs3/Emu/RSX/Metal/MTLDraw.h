#pragma once

// Draw submission for the Metal backend (MTLDraw.mm). Plain C++ interface: the renderer decodes the
// RSX state into these structures, the Objective-C++ side turns them into Metal objects and commands.

#include "MTLDevice.h"
#include "MTLShaderCompiler.h"

#include <array>
#include <vector>

namespace mtl
{
	using u16 = std::uint16_t;
	using u64 = std::uint64_t;

	// ---------------------------------------------------------------------------------------------
	// Per-draw data ring (vertex data, indices, constants, draw parameters)
	// ---------------------------------------------------------------------------------------------

	struct ring_allocation
	{
		u8* ptr = nullptr;  // CPU address (shared storage), nullptr if the allocation failed
		u32 offset = 0;     // Offset in the ring buffer
	};

	// Allocates 'size' bytes. Waits for the GPU if the ring wraps onto data that is still in use.
	ring_allocation ring_alloc(u32 size, u32 alignment = 256);

	// ---------------------------------------------------------------------------------------------
	// Fixed-function state. Enum values mirror the Metal enums (checked with static_assert in MTLDraw.mm)
	// ---------------------------------------------------------------------------------------------

	enum class compare_func : u8 { never = 0, less, equal, less_equal, greater, not_equal, greater_equal, always };
	enum class stencil_op : u8 { keep = 0, zero, replace, incr_clamp, decr_clamp, invert, incr_wrap, decr_wrap };
	enum class blend_factor : u8
	{
		zero = 0, one, src_color, one_minus_src_color, src_alpha, one_minus_src_alpha,
		dst_color, one_minus_dst_color, dst_alpha, one_minus_dst_alpha, src_alpha_saturated,
		constant_color, one_minus_constant_color, constant_alpha, one_minus_constant_alpha
	};
	enum class blend_op : u8 { add = 0, subtract, reverse_subtract, min, max };
	enum class cull_mode : u8 { none = 0, front, back };
	enum class winding : u8 { clockwise = 0, counter_clockwise };
	enum class primitive : u8 { points = 0, lines, line_strip, triangles, triangle_strip };

	enum color_write_bits : u8
	{
		write_r = 8,
		write_g = 4,
		write_b = 2,
		write_a = 1,
		write_all = 15,
	};

	// Everything that is baked into an MTLRenderPipelineState besides the shaders.
	// Plain u32 fields only: it is hashed bytewise by the program cache.
	struct render_pipeline_state
	{
		u32 color_formats[4]{};   // pixel_format of each color attachment
		u32 depth_format = 0;     // pixel_format of the depth/stencil attachment
		u32 write_masks = 0;      // 4 bits (color_write_bits) per attachment
		u32 blend_enable = 0;     // Bit per attachment
		u32 blend_factors = 0;    // src_rgb | dst_rgb << 8 | src_a << 16 | dst_a << 24
		u32 blend_ops = 0;        // op_rgb | op_a << 8
		u32 alpha_to_coverage = 0;

		bool operator==(const render_pipeline_state& other) const = default;
	};

	// Opaque id<MTLRenderPipelineState>
	class render_pipeline
	{
	public:
		render_pipeline() = default;
		render_pipeline(const render_pipeline&) = delete;
		render_pipeline& operator=(const render_pipeline&) = delete;
		~render_pipeline();

		bool create(const shader_function& vs, const shader_function& fs, const render_pipeline_state& state, std::string& error);
		bool valid() const { return m_handle != nullptr; }
		void* native() const { return m_handle; }

	private:
		void* m_handle = nullptr;
	};

	struct stencil_face
	{
		stencil_op fail = stencil_op::keep;
		stencil_op depth_fail = stencil_op::keep;
		stencil_op pass = stencil_op::keep;
		compare_func func = compare_func::always;
		u8 read_mask = 0xff;
		u8 write_mask = 0xff;
		u8 reference = 0;
	};

	struct depth_stencil_state
	{
		bool depth_test = false;
		bool depth_write = false;
		compare_func depth_func = compare_func::always;
		bool stencil_test = false;
		stencil_face front{};
		stencil_face back{};
	};

	// ---------------------------------------------------------------------------------------------
	// Draw description
	// ---------------------------------------------------------------------------------------------

	enum class binding_source : u8
	{
		ring_buffer,     // Buffer at 'offset' in the data ring
		ring_texels,     // R8Uint texture buffer over the whole data ring (vertex pulling streams)
		dummy_texture,   // Placeholder texture matching the declared type (unbound texture units)
		texture,         // 'texture_handle' (id<MTLTexture>) with 'sampler_handle' (id<MTLSamplerState>)
	};

	struct resource_binding
	{
		shader_stage stage = shader_stage::vertex;
		binding_source source = binding_source::ring_buffer;
		u32 index = 0;          // Metal buffer/texture index
		u32 offset = 0;         // ring_buffer only
		u32 sampler = 0;        // Sampler index (dummy_texture and texture)
		void* texture_handle = nullptr;
		void* sampler_handle = nullptr;
		texture_dimension dimension = texture_dimension::dim_2d;
		bool depth = false;
		bool multisampled = false;
	};

	struct draw_range
	{
		u32 first = 0;  // First vertex or index
		u32 count = 0;
	};

	struct draw_desc
	{
		// Attachments (nullptr = unused)
		std::array<texture*, 4> color{};
		texture* depth_target = nullptr;

		const render_pipeline* pipeline = nullptr;
		depth_stencil_state depth_stencil{};

		// Viewport and scissor, in attachment pixels
		float viewport[4]{};      // x, y, width, height
		u32 scissor[4]{};         // x, y, width, height

		cull_mode cull = cull_mode::none;
		winding front_face = winding::counter_clockwise;
		bool depth_clamp = false;
		float depth_bias = 0.f;
		float depth_slope_scale = 0.f;
		float blend_color[4]{};

		primitive topology = primitive::triangles;
		bool indexed = false;
		bool index_32bit = false;
		u32 index_offset = 0;     // Ring offset of the index data

		std::vector<resource_binding> bindings;
		std::vector<draw_range> ranges;

		// Occlusion query the draw counts samples for (nullptr = none)
		struct occlusion_query* occlusion = nullptr;
	};

	// Occlusion query: one visibility-buffer slot per render pass the query spans (Metal writes one
	// result per pass and offset). Results are summed when read.
	struct occlusion_query
	{
		std::vector<u32> slots;
		u64 pass_generation = ~0ull; // Render pass the last slot belongs to
	};

	// True when every slot has been written by the GPU (false while a slot's command buffer is pending)
	bool occlusion_ready(const occlusion_query& query);
	// Waits for the GPU if needed, then returns the sum of passed samples (stops at the first hit if !precise)
	u64 occlusion_result(const occlusion_query& query, bool precise);
	// Returns the slots to the pool
	void occlusion_release(occlusion_query& query);

	// Records the draw into the pending command buffer (reusing the open render pass when the
	// attachments did not change)
	void draw(const draw_desc& desc);

	// Releases the ring, placeholder textures and cached state objects
	void shutdown_draw_resources();
}
