#include "stdafx.h"
#include "MTLGSRender.h"

#include "Emu/RSX/Common/BufferUtils.h"
#include "Emu/RSX/rsx_methods.h"
#include "Emu/system_config.h"
#include "Emu/RSX/Utils/color_utils.hpp"
#include "Emu/RSX/Utils/rsx_utils.h"

#include <unordered_set>

// Draw path of the Metal backend. Mirrors VKGSRender (VKDraw.cpp, VKVertexBuffers.cpp, load_program_env)
// since the shaders come from the Vulkan decompilers and expect the same data layout:
// - vertex data is pulled from two R8Uint texel buffers (persistent and volatile streams),
// - per-draw parameters (draw_parameters_t) describe attribute layouts,
// - environment blocks (vertex/fragment contexts, constants, texture parameters) are uniform buffers.
// Instead of dynamic offsets into shared heaps, every block is bound at its own ring offset and all
// the *_offset fields of draw_parameters_t are zero.

namespace
{
	struct primitive_info
	{
		mtl::primitive topology;
		bool emulated; // Needs an index buffer that converts the primitive to a native one
	};

	primitive_info get_primitive_info(rsx::primitive_type mode)
	{
		switch (mode)
		{
		case rsx::primitive_type::points: return { mtl::primitive::points, false };
		case rsx::primitive_type::lines: return { mtl::primitive::lines, false };
		case rsx::primitive_type::line_loop: return { mtl::primitive::line_strip, true };
		case rsx::primitive_type::line_strip: return { mtl::primitive::line_strip, false };
		case rsx::primitive_type::triangles: return { mtl::primitive::triangles, false };
		case rsx::primitive_type::triangle_strip:
		case rsx::primitive_type::quad_strip: return { mtl::primitive::triangle_strip, false };
		// Metal has no triangle fans
		case rsx::primitive_type::triangle_fan:
		case rsx::primitive_type::quads:
		case rsx::primitive_type::polygon: return { mtl::primitive::triangles, true };
		default:
			fmt::throw_exception("Unsupported primitive topology 0x%x", static_cast<u8>(mode));
		}
	}

	bool is_native_on_metal(rsx::primitive_type mode)
	{
		return !get_primitive_info(mode).emulated;
	}

	mtl::compare_func get_compare_func(rsx::comparison_function op)
	{
		switch (op)
		{
		case rsx::comparison_function::never: return mtl::compare_func::never;
		case rsx::comparison_function::greater: return mtl::compare_func::greater;
		case rsx::comparison_function::less: return mtl::compare_func::less;
		case rsx::comparison_function::less_or_equal: return mtl::compare_func::less_equal;
		case rsx::comparison_function::greater_or_equal: return mtl::compare_func::greater_equal;
		case rsx::comparison_function::equal: return mtl::compare_func::equal;
		case rsx::comparison_function::not_equal: return mtl::compare_func::not_equal;
		case rsx::comparison_function::always: return mtl::compare_func::always;
		default: return mtl::compare_func::always;
		}
	}

	mtl::stencil_op get_stencil_op(rsx::stencil_op op)
	{
		switch (op)
		{
		case rsx::stencil_op::keep: return mtl::stencil_op::keep;
		case rsx::stencil_op::zero: return mtl::stencil_op::zero;
		case rsx::stencil_op::replace: return mtl::stencil_op::replace;
		case rsx::stencil_op::incr: return mtl::stencil_op::incr_clamp;
		case rsx::stencil_op::decr: return mtl::stencil_op::decr_clamp;
		case rsx::stencil_op::invert: return mtl::stencil_op::invert;
		case rsx::stencil_op::incr_wrap: return mtl::stencil_op::incr_wrap;
		case rsx::stencil_op::decr_wrap: return mtl::stencil_op::decr_wrap;
		default: return mtl::stencil_op::keep;
		}
	}

	mtl::blend_factor get_blend_factor(rsx::blend_factor factor)
	{
		switch (factor)
		{
		case rsx::blend_factor::one: return mtl::blend_factor::one;
		case rsx::blend_factor::zero: return mtl::blend_factor::zero;
		case rsx::blend_factor::src_alpha: return mtl::blend_factor::src_alpha;
		case rsx::blend_factor::dst_alpha: return mtl::blend_factor::dst_alpha;
		case rsx::blend_factor::src_color: return mtl::blend_factor::src_color;
		case rsx::blend_factor::dst_color: return mtl::blend_factor::dst_color;
		case rsx::blend_factor::constant_color: return mtl::blend_factor::constant_color;
		case rsx::blend_factor::constant_alpha: return mtl::blend_factor::constant_alpha;
		case rsx::blend_factor::one_minus_src_color: return mtl::blend_factor::one_minus_src_color;
		case rsx::blend_factor::one_minus_dst_color: return mtl::blend_factor::one_minus_dst_color;
		case rsx::blend_factor::one_minus_src_alpha: return mtl::blend_factor::one_minus_src_alpha;
		case rsx::blend_factor::one_minus_dst_alpha: return mtl::blend_factor::one_minus_dst_alpha;
		case rsx::blend_factor::one_minus_constant_alpha: return mtl::blend_factor::one_minus_constant_alpha;
		case rsx::blend_factor::one_minus_constant_color: return mtl::blend_factor::one_minus_constant_color;
		case rsx::blend_factor::src_alpha_saturate: return mtl::blend_factor::src_alpha_saturated;
		default: return mtl::blend_factor::one;
		}
	}

	mtl::blend_op get_blend_op(rsx::blend_equation op)
	{
		switch (op)
		{
		case rsx::blend_equation::add:
		case rsx::blend_equation::add_signed:
		case rsx::blend_equation::reverse_add_signed: return mtl::blend_op::add;
		case rsx::blend_equation::subtract: return mtl::blend_op::subtract;
		case rsx::blend_equation::reverse_subtract:
		case rsx::blend_equation::reverse_subtract_signed: return mtl::blend_op::reverse_subtract;
		case rsx::blend_equation::min: return mtl::blend_op::min;
		case rsx::blend_equation::max: return mtl::blend_op::max;
		default: return mtl::blend_op::add;
		}
	}

#pragma pack(push, 1)
	// Must match draw_parameters_t (RSXDefines2.glsl)
	struct draw_parameters_t
	{
		u32 vertex_base_index;
		u32 vertex_index_offset;
		u32 draw_id;
		u32 xform_constants_offset;
		u32 vs_context_offset;
		u32 fs_constants_offset;
		u32 fs_context_offset;
		u32 fs_texture_base_index;
		u32 fs_stipple_pattern_offset;
		u32 reserved;
		s32 attrib_data[32];
	};
#pragma pack(pop)

	static_assert(sizeof(draw_parameters_t) == 168);
}

struct MTLGSRender::vertex_upload_info
{
	mtl::primitive topology = mtl::primitive::triangles;
	u32 vertex_draw_count = 0;       // Vertices or indices to draw
	u32 allocated_vertex_count = 0;
	u32 first_vertex = 0;            // First vertex in stream
	u32 vertex_index_base = 0;       // Index of vertex at data location 0
	u32 vertex_index_offset = 0;
	u32 persistent_base = umax;      // Byte offset of the persistent stream in the ring
	u32 volatile_base = umax;

	bool indexed = false;
	bool index_32bit = false;
	u32 index_offset = 0;            // Ring offset of the index data
};

struct MTLGSRender::draw_env
{
	u32 vertex_context = 0;
	u32 vertex_constants = 0;
	u32 fragment_constants = 0;
	u32 fragment_context = 0;
	u32 texture_parameters = 0;
	u32 stipple_pattern = 0;
	u32 zero_block = 0;   // Zeroed block for push constants and unknown buffers

	std::vector<mtl::resource_binding> vs_textures;
	std::vector<mtl::resource_binding> fs_textures;
};

void MTLGSRender::load_texture_env()
{
	mtl::command_context cmd;
	std::lock_guard lock(m_sampler_mutex);

	using descriptor_t = mtl::texture_cache::sampled_image_descriptor;

	for (u32 textures_ref = current_fp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			continue;
		}

		if (!fs_sampler_state[i])
		{
			fs_sampler_state[i] = std::make_unique<descriptor_t>();
		}

		auto sampler_state = static_cast<descriptor_t*>(fs_sampler_state[i].get());
		const auto& tex = rsx::method_registers.fragment_textures[i];
		const auto previous_format_class = sampler_state->format_class;

		if (!m_samplers_dirty &&
			!m_textures_dirty[i] &&
			!m_texture_cache.test_if_descriptor_expired(cmd, m_rtts, sampler_state, tex))
		{
			continue;
		}

		const bool is_sampler_dirty = m_textures_dirty[i];
		m_textures_dirty[i] = false;

		if (!tex.enabled())
		{
			*sampler_state = {};
			continue;
		}

		*sampler_state = m_texture_cache.upload_texture(cmd, tex, m_rtts);
		if (!sampler_state->validate())
		{
			continue;
		}

		if (!is_sampler_dirty)
		{
			if (sampler_state->format_class != previous_format_class)
			{
				// Host details changed but RSX is not aware
				m_graphics_state |= rsx::fragment_program_state_dirty;
			}

			if (sampler_state->format_ex && m_fs_samplers[i])
			{
				// Nothing to change, use the cached sampler
				continue;
			}
		}

		sampler_state->format_ex = tex.format_ex();

		u32 actual_mipcount = 1;
		if (sampler_state->upload_context == rsx::texture_upload_context::shader_read)
		{
			actual_mipcount = tex.get_exact_mipmap_count();
		}
		else if (sampler_state->external_subresource_desc.op != rsx::deferred_request_command::nop)
		{
			actual_mipcount = sampler_state->external_subresource_desc.exact_mip_count();
		}

		m_fs_samplers[i] = mtl::get_sampler(mtl::make_sampler_desc(tex, sampler_state, actual_mipcount));
	}

	for (u32 textures_ref = current_vp_metadata.referenced_textures_mask, i = 0; textures_ref; textures_ref >>= 1, ++i)
	{
		if (!(textures_ref & 1))
		{
			continue;
		}

		if (!vs_sampler_state[i])
		{
			vs_sampler_state[i] = std::make_unique<descriptor_t>();
		}

		auto sampler_state = static_cast<descriptor_t*>(vs_sampler_state[i].get());
		const auto& tex = rsx::method_registers.vertex_textures[i];
		const auto previous_format_class = sampler_state->format_class;

		if (!m_samplers_dirty &&
			!m_vertex_textures_dirty[i] &&
			!m_texture_cache.test_if_descriptor_expired(cmd, m_rtts, sampler_state, tex))
		{
			continue;
		}

		const bool is_sampler_dirty = m_vertex_textures_dirty[i];
		m_vertex_textures_dirty[i] = false;

		if (!tex.enabled())
		{
			*sampler_state = {};
			continue;
		}

		*sampler_state = m_texture_cache.upload_texture(cmd, tex, m_rtts);

		if (!sampler_state->validate())
		{
			continue;
		}

		if (is_sampler_dirty || !m_vs_samplers[i])
		{
			m_vs_samplers[i] = mtl::get_sampler(mtl::make_sampler_desc(tex, sampler_state, 1));
		}
		else if (sampler_state->format_class != previous_format_class)
		{
			m_graphics_state |= rsx::vertex_program_state_dirty;
		}
	}

	m_samplers_dirty.store(false);
}

namespace
{
	bool view_matches_declaration(const mtl::texture_view* view, const mtl::shader_resource& res)
	{
		const auto image = view->image();

		mtl::texture_dimension dimension = mtl::texture_dimension::dim_2d;
		switch (image->type())
		{
		case mtl::texture_type::tex_1d: dimension = mtl::texture_dimension::dim_1d; break;
		case mtl::texture_type::tex_3d: dimension = mtl::texture_dimension::dim_3d; break;
		case mtl::texture_type::tex_cube: dimension = mtl::texture_dimension::dim_cube; break;
		default: break;
		}

		// 1D textures are declared as 2D by some decompiler paths and vice versa; Metal needs an exact match
		if (dimension != res.dimension)
		{
			return false;
		}

		// Shadow samplers need depth images; stencil views need integer textures
		if (res.depth && !image->is_depth())
		{
			return false;
		}

		return !res.multisampled;
	}

	// Parses "tex3", "tex3_stencil", "vtex1"
	bool parse_texture_name(const std::string& name, std::string_view prefix, u32& index, bool& stencil)
	{
		if (!name.starts_with(prefix))
		{
			return false;
		}

		std::string_view rest = std::string_view(name).substr(prefix.size());
		stencil = rest.ends_with("_stencil");
		if (stencil)
		{
			rest.remove_suffix(8);
		}

		if (rest.empty() || rest.size() > 2 || !std::all_of(rest.begin(), rest.end(), [](char c) { return c >= '0' && c <= '9'; }))
		{
			return false;
		}

		index = static_cast<u32>(std::stoul(std::string(rest)));
		return true;
	}
}

void MTLGSRender::bind_texture_env(std::vector<mtl::resource_binding>& vs_textures, std::vector<mtl::resource_binding>& fs_textures)
{
	mtl::command_context cmd;
	using descriptor_t = mtl::texture_cache::sampled_image_descriptor;

	vs_textures.clear();
	fs_textures.clear();

	const auto resolve = [&](rsx::sampled_image_descriptor_base* base, bool enabled) -> mtl::texture_view*
	{
		auto sampler_state = static_cast<descriptor_t*>(base);
		if (!enabled || !sampler_state || !sampler_state->validate())
		{
			return nullptr;
		}

		if (sampler_state->image_handle)
		{
			return sampler_state->image_handle;
		}

		return m_texture_cache.create_temporary_subresource(cmd, sampler_state->external_subresource_desc);
	};

	const auto make_binding = [](mtl::shader_stage stage, const mtl::shader_resource& res) -> mtl::resource_binding
	{
		mtl::resource_binding b{};
		b.stage = stage;
		b.source = mtl::binding_source::dummy_texture;
		b.index = res.msl_index;
		b.sampler = res.msl_sampler;
		b.dimension = res.dimension;
		b.depth = res.depth;
		b.multisampled = res.multisampled;
		return b;
	};

	for (const auto& res : m_fragment_prog->result.resources)
	{
		if (res.kind != mtl::resource_kind::texture)
		{
			continue;
		}

		auto b = make_binding(mtl::shader_stage::fragment, res);
		u32 unit = 0;
		bool stencil = false;

		if (parse_texture_name(res.name, "tex", unit, stencil) && unit < rsx::limits::fragment_textures_count)
		{
			if (auto view = resolve(fs_sampler_state[unit].get(), rsx::method_registers.fragment_textures[unit].enabled()))
			{
				if (stencil)
				{
					auto image = static_cast<mtl::viewable_image*>(view->image());
					view = mtl::has_stencil(image->format()) ? image->get_view(rsx::default_remap_vector, mtl::image_aspect::stencil) : nullptr;
				}

				if (view && (stencil || view_matches_declaration(view, res)))
				{
					b.source = mtl::binding_source::texture;
					b.texture_handle = view->native();
					b.sampler_handle = stencil ? nullptr : m_fs_samplers[unit];
				}
				else if (view)
				{
					// Report each program/unit combination once: the draw samples a placeholder instead
					static std::unordered_set<u64> s_reported;
					const auto image = view->image();
					if (s_reported.insert((u64{m_fragment_prog->id} << 8) | unit).second)
					{
						rsx_log.warning("Metal: fp %u %s: image (type %d, format %d, depth %d) does not match the declaration (dim %d, depth %d, ms %d); using a placeholder",
							m_fragment_prog->id, res.name, static_cast<int>(image->type()), static_cast<int>(image->format()), image->is_depth(),
							static_cast<int>(res.dimension), res.depth, res.multisampled);
					}
				}
			}
		}

		fs_textures.push_back(b);
	}

	for (const auto& res : m_vertex_prog->result.resources)
	{
		if (res.kind != mtl::resource_kind::texture)
		{
			continue;
		}

		auto b = make_binding(mtl::shader_stage::vertex, res);
		u32 unit = 0;
		bool stencil = false;

		if (parse_texture_name(res.name, "vtex", unit, stencil) && unit < rsx::limits::vertex_textures_count && !stencil)
		{
			if (auto view = resolve(vs_sampler_state[unit].get(), rsx::method_registers.vertex_textures[unit].enabled());
				view && view_matches_declaration(view, res))
			{
				b.source = mtl::binding_source::texture;
				b.texture_handle = view->native();
				b.sampler_handle = m_vs_samplers[unit];
			}
		}

		vs_textures.push_back(b);
	}
}

void MTLGSRender::fill_pipeline_properties()
{
	const auto& regs = rsx::method_registers;
	mtl::pipeline_props props{};

	u32 attachment = 0;
	m_color_attachments = {};

	for (const u8 index : m_rtts.m_bound_render_target_ids)
	{
		if (auto surface = std::get<1>(m_rtts.m_bound_render_targets[index]); surface && attachment < 4)
		{
			m_color_attachments[attachment] = surface;
			props.color_formats[attachment] = static_cast<u32>(surface->format());
			attachment++;
		}
	}

	if (auto ds = std::get<1>(m_rtts.m_bound_depth_stencil))
	{
		props.depth_format = static_cast<u32>(ds->format());
	}

	// Color write masks (same rules as the Vulkan backend)
	const auto host_write_mask = rsx::get_write_output_mask(regs.surface_color());

	for (u32 index = 0; index < attachment; ++index)
	{
		bool r = regs.color_mask_r(index);
		bool g = regs.color_mask_g(index);
		bool b = regs.color_mask_b(index);
		bool a = regs.color_mask_a(index);

		switch (regs.surface_color())
		{
		case rsx::surface_color_format::b8:
			rsx::get_b8_colormask(r, g, b, a);
			break;
		case rsx::surface_color_format::g8b8:
			rsx::get_g8b8_r8g8_colormask(r, g, b, a);
			break;
		default:
			break;
		}

		u32 mask = 0;
		if (r && host_write_mask[0]) mask |= mtl::write_r;
		if (g && host_write_mask[1]) mask |= mtl::write_g;
		if (b && host_write_mask[2]) mask |= mtl::write_b;
		if (a && host_write_mask[3]) mask |= mtl::write_a;

		props.write_masks |= mask << (index * 4);
	}

	// Logic ops do not exist in Metal; blending is skipped like the Vulkan backend does when they are on
	if (!regs.logic_op_enabled())
	{
		if (const auto blend_enabled = regs.blend_enabled_mask())
		{
			props.blend_factors =
				static_cast<u32>(get_blend_factor(regs.blend_func_sfactor_rgb())) |
				(static_cast<u32>(get_blend_factor(regs.blend_func_dfactor_rgb())) << 8) |
				(static_cast<u32>(get_blend_factor(regs.blend_func_sfactor_a())) << 16) |
				(static_cast<u32>(get_blend_factor(regs.blend_func_dfactor_a())) << 24);

			props.blend_ops =
				static_cast<u32>(get_blend_op(regs.blend_equation_rgb())) |
				(static_cast<u32>(get_blend_op(regs.blend_equation_a())) << 8);

			props.blend_enable = blend_enabled & ((1u << attachment) - 1);
		}
	}

	props.alpha_to_coverage = 0;

	m_pipeline_properties = props;
}

bool MTLGSRender::load_program()
{
	if (!m_prog_buffer)
	{
		return false;
	}

	if (m_graphics_state & rsx::pipeline_state::invalidate_pipeline_bits)
	{
		get_current_fragment_program(fs_sampler_state);
		ensure(current_fragment_program.valid);

		get_current_vertex_program(vs_sampler_state);

		m_graphics_state.clear(rsx::pipeline_state::invalidate_pipeline_bits);
	}

	fill_pipeline_properties();

	// Async shader modes: unknown pipelines are built on worker threads and their draws are skipped meanwhile
	// (there is no shader interpreter on Metal). Synchronous builds stall the RSX thread for tens of ms,
	// long enough for games such as UC2 to time out waiting on RSX progress.
	const bool compile_async = g_cfg.video.shadermode != shader_mode::recompiler;

	const auto [pipeline, vp, fp] = m_prog_buffer->get_graphics_pipeline(
		nullptr, current_vertex_program, current_fragment_program, m_pipeline_properties, compile_async, false);

	m_pipeline = pipeline;
	m_vertex_prog = vp;
	m_fragment_prog = fp;

	return m_pipeline && m_pipeline->valid && m_vertex_prog && m_fragment_prog;
}

bool MTLGSRender::upload_draw_env(draw_env& env)
{
	const auto& regs = rsx::method_registers;

	// Vertex context (vertex_context_t, 96 bytes)
	{
		auto mem = mtl::ring_alloc(96);
		if (!mem.ptr) return false;

		// flip_y: the shaders follow the Vulkan convention (NDC y down), Metal's NDC y points up
		m_draw_processor.fill_scale_offset_data(mem.ptr, true);
		m_draw_processor.fill_user_clip_data(mem.ptr + 64);
		*reinterpret_cast<u32*>(mem.ptr + 68) = regs.transform_branch_bits();
		*reinterpret_cast<f32*>(mem.ptr + 72) = regs.point_size() * resolution_scaling_config.scale_factor();
		*reinterpret_cast<f32*>(mem.ptr + 76) = regs.clip_min();
		*reinterpret_cast<f32*>(mem.ptr + 80) = regs.clip_max();
		env.vertex_context = mem.offset;
	}

	// Transform constants
	{
		const usz size = m_vertex_prog->has_indexed_constants ? 8192 : m_vertex_prog->constant_ids.size() * 16;
		auto mem = mtl::ring_alloc(static_cast<u32>(std::max<usz>(size, 16)));
		if (!mem.ptr) return false;

		if (size)
		{
			const auto ids = (size == 8192) ? std::span<const u16>{} : std::span<const u16>(m_vertex_prog->constant_ids);
			m_draw_processor.fill_vertex_program_constants_data(mem.ptr, ids);
		}

		env.vertex_constants = mem.offset;
	}

	// Fragment constants
	{
		const auto& offsets = m_fragment_prog->host.constant_offsets;
		const usz size = std::max<usz>(current_fp_metadata.program_constants_buffer_length, offsets.size() * 16);
		auto mem = mtl::ring_alloc(static_cast<u32>(std::max<usz>(size, 16)));
		if (!mem.ptr) return false;

		if (!offsets.empty())
		{
			rsx::write_fragment_constants_to_buffer({ reinterpret_cast<f32*>(mem.ptr), size / 4 }, current_fragment_program, offsets, true);
		}

		env.fragment_constants = mem.offset;
	}

	// Fragment context (fragment_context_t, 32 bytes)
	{
		auto mem = mtl::ring_alloc(32);
		if (!mem.ptr) return false;

		m_draw_processor.fill_fragment_state_buffer(mem.ptr, current_fragment_program);
		env.fragment_context = mem.offset;
	}

	// Texture parameters (16 x sampler_info)
	{
		auto mem = mtl::ring_alloc(768);
		if (!mem.ptr) return false;

		std::memset(mem.ptr, 0, 768);
		current_fragment_program.texture_params.write_to(mem.ptr, current_fp_metadata.referenced_textures_mask);
		env.texture_parameters = mem.offset;
	}

	// Polygon stipple pattern
	{
		auto mem = mtl::ring_alloc(128);
		if (!mem.ptr) return false;

		std::memcpy(mem.ptr, regs.polygon_stipple_pattern(), 128);
		env.stipple_pattern = mem.offset;
	}

	// Zeroed block (push constants: draw_parameters_offset = 0)
	{
		auto mem = mtl::ring_alloc(256);
		if (!mem.ptr) return false;

		std::memset(mem.ptr, 0, 256);
		env.zero_block = mem.offset;
	}

	return true;
}

bool MTLGSRender::upload_vertex_data(vertex_upload_info& info)
{
	const auto& regs = rsx::method_registers;
	const auto& clause = regs.current_draw_clause;
	const auto prim = get_primitive_info(clause.primitive);

	info.topology = prim.topology;

	u32 min_index = 0, max_index = 0;
	bool index_rebase = false;

	const auto command = m_draw_processor.get_draw_command(regs);

	if (std::holds_alternative<rsx::draw_indexed_array_command>(command))
	{
		const auto& indexed = std::get<rsx::draw_indexed_array_command>(command);

		const rsx::index_array_type index_type = clause.is_immediate_draw ? rsx::index_array_type::u32 : regs.index_type();
		const u32 type_size = get_index_type_size(index_type);

		u32 index_count = clause.get_elements_count();
		if (prim.emulated)
		{
			index_count = get_index_count(clause.primitive, index_count);
		}

		const u32 upload_size = index_count * type_size;
		auto mem = mtl::ring_alloc(std::max(upload_size, 4u));
		if (!mem.ptr) return false;

		std::tie(min_index, max_index, index_count) = write_index_array_data_to_buffer(
			std::span<std::byte>(reinterpret_cast<std::byte*>(mem.ptr), upload_size),
			indexed.raw_index_buffer, index_type,
			clause.primitive,
			regs.restart_index_enabled(),
			regs.restart_index(),
			[](auto p) { return !is_native_on_metal(p); });

		if (min_index >= max_index)
		{
			// Empty set, do not draw
			return true;
		}

		info.indexed = true;
		info.index_32bit = index_type == rsx::index_array_type::u32;
		info.index_offset = mem.offset;
		info.vertex_draw_count = index_count;
		info.vertex_index_offset = regs.vertex_data_base_index();
		index_rebase = true;
	}
	else
	{
		u32 vertex_count = 0;

		if (std::holds_alternative<rsx::draw_inlined_array>(command))
		{
			const auto stream_length = clause.inline_vertex_array.size();
			vertex_count = static_cast<u32>(stream_length * sizeof(u32)) / m_vertex_layout.interleaved_blocks[0]->attribute_stride;
			min_index = 0;
		}
		else
		{
			vertex_count = clause.get_elements_count();
			min_index = clause.min_index();
		}

		if (!vertex_count)
		{
			return true;
		}

		max_index = min_index + vertex_count - 1;
		info.vertex_draw_count = vertex_count;

		if (prim.emulated)
		{
			const u32 index_count = get_index_count(clause.primitive, vertex_count);
			auto mem = mtl::ring_alloc(std::max(index_count * 2, 4u));
			if (!mem.ptr) return false;

			write_index_array_for_non_indexed_non_native_primitive_to_buffer(reinterpret_cast<char*>(mem.ptr), clause.primitive, vertex_count);

			info.indexed = true;
			info.index_32bit = false;
			info.index_offset = mem.offset;
			info.vertex_draw_count = index_count;
		}
	}

	const u32 vertex_count = (max_index - min_index) + 1;
	u32 vertex_base = min_index;
	u32 index_base = 0;

	if (index_rebase)
	{
		vertex_base = rsx::get_index_from_base(vertex_base, regs.vertex_data_base_index());
		index_base = min_index;
	}

	const auto required = calculate_memory_requirements(m_vertex_layout, vertex_base, vertex_count);

	u8* persistent = nullptr;
	u8* volatile_ = nullptr;

	if (required.first > 0)
	{
		auto mem = mtl::ring_alloc(required.first);
		if (!mem.ptr) return false;

		persistent = mem.ptr;
		info.persistent_base = mem.offset;
	}

	if (required.second > 0)
	{
		auto mem = mtl::ring_alloc(required.second);
		if (!mem.ptr) return false;

		volatile_ = mem.ptr;
		info.volatile_base = mem.offset;
	}

	if (persistent || volatile_)
	{
		m_draw_processor.write_vertex_data_to_memory(m_vertex_layout, vertex_base, vertex_count, persistent, volatile_);
	}

	info.allocated_vertex_count = vertex_count;
	info.first_vertex = vertex_base;
	info.vertex_index_base = index_base;
	return true;
}

void MTLGSRender::fill_fixed_function_state(mtl::draw_desc& desc)
{
	const auto& regs = rsx::method_registers;

	// Viewport: the scale/offset matrix already contains the RSX viewport transform
	const auto [clip_width, clip_height] = rsx::apply_resolution_scale<true>(
		resolution_scaling_config, regs.surface_clip_width(), regs.surface_clip_height());

	desc.viewport[0] = 0.f;
	desc.viewport[1] = 0.f;
	desc.viewport[2] = static_cast<f32>(clip_width);
	desc.viewport[3] = static_cast<f32>(clip_height);

	desc.scissor[0] = m_scissor.x1;
	desc.scissor[1] = m_scissor.y1;
	desc.scissor[2] = m_scissor.width();
	desc.scissor[3] = m_scissor.height();

	desc.front_face = regs.front_face_mode() == rsx::front_face::cw ? mtl::winding::clockwise : mtl::winding::counter_clockwise;
	desc.depth_clamp = regs.depth_clamp_enabled() || !regs.depth_clip_enabled();

	if (regs.cull_face_enabled())
	{
		switch (regs.cull_face_mode())
		{
		case rsx::cull_face::front: desc.cull = mtl::cull_mode::front; break;
		case rsx::cull_face::back: desc.cull = mtl::cull_mode::back; break;
		default: desc.cull = mtl::cull_mode::none; break; // front_and_back is handled by the caller
		}
	}

	if (regs.poly_offset_fill_enabled())
	{
		desc.depth_bias = regs.poly_offset_bias();
		desc.depth_slope_scale = regs.poly_offset_scale();
	}

	const auto blend_colors = rsx::get_constant_blend_colors();
	std::copy(blend_colors.begin(), blend_colors.end(), desc.blend_color);

	auto& ds = desc.depth_stencil;

	if (desc.depth_target && regs.depth_test_enabled())
	{
		ds.depth_test = true;
		ds.depth_write = regs.depth_write_enabled();
		ds.depth_func = get_compare_func(regs.depth_func());
	}

	if (desc.depth_target && mtl::has_stencil(desc.depth_target->format()) && regs.stencil_test_enabled())
	{
		ds.stencil_test = true;

		ds.front.fail = get_stencil_op(regs.stencil_op_fail());
		ds.front.depth_fail = get_stencil_op(regs.stencil_op_zfail());
		ds.front.pass = get_stencil_op(regs.stencil_op_zpass());
		ds.front.func = get_compare_func(regs.stencil_func());
		ds.front.read_mask = static_cast<u8>(regs.stencil_func_mask());
		ds.front.write_mask = static_cast<u8>(regs.stencil_mask());
		ds.front.reference = static_cast<u8>(regs.stencil_func_ref());

		if (regs.two_sided_stencil_test_enabled())
		{
			ds.back.fail = get_stencil_op(regs.back_stencil_op_fail());
			ds.back.depth_fail = get_stencil_op(regs.back_stencil_op_zfail());
			ds.back.pass = get_stencil_op(regs.back_stencil_op_zpass());
			ds.back.func = get_compare_func(regs.back_stencil_func());
			ds.back.read_mask = static_cast<u8>(regs.back_stencil_func_mask());
			ds.back.write_mask = static_cast<u8>(regs.back_stencil_mask());
			ds.back.reference = static_cast<u8>(regs.back_stencil_func_ref());
		}
		else
		{
			ds.back = ds.front;
		}
	}
}

void MTLGSRender::emit_draw(u32 sub_index, const draw_env& env)
{
	auto& draw_call = rsx::method_registers.current_draw_clause;

	const rsx::flags32_t vertex_state_mask = rsx::vertex_base_changed | rsx::vertex_arrays_changed;
	const rsx::flags32_t state_flags = (sub_index == 0) ? rsx::vertex_arrays_changed : draw_call.execute_pipeline_dependencies(m_ctx);

	if (state_flags & rsx::vertex_arrays_changed)
	{
		m_draw_processor.analyse_inputs_interleaved(m_vertex_layout, current_vp_metadata);
	}
	else if (state_flags & rsx::vertex_base_changed)
	{
		for (auto& block : m_vertex_layout.interleaved_blocks)
		{
			block->vertex_range.second = 0;
			const auto vertex_base_offset = rsx::method_registers.vertex_data_base_offset();
			block->real_offset_address = rsx::get_address(rsx::get_vertex_offset_from_base(vertex_base_offset, block->base_offset), block->memory_location);
		}
	}
	else
	{
		for (auto& block : m_vertex_layout.interleaved_blocks)
		{
			block->vertex_range.second = 0;
		}
	}

	if ((state_flags & vertex_state_mask) && !m_vertex_layout.validate())
	{
		// No vertex inputs enabled: execute the remaining pipeline barriers with NOP draws
		do
		{
			draw_call.execute_pipeline_dependencies(m_ctx);
		}
		while (draw_call.next());

		draw_call.end();
		return;
	}

	vertex_upload_info upload{};
	if (!upload_vertex_data(upload))
	{
		rsx_log.error("Metal: out of ring memory for vertex data");
		return;
	}

	if (!upload.vertex_draw_count)
	{
		return;
	}

	// Per-draw parameters
	auto params_mem = mtl::ring_alloc(sizeof(draw_parameters_t));
	if (!params_mem.ptr)
	{
		return;
	}

	auto params = reinterpret_cast<draw_parameters_t*>(params_mem.ptr);
	std::memset(params, 0, sizeof(draw_parameters_t));
	params->vertex_base_index = upload.vertex_index_base;
	params->vertex_index_offset = upload.vertex_index_offset;
	params->draw_id = sub_index;

	m_draw_processor.fill_vertex_layout_state(
		m_vertex_layout,
		current_vp_metadata,
		upload.first_vertex,
		upload.allocated_vertex_count,
		params->attrib_data,
		upload.persistent_base,
		upload.volatile_base);

	mtl::draw_desc desc{};
	desc.color = m_color_attachments;
	desc.depth_target = std::get<1>(m_rtts.m_bound_depth_stencil);
	desc.pipeline = &m_pipeline->state;
	desc.topology = upload.topology;
	desc.indexed = upload.indexed;
	desc.index_32bit = upload.index_32bit;
	desc.index_offset = upload.index_offset;

	fill_fixed_function_state(desc);

	if (rsx::method_registers.cull_face_enabled() && rsx::method_registers.cull_face_mode() == rsx::cull_face::front_and_back)
	{
		// Everything is culled; only the vertex fetch side effects matter (none)
		return;
	}

	// Resources
	const auto bind = [&](mtl::shader_stage stage, const std::vector<mtl::shader_resource>& resources)
	{
		for (const auto& res : resources)
		{
			mtl::resource_binding b{};
			b.stage = stage;
			b.index = res.msl_index;

			switch (res.kind)
			{
			case mtl::resource_kind::texel_buffer:
				b.source = mtl::binding_source::ring_texels;
				break;

			case mtl::resource_kind::texture:
				// Resolved once per draw call by bind_texture_env
				continue;

			case mtl::resource_kind::push_constant:
				b.source = mtl::binding_source::ring_buffer;
				b.offset = env.zero_block;
				break;

			case mtl::resource_kind::buffer:
				b.source = mtl::binding_source::ring_buffer;

				if (res.name == "DrawParametersBuffer") b.offset = params_mem.offset;
				else if (res.name == "VertexContextBuffer") b.offset = env.vertex_context;
				else if (res.name == "VertexConstantsBuffer") b.offset = env.vertex_constants;
				else if (res.name == "FragmentConstantsBuffer") b.offset = env.fragment_constants;
				else if (res.name == "FragmentStateBuffer") b.offset = env.fragment_context;
				else if (res.name == "TextureParametersBuffer") b.offset = env.texture_parameters;
				else if (res.name == "RasterizerHeap") b.offset = env.stipple_pattern;
				else
				{
					static std::unordered_set<std::string> s_reported;
					if (s_reported.insert(res.name).second)
					{
						rsx_log.todo("Metal: unhandled shader buffer '%s'", res.name);
					}
					b.offset = env.zero_block;
				}
				break;
			}

			desc.bindings.push_back(b);
		}
	};

	bind(mtl::shader_stage::vertex, m_vertex_prog->result.resources);
	bind(mtl::shader_stage::fragment, m_fragment_prog->result.resources);

	desc.bindings.insert(desc.bindings.end(), env.vs_textures.begin(), env.vs_textures.end());
	desc.bindings.insert(desc.bindings.end(), env.fs_textures.begin(), env.fs_textures.end());

	// Draw ranges
	if (draw_call.is_single_draw())
	{
		desc.ranges.push_back({ 0, upload.vertex_draw_count });
	}
	else
	{
		u32 offset = 0;
		for (const auto& range : draw_call.get_subranges())
		{
			const u32 count = desc.indexed ? get_index_count(draw_call.primitive, range.count) : range.count;
			desc.ranges.push_back({ offset, count });
			offset += count;
		}
	}

	mtl::draw(desc);
	m_draws_submitted++;
}

void MTLGSRender::end()
{
	if (skip_current_frame || !m_device_ready || !m_prog_buffer)
	{
		execute_nop_draw();
		rsx::thread::end();
		return;
	}

	mtl::stall_probe probe_draw("draw");

	{
		mtl::stall_probe probe("init_buffers");
		init_buffers(rsx::framebuffer_creation_context::context_draw);
	}

	if (areau scissor; get_scissor(scissor, true))
	{
		m_scissor = scissor;
	}

	if (!m_graphics_state.test(rsx::rtt_config_valid))
	{
		execute_nop_draw();
		rsx::thread::end();
		return;
	}

	analyse_current_rsx_pipeline();

	// Texture descriptors feed the program analysis (format classes, texture parameters)
	{
		mtl::stall_probe probe("load_texture_env");
		load_texture_env();
	}

	bool program_ready;
	{
		mtl::stall_probe probe("load_program");
		program_ready = load_program();
	}

	if (!program_ready)
	{
		execute_nop_draw();
		rsx::thread::end();
		return;
	}

	draw_env env{};
	if (!upload_draw_env(env))
	{
		rsx_log.error("Metal: out of ring memory for draw constants");
		execute_nop_draw();
		rsx::thread::end();
		return;
	}

	// Resolve inherited surface contents before rendering into them
	mtl::command_context cmd;

	{
		mtl::stall_probe probe("surface write barriers");

		if (auto ds = std::get<1>(m_rtts.m_bound_depth_stencil))
		{
			ds->write_barrier(cmd);
		}

		for (auto& rtt : m_rtts.m_bound_render_targets)
		{
			if (auto surface = std::get<1>(rtt))
			{
				surface->write_barrier(cmd);
			}
		}
	}

	// Bind textures after the barriers; temporary copies (cyclic references) are made here
	{
		mtl::stall_probe probe("bind_texture_env");
		bind_texture_env(env.vs_textures, env.fs_textures);
	}

	if (m_dump)
	{
		dump_draw(env.fs_textures);
	}

	auto& draw_call = rsx::method_registers.current_draw_clause;
	draw_call.begin();

	u32 sub_index = 0;
	do
	{
		emit_draw(sub_index++, env);
	}
	while (draw_call.next());

	m_texture_cache.release_uncached_temporary_subresources();

	m_rtts.on_write(m_framebuffer_layout.color_write_enabled, m_framebuffer_layout.zeta_write_enabled);

	rsx::thread::end();
}
