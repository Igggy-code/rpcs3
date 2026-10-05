#pragma once

// RSX render targets for the Metal backend: mtl::render_target plugs Metal textures into the
// backend-agnostic rsx::surface_store, following the OpenGL backend (GL/GLRenderTargets.h).

#include "MTLDevice.h"
#include "MTLTexture.h"

#include "Emu/RSX/Common/surface_store.h"
#include "Emu/RSX/Common/TextureUtils.h"
#include "Emu/RSX/Utils/rsx_utils.h"
#include "Emu/system_config.h"

namespace mtl
{
	// Command list token passed through the surface store (all Metal work goes to the device's pending command buffer)
	struct command_context
	{
	};

	// Placeholder texture descriptor until the texture cache exists (program analysis dereferences these)
	class null_sampled_image : public rsx::sampled_image_descriptor_base
	{
	public:
		u32 encoded_component_map() const override { return 0; }
	};

	pixel_format surface_color_format_to_mtl(rsx::surface_color_format format);
	// Native component layout (RSX ARGB order) of a color surface stored in its Metal format
	std::array<swizzle, 4> surface_color_component_layout(rsx::surface_color_format format);
	pixel_format surface_depth_format_to_mtl(rsx::surface_depth_format2 format);

	class render_target : public viewable_image, public rsx::render_target_descriptor<texture*>
	{
		void initialize_memory(command_context& cmd, rsx::surface_access access);
		void load_memory(command_context& cmd);

	public:
		render_target(u32 width, u32 height, pixel_format format, rsx::format_class format_class)
			: viewable_image(width, height, format, usage_sampled | usage_render_target, format_class)
		{
		}

		void set_native_pitch(u32 pitch) { native_pitch = pitch; }
		void set_rsx_pitch(u32 pitch) { rsx_pitch = pitch; }

		void set_surface_dimensions(u16 w, u16 h, u32 pitch)
		{
			surface_width = w;
			surface_height = h;
			rsx_pitch = pitch;
		}

		bool is_depth_surface() const override
		{
			return is_depth();
		}

		viewable_image* get_surface(rsx::surface_access /*access_type*/) override
		{
			return this;
		}

		bool matches_dimensions(u16 _width, u16 _height) const
		{
			const auto [scaled_w, scaled_h] = rsx::apply_resolution_scale<true>(resolution_scaling_config, _width, _height);
			return scaled_w == width() && scaled_h == height();
		}

		// Copies inherited contents (same format, scaled or bit-cast)
		void transfer_contents(render_target& src, const areai& src_area, const areai& dst_area);

		// Resolves pending inherited contents (old_contents) and initializes fresh surfaces
		void memory_barrier(command_context& cmd, rsx::surface_access access);
		void read_barrier(command_context& cmd) { memory_barrier(cmd, rsx::surface_access::shader_read); }
		void write_barrier(command_context& cmd) { memory_barrier(cmd, rsx::surface_access::shader_write); }
	};

	static inline render_target* as_rtt(texture* t)
	{
		return ensure(dynamic_cast<render_target*>(t));
	}

	static inline const render_target* as_rtt(const texture* t)
	{
		return ensure(dynamic_cast<const render_target*>(t));
	}
}

struct mtl_render_target_traits
{
	using surface_storage_type = std::unique_ptr<mtl::render_target>;
	using surface_type = mtl::render_target*;
	using buffer_object_storage_type = std::unique_ptr<std::vector<u8>>;
	using buffer_object_type = std::vector<u8>*;
	using command_list_type = mtl::command_context&;
	using download_buffer_object = std::vector<u8>;
	using barrier_descriptor_t = rsx::deferred_clipped_region<mtl::render_target*>;

	static std::unique_ptr<mtl::render_target> create_new_surface(
		mtl::command_context&,
		u32 address,
		rsx::surface_color_format surface_color_format,
		usz width, usz height, usz pitch,
		rsx::surface_antialiasing antialias,
		const rsx::surface_scaling_config_t& resolution_scaling_config)
	{
		const auto [width_, height_] = rsx::apply_resolution_scale<true>(resolution_scaling_config, static_cast<u16>(width), static_cast<u16>(height));

		// MSAA is not implemented yet: surfaces are always single-sampled (like msaa_level::none)
		auto result = std::make_unique<mtl::render_target>(width_, height_, mtl::surface_color_format_to_mtl(surface_color_format), rsx::RSX_FORMAT_CLASS_COLOR);

		result->set_label(fmt::format("RTV@0x%x", address));
		result->set_native_component_layout(mtl::surface_color_component_layout(surface_color_format));
		result->set_aa_mode(antialias);
		result->set_resolution_scaling_config(resolution_scaling_config);
		result->set_native_pitch(static_cast<u32>(width) * get_format_block_size_in_bytes(surface_color_format) * result->samples_x);
		result->set_surface_dimensions(static_cast<u16>(width), static_cast<u16>(height), static_cast<u32>(pitch));
		result->set_format(surface_color_format);

		result->memory_usage_flags = rsx::surface_usage_flags::attachment;
		result->state_flags = rsx::surface_state_flags::erase_bkgnd;
		result->sample_layout = rsx::surface_sample_layout::null;
		result->queue_tag(address);
		result->add_ref();
		return result;
	}

	static std::unique_ptr<mtl::render_target> create_new_surface(
		mtl::command_context&,
		u32 address,
		rsx::surface_depth_format2 surface_depth_format,
		usz width, usz height, usz pitch,
		rsx::surface_antialiasing antialias,
		const rsx::surface_scaling_config_t& resolution_scaling_config)
	{
		const auto [width_, height_] = rsx::apply_resolution_scale<true>(resolution_scaling_config, static_cast<u16>(width), static_cast<u16>(height));

		auto result = std::make_unique<mtl::render_target>(width_, height_, mtl::surface_depth_format_to_mtl(surface_depth_format), rsx::classify_format(surface_depth_format));

		result->set_label(fmt::format("DSV@0x%x", address));
		result->set_aa_mode(antialias);
		result->set_resolution_scaling_config(resolution_scaling_config);
		result->set_surface_dimensions(static_cast<u16>(width), static_cast<u16>(height), static_cast<u32>(pitch));
		result->set_format(surface_depth_format);
		result->set_native_pitch(static_cast<u32>(width) * get_format_block_size_in_bytes(surface_depth_format) * result->samples_x);

		result->memory_usage_flags = rsx::surface_usage_flags::attachment;
		result->state_flags = rsx::surface_state_flags::erase_bkgnd;
		result->sample_layout = rsx::surface_sample_layout::null;
		result->queue_tag(address);
		result->add_ref();
		return result;
	}

	static bool is_reusable_surface(const mtl::render_target* surface, const mtl::render_target* ref)
	{
		return surface->valid() && surface->format() == ref->format();
	}

	static void prepare_for_reuse(mtl::command_context&, mtl::render_target*)
	{
	}

	static void clone_surface(
		mtl::command_context& cmd,
		std::unique_ptr<mtl::render_target>& sink, mtl::render_target* ref,
		u32 address, barrier_descriptor_t& prev,
		const rsx::surface_scaling_config_t& scaling_config)
	{
		const bool initialize = !sink || !sink->has_refs();
		if (sink && initialize)
		{
			prepare_for_reuse(cmd, sink.get());
		}

		if (!sink)
		{
			const auto [new_w, new_h] = rsx::apply_resolution_scale<true>(
				scaling_config,
				prev.width, prev.height,
				ref->get_surface_width<rsx::surface_metrics::pixels>(),
				ref->get_surface_height<rsx::surface_metrics::pixels>());

			sink = std::make_unique<mtl::render_target>(new_w, new_h, ref->format(), ref->format_class());
			sink->set_native_component_layout(ref->native_component_layout());
		}

		if (initialize)
		{
			sink->reset();
			sink->msaa_flags = rsx::surface_state_flags::ready;
			sink->stencil_init_flags = ref->stencil_init_flags;
			sink->add_ref();

			sink->memory_usage_flags = rsx::surface_usage_flags::storage;
			sink->state_flags = rsx::surface_state_flags::erase_bkgnd;
			sink->format_info = ref->format_info;

			sink->sample_layout = ref->sample_layout;
			sink->resolution_scaling_config = scaling_config;

			sink->set_label(fmt::format("SINK@0x%x", address));
			sink->set_aa_mode(ref->get_aa_mode());
			sink->set_native_pitch(static_cast<u32>(prev.width) * ref->get_bpp() * ref->samples_x);
			sink->set_rsx_pitch(ref->get_rsx_pitch());
			sink->set_surface_dimensions(prev.width, prev.height, ref->get_rsx_pitch());
			sink->queue_tag(address);
		}

		sink->on_clone_from(ref);

		if (!sink->old_contents.empty())
		{
			if (sink->surface_width > prev.width || sink->surface_height > prev.height)
			{
				sink->write_barrier(cmd);
			}
			else
			{
				sink->clear_rw_barrier();
			}
		}

		prev.target = sink.get();
		sink->set_old_contents_region(prev, false);
	}

	static std::unique_ptr<mtl::render_target> convert_pitch(
		mtl::command_context& /*cmd*/,
		std::unique_ptr<mtl::render_target>& src,
		usz /*out_pitch*/)
	{
		// TODO (same as OpenGL)
		src->state_flags = rsx::surface_state_flags::erase_bkgnd;
		return {};
	}

	static bool is_compatible_surface(const mtl::render_target* surface, const mtl::render_target* ref, u16 width, u16 height, u8 sample_count)
	{
		return surface->format() == ref->format() &&
			surface->get_spp() == sample_count &&
			surface->get_surface_width<rsx::surface_metrics::pixels>() == width &&
			surface->get_surface_height<rsx::surface_metrics::pixels>() == height;
	}

	static void prepare_surface_for_drawing(mtl::command_context& cmd, mtl::render_target* surface)
	{
		surface->memory_barrier(cmd, rsx::surface_access::gpu_reference);
		surface->memory_usage_flags |= rsx::surface_usage_flags::attachment;
	}

	static void prepare_surface_for_sampling(mtl::command_context&, mtl::render_target*)
	{
	}

	static bool surface_is_pitch_compatible(const std::unique_ptr<mtl::render_target>& surface, usz pitch)
	{
		return surface->get_rsx_pitch() == pitch;
	}

	static void int_invalidate_surface_contents(mtl::command_context&, mtl::render_target* surface, u32 address, usz pitch)
	{
		surface->set_rsx_pitch(static_cast<u32>(pitch));
		surface->queue_tag(address);
		surface->last_use_tag = 0;
		surface->stencil_init_flags = 0;
		surface->memory_usage_flags = rsx::surface_usage_flags::unknown;
		surface->raster_type = rsx::surface_raster_type::linear;
	}

	static void invalidate_surface_contents(
		mtl::command_context& cmd,
		mtl::render_target* surface,
		rsx::surface_color_format format,
		u32 address,
		usz pitch)
	{
		surface->set_format(format);
		surface->set_native_component_layout(mtl::surface_color_component_layout(format));
		surface->set_label(fmt::format("RTV@0x%x", address));
		int_invalidate_surface_contents(cmd, surface, address, pitch);
	}

	static void invalidate_surface_contents(
		mtl::command_context& cmd,
		mtl::render_target* surface,
		rsx::surface_depth_format2 format,
		u32 address,
		usz pitch)
	{
		surface->set_format(format);
		surface->set_label(fmt::format("DSV@0x%x", address));
		int_invalidate_surface_contents(cmd, surface, address, pitch);
	}

	static void notify_surface_invalidated(const std::unique_ptr<mtl::render_target>& surface)
	{
		if (!surface->old_contents.empty())
		{
			surface->clear_rw_barrier();
		}

		surface->release();
	}

	static void notify_surface_persist(const std::unique_ptr<mtl::render_target>& /*surface*/)
	{
	}

	static void notify_surface_reused(const std::unique_ptr<mtl::render_target>& surface)
	{
		surface->state_flags |= rsx::surface_state_flags::erase_bkgnd;
		surface->add_ref();
	}

	static bool int_surface_matches_properties(
		const std::unique_ptr<mtl::render_target>& surface,
		mtl::pixel_format format,
		usz width, usz height,
		rsx::surface_antialiasing antialias,
		const rsx::surface_scaling_config_t& scaling_config,
		bool check_refs = false)
	{
		if (check_refs && surface->has_refs())
			return false;

		return surface->format() == format &&
			surface->get_spp() == get_format_sample_count(antialias) &&
			surface->matches_dimensions(static_cast<u16>(width), static_cast<u16>(height)) &&
			surface->resolution_scaling_config == scaling_config;
	}

	static bool surface_matches_properties(
		const std::unique_ptr<mtl::render_target>& surface,
		rsx::surface_color_format format,
		usz width, usz height,
		rsx::surface_antialiasing antialias,
		const rsx::surface_scaling_config_t& scaling_config,
		bool check_refs = false)
	{
		return int_surface_matches_properties(surface, mtl::surface_color_format_to_mtl(format), width, height, antialias, scaling_config, check_refs);
	}

	static bool surface_matches_properties(
		const std::unique_ptr<mtl::render_target>& surface,
		rsx::surface_depth_format2 format,
		usz width, usz height,
		rsx::surface_antialiasing antialias,
		const rsx::surface_scaling_config_t& scaling_config,
		bool check_refs = false)
	{
		return int_surface_matches_properties(surface, mtl::surface_depth_format_to_mtl(format), width, height, antialias, scaling_config, check_refs);
	}

	static void spill_buffer(std::unique_ptr<std::vector<u8>>& /*bo*/)
	{
	}

	static void unspill_buffer(std::unique_ptr<std::vector<u8>>& /*bo*/)
	{
	}

	static void write_render_target_to_memory(
		mtl::command_context&,
		std::vector<u8>*,
		mtl::render_target*,
		u64, u64, u64)
	{
		// TODO: GPU -> guest memory write-back (Write Color/Depth Buffers)
	}

	template <int BlockSize>
	static std::vector<u8>* merge_bo_list(mtl::command_context&, const std::vector<std::vector<u8>*>& /*list*/)
	{
		return nullptr;
	}

	template <typename T>
	static T* get(const std::unique_ptr<T>& in)
	{
		return in.get();
	}
};

struct mtl_render_targets : public rsx::surface_store<mtl_render_target_traits>
{
	// Debugging: visits every color and depth surface
	template <typename F>
	void for_each_surface(F&& func)
	{
		const auto all = rsx::address_range32::start_end(0, 0xfffffffe);
		for (auto data : { &m_render_targets_storage, &m_depth_stencil_storage })
		{
			for (auto it = data->begin_range(all); it != data->end(); ++it)
			{
				func(it->first, mtl_render_target_traits::get(it->second));
			}
		}
	}

	void destroy()
	{
		invalidate_all();
		invalidated_resources.clear();
	}

	void trim(mtl::command_context& cmd)
	{
		run_cleanup_internal(cmd, rsx::problem_severity::moderate, 256, [](mtl::command_context&) {});

		invalidated_resources.remove_if([](auto& rtt)
		{
			return rtt->unused_check_count() >= 2;
		});
	}
};
