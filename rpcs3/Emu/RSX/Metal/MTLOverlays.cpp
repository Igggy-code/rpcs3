#include "stdafx.h"
#include "MTLOverlays.h"

#include "Emu/RSX/Overlays/overlays.h"
#include "Emu/RSX/Overlays/overlay_controls.h"
#include "Emu/RSX/Program/RSXOverlay.h"
#include "Emu/Cell/timers.hpp"

namespace mtl
{
	namespace
	{
		// Resource images (icons) use keys 1..N, fonts and temporary images their object address
		constexpr u32 no_owner = umax;

		void copy4(float* dst, f32 a, f32 b, f32 c, f32 d)
		{
			dst[0] = a;
			dst[1] = b;
			dst[2] = c;
			dst[3] = d;
		}
	}

	ui_overlay_renderer::~ui_overlay_renderer()
	{
		destroy();
	}

	bool ui_overlay_renderer::create()
	{
		if (m_renderer)
		{
			return true;
		}

		std::string error;
		m_renderer = overlay::create_renderer(error);

		if (!m_renderer)
		{
			rsx_log.error("Metal: overlay renderer could not be created: %s", error);
			return false;
		}

		rsx::overlays::resource_config configuration;
		configuration.load_files();

		u64 key = 1;
		for (const auto& res : configuration.texture_raw_data)
		{
			overlay::upload_texture(m_renderer, key++, res->w, res->h, 1, false, res->get_data(), no_owner);
		}

		configuration.free_resources();
		return true;
	}

	void ui_overlay_renderer::destroy()
	{
		overlay::destroy_renderer(std::exchange(m_renderer, nullptr));
	}

	void ui_overlay_renderer::remove_temp_resources(u32 owner_uid)
	{
		if (m_renderer)
		{
			overlay::remove_textures_of_owner(m_renderer, owner_uid);
		}
	}

	void ui_overlay_renderer::run(void* encoder, const areau& viewport, rsx::overlays::overlay& ui)
	{
		if (!m_renderer || !viewport.width() || !viewport.height())
		{
			return;
		}

		ui.set_render_viewport(
			static_cast<u16>(std::min<u32>(viewport.width(), 0xffff)),
			static_cast<u16>(std::min<u32>(viewport.height(), 0xffff)));

		if (ui.status_flags & rsx::overlays::status_bits::invalidate_image_cache)
		{
			remove_temp_resources(ui.uid);
			ui.status_flags.clear(rsx::overlays::status_bits::invalidate_image_cache);
		}

		const f32 ui_w = ui.get_virtual_width();
		const f32 ui_h = ui.get_virtual_height();
		const float vp[4] = { static_cast<f32>(viewport.x1), static_cast<f32>(viewport.y1), static_cast<f32>(viewport.width()), static_cast<f32>(viewport.height()) };

		for (auto& command : ui.get_compiled().draw_commands)
		{
			const auto& config = command.config;

			overlay::draw_cmd cmd{};
			cmd.vertices = reinterpret_cast<const float*>(command.verts.data());
			cmd.vertex_count = ::size32(command.verts);

			switch (config.primitives)
			{
			case rsx::overlays::primitive_type::quad_list: cmd.type = overlay::primitive::quad_list; break;
			case rsx::overlays::primitive_type::triangle_strip: cmd.type = overlay::primitive::triangle_strip; break;
			case rsx::overlays::primitive_type::line_list: cmd.type = overlay::primitive::line_list; break;
			case rsx::overlays::primitive_type::line_strip: cmd.type = overlay::primitive::line_strip; break;
			case rsx::overlays::primitive_type::triangle_fan: cmd.type = overlay::primitive::triangle_fan; break;
			}

			auto texture_mode = rsx::overlays::texture_sampling_mode::texture2D;

			switch (config.texture_ref)
			{
			case rsx::overlays::image_resource_id::game_icon:
			case rsx::overlays::image_resource_id::backbuffer:
				// Not supported (same as the other backends)
			case rsx::overlays::image_resource_id::none:
				texture_mode = rsx::overlays::texture_sampling_mode::none;
				break;
			case rsx::overlays::image_resource_id::font_file:
			{
				const auto font = config.font_ref;
				const auto size = font->get_glyph_data_dimensions();
				const u64 key = reinterpret_cast<u64>(font);

				if (!overlay::has_texture(m_renderer, key, size.width, size.height, size.depth))
				{
					overlay::upload_texture(m_renderer, key, size.width, size.height, size.depth, true, font->get_glyph_data().data(), no_owner);
				}

				texture_mode = rsx::overlays::texture_sampling_mode::font3D;
				cmd.source = overlay::sampler_source::font;
				cmd.texture_key = key;
				break;
			}
			case rsx::overlays::image_resource_id::raw_image:
			{
				const auto desc = static_cast<const rsx::overlays::image_info_base*>(config.external_data_ref);
				const bool dirty = std::exchange(desc->dirty, false);
				const u64 key = reinterpret_cast<u64>(desc);

				if (dirty || !overlay::has_texture(m_renderer, key, desc->w, desc->h, 1))
				{
					overlay::upload_texture(m_renderer, key, desc->w, desc->h, 1, false, desc->get_data(), ui.uid);
				}

				cmd.source = overlay::sampler_source::image;
				cmd.texture_key = key;
				break;
			}
			default:
				cmd.source = overlay::sampler_source::image;
				cmd.texture_key = config.texture_ref;
				break;
			}

			copy4(cmd.vs.ui_scale, ui_w, ui_h, 1.f, 1.f);
			copy4(cmd.vs.albedo, config.color.r, config.color.g, config.color.b, config.color.a);
			copy4(cmd.vs.viewport, vp[2], vp[3], vp[0], vp[1]);
			copy4(cmd.vs.clip_bounds, config.clip_rect.x1, config.clip_rect.y1, config.clip_rect.x2, config.clip_rect.y2);
			cmd.vs.options = rsx::overlays::vertex_options{}
				.disable_vertex_snap(config.disable_vertex_snap)
				.enable_vertical_flip(false)
				.get();

			cmd.fs.options = rsx::overlays::fragment_options{}
				.texture_mode(texture_mode)
				.clip_fragments(config.clip_region)
				.pulse_glow(config.pulse_glow)
				.set_sdf(config.sdf_config.func)
				.get();
			cmd.fs.timestamp = config.get_sinus_value();
			cmd.fs.blur_intensity = static_cast<f32>(config.blur_strength) * 0.01f;

			if (config.sdf_config.func != rsx::overlays::sdf_function::none)
			{
				auto sdf = config.sdf_config;
				sdf.transform(static_cast<areaf>(viewport), { ui_w, ui_h });

				copy4(cmd.fs.sdf_params, sdf.hx, sdf.hy, sdf.br, sdf.bw);
				copy4(cmd.fs.sdf_origin, sdf.cx, sdf.cy, 0.f, 0.f);
				copy4(cmd.fs.sdf_border_color, sdf.border_color.r, sdf.border_color.g, sdf.border_color.b, sdf.border_color.a);
			}

			overlay::draw(m_renderer, encoder, vp, cmd);
		}

		ui.update(get_system_time());
	}
}
