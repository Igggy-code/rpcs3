#pragma once

// RSX texture cache for the Metal backend: plugs Metal images into the backend-agnostic
// rsx::texture_cache (Common/texture_cache.h). Ported from the OpenGL backend (GLTextureCache.h).

#include "MTLTexture.h"
#include "MTLRenderTargets.h"

#include "Emu/RSX/Common/texture_cache.h"

#include <memory>
#include <vector>

namespace mtl
{
	class cached_texture_section;
	class texture_cache;

	struct texture_cache_traits
	{
		using commandbuffer_type      = mtl::command_context;
		using section_storage_type    = mtl::cached_texture_section;
		using texture_cache_type      = mtl::texture_cache;
		using texture_cache_base_type = rsx::texture_cache<texture_cache_type, texture_cache_traits>;
		using image_resource_type     = mtl::texture*;
		using image_view_type         = mtl::texture_view*;
		using image_storage_type      = mtl::texture;
		using texture_format          = mtl::pixel_format;
		using viewable_image_type     = mtl::viewable_image*;
	};

	// Scaled/typeless blits for upload_scaled_image (same role as gl::blitter)
	struct blitter
	{
		void scale_image(command_context& cmd, texture* src, texture* dst, areai src_rect, areai dst_rect,
			bool linear_interpolation, const rsx::typeless_xfer& xfer_info);
	};

	class cached_texture_section : public rsx::cached_texture_section<mtl::cached_texture_section, mtl::texture_cache_traits>
	{
		using baseclass = rsx::cached_texture_section<mtl::cached_texture_section, mtl::texture_cache_traits>;
		friend baseclass;

		viewable_image* vram_texture = nullptr;
		std::unique_ptr<viewable_image> managed_texture;

		// Guest-layout copy of the image for CPU readback (filled by copy_texture)
		std::vector<u8> m_flush_buffer;

	public:
		using baseclass::cached_texture_section;

		void create(u16 w, u16 h, u16 depth, u16 mipmaps, texture* image, u32 rsx_pitch, bool managed);
		void create(u16 w, u16 h, u16 depth, u16 mipmaps, texture* image, u32 rsx_pitch, bool managed, const render_target* surface);

		void set_dimensions(u32 width, u32 height, u32 /*depth*/, u32 pitch)
		{
			this->width = static_cast<u16>(width);
			this->height = static_cast<u16>(height);
			rsx_pitch = pitch;
		}

		// GPU -> guest layout copy
		void copy_texture(command_context& cmd, bool miss);
		// Copy of another image into this (DMA-only) section's guest layout
		void dma_transfer(command_context& cmd, texture* src, const areai& src_area, const utils::address_range32& valid_range, u32 pitch);

		void* map_synchronized(u32 offset, u32 size);
		void finish_flush();

		void destroy();
		void sync_surface_memory(const rsx::simple_array<cached_texture_section*>& surfaces);

		bool exists() const
		{
			return vram_texture != nullptr;
		}

		bool is_managed() const
		{
			return !exists() || !!managed_texture;
		}

		pixel_format get_format() const
		{
			return vram_texture ? vram_texture->format() : pixel_format::invalid;
		}

		texture_view* get_view(const rsx::texture_channel_remap_t& remap)
		{
			return vram_texture->get_view(remap);
		}

		viewable_image* get_raw_texture() const
		{
			return managed_texture.get();
		}

		render_target* get_render_target() const
		{
			return as_rtt(vram_texture);
		}

		texture_view* get_raw_view()
		{
			return vram_texture->get_raw_view();
		}

		bool is_depth_texture() const
		{
			return vram_texture && vram_texture->is_depth();
		}

		bool has_compatible_format(texture* tex) const
		{
			return vram_texture && tex->format() == vram_texture->format();
		}
	};

	class texture_cache : public rsx::texture_cache<mtl::texture_cache, mtl::texture_cache_traits>
	{
	private:
		using baseclass = rsx::texture_cache<mtl::texture_cache, mtl::texture_cache_traits>;
		friend baseclass;

		struct temporary_image_t : public viewable_image, public rsx::ref_counted
		{
			u64 properties_encoding = 0;

			using viewable_image::viewable_image;
		};

		std::vector<std::unique_ptr<temporary_image_t>> m_temporary_surfaces;
		const u32 max_cached_image_pool_size = 256;

		void clear();
		void clear_temporary_subresources();

		void initialize_subresource_from_memory(command_context& cmd, texture* dst, const deferred_subresource& desc, rsx::texture_dimension_extended type) const;

		texture_view* create_temporary_subresource_impl(command_context& cmd, texture* src, pixel_format format, rsx::texture_dimension_extended dst_type, u32 gcm_format,
			u16 width, u16 height, u16 depth, u8 mipmaps, const rsx::texture_channel_remap_t& remap, const copy_region_descriptor* copy = nullptr);

		std::array<swizzle, 4> get_component_mapping_for(u32 gcm_format, rsx::component_order flags) const;

		void copy_transfer_regions_impl(command_context& cmd, texture* dst_image, const rsx::simple_array<copy_region_descriptor>& sources) const;

		texture* get_template_from_collection_impl(const rsx::simple_array<copy_region_descriptor>& sections_to_transfer) const;

	protected:
		texture_view* create_temporary_subresource_view(command_context& cmd, const deferred_subresource& desc) override;
		texture_view* generate_cubemap_from_images(command_context& cmd, const deferred_subresource& desc) override;
		texture_view* generate_3d_from_2d_images(command_context& cmd, const deferred_subresource& desc) override;
		texture_view* generate_atlas_from_images(command_context& cmd, const deferred_subresource& desc) override;
		texture_view* generate_2d_mipmaps_from_images(command_context& cmd, const deferred_subresource& desc) override;
		void release_temporary_subresource(texture_view* view) override;
		void update_image_contents(command_context& cmd, texture_view* dst, const deferred_subresource& desc) override;

		cached_texture_section* create_new_texture(command_context& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch,
			u32 gcm_format, rsx::texture_upload_context context, rsx::texture_dimension_extended type, bool swizzled, rsx::component_order swizzle_flags, rsx::flags32_t flags) override;

		cached_texture_section* create_nul_section(command_context& cmd, const utils::address_range32& rsx_range, const rsx::image_section_attributes_t& attrs,
			const rsx::GCM_tile_reference& tile, bool memory_load) override;

		cached_texture_section* upload_image_from_cpu(command_context& cmd, const utils::address_range32& rsx_range, u16 width, u16 height, u16 depth, u16 mipmaps, u32 pitch, u32 gcm_format,
			rsx::texture_upload_context context, const std::vector<rsx::subresource_layout>& subresource_layout, rsx::texture_dimension_extended type, bool input_swizzled) override;

		void set_component_order(cached_texture_section& section, u32 gcm_format, rsx::component_order flags) override;

		void insert_texture_barrier(command_context&, texture*, bool) override
		{
			// Metal tracks hazards between passes; sampling a bound attachment is avoided by the cache (copies)
		}

		bool render_target_format_is_compatible(texture* tex, u32 gcm_format) override;

		void prepare_for_dma_transfers(command_context&) override {}
		void cleanup_after_dma_transfers(command_context&) override {}

	public:
		using baseclass::texture_cache;

		void initialize() {}

		void destroy() override
		{
			clear();
		}

		bool is_depth_texture(u32 rsx_address, u32 rsx_size) override;

		void on_frame_end() override;

		bool blit(command_context& cmd, const rsx::blit_src_info& src, const rsx::blit_dst_info& dst, bool linear_interpolate, mtl_render_targets& rtts);
	};
}
