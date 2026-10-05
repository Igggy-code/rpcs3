#pragma once

// Plain C++ interface to the Metal device layer (MTLDevice.mm, MTLTextureOps.mm).
// Everything that touches Objective-C lives in the .mm files; the rest of the backend only sees
// opaque handles, so it compiles as regular C++ together with the RPCS3 headers.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace mtl
{
	using u8 = std::uint8_t;
	using u16 = std::uint16_t;
	using u32 = std::uint32_t;
	using u64 = std::uint64_t;

	enum class pixel_format : u32
	{
		invalid = 0,

		// Color (render targets)
		bgra8,     // RSX A8R8G8B8 / X8R8G8B8
		rgba8,     // RSX A8B8G8R8 / X8B8G8R8
		b5g6r5,    // RSX R5G6B5 (same bit layout as VK R5G6B5_UNORM_PACK16)
		bgr5a1,    // RSX X1R5G5B5 / A1R5G5B5 (same bit layout as VK A1R5G5B5_UNORM_PACK16)
		r8,        // RSX B8
		rg8,       // RSX G8B8
		rgba16f,   // RSX W16Z16Y16X16
		rgba32f,   // RSX W32Z32Y32X32
		r32f,      // RSX X32

		// Depth / stencil
		depth16,          // RSX Z16 (fixed point)
		depth32f,         // RSX Z16 (float)
		depth32f_stencil8,// RSX Z24S8 (Apple GPUs have no D24S8)

		// Sampled-only formats (textures)
		a1bgr5,    // VK R5G5B5A1_UNORM_PACK16
		abgr4,     // VK R4G4B4A4_UNORM_PACK16
		rg8_snorm,
		r16,
		rg16,
		rg16f,
		bc1,
		bc2,
		bc3,
		r8_uint,   // Typeless helpers
		r16_uint,
		r32_uint,
		rg32_uint,
		rgba32_uint,
		x32_stencil8, // Stencil view of depth32f_stencil8
	};

	bool is_depth_format(pixel_format format);
	bool has_stencil(pixel_format format);
	bool is_compressed_format(pixel_format format);
	// Bytes per texel (per 4x4 block for compressed formats)
	u32 get_format_block_size(pixel_format format);

	enum texture_usage : u32
	{
		usage_sampled = 1,
		usage_render_target = 2,
		usage_format_view = 4, // Views may reinterpret the pixel format
	};

	enum class texture_type : u8
	{
		tex_1d,
		tex_2d,
		tex_3d,
		tex_cube,
	};

	// Same numbering as MTLTextureSwizzle
	enum class swizzle : u8
	{
		zero = 0,
		one = 1,
		red = 2,
		green = 3,
		blue = 4,
		alpha = 5,
	};

	// Swizzle in RGBA order (what Metal expects)
	using swizzle_rgba = std::array<swizzle, 4>;
	constexpr swizzle_rgba identity_swizzle = { swizzle::red, swizzle::green, swizzle::blue, swizzle::alpha };

	enum class image_aspect : u8
	{
		color = 1,
		depth = 2,
		stencil = 4,
	};

	struct texture_desc
	{
		texture_type type = texture_type::tex_2d;
		u32 width = 1;
		u32 height = 1;
		u32 depth = 1;   // 3D textures only
		u32 levels = 1;
		pixel_format format = pixel_format::invalid;
		u32 usage = usage_sampled;
	};

	class texture;

	// Opaque view of a texture (id<MTLTexture> made with newTextureViewWithPixelFormat:...:swizzle:)
	class texture_view
	{
	public:
		texture_view(texture* image, void* handle, const swizzle_rgba& mapping, image_aspect aspect, u32 remap_encoding)
			: m_image(image), m_handle(handle), m_mapping(mapping), m_aspect(aspect), m_remap_encoding(remap_encoding)
		{
		}

		texture_view(const texture_view&) = delete;
		texture_view& operator=(const texture_view&) = delete;
		~texture_view();

		texture* image() const { return m_image; }
		void* native() const { return m_handle; }
		const swizzle_rgba& mapping() const { return m_mapping; }
		image_aspect aspect() const { return m_aspect; }

		// RSX channel remap this view was created for (texture cache bookkeeping)
		u32 encoded_component_map() const { return m_remap_encoding; }

	private:
		texture* m_image = nullptr;
		void* m_handle = nullptr;
		swizzle_rgba m_mapping = identity_swizzle;
		image_aspect m_aspect = image_aspect::color;
		u32 m_remap_encoding = 0;
	};

	// Opaque Metal texture. Owns one retained id<MTLTexture> and the views made from it.
	class texture
	{
	public:
		texture(const texture_desc& desc);
		texture(u32 width, u32 height, pixel_format format, u32 usage);
		texture(const texture&) = delete;
		texture& operator=(const texture&) = delete;
		virtual ~texture();

		u32 width() const { return m_desc.width; }
		u32 height() const { return m_desc.height; }
		u32 depth() const { return m_desc.depth; }
		u32 layers() const { return m_desc.type == texture_type::tex_cube ? 6 : 1; }
		u32 levels() const { return m_desc.levels; }
		u8 samples() const { return 1; }
		texture_type type() const { return m_desc.type; }
		pixel_format format() const { return m_desc.format; }
		u32 usage() const { return m_desc.usage; }
		bool is_depth() const { return is_depth_format(m_desc.format); }
		bool valid() const { return m_handle != nullptr; }

		// Bytes per texel row unit (per block for compressed formats)
		u32 block_size() const { return get_format_block_size(m_desc.format); }

		void* native() const { return m_handle; }
		void set_label(const std::string& label);

		// View with the given RGBA swizzle. Depth/stencil textures can be viewed as depth or stencil.
		// 'remap_encoding' identifies the RSX remap the swizzle was derived from.
		texture_view* get_view(const swizzle_rgba& mapping, image_aspect aspect, u32 remap_encoding);

		// Native component layout in RSX ARGB order: which host channel holds A, R, G and B.
		// Views apply the RSX remap on top of it.
		const std::array<swizzle, 4>& native_component_layout() const { return m_native_layout; }
		void set_native_component_layout(const std::array<swizzle, 4>& layout);

	private:
		void* m_handle = nullptr;
		texture_desc m_desc{};
		std::array<swizzle, 4> m_native_layout = { swizzle::alpha, swizzle::red, swizzle::green, swizzle::blue };
		std::unordered_map<u64, std::unique_ptr<texture_view>> m_views;
	};

	struct clear_rect
	{
		u32 x = 0, y = 0, width = 0, height = 0;
	};

	// Device lifetime (one device for the whole backend)
	bool init_device(std::string& device_name, std::string& error);
	void shutdown_device();

	// Commands are recorded into one pending command buffer, committed by flush() or present.
	void clear_color(texture& dst, const float rgba[4], u32 write_mask);
	void clear_depth_stencil(texture& dst, bool clear_depth, float depth, bool clear_stencil, u8 stencil);
	// Same-format region copy (used when surfaces inherit contents from older surfaces)
	bool copy_region(texture& src, texture& dst, u32 src_x, u32 src_y, u32 dst_x, u32 dst_y, u32 width, u32 height);
	void flush();

	// Error messages from the Objective-C++ layer (which cannot use the RPCS3 log channels)
	using log_handler = void (*)(const char* message);
	void set_log_handler(log_handler handler);
	// Submits pending work and waits for the GPU to finish it
	void finish();
}
