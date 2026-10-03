#pragma once

// Plain C++ interface to the Metal device layer (MTLDevice.mm).
// Everything that touches Objective-C lives in the .mm files; the rest of the backend only sees
// opaque handles, so it compiles as regular C++ together with the RPCS3 headers.

#include <cstdint>
#include <string>

namespace mtl
{
	using u8 = std::uint8_t;
	using u32 = std::uint32_t;

	enum class pixel_format : u32
	{
		invalid = 0,

		// Color
		bgra8,     // RSX A8R8G8B8 / X8R8G8B8
		rgba8,     // RSX A8B8G8R8 / X8B8G8R8
		b5g6r5,    // RSX R5G6B5
		bgr5a1,    // RSX X1R5G5B5 (approximation, alpha bit ignored)
		r8,        // RSX B8
		rg8,       // RSX G8B8
		rgba16f,   // RSX W16Z16Y16X16
		rgba32f,   // RSX W32Z32Y32X32
		r32f,      // RSX X32

		// Depth / stencil
		depth16,          // RSX Z16 (fixed point)
		depth32f,         // RSX Z16 (float)
		depth32f_stencil8 // RSX Z24S8 (Apple GPUs have no D24S8)
	};

	bool is_depth_format(pixel_format format);
	bool has_stencil(pixel_format format);

	enum texture_usage : u32
	{
		usage_sampled = 1,
		usage_render_target = 2,
	};

	// Opaque Metal texture. Owns one retained id<MTLTexture>.
	class texture
	{
	public:
		texture(u32 width, u32 height, pixel_format format, u32 usage);
		texture(const texture&) = delete;
		texture& operator=(const texture&) = delete;
		virtual ~texture();

		u32 width() const { return m_width; }
		u32 height() const { return m_height; }
		u8 samples() const { return 1; }
		pixel_format format() const { return m_format; }
		bool is_depth() const { return is_depth_format(m_format); }
		bool valid() const { return m_handle != nullptr; }

		void* native() const { return m_handle; }
		void set_label(const std::string& label);

	private:
		void* m_handle = nullptr;
		u32 m_width = 0;
		u32 m_height = 0;
		pixel_format m_format = pixel_format::invalid;
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
}
