#include "stdafx.h"
#include "MTLDebugDump.h"
#include "MTLTextureOps.h"

#include "Utilities/File.h"

#include <zlib.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <vector>

namespace mtl::debug
{
	atomic_t<bool> g_frame_dump_active{false};

	namespace
	{
		f32 half_to_float(u16 h)
		{
			const u32 sign = (h & 0x8000u) << 16;
			u32 exponent = (h >> 10) & 0x1f;
			u32 mantissa = h & 0x3ff;

			if (exponent == 0)
			{
				if (!mantissa) return std::bit_cast<f32>(sign);
				while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
				exponent++;
				mantissa &= 0x3ff;
			}
			else if (exponent == 31)
			{
				return std::bit_cast<f32>(sign | 0x7f800000u | (mantissa << 13));
			}

			return std::bit_cast<f32>(sign | ((exponent + 112) << 23) | (mantissa << 13));
		}

		u8 unorm8(f32 v)
		{
			if (!(v == v)) return 255; // NaN shows as white
			return static_cast<u8>(std::clamp(v, 0.f, 1.f) * 255.f + 0.5f);
		}

		void append_chunk(std::vector<u8>& out, const char* type, const u8* data, u32 length)
		{
			const u8 len_be[4] = { u8(length >> 24), u8(length >> 16), u8(length >> 8), u8(length) };
			out.insert(out.end(), len_be, len_be + 4);

			const usz type_pos = out.size();
			out.insert(out.end(), type, type + 4);
			if (length) out.insert(out.end(), data, data + length);

			const uLong crc = crc32(0, out.data() + type_pos, static_cast<uInt>(4 + length));
			const u8 crc_be[4] = { u8(crc >> 24), u8(crc >> 16), u8(crc >> 8), u8(crc) };
			out.insert(out.end(), crc_be, crc_be + 4);
		}

		// Gray image with the value range stretched to [0, 255] (depth buffers sit close to 1.0)
		std::vector<u8> stretch(const std::vector<f32>& values, f32& lo, f32& hi)
		{
			lo = 1e30f;
			hi = -1e30f;
			for (f32 v : values)
			{
				if (v == v && std::isfinite(v)) { lo = std::min(lo, v); hi = std::max(hi, v); }
			}

			std::vector<u8> out(values.size());
			const f32 range = hi > lo ? hi - lo : 1.f;
			for (usz i = 0; i < values.size(); ++i)
			{
				out[i] = unorm8((values[i] - lo) / range);
			}

			return out;
		}
	}

	bool write_png(const std::string& path, u32 width, u32 height, u32 channels, const u8* data)
	{
		const u32 row_bytes = width * channels;
		std::vector<u8> raw(static_cast<usz>(row_bytes + 1) * height);
		for (u32 y = 0; y < height; ++y)
		{
			raw[static_cast<usz>(y) * (row_bytes + 1)] = 0; // Filter: none
			std::memcpy(&raw[static_cast<usz>(y) * (row_bytes + 1) + 1], data + static_cast<usz>(y) * row_bytes, row_bytes);
		}

		uLongf packed_size = compressBound(static_cast<uLong>(raw.size()));
		std::vector<u8> packed(packed_size);
		if (compress2(packed.data(), &packed_size, raw.data(), static_cast<uLong>(raw.size()), 1) != Z_OK)
		{
			return false;
		}

		std::vector<u8> png = { 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };

		const u8 ihdr[13] =
		{
			u8(width >> 24), u8(width >> 16), u8(width >> 8), u8(width),
			u8(height >> 24), u8(height >> 16), u8(height >> 8), u8(height),
			8, static_cast<u8>(channels == 3 ? 2 : 0), 0, 0, 0
		};

		append_chunk(png, "IHDR", ihdr, 13);
		append_chunk(png, "IDAT", packed.data(), static_cast<u32>(packed_size));
		append_chunk(png, "IEND", nullptr, 0);

		return fs::write_file(path, fs::rewrite, png);
	}

	const char* format_name(pixel_format format)
	{
		switch (format)
		{
		case pixel_format::invalid: return "invalid";
		case pixel_format::bgra8: return "bgra8";
		case pixel_format::rgba8: return "rgba8";
		case pixel_format::b5g6r5: return "b5g6r5";
		case pixel_format::bgr5a1: return "bgr5a1";
		case pixel_format::r8: return "r8";
		case pixel_format::rg8: return "rg8";
		case pixel_format::rgba16f: return "rgba16f";
		case pixel_format::rgba32f: return "rgba32f";
		case pixel_format::r32f: return "r32f";
		case pixel_format::depth16: return "depth16";
		case pixel_format::depth32f: return "depth32f";
		case pixel_format::depth32f_stencil8: return "depth32f_stencil8";
		case pixel_format::a1bgr5: return "a1bgr5";
		case pixel_format::abgr4: return "abgr4";
		case pixel_format::rg8_snorm: return "rg8_snorm";
		case pixel_format::r16: return "r16";
		case pixel_format::rg16: return "rg16";
		case pixel_format::rg16f: return "rg16f";
		case pixel_format::bc1: return "bc1";
		case pixel_format::bc2: return "bc2";
		case pixel_format::bc3: return "bc3";
		case pixel_format::r8_uint: return "r8_uint";
		case pixel_format::r16_uint: return "r16_uint";
		case pixel_format::r32_uint: return "r32_uint";
		case pixel_format::rg32_uint: return "rg32_uint";
		case pixel_format::rgba32_uint: return "rgba32_uint";
		case pixel_format::x32_stencil8: return "x32_stencil8";
		}

		return "?";
	}

	std::string dump_texture(texture& tex, const std::string& path, bool with_alpha)
	{
		const u32 w = tex.width();
		const u32 h = tex.height();
		const usz n = static_cast<usz>(w) * h;

		if (!tex.valid() || !w || !h || is_compressed_format(tex.format()) || tex.type() == texture_type::tex_3d)
		{
			return "not dumped";
		}

		std::vector<u8> rgb(n * 3), alpha(n, 255);
		bool has_alpha = false;

		const auto store = [&](usz i, f32 r, f32 g, f32 b)
		{
			rgb[i * 3 + 0] = unorm8(r);
			rgb[i * 3 + 1] = unorm8(g);
			rgb[i * 3 + 2] = unorm8(b);
		};

		switch (tex.format())
		{
		case pixel_format::depth32f_stencil8:
		case pixel_format::depth32f:
		case pixel_format::depth16:
		{
			std::vector<f32> depth(n);

			if (tex.format() == pixel_format::depth16)
			{
				std::vector<u16> raw(n);
				if (!download_texture(tex, 0, 0, 0, 0, w, h, image_aspect::depth, raw.data(), w * 2)) return "readback failed";
				for (usz i = 0; i < n; ++i) depth[i] = raw[i] / 65535.f;
			}
			else if (!download_texture(tex, 0, 0, 0, 0, w, h, image_aspect::depth, depth.data(), w * 4))
			{
				return "readback failed";
			}

			f32 lo, hi;
			const auto gray = stretch(depth, lo, hi);
			write_png(path + ".png", w, h, 1, gray.data());

			if (tex.format() == pixel_format::depth32f_stencil8)
			{
				std::vector<u8> stencil(n);
				if (download_texture(tex, 0, 0, 0, 0, w, h, image_aspect::stencil, stencil.data(), w))
				{
					write_png(path + "_stencil.png", w, h, 1, stencil.data());
				}
			}

			return fmt::format("depth range [%f, %f] stretched", lo, hi);
		}
		default:
			break;
		}

		const u32 bpp = tex.block_size();
		std::vector<u8> raw(n * bpp);
		if (!download_texture(tex, 0, 0, 0, 0, w, h, image_aspect::color, raw.data(), w * bpp))
		{
			return "readback failed";
		}

		const auto u16_at = [&](usz i, u32 c) { u16 v; std::memcpy(&v, &raw[i * bpp + c * 2], 2); return v; };
		const auto f32_at = [&](usz i, u32 c) { f32 v; std::memcpy(&v, &raw[i * bpp + c * 4], 4); return v; };

		f32 lo = 0.f, hi = 1.f;
		std::vector<f32> scalar;

		for (usz i = 0; i < n; ++i)
		{
			const u8* p = &raw[i * bpp];

			switch (tex.format())
			{
			case pixel_format::bgra8:
				store(i, p[2] / 255.f, p[1] / 255.f, p[0] / 255.f);
				alpha[i] = p[3];
				has_alpha = true;
				break;
			case pixel_format::rgba8:
				store(i, p[0] / 255.f, p[1] / 255.f, p[2] / 255.f);
				alpha[i] = p[3];
				has_alpha = true;
				break;
			case pixel_format::r8:
			case pixel_format::r8_uint:
				store(i, p[0] / 255.f, p[0] / 255.f, p[0] / 255.f);
				break;
			case pixel_format::rg8:
			case pixel_format::rg8_snorm:
				store(i, p[0] / 255.f, p[1] / 255.f, 0.f);
				break;
			case pixel_format::b5g6r5:
			{
				const u16 v = u16_at(i, 0);
				store(i, ((v >> 11) & 31) / 31.f, ((v >> 5) & 63) / 63.f, (v & 31) / 31.f);
				break;
			}
			case pixel_format::bgr5a1:
			{
				const u16 v = u16_at(i, 0);
				store(i, ((v >> 10) & 31) / 31.f, ((v >> 5) & 31) / 31.f, (v & 31) / 31.f);
				alpha[i] = (v & 0x8000) ? 255 : 0;
				has_alpha = true;
				break;
			}
			case pixel_format::a1bgr5:
			{
				const u16 v = u16_at(i, 0);
				store(i, (v & 31) / 31.f, ((v >> 5) & 31) / 31.f, ((v >> 10) & 31) / 31.f);
				break;
			}
			case pixel_format::abgr4:
			{
				const u16 v = u16_at(i, 0);
				store(i, (v & 15) / 15.f, ((v >> 4) & 15) / 15.f, ((v >> 8) & 15) / 15.f);
				alpha[i] = static_cast<u8>(((v >> 12) & 15) * 17);
				has_alpha = true;
				break;
			}
			case pixel_format::r16:
			case pixel_format::r16_uint:
				store(i, u16_at(i, 0) / 65535.f, u16_at(i, 0) / 65535.f, u16_at(i, 0) / 65535.f);
				break;
			case pixel_format::rg16:
				store(i, u16_at(i, 0) / 65535.f, u16_at(i, 1) / 65535.f, 0.f);
				break;
			case pixel_format::rg16f:
				store(i, half_to_float(u16_at(i, 0)), half_to_float(u16_at(i, 1)), 0.f);
				break;
			case pixel_format::rgba16f:
				store(i, half_to_float(u16_at(i, 0)), half_to_float(u16_at(i, 1)), half_to_float(u16_at(i, 2)));
				alpha[i] = unorm8(half_to_float(u16_at(i, 3)));
				has_alpha = true;
				break;
			case pixel_format::rgba32f:
				store(i, f32_at(i, 0), f32_at(i, 1), f32_at(i, 2));
				alpha[i] = unorm8(f32_at(i, 3));
				has_alpha = true;
				break;
			case pixel_format::r32f:
			case pixel_format::r32_uint:
				if (scalar.empty()) scalar.resize(n);
				scalar[i] = f32_at(i, 0);
				break;
			default:
				return "unsupported format";
			}
		}

		if (!scalar.empty())
		{
			const auto gray = stretch(scalar, lo, hi);
			write_png(path + ".png", w, h, 1, gray.data());
			return fmt::format("float range [%f, %f] stretched", lo, hi);
		}

		write_png(path + ".png", w, h, 3, rgb.data());

		if (with_alpha && has_alpha)
		{
			write_png(path + "_alpha.png", w, h, 1, alpha.data());
		}

		return "ok";
	}
}
