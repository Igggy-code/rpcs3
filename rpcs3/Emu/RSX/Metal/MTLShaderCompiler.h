#pragma once

// RSX program -> Metal shader translation.
//
// The RSX programs are decompiled to Vulkan-flavoured GLSL by the existing Vulkan decompilers,
// compiled to SPIR-V with glslang (Program/SPIRVCommon), converted to MSL with SPIRV-Cross,
// and finally compiled by the Metal driver (MTLDevice.mm).

#include "MTLDevice.h"

#include <memory>
#include <string>
#include <vector>

namespace mtl
{
	enum class shader_stage : u32
	{
		vertex,
		fragment,
	};

	// Opaque compiled Metal function (id<MTLFunction>)
	class shader_function
	{
	public:
		shader_function() = default;
		shader_function(const shader_function&) = delete;
		shader_function& operator=(const shader_function&) = delete;
		~shader_function();

		bool valid() const { return m_handle != nullptr; }
		void* native() const { return m_handle; }

		// Compiles MSL source and looks up 'entry'. Returns false and fills 'error' on failure.
		bool create(const std::string& msl, const std::string& entry, std::string& error);

	private:
		void* m_handle = nullptr;
	};

	enum class resource_kind : u8
	{
		buffer,        // Uniform/storage block -> [[buffer(n)]]
		push_constant, // Push constant block -> [[buffer(n)]]
		texel_buffer,  // (u)samplerBuffer -> texture_buffer [[texture(n)]]
		texture,       // Sampled image -> [[texture(n)]] + [[sampler(m)]]
	};

	enum class texture_dimension : u8
	{
		dim_1d,
		dim_2d,
		dim_3d,
		dim_cube,
	};

	// One resource of a translated shader and the Metal argument slots it was assigned
	struct shader_resource
	{
		std::string name;     // GLSL block or variable name (e.g. "VertexContextBuffer", "tex0")
		resource_kind kind = resource_kind::buffer;
		u32 binding = 0;      // Vulkan binding in the GLSL source
		u32 msl_index = 0;    // Metal buffer or texture index
		u32 msl_sampler = 0;  // Metal sampler index (textures only)
		texture_dimension dimension = texture_dimension::dim_2d;
		bool depth = false;        // Shadow sampler (depth2d)
		bool multisampled = false;
	};

	// Metal buffer index used for push constants (vertex pulling never uses stage_in buffers)
	constexpr u32 push_constant_buffer_index = 30;

	// True when the build has SPIRV-Cross (HAVE_SPIRV_CROSS)
	bool shader_translation_available();

	// SPIR-V -> MSL. On success 'msl' holds the source, 'entry' the (cleansed) entry point name and
	// 'resources' the Metal slots of every buffer and texture the shader declares.
	bool spirv_to_msl(const std::vector<u32>& spirv, shader_stage stage, std::string& msl, std::string& entry,
		std::vector<shader_resource>& resources, std::string& error);
}
