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

	// True when the build has SPIRV-Cross (HAVE_SPIRV_CROSS)
	bool shader_translation_available();

	// SPIR-V -> MSL. On success 'msl' holds the source and 'entry' the (cleansed) entry point name.
	bool spirv_to_msl(const std::vector<u32>& spirv, shader_stage stage, std::string& msl, std::string& entry, std::string& error);
}
