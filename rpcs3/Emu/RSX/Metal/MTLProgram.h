#pragma once

#include "MTLShaderCompiler.h"
#include "MTLDraw.h"

#include "Emu/RSX/Program/ProgramStateCache.h"
#include "Emu/RSX/VK/VKVertexProgram.h"
#include "Emu/RSX/VK/VKFragmentProgram.h"

namespace mtl
{
	// Translation statistics (vertex + fragment programs)
	extern atomic_t<u32> g_programs_ok;
	extern atomic_t<u32> g_programs_failed;

	// Result of translating one RSX program: GLSL (from the Vulkan decompiler) -> SPIR-V -> MSL -> MTLFunction
	struct translated_program
	{
		std::string glsl;
		std::string msl;
		std::string entry;
		std::string error;     // Empty on success
		shader_function function;
		std::vector<shader_resource> resources;

		bool ok() const { return error.empty() && function.valid(); }
	};

	// Pipeline state that selects a Metal render pipeline
	using pipeline_props = render_pipeline_state;

	struct pipeline
	{
		render_pipeline state;
		bool valid = false;
	};

	// Creates the Metal pipeline for a program pair (logs failures)
	std::unique_ptr<pipeline> create_pipeline(const translated_program& vp, const translated_program& fp, const pipeline_props& props);
}

// Reuses the Vulkan program objects as decompiler hosts: they own the binding tables the GLSL refers to.
// Nothing Vulkan-specific (devices, VkShaderModule) is touched.
class MTLVertexProgram : public rsx::VertexProgramBase
{
public:
	VKVertexProgram host;
	mtl::translated_program result;

	void Decompile(const RSXVertexProgram& prog);
	void Compile();
};

class MTLFragmentProgram
{
public:
	u32 id = 0;
	VKFragmentProgram host;
	mtl::translated_program result;

	void Decompile(const RSXFragmentProgram& prog);
	void Compile();
};

struct MTLProgramTraits
{
	using vertex_program_type = MTLVertexProgram;
	using fragment_program_type = MTLFragmentProgram;
	using pipeline_type = mtl::pipeline;
	using pipeline_storage_type = std::unique_ptr<mtl::pipeline>;
	using pipeline_properties = mtl::pipeline_props;

	static void recompile_fragment_program(const RSXFragmentProgram& RSXFP, fragment_program_type& fragmentProgramData, usz ID)
	{
		fragmentProgramData.Decompile(RSXFP);
		fragmentProgramData.id = static_cast<u32>(ID);
		fragmentProgramData.Compile();
	}

	static void recompile_vertex_program(const RSXVertexProgram& RSXVP, vertex_program_type& vertexProgramData, usz ID)
	{
		vertexProgramData.Decompile(RSXVP);
		vertexProgramData.id = static_cast<u32>(ID);
		vertexProgramData.Compile();
	}

	static void validate_pipeline_properties(const MTLVertexProgram&, const MTLFragmentProgram&, mtl::pipeline_props&)
	{
	}

	static pipeline_type* build_pipeline(
		const vertex_program_type& vertexProgramData,
		const fragment_program_type& fragmentProgramData,
		const mtl::pipeline_props& pipelineProperties,
		bool /*compile_async*/,
		std::function<pipeline_type*(pipeline_storage_type&)> callback)
	{
		auto result = mtl::create_pipeline(vertexProgramData.result, fragmentProgramData.result, pipelineProperties);
		return callback(result);
	}
};

struct MTLProgramBuffer : public program_state_cache<MTLProgramTraits>
{
};
