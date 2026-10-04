#include "stdafx.h"
#include "MTLProgram.h"

#include "Emu/RSX/Program/SPIRVCommon.h"
#include "Emu/system_config.h"

namespace mtl
{
	atomic_t<u32> g_programs_ok{0};
	atomic_t<u32> g_programs_failed{0};

	std::unique_ptr<pipeline> create_pipeline(const translated_program& vp, const translated_program& fp, const pipeline_props& props)
	{
		auto result = std::make_unique<pipeline>();

		if (!vp.ok() || !fp.ok())
		{
			return result;
		}

		std::string error;
		if (!result->state.create(vp.function, fp.function, props, error))
		{
			rsx_log.error("Metal: render pipeline creation failed: %s", error);
			return result;
		}

		result->valid = true;
		return result;
	}
}

namespace
{
	void translate(const std::string& glsl, ::glsl::program_domain domain, mtl::shader_stage stage, u32 id, mtl::translated_program& out)
	{
		out.glsl = glsl;

		const char* const kind = stage == mtl::shader_stage::vertex ? "Vertex" : "Fragment";

		if (g_cfg.video.log_programs)
		{
			fs::write_file(fs::get_cache_dir() + fmt::format("shaderlog/Metal%sProgram%u.glsl", kind, id), fs::rewrite, glsl);
		}

		std::vector<u32> spirv;
		std::string source = glsl; // compile_glsl_to_spv takes a mutable string

		if (!spirv::compile_glsl_to_spv(spirv, source, domain, ::glsl::glsl_rules_vulkan))
		{
			out.error = "GLSL -> SPIR-V failed";
		}
		else if (!mtl::spirv_to_msl(spirv, stage, out.msl, out.entry, out.resources, out.error))
		{
			// error filled by spirv_to_msl
		}
		else
		{
			if (g_cfg.video.log_programs)
			{
				fs::write_file(fs::get_cache_dir() + fmt::format("shaderlog/Metal%sProgram%u.metal", kind, id), fs::rewrite, out.msl);
			}

			std::string error;
			if (!out.function.create(out.msl, out.entry, error))
			{
				out.error = "Metal compilation failed: " + error;
			}
		}

		(out.error.empty() ? mtl::g_programs_ok : mtl::g_programs_failed)++;

		if (!out.error.empty())
		{
			rsx_log.error("Metal: %s program %u: %s", kind, id, out.error);

			if (!out.msl.empty())
			{
				rsx_log.notice("Metal: failing MSL:\n%s", out.msl);
			}
			else
			{
				rsx_log.notice("Metal: failing GLSL:\n%s", glsl);
			}
		}
	}
}

void MTLVertexProgram::Decompile(const RSXVertexProgram& prog)
{
	// The Vulkan decompiler has no device dependencies for vertex programs
	host.Decompile(prog);
	has_indexed_constants = host.has_indexed_constants;
	constant_ids = host.constant_ids;
}

void MTLVertexProgram::Compile()
{
	translate(host.shader.get_source(), ::glsl::program_domain::glsl_vertex_program, mtl::shader_stage::vertex, id, result);
}

void MTLFragmentProgram::Decompile(const RSXFragmentProgram& prog)
{
	// Same as VKFragmentProgram::Decompile, without querying a Vulkan device
	u32 size;
	std::string source;
	VKFragmentDecompilerThread decompiler(source, host.parr, prog, size, host);

	decompiler.device_props.has_native_half_support = false;
	decompiler.device_props.emulate_depth_compare = false;
	decompiler.device_props.has_low_precision_rounding = false;
	decompiler.Task();

	host.constant_offsets = std::move(decompiler.properties.constant_offsets);
	host.shader.create(::glsl::program_domain::glsl_fragment_program, source);
}

void MTLFragmentProgram::Compile()
{
	translate(host.shader.get_source(), ::glsl::program_domain::glsl_fragment_program, mtl::shader_stage::fragment, id, result);
}
