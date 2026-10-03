#include "stdafx.h"
#include "MTLShaderCompiler.h"

#ifdef HAVE_SPIRV_CROSS
#include <spirv_cross/spirv_cross_c.h>
#endif

namespace mtl
{
	bool shader_translation_available()
	{
#ifdef HAVE_SPIRV_CROSS
		return true;
#else
		return false;
#endif
	}

#ifdef HAVE_SPIRV_CROSS
	namespace
	{
		// RAII for the SPIRV-Cross C API context (frees every object created from it)
		struct spvc_context_holder
		{
			spvc_context ctx = nullptr;

			spvc_context_holder()
			{
				spvc_context_create(&ctx);
			}

			~spvc_context_holder()
			{
				if (ctx) spvc_context_destroy(ctx);
			}
		};
	}

	bool spirv_to_msl(const std::vector<u32>& spirv, shader_stage stage, std::string& msl, std::string& entry, std::string& error)
	{
		spvc_context_holder holder;
		spvc_context ctx = holder.ctx;

		if (!ctx)
		{
			error = "spvc_context_create failed";
			return false;
		}

		const auto fail = [&](const char* what)
		{
			error = fmt::format("%s: %s", what, spvc_context_get_last_error_string(ctx));
			return false;
		};

		spvc_parsed_ir ir = nullptr;
		if (spvc_context_parse_spirv(ctx, spirv.data(), spirv.size(), &ir) != SPVC_SUCCESS)
		{
			return fail("SPIR-V parse");
		}

		spvc_compiler compiler = nullptr;
		if (spvc_context_create_compiler(ctx, SPVC_BACKEND_MSL, ir, SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, &compiler) != SPVC_SUCCESS)
		{
			return fail("MSL compiler creation");
		}

		spvc_compiler_options options = nullptr;
		if (spvc_compiler_create_compiler_options(compiler, &options) != SPVC_SUCCESS)
		{
			return fail("MSL options");
		}

		// Apple Silicon Macs: MSL 2.4 covers texture buffers and framebuffer fetch (subpass inputs)
		spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_VERSION, SPVC_MAKE_MSL_VERSION(2, 4, 0));
		spvc_compiler_options_set_uint(options, SPVC_COMPILER_OPTION_MSL_PLATFORM, SPVC_MSL_PLATFORM_MACOS);
		spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_FRAMEBUFFER_FETCH_SUBPASS, static_cast<spvc_bool>(1));
		spvc_compiler_options_set_bool(options, SPVC_COMPILER_OPTION_MSL_TEXTURE_BUFFER_NATIVE, static_cast<spvc_bool>(1));

		if (spvc_compiler_install_compiler_options(compiler, options) != SPVC_SUCCESS)
		{
			return fail("MSL options install");
		}

		const char* source = nullptr;
		if (spvc_compiler_compile(compiler, &source) != SPVC_SUCCESS || !source)
		{
			return fail("MSL generation");
		}

		msl = source;

		const SpvExecutionModel model = stage == shader_stage::vertex ? SpvExecutionModelVertex : SpvExecutionModelFragment;
		const char* cleansed = spvc_compiler_get_cleansed_entry_point_name(compiler, "main", model);
		entry = cleansed ? cleansed : "main0";
		return true;
	}
#else
	bool spirv_to_msl(const std::vector<u32>&, shader_stage, std::string&, std::string&, std::string& error)
	{
		error = "RPCS3 was built without SPIRV-Cross (brew install spirv-cross)";
		return false;
	}
#endif
}
