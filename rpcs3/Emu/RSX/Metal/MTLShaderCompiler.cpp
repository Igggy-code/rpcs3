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

	bool spirv_to_msl(const std::vector<u32>& spirv, shader_stage stage, std::string& msl, std::string& entry,
		std::vector<shader_resource>& resources, std::string& error)
	{
		resources.clear();

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

		const SpvExecutionModel model = stage == shader_stage::vertex ? SpvExecutionModelVertex : SpvExecutionModelFragment;

		// Assign explicit Metal slots: buffers and textures keep their Vulkan binding number (bindings are
		// unique per stage in the RSX decompilers), samplers are numbered in declaration order.
		spvc_resources shader_resources = nullptr;
		if (spvc_compiler_create_shader_resources(compiler, &shader_resources) != SPVC_SUCCESS)
		{
			return fail("Resource reflection");
		}

		const auto resource_name = [&](const spvc_reflected_resource& res) -> std::string
		{
			if (res.name && res.name[0])
			{
				return res.name;
			}

			// Anonymous blocks are reflected by their type name
			const char* type_name = spvc_compiler_get_name(compiler, res.base_type_id);
			return type_name ? type_name : "";
		};

		const auto add_binding = [&](u32 desc_set, u32 binding, u32 msl_buffer, u32 msl_texture, u32 msl_sampler) -> bool
		{
			spvc_msl_resource_binding_2 b;
			spvc_msl_resource_binding_init_2(&b);
			b.stage = model;
			b.desc_set = desc_set;
			b.binding = binding;
			b.count = 1;
			b.msl_buffer = msl_buffer;
			b.msl_texture = msl_texture;
			b.msl_sampler = msl_sampler;
			return spvc_compiler_msl_add_resource_binding_2(compiler, &b) == SPVC_SUCCESS;
		};

		u32 next_sampler = 0;

		const spvc_resource_type types[] =
		{
			SPVC_RESOURCE_TYPE_UNIFORM_BUFFER,
			SPVC_RESOURCE_TYPE_STORAGE_BUFFER,
			SPVC_RESOURCE_TYPE_PUSH_CONSTANT,
			SPVC_RESOURCE_TYPE_SAMPLED_IMAGE,
		};

		for (const spvc_resource_type type : types)
		{
			const spvc_reflected_resource* list = nullptr;
			size_t count = 0;

			if (spvc_resources_get_resource_list_for_type(shader_resources, type, &list, &count) != SPVC_SUCCESS)
			{
				return fail("Resource list");
			}

			for (size_t i = 0; i < count; ++i)
			{
				const spvc_reflected_resource& res = list[i];

				shader_resource out{};
				out.name = resource_name(res);

				if (type == SPVC_RESOURCE_TYPE_PUSH_CONSTANT)
				{
					out.kind = resource_kind::push_constant;
					out.msl_index = push_constant_buffer_index;

					if (!add_binding(SPVC_MSL_PUSH_CONSTANT_DESC_SET, SPVC_MSL_PUSH_CONSTANT_BINDING, out.msl_index, 0, 0))
					{
						return fail("Push constant binding");
					}

					resources.push_back(std::move(out));
					continue;
				}

				const u32 desc_set = spvc_compiler_get_decoration(compiler, res.id, SpvDecorationDescriptorSet);
				out.binding = spvc_compiler_get_decoration(compiler, res.id, SpvDecorationBinding);
				out.msl_index = out.binding;

				if (type == SPVC_RESOURCE_TYPE_SAMPLED_IMAGE)
				{
					const spvc_type image_type = spvc_compiler_get_type_handle(compiler, res.type_id);
					const SpvDim dim = spvc_type_get_image_dimension(image_type);

					if (dim == SpvDimBuffer)
					{
						out.kind = resource_kind::texel_buffer;
					}
					else
					{
						out.kind = resource_kind::texture;
						out.msl_sampler = next_sampler++;
						out.depth = !!spvc_type_get_image_is_depth(image_type);
						out.multisampled = !!spvc_type_get_image_multisampled(image_type);

						switch (dim)
						{
						case SpvDim1D: out.dimension = texture_dimension::dim_1d; break;
						case SpvDim3D: out.dimension = texture_dimension::dim_3d; break;
						case SpvDimCube: out.dimension = texture_dimension::dim_cube; break;
						default: out.dimension = texture_dimension::dim_2d; break;
						}
					}

					if (!add_binding(desc_set, out.binding, 0, out.msl_index, out.msl_sampler))
					{
						return fail("Texture binding");
					}
				}
				else
				{
					out.kind = resource_kind::buffer;

					if (!add_binding(desc_set, out.binding, out.msl_index, 0, 0))
					{
						return fail("Buffer binding");
					}
				}

				if (out.msl_index >= push_constant_buffer_index || out.msl_sampler >= 16)
				{
					error = fmt::format("Resource '%s' (binding %u) does not fit the Metal argument table", out.name, out.binding);
					return false;
				}

				resources.push_back(std::move(out));
			}
		}

		const char* source = nullptr;
		if (spvc_compiler_compile(compiler, &source) != SPVC_SUCCESS || !source)
		{
			return fail("MSL generation");
		}

		msl = source;

		const char* cleansed = spvc_compiler_get_cleansed_entry_point_name(compiler, "main", model);
		entry = cleansed ? cleansed : "main0";
		return true;
	}
#else
	bool spirv_to_msl(const std::vector<u32>&, shader_stage, std::string&, std::string&, std::vector<shader_resource>&, std::string& error)
	{
		error = "RPCS3 was built without SPIRV-Cross (brew install spirv-cross)";
		return false;
	}
#endif
}
