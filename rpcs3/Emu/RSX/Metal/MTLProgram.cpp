#include "stdafx.h"
#include "MTLProgram.h"

#include "Emu/RSX/Program/SPIRVCommon.h"
#include "Emu/system_config.h"
#include "Emu/Cell/timers.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <thread>

namespace mtl
{
	atomic_t<u32> g_programs_ok{0};
	atomic_t<u32> g_programs_failed{0};

	const shader_function* translated_program::get_function() const
	{
		if (!ok())
		{
			return nullptr;
		}

		std::lock_guard lock(compiled->lock);

		if (!compiled->attempted)
		{
			compiled->attempted = true;

			const u64 start = get_system_time();
			const char* const kind = is_vertex ? "Vertex" : "Fragment";
			const auto stage = is_vertex ? shader_stage::vertex : shader_stage::fragment;
			const auto domain = is_vertex ? ::glsl::program_domain::glsl_vertex_program : ::glsl::program_domain::glsl_fragment_program;

			std::vector<u32> spirv;
			std::string source = glsl; // compile_glsl_to_spv takes a mutable string

			if (!spirv::compile_glsl_to_spv(spirv, source, domain, ::glsl::glsl_rules_vulkan))
			{
				error = "GLSL -> SPIR-V failed";
			}
			else if (spirv_to_msl(spirv, stage, msl, entry, resources, error))
			{
				if (g_cfg.video.log_programs)
				{
					fs::write_file(fs::get_cache_dir() + fmt::format("shaderlog/Metal%sProgram%u.metal", kind, id), fs::rewrite, msl);
				}

				if (!compiled->function.create(msl, entry, compiled->error))
				{
					error = "Metal compilation failed: " + compiled->error;
				}
			}

			if (!error.empty())
			{
				g_programs_failed++;
				rsx_log.error("Metal: %s program %u: %s", kind, id, error);
				rsx_log.notice("Metal: failing %s:\n%s", msl.empty() ? "GLSL" : "MSL", msl.empty() ? glsl : msl);
			}
			else
			{
				g_programs_ok++;
			}

			if (const u64 elapsed = get_system_time() - start; elapsed > 20'000)
			{
				rsx_log.notice("Metal: %s program %u translated and compiled in %llu ms", kind, id, elapsed / 1000);
			}
		}

		return compiled->function.valid() ? &compiled->function : nullptr;
	}

	namespace
	{
		struct pipeline_compiler
		{
			std::mutex lock;
			std::condition_variable cv;
			std::deque<std::function<void()>> jobs;
			std::vector<std::thread> workers;
			u32 busy = 0;
			bool stopping = false;

			void start()
			{
				const u32 count = std::clamp(std::thread::hardware_concurrency() / 3, 1u, 3u);
				for (u32 i = 0; i < count; ++i)
				{
					workers.emplace_back([this]()
					{
						for (;;)
						{
							std::function<void()> job;
							{
								std::unique_lock guard(lock);
								cv.wait(guard, [this]() { return stopping || !jobs.empty(); });

								if (stopping)
								{
									return;
								}

								job = std::move(jobs.front());
								jobs.pop_front();
								busy++;
							}

							job();

							std::lock_guard guard(lock);
							busy--;
						}
					});
				}
			}

			void stop()
			{
				{
					std::lock_guard guard(lock);
					stopping = true;
					jobs.clear();
				}

				cv.notify_all();

				for (auto& worker : workers)
				{
					worker.join();
				}

				workers.clear();
				stopping = false;
			}
		};

		std::mutex s_compiler_lock;
		std::unique_ptr<pipeline_compiler> s_compiler;
	}

	void post_pipeline_job(std::function<void()> job)
	{
		std::lock_guard lock(s_compiler_lock);

		if (!s_compiler)
		{
			s_compiler = std::make_unique<pipeline_compiler>();
			s_compiler->start();
		}

		{
			std::lock_guard guard(s_compiler->lock);
			s_compiler->jobs.emplace_back(std::move(job));
		}

		s_compiler->cv.notify_one();
	}

	void shutdown_pipeline_compiler()
	{
		std::lock_guard lock(s_compiler_lock);

		if (s_compiler)
		{
			s_compiler->stop();
			s_compiler.reset();
		}
	}

	std::unique_ptr<pipeline> create_pipeline(const translated_program& vp, const translated_program& fp, const pipeline_props& props)
	{
		auto result = std::make_unique<pipeline>();

		const shader_function* vs = vp.get_function();
		const shader_function* fs = fp.get_function();

		if (!vs || !fs)
		{
			return result;
		}

		std::string error;
		if (!result->state.create(*vs, *fs, props, error))
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
	void translate(const std::string& glsl, ::glsl::program_domain, mtl::shader_stage stage, u32 id, mtl::translated_program& out)
	{
		out.id = id;
		out.is_vertex = stage == mtl::shader_stage::vertex;
		out.glsl = glsl;

		if (g_cfg.video.log_programs)
		{
			fs::write_file(fs::get_cache_dir() + fmt::format("shaderlog/Metal%sProgram%u.glsl", out.is_vertex ? "Vertex" : "Fragment", id), fs::rewrite, glsl);
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
