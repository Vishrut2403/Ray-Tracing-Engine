#include "app/render_config.h"
#include "app/cli.h"
#include "scenes/scene_factory.h"
#include "core/camera_factory.h"
#include "render/renderer.h"
#include "render/ppm_renderer.h"
#include "render/framebuffer.h"
#include "io/image_writer.h"
#include "io/denoiser.h"
#include "viewer/preview_window.h"
#include "cuda/cuda_renderer.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <iostream>
#include <string>

static bool has_flag(int argc, char** argv, const char* flag) {
	for (int i = 1; i < argc; ++i)
		if (strcmp(argv[i], flag) == 0) return true;
	return false;
}

static bool get_flag_value(int argc, char** argv,
							const char* flag, const char* value) {
	for (int i = 1; i < argc-1; ++i)
		if (strcmp(argv[i], flag) == 0 && strcmp(argv[i+1], value) == 0)
			return true;
	return false;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
	return s.size() >= suffix.size() &&
		   s.compare(s.size()-suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char** argv)
{
	RenderConfig config  = parse_cli(argc, argv);
	bool use_gpu         = get_flag_value(argc, argv, "--device", "gpu");
	bool no_preview      = has_flag(argc, argv, "--no-preview");
	// An explicit --integrator wins over the scene's own default.
	bool pick_ppm        = get_flag_value(argc, argv, "--integrator", "ppm");
	bool pick_other      = get_flag_value(argc, argv, "--integrator", "pt")
						   || get_flag_value(argc, argv, "--integrator", "bdpt")
						   || get_flag_value(argc, argv, "--integrator", "restir");
	bool use_ppm         = pick_ppm
						   || (config.feature == "ppm" && !pick_other);
	bool use_denoise     = has_flag(argc, argv, "--denoise");
	bool want_restir     = get_flag_value(argc, argv, "--integrator", "restir");

	if (config.feature == "furnace")
		apply_furnace_preset(config);

	// The GPU covers only a subset of scenes. An unknown name used to fall
	// through to a Cornell box and look like it worked; say so and use the CPU.
	if (use_gpu && use_ppm) {
		std::cerr << "[note] progressive photon mapping is CPU-only; "
					 "ignoring --device gpu\n";
		use_gpu = false;
	} else if (use_gpu && !cuda_supports_scene(config.feature)) {
		std::cerr << "[note] scene '" << config.feature
				  << "' has no GPU implementation; rendering on the CPU instead\n";
		use_gpu = false;
	}

	GpuIntegrator gpu_integrator = want_restir ? GpuIntegrator::RESTIR
											   : GpuIntegrator::PATH_TRACER;

	Scene  scene = SceneFactory::build(config.feature);
	// Scenes carry a default integrator; --integrator overrides it so the same
	// scene can be rendered by independent estimators and compared.
	if      (get_flag_value(argc, argv, "--integrator", "bdpt")) scene.use_bdpt = true;
	else if (get_flag_value(argc, argv, "--integrator", "pt"))   scene.use_bdpt = false;
	camera cam   = CameraFactory::build(config);
	Framebuffer fb(config.width, config.height);

	if (use_ppm) {
		PPMRenderer ppm(
			config.samples,
			1000000,
			config.max_depth,
			50.0,
			0.7
		);
		ppm.render(scene, fb, cam, config.background);

	} else if (use_gpu) {
		if (no_preview) {
			cuda_render(scene, fb, cam, config.background,
						config.samples, config.max_depth,
						nullptr, config.feature, gpu_integrator);
		} else {
			PreviewWindow preview(config.width, config.height);
			std::thread render_thread([&]() {
				cuda_render(scene, fb, cam, config.background,
							config.samples, config.max_depth,
							&preview, config.feature, gpu_integrator);
			});
			while (!preview.should_close()) {
				preview.poll_events();
				std::lock_guard<std::mutex> lock(fb.mtx);
				preview.update(fb.raw_data());
			}
			render_thread.join();
		}
	} else {
		Renderer renderer(config.samples, config.max_depth, config.tile_size);
		if (no_preview) {
			renderer.render(scene, fb, cam, config.background);
		} else {
			PreviewWindow preview(config.width, config.height);
			std::thread render_thread([&]() {
				renderer.render(scene, fb, cam, config.background);
			});
			while (!preview.should_close()) {
				preview.poll_events();
				std::lock_guard<std::mutex> lock(fb.mtx);
				preview.update(fb.raw_data());
			}
			render_thread.join();
		}
	}

	if (use_denoise) {
		OIDNDenoiser::denoise(fb);
	}

	if (ends_with(config.output_path, ".exr"))
		ImageWriter::write_exr(config.output_path, fb);
	else
		ImageWriter::write_ppm(config.output_path, fb);

	return 0;
}