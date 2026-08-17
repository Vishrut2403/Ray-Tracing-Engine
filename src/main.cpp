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
#include <functional>
#include <iostream>
#include <string>
#include <vector>

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

// The lock is held for the snapshot copy only: drawing under it stalls every
// render thread until the compositor accepts the frame, which on Wayland never
// happens while the window is hidden.
static void render_with_preview(Framebuffer& fb, int width, int height,
								const std::function<void()>& render) {
	PreviewWindow preview(width, height);
	std::atomic<bool> done{false};
	std::thread render_thread([&]() { render(); done = true; });

	std::vector<float> snapshot((size_t)width * height * 3, 0.0f);
	bool have_final = false;
	while (!preview.should_close()) {
		if (!have_final) {
			// Sampled before the copy, so the copy that follows a true reading
			// is the final image.
			bool finished = done;
			{
				std::lock_guard<std::mutex> lock(fb.mtx);
				std::memcpy(snapshot.data(), fb.raw_data(),
							snapshot.size() * sizeof(float));
			}
			have_final = finished;
		}
		preview.update(snapshot.data());
		preview.wait_events(1.0/30.0);
	}
	render_thread.join();
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
		auto go = [&](bool stage_frames) {
			cuda_render(scene, fb, cam, config.background,
						config.samples, config.max_depth,
						stage_frames, config.feature, gpu_integrator);
		};
		if (no_preview) go(false);
		else render_with_preview(fb, config.width, config.height,
								 [&]() { go(true); });
	} else {
		Renderer renderer(config.samples, config.max_depth, config.tile_size);
		auto go = [&]() { renderer.render(scene, fb, cam, config.background); };
		if (no_preview) go();
		else render_with_preview(fb, config.width, config.height, go);
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