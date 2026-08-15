// Cross-backend render checks. The analytic suites cover materials and the GPU
// medium primitives but never run an integrator, so a transport bug on one
// backend alone slips past them. These render both backends and compare.

#include "tests/test_util.h"
#include "app/render_config.h"
#include "scenes/scene_factory.h"
#include "core/camera_factory.h"
#include "render/renderer.h"
#include "render/framebuffer.h"
#include "cuda/cuda_renderer.h"

#include <cstdio>
#include <cmath>
#include <string>

namespace {

struct Stats { double mean; double lower_half; double upper_half; };

double lum(const color& c) {
	return 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
}

// Split the frame so a localised difference cannot average away against the
// rest of the image.
Stats measure(const Framebuffer& fb, int W, int H) {
	double all = 0.0, lo = 0.0, hi = 0.0;
	for (int j = 0; j < H; ++j)
		for (int i = 0; i < W; ++i) {
			double v = lum(fb.get(i, j));
			if (!std::isfinite(v)) v = 0.0;
			all += v;
			if (j < H/2) lo += v; else hi += v;
		}
	double n = (double)(W*H), h = n/2.0;
	return { all/n, lo/h, hi/h };
}

void compare_scene(const std::string& scene_name, int spp, int depth,
					double tol) {
	const int W = 96, H = 96;

	RenderConfig cfg;
	cfg.feature = scene_name;
	cfg.width = W; cfg.height = H;
	cfg.samples = spp; cfg.max_depth = depth;

	Scene  scene = SceneFactory::build(scene_name);
	camera cam   = CameraFactory::build(cfg);

	Framebuffer fb_cpu(W, H), fb_gpu(W, H);

	Renderer renderer(spp, depth, 32);
	renderer.render(scene, fb_cpu, cam, cfg.background);
	cuda_render(scene, fb_gpu, cam, cfg.background, spp, depth,
				nullptr, scene_name, GpuIntegrator::PATH_TRACER);

	Stats c = measure(fb_cpu, W, H), g = measure(fb_gpu, W, H);

	auto rel = [](double a, double b) {
		double m = std::max(std::abs(a), std::abs(b));
		return m > 1e-9 ? std::abs(a - b) / m : 0.0;
	};

	check(rel(c.mean, g.mean) < tol, scene_name + ": CPU/GPU mean agree",
		  c.mean, g.mean, tol);
	check(rel(c.lower_half, g.lower_half) < tol,
		  scene_name + ": CPU/GPU top half agree", c.lower_half, g.lower_half, tol);
	check(rel(c.upper_half, g.upper_half) < tol,
		  scene_name + ": CPU/GPU bottom half agree",
		  c.upper_half, g.upper_half, tol);
}

} // namespace

void run_render_tests() {
	std::printf("Cross-backend render checks\n");
	// volume is the one that exercises the medium transport path on both sides.
	compare_scene("volume",  192, 12, 0.04);
	compare_scene("cornell", 128, 10, 0.04);
}
