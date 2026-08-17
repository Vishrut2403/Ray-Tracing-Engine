// Cross-backend render checks. The analytic suites cover materials and the GPU
// medium primitives but never run an integrator, so a transport bug on one
// backend alone slips past them. These render both backends and compare.

#include "tests/test_util.h"
#include "app/render_config.h"
#include "scenes/scene_factory.h"
#include "core/camera_factory.h"
#include "render/renderer.h"
#include "render/ppm_renderer.h"
#include "render/framebuffer.h"
#include "cuda/cuda_renderer.h"
#include "io/denoiser.h"

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
				false, scene_name, GpuIntegrator::PATH_TRACER);

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

// Path tracing vs BDPT on the same scene. These are independent estimators --
// different path construction, different MIS weights -- so agreement is real
// evidence of correctness, unlike CPU/GPU agreement on a shared algorithm.
//
// Not every scene qualifies: BDPT has no participating-medium support, and on
// caustics PT converges too slowly through specular chains to compare at a
// sane sample count (measured ~8% low, concentrated in the caustic itself).
void compare_integrators(const std::string& scene_name, int spp, int depth,
						  double tol) {
	// Lower resolution rather than fewer samples: the comparison is on image
	// means, whose noise depends on the total sample count.
	const int W = 64, H = 64;

	RenderConfig cfg;
	cfg.feature = scene_name;
	cfg.width = W; cfg.height = H;
	cfg.samples = spp; cfg.max_depth = depth;

	camera cam = CameraFactory::build(cfg);
	Framebuffer fb_pt(W, H), fb_bd(W, H);
	Renderer renderer(spp, depth, 32);

	Scene s_pt = SceneFactory::build(scene_name);
	s_pt.use_bdpt = false;
	renderer.render(s_pt, fb_pt, cam, cfg.background);

	Scene s_bd = SceneFactory::build(scene_name);
	s_bd.use_bdpt = true;
	renderer.render(s_bd, fb_bd, cam, cfg.background);

	Stats a = measure(fb_pt, W, H), b = measure(fb_bd, W, H);
	auto rel = [](double x, double y) {
		double m = std::max(std::abs(x), std::abs(y));
		return m > 1e-9 ? std::abs(x - y) / m : 0.0;
	};

	check(rel(a.mean, b.mean) < tol, scene_name + ": PT/BDPT mean agree",
		  a.mean, b.mean, tol);
	check(rel(a.lower_half, b.lower_half) < tol,
		  scene_name + ": PT/BDPT top half agree",
		  a.lower_half, b.lower_half, tol);
	check(rel(a.upper_half, b.upper_half) < tol,
		  scene_name + ": PT/BDPT bottom half agree",
		  a.upper_half, b.upper_half, tol);
}

// Closed enclosure with emitting, reflecting walls: L = Le/(1-rho) exactly.
// The open furnace scene is a convex sphere and validates a single bounce; this
// one pins the whole multi-bounce chain against a closed form.
void test_closed_furnace() {
	const int W = 48, H = 48, spp = 400, depth = 30;

	RenderConfig cfg;
	cfg.feature = "closed_furnace";
	cfg.width = W; cfg.height = H;
	cfg.samples = spp; cfg.max_depth = depth;

	Scene  scene = SceneFactory::build(cfg.feature);
	camera cam   = CameraFactory::build(cfg);
	Framebuffer fb(W, H);
	Renderer(spp, depth, 32).render(scene, fb, cam, cfg.background);

	Stats st = measure(fb, W, H);
	check(std::abs(st.mean - 1.0) < 0.01,
		  "closed furnace: L = Le/(1-rho) = 1", st.mean, 1.0, 0.01);
	check(std::abs(st.lower_half - 1.0) < 0.015,
		  "closed furnace: top half = 1", st.lower_half, 1.0, 0.015);
	check(std::abs(st.upper_half - 1.0) < 0.015,
		  "closed furnace: bottom half = 1", st.upper_half, 1.0, 0.015);
}

// PPM normalises by every photon ever emitted, so its result must not depend on
// the iteration count -- more passes mean less noise, not more light. It is
// also a third estimator, independent of both PT and BDPT.
void test_ppm() {
	const int W = 48, H = 48, depth = 10;

	RenderConfig cfg;
	cfg.feature = "cornell";
	cfg.width = W; cfg.height = H; cfg.max_depth = depth;
	camera cam = CameraFactory::build(cfg);

	double means[2];
	const int iters[2] = { 6, 18 };
	for (int k = 0; k < 2; ++k) {
		Scene scene = SceneFactory::build(cfg.feature);
		Framebuffer fb(W, H);
		PPMRenderer(iters[k], 150000, depth, 45.0, 0.7)
			.render(scene, fb, cam, cfg.background);
		means[k] = measure(fb, W, H).mean;
	}
	double rel = std::abs(means[0] - means[1])
			   / std::max(means[0], means[1]);
	check(rel < 0.06, "ppm: invariant to iteration count",
		  means[0], means[1], 0.06);

	Scene s_bd = SceneFactory::build(cfg.feature);
	s_bd.use_bdpt = true;
	Framebuffer fb_bd(W, H);
	Renderer(400, depth, 32).render(s_bd, fb_bd, cam, cfg.background);
	double mb = measure(fb_bd, W, H).mean;

	double r2 = std::abs(means[1] - mb) / std::max(means[1], mb);
	check(r2 < 0.08, "ppm: agrees with BDPT", means[1], mb, 0.08);
}

// ReSTIR is biased by design, so this is a guard rather than a correctness
// check: it catches black frames, NaNs and gross regressions. The band is
// tight enough to notice a repeat of the reservoir-weight clamp that used to
// cost ~20% of the indirect light on diffuse scenes.
void check_restir(const std::string& scene_name, int spp, int depth,
				   double lo, double hi) {
	const int W = 64, H = 64;

	RenderConfig cfg;
	cfg.feature = scene_name;
	cfg.width = W; cfg.height = H;
	cfg.samples = spp; cfg.max_depth = depth;

	Scene  scene = SceneFactory::build(scene_name);
	camera cam   = CameraFactory::build(cfg);
	Framebuffer fb_pt(W, H), fb_rs(W, H);

	Renderer(spp, depth, 32).render(scene, fb_pt, cam, cfg.background);
	cuda_render(scene, fb_rs, cam, cfg.background, spp, depth,
				false, scene_name, GpuIntegrator::RESTIR);

	Stats p = measure(fb_pt, W, H), r = measure(fb_rs, W, H);
	double ratio = p.mean > 1e-9 ? r.mean / p.mean : 0.0;

	check(r.mean > 0.01, scene_name + ": ReSTIR not black", r.mean, 1.0, 0.0);
	check(ratio > lo && ratio < hi,
		  scene_name + ": ReSTIR within band of PT", ratio, 1.0, hi - 1.0);
}

// Nothing else here renders through --denoise, so a denoiser that shifted
// energy would be invisible. The real question is whether it moves the image
// closer to ground truth, not just whether it looks smoother -- a filter can
// blur beautifully and still bias the result. So compare both against a
// converged reference.
void test_denoiser() {
	const int W = 64, H = 64, depth = 8;

	RenderConfig cfg;
	cfg.feature = "cornell";
	cfg.width = W; cfg.height = H; cfg.max_depth = depth;

	Scene  scene = SceneFactory::build(cfg.feature);
	camera cam   = CameraFactory::build(cfg);

	Framebuffer ref(W, H), noisy(W, H);
	Renderer(1500, depth, 32).render(scene, ref,   cam, cfg.background);
	Renderer(16,   depth, 32).render(scene, noisy, cam, cfg.background);

	// Measured in display space, matching image_writer. In linear HDR the
	// emitter dominates: 0.8% of pixels at radiance ~30 swamp the squared
	// error, and blurring their edges hides a large gain everywhere else.
	auto encode = [](double x) {
		const double a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
		double t = (x*(a*x+b)) / (x*(c*x+d)+e);
		t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
		return std::pow(t, 1.0/2.2);
	};
	auto rmse_vs_ref = [&](const Framebuffer& f) {
		double acc = 0.0;
		for (int j = 0; j < H; ++j)
			for (int i = 0; i < W; ++i) {
				color a = f.get(i,j), b = ref.get(i,j);
				for (int k = 0; k < 3; ++k) {
					double diff = encode(a[k]) - encode(b[k]);
					acc += diff*diff;
				}
			}
		return std::sqrt(acc / (W*H*3));
	};

	double mean_before = measure(noisy, W, H).mean;
	double rmse_before = rmse_vs_ref(noisy);

	OIDNDenoiser::denoise(noisy);

	double mean_after = measure(noisy, W, H).mean;
	double rmse_after = rmse_vs_ref(noisy);

	bool finite = true, nonneg = true;
	for (int j = 0; j < H; ++j)
		for (int i = 0; i < W; ++i) {
			color c = noisy.get(i,j);
			for (int k = 0; k < 3; ++k) {
				if (!std::isfinite(c[k])) finite = false;
				if (c[k] < -1e-6)         nonneg = false;
			}
		}

	check(finite, "denoiser: output is finite", 1.0, 1.0, 0.0);
	check(nonneg, "denoiser: output is non-negative", 1.0, 1.0, 0.0);

	double rel = std::abs(mean_after - mean_before)
			   / std::max(mean_before, 1e-9);
	check(rel < 0.05, "denoiser: preserves mean energy",
		  mean_after, mean_before, 0.05);
	check(rmse_after < 0.8 * rmse_before,
		  "denoiser: moves image closer to reference",
		  rmse_after, rmse_before, 0.0);
}

} // namespace

void run_render_tests() {
	std::printf("Cross-backend render checks\n");
	// volume is the one that exercises the medium transport path on both sides.
	compare_scene("volume",  192, 12, 0.04);
	compare_scene("cornell", 128, 10, 0.04);

	compare_integrators("cornell", 320, 10, 0.03);
	compare_integrators("glass",   320, 10, 0.03);

	// caustics defaults to BDPT on both backends, so this is the only check
	// that exercises the GPU BDPT kernels.
	compare_scene("caustics", 256, 12, 0.05);

	test_closed_furnace();
	test_ppm();

	check_restir("cornell", 300, 10, 0.88, 1.15);
	check_restir("glass",   300, 10, 0.88, 1.15);
	check_restir("ggx",     300, 10, 0.88, 1.15);

	test_denoiser();
}
