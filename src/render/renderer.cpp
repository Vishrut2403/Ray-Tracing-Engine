#include "render/renderer.h"

#include <omp.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <mutex>

#include "render/integrator.h"
#include "render/tile.h"
#include "lights/env_light.h"

Renderer::Renderer(int spp, int depth, int tile_size)
	: samples_per_pixel(spp), max_depth(depth), tile_size(tile_size) {}

void Renderer::render(
	const Scene& scene,
	Framebuffer& fb,
	const camera& cam,
	const color& background,
	const std::atomic<bool>* cancel
) {
	auto world  = scene.world;
	auto lights = scene.lights;
	auto env    = scene.env;
	bool use_bdpt = scene.use_bdpt;

	const int W = fb.get_width();
	const int H = fb.get_height();

	auto tiles = generate_tiles(W, H, tile_size);

	// t == 1 deposits into arbitrary pixels, so it cannot use the tile buffers.
	// One bucket per pixel, which is what makes the merge order fixed.
	BDPTSplatBuffer splat;
	if (use_bdpt) splat.resize(W, H, W * H);

	float cx = W*0.5f, cy = H*0.5f;
	std::sort(tiles.begin(), tiles.end(),
		[cx,cy](const Tile& a, const Tile& b){
			float ax=(a.x0+a.x1)*0.5f, ay=(a.y0+a.y1)*0.5f;
			float bx=(b.x0+b.x1)*0.5f, by=(b.y0+b.y1)*0.5f;
			return (ax-cx)*(ax-cx)+(ay-cy)*(ay-cy)
				 < (bx-cx)*(bx-cx)+(by-cy)*(by-cy);
		});

	// One pass over the whole image per sample, rather than every sample of a
	// tile before moving on. The image then refines everywhere at once, which
	// is what a viewport needs, and it can be stopped between passes with a
	// complete estimate in hand. Each sample is keyed by (pixel, index), so
	// the order the passes are walked in does not change a single value.
	std::vector<color> accum((size_t)W * H);

	for (int s = 0; s < samples_per_pixel; ++s) {
		if (cancel && cancel->load()) break;

#pragma omp parallel for schedule(dynamic)
		for (size_t t = 0; t < tiles.size(); ++t) {
			if (cancel && cancel->load()) continue;
			const Tile& tile = tiles[t];

			for (int j = tile.y0; j < tile.y1; ++j) {
				for (int i = tile.x0; i < tile.x1; ++i) {
					sampler_begin_sample((uint32_t)(j*W + i), (uint32_t)s);
					real u = (i + random_double()) / (W - 1);
					real v = (j + random_double()) / (H - 1);
					ray r = cam.get_ray(u, v);

					color sample;
					if (use_bdpt)
						sample = bdpt_Li(r, cam, world, lights, max_depth,
										 splat, j*W + i);
					else
						sample = Li(r, background, world, lights, max_depth, env);

					accum[(size_t)j*W + i] += sample;
					sampler_end_sample();
				}
			}
		}

		if (cancel && cancel->load()) break;

		// Fold this pass's light-tracing splats in, in tile order.
		if (use_bdpt) splat.flush();

		// The running mean, so a render stopped early still shows an unbiased
		// estimate rather than a part-finished one.
		{
			std::lock_guard<std::mutex> lock(fb.mtx);
			for (int j = 0; j < H; ++j)
				for (int i = 0; i < W; ++i) {
					color c = accum[(size_t)j*W + i];
					c /= real(s + 1);
					fb.set(i, j, c);
				}

			// Same 1/n as the camera paths, applied to the light-tracing splats.
			if (use_bdpt) {
				real inv = 1.0 / real(s + 1);
				for (int j = 0; j < H; ++j)
					for (int i = 0; i < W; ++i)
						fb.set(i, j, fb.get(i, j) + splat.total(i, j) * inv);
			}
		}

		if (verbose)
			std::cerr << "\rPass: " << (s+1) << "/" << samples_per_pixel
					  << std::flush;
	}

	if (verbose) std::cerr << "\nDone.\n";
}
