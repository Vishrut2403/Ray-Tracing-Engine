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
	const color& background
) {
	auto world  = scene.world;
	auto lights = scene.lights;
	auto env    = scene.env;
	bool use_bdpt = scene.use_bdpt;

	const int W = fb.get_width();
	const int H = fb.get_height();
	int tiles_done = 0;

	// t == 1 deposits into arbitrary pixels, so it cannot use the tile buffers.
	BDPTSplatBuffer splat;
	if (use_bdpt) splat.resize(W, H);

	auto tiles = generate_tiles(W, H, tile_size);

	float cx = W*0.5f, cy = H*0.5f;
	std::sort(tiles.begin(), tiles.end(),
		[cx,cy](const Tile& a, const Tile& b){
			float ax=(a.x0+a.x1)*0.5f, ay=(a.y0+a.y1)*0.5f;
			float bx=(b.x0+b.x1)*0.5f, by=(b.y0+b.y1)*0.5f;
			return (ax-cx)*(ax-cx)+(ay-cy)*(ay-cy)
				 < (bx-cx)*(bx-cx)+(by-cy)*(by-cy);
		});

#pragma omp parallel for schedule(dynamic)
	for (size_t t = 0; t < tiles.size(); ++t) {
		const Tile& tile = tiles[t];
		int tw = tile.x1 - tile.x0, th = tile.y1 - tile.y0;
		std::vector<color> buf(tw * th);

		for (int j = tile.y0; j < tile.y1; ++j) {
			for (int i = tile.x0; i < tile.x1; ++i) {
				color pixel(0,0,0);
				for (int s = 0; s < samples_per_pixel; ++s) {
					sampler_begin_sample((uint32_t)(j*W + i), (uint32_t)s);
					real u = (i + random_double()) / (W - 1);
					real v = (j + random_double()) / (H - 1);
					ray r = cam.get_ray(u, v);

					color sample;
					if (use_bdpt)
						sample = bdpt_Li(r, cam, world, lights, max_depth, splat);
					else
						sample = Li(r, background, world, lights, max_depth, env);

					pixel += sample;
				}
				sampler_end_sample();
				pixel /= real(samples_per_pixel);
				buf[(j-tile.y0)*tw + (i-tile.x0)] = pixel;
			}
		}

		{
			std::lock_guard<std::mutex> lock(fb.mtx);
			for (int j = tile.y0; j < tile.y1; ++j)
				for (int i = tile.x0; i < tile.x1; ++i)
					fb.set(i, j, buf[(j-tile.y0)*tw+(i-tile.x0)]);
		}

#pragma omp critical
		{
			++tiles_done;
			std::cerr << "\rTiles: " << tiles_done << "/" << tiles.size() << std::flush;
		}
	}

	// Merge the light-tracing splats; same 1/spp as the camera paths.
	if (use_bdpt) {
		std::lock_guard<std::mutex> lock(fb.mtx);
		real inv_spp = 1.0 / real(samples_per_pixel);
		for (int j = 0; j < H; ++j) {
			for (int i = 0; i < W; ++i) {
				size_t k = ((size_t)j*W + i)*3;
				fb.set(i, j, fb.get(i, j)
							 + color(splat.data[k+0], splat.data[k+1],
									 splat.data[k+2]) * inv_spp);
			}
		}
	}

	std::cerr << "\nDone.\n";
}