#pragma once

#include <memory>
#include <vector>
#include "core/rtweekend.h"
#include "core/ray.h"
#include "core/camera.h"
#include "hittables/hittable.h"
#include "hittables/hittable_list.h"
#include "lights/env_light.h"

class Framebuffer;

// Accumulator for BDPT's t == 1 strategy, which deposits into whatever pixel a
// light vertex projects onto rather than the calling thread's. Atomic because
// the tile loop is parallel.
struct BDPTSplatBuffer {
	int W = 0, H = 0;
	std::vector<real> data;

	void resize(int w, int h) { W = w; H = h; data.assign((size_t)w*h*3, 0.0); }
	void add(int x, int y, const color& c) {
		if (x < 0 || x >= W || y < 0 || y >= H) return;
		size_t i = ((size_t)y*W + x)*3;
		#pragma omp atomic
		data[i+0] += c.x();
		#pragma omp atomic
		data[i+1] += c.y();
		#pragma omp atomic
		data[i+2] += c.z();
	}
};

// Returns this pixel's contribution; t == 1 contributions go to `splat`.
color bdpt_Li(
	const ray& camera_ray,
	const camera& cam,
	const std::shared_ptr<hittable>& world,
	const std::shared_ptr<hittable_list>& lights,
	int max_depth,
	BDPTSplatBuffer& splat
);

color Li(
	const ray& r,
	const color& background,
	const std::shared_ptr<hittable>& world,
	const std::shared_ptr<hittable_list>& lights,
	int max_depth,
	const std::shared_ptr<env_light>& env = nullptr
);