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
// light vertex projects onto rather than the calling thread's.
//
// Contributions are parked in a bucket per originating pixel and folded into
// the running total between passes, in pixel order. Float addition is not
// associative, so summing them under an atomic left the result at the mercy of
// the thread schedule -- eight runs of `caustics` gave two distinct images.
// The bucket is keyed to the pixel that generated the light path, not to the
// thread or the tile: which thread picks up a tile is up to OpenMP, and the
// tiling itself moves with --tile, while the pixel order is fixed.
struct BDPTSplatBuffer {
	struct Splat { int pixel; real r, g, b; };

	int W = 0, H = 0;
	std::vector<real> accum;                  // W*H*3, folded between passes
	std::vector<std::vector<Splat>> pending;  // one bucket per tile

	void resize(int w, int h, int n_buckets) {
		W = w; H = h;
		accum.assign((size_t)w * h * 3, 0.0);
		pending.assign(n_buckets > 0 ? n_buckets : 1, {});
	}

	// A pixel's sample runs on one thread, so a bucket has a single writer
	// and needs no lock.
	void add(int bucket, int x, int y, const color& c) {
		if (x < 0 || x >= W || y < 0 || y >= H) return;
		pending[bucket].push_back({y * W + x, c.x(), c.y(), c.z()});
	}

	// Between passes, single-threaded.
	void flush() {
		for (auto& bucket : pending) {
			for (const Splat& s : bucket) {
				size_t i = (size_t)s.pixel * 3;
				accum[i+0] += s.r;
				accum[i+1] += s.g;
				accum[i+2] += s.b;
			}
			bucket.clear();
		}
	}

	color total(int x, int y) const {
		size_t i = ((size_t)y * W + x) * 3;
		return color(accum[i+0], accum[i+1], accum[i+2]);
	}
};

// Returns this pixel's contribution; t == 1 contributions go to `splat`.
color bdpt_Li(
	const ray& camera_ray,
	const camera& cam,
	const std::shared_ptr<hittable>& world,
	const std::shared_ptr<hittable_list>& lights,
	int max_depth,
	BDPTSplatBuffer& splat,
	// Which bucket this caller owns -- the index of the pixel being sampled.
	// Passed in so the integrator needs no notion of tiles or threads.
	int splat_bucket
);

color Li(
	const ray& r,
	const color& background,
	const std::shared_ptr<hittable>& world,
	const std::shared_ptr<hittable_list>& lights,
	int max_depth,
	const std::shared_ptr<env_light>& env = nullptr
);