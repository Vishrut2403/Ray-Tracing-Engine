#pragma once

#include "render/framebuffer.h"
#include "core/camera.h"
#include "scenes/cornell_scene.h"
#include "photon/photon_map.h"

class PPMRenderer {
public:
	int    n_iterations;
	int    photons_per_iter;
	int    max_depth;  
	double initial_radius;
	double alpha;    

	PPMRenderer(int iterations    = 64,
				int photons       = 100000,
				int depth         = 10,
				double radius     = 15.0,
				double alpha_val  = 0.7)
		: n_iterations(iterations)
		, photons_per_iter(photons)
		, max_depth(depth)
		, initial_radius(radius)
		, alpha(alpha_val) {}

	void render(
		const Scene& scene,
		Framebuffer& fb,
		const camera& cam,
		const color& background
	);

};