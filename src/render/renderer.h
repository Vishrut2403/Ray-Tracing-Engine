#pragma once
#include <atomic>
#include "render/framebuffer.h"
#include "render/tile.h"
#include "core/camera.h"
#include "scenes/cornell_scene.h"

class Renderer {
public:
	Renderer(int spp, int depth, int tile_size);

	void render(
		const Scene& scene,
		Framebuffer& fb,
		const camera& cam,
		const color& background,
		// Checked between passes and between tiles. A cancelled render leaves
		// the mean of the passes that did finish.
		const std::atomic<bool>* cancel = nullptr
	);

	// Off when a viewport drives the render and restarts it on every nudge.
	bool verbose = true;

private:
	int samples_per_pixel;
	int max_depth;
	int tile_size;
};