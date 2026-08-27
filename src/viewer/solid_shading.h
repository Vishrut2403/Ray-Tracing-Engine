#pragma once

#include <algorithm>
#include <memory>
#include "materials/material.h"

// Emitters carry radiance, not a colour, so the brightest channel sets the
// exposure and what survives is the emitter's hue.
inline color display_rgb(const std::shared_ptr<material>& m) {
	if (!m) return color(0.8, 0.8, 0.8);

	color c = m->display_color();
	for (int i = 0; i < 3; ++i)
		if (!(c[i] > 0)) c[i] = 0;

	real peak = c.max_component();
	return peak > 1 ? c / peak : c;
}
