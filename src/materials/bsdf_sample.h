#pragma once

#include "core/vec3.h"

struct BSDFSample {
	vec3 wi;
	color f;
	double pdf;
	bool is_delta;
	// Phase-function sample: no surface normal, so no cosine factor.
	bool is_phase = false;
};