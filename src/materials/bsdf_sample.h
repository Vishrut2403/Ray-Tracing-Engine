#pragma once

#include "core/vec3.h"

struct BSDFSample {
	vec3 wi;
	color f;
	real pdf;
	bool is_delta;
	// Phase-function sample: no surface normal, so no cosine factor.
	bool is_phase = false;
	// A dispersive refraction sends each wavelength somewhere different, so a
	// path carrying three of them cannot continue. The material drops the two
	// secondaries and scales the survivor to keep the estimator unbiased; this
	// tells the integrator to stop handing out the other two wavelengths.
	bool terminate_secondary = false;
};