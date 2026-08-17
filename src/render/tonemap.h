#pragma once
#include <cmath>
#include "core/vec3.h"

// Display transform for both the image writer and the preview window. The GLSL
// below must stay in step with the C++ above it.

static constexpr double TM_EXPOSURE      = 1.0;
static constexpr double TM_FIREFLY_CLAMP = 20.0;

inline double tm_clamp01(double x) { return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x); }

inline double tm_aces(double x) {
	const double a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
	return tm_clamp01((x*(a*x+b))/(x*(c*x+d)+e));
}

inline color tonemap_display(color c) {
	double lum = 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
	if (lum > TM_FIREFLY_CLAMP) c = c * (real)(TM_FIREFLY_CLAMP / lum);
	c = c * (real)TM_EXPOSURE;
	return color((real)std::pow(tm_aces(c.x()), 1.0/2.2),
				 (real)std::pow(tm_aces(c.y()), 1.0/2.2),
				 (real)std::pow(tm_aces(c.z()), 1.0/2.2));
}

inline const char* tonemap_glsl() {
	return R"(
		const float TM_EXPOSURE      = 1.0;
		const float TM_FIREFLY_CLAMP = 20.0;

		float tm_aces(float x) {
			const float a=2.51, b=0.03, c=2.43, d=0.59, e=0.14;
			return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
		}

		vec3 tonemap_display(vec3 c) {
			float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
			if (lum > TM_FIREFLY_CLAMP) c *= TM_FIREFLY_CLAMP / lum;
			c *= TM_EXPOSURE;
			c = vec3(tm_aces(c.r), tm_aces(c.g), tm_aces(c.b));
			return pow(c, vec3(1.0 / 2.2));
		}
	)";
}
