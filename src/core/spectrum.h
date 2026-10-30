#pragma once

#include <cmath>
#include "core/vec3.h"

// Visible range, matching the CIE 1931 2-degree observer's tabulated support.
inline constexpr real kLambdaMin = 360;
inline constexpr real kLambdaMax = 830;

// Wyman, Sloan and Shirley's multi-lobe Gaussian fits to the colour matching
// functions (JCGT 2013). Three asymmetric Gaussians instead of a 471-entry
// table, within about 1% of the tabulated curves.
HD inline real cie_lobe(real lambda, real mu, real s_lo, real s_hi) {
	real t = (lambda - mu) * (lambda < mu ? s_lo : s_hi);
	return std::exp((real)-0.5 * t * t);
}

HD inline real cie_x(real l) {
	return (real) 0.362 * cie_lobe(l, 442.0, 0.0624, 0.0374)
		 + (real) 1.056 * cie_lobe(l, 599.8, 0.0264, 0.0323)
		 - (real) 0.065 * cie_lobe(l, 501.1, 0.0490, 0.0382);
}

HD inline real cie_y(real l) {
	return (real) 0.821 * cie_lobe(l, 568.8, 0.0213, 0.0247)
		 + (real) 0.286 * cie_lobe(l, 530.9, 0.0613, 0.0322);
}

HD inline real cie_z(real l) {
	return (real) 1.217 * cie_lobe(l, 437.0, 0.0845, 0.0278)
		 + (real) 0.681 * cie_lobe(l, 459.0, 0.0385, 0.0725);
}

// Integral of the luminance curve over the visible range. Dividing by it is
// what makes a unit reflector under an equal-energy illuminant come back as
// unit luminance rather than a hundred and six of them.
inline constexpr real kCieYIntegral = 106.856895;

HD inline vec3 cie_xyz(real lambda) {
	return vec3(cie_x(lambda), cie_y(lambda), cie_z(lambda));
}

// One wavelength's contribution to the film, with 1/pdf already applied by the
// caller. Spectral radiance is a density in lambda, so the CIE weights are a
// quadrature rule, not a colour.
HD inline vec3 spectral_to_xyz(real lambda, real radiance) {
	return cie_xyz(lambda) * (radiance / kCieYIntegral);
}

// sRGB primaries, D65 white. Linear: the display transform comes later.
HD inline color xyz_to_rgb(const vec3& c) {
	return color(
		(real) 3.2406 * c.x() - (real)1.5372 * c.y() - (real)0.4986 * c.z(),
		(real)-0.9689 * c.x() + (real)1.8758 * c.y() + (real)0.0415 * c.z(),
		(real) 0.0557 * c.x() - (real)0.2040 * c.y() + (real)1.0570 * c.z());
}
