#pragma once

#include <algorithm>
#include "core/vec3.h"

// Smits' RGB-to-reflectance conversion (JGT 1999). An RGB triple names a point
// in a three-dimensional space; a reflectance spectrum is a function. The
// inverse is underdetermined, so this picks the smoothest plausible answer:
// decompose the colour onto white, the two secondaries and the primary that
// bound it, each a measured smooth basis, which keeps the result inside [0,1]
// and free of the spikes a naive fit produces.
//
// Jakob and Hanika's sigmoid-polynomial fit round-trips more tightly, but it
// needs a precomputed coefficient table; this needs seventy values.

inline constexpr int  kSmitsN   = 10;
inline constexpr real kSmitsMin = 400;
inline constexpr real kSmitsMax = 700;

inline constexpr real kSmitsWhite[kSmitsN] = {
	1.0000, 1.0000, 0.9999, 0.9993, 0.9992, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000 };
inline constexpr real kSmitsCyan[kSmitsN] = {
	0.9710, 0.9426, 1.0007, 1.0007, 1.0007, 1.0007, 0.1564, 0.0000, 0.0000, 0.0000 };
inline constexpr real kSmitsMagenta[kSmitsN] = {
	1.0000, 1.0000, 0.9685, 0.2229, 0.0000, 0.0458, 0.8369, 1.0000, 1.0000, 0.9959 };
inline constexpr real kSmitsYellow[kSmitsN] = {
	0.0001, 0.0000, 0.1088, 0.6651, 1.0000, 1.0000, 0.9996, 0.9586, 0.9685, 0.9840 };
inline constexpr real kSmitsRed[kSmitsN] = {
	0.1012, 0.0515, 0.0000, 0.0000, 0.0000, 0.0000, 0.8325, 1.0149, 1.0149, 1.0149 };
inline constexpr real kSmitsGreen[kSmitsN] = {
	0.0000, 0.0000, 0.0273, 0.7937, 1.0000, 0.9418, 0.1719, 0.0000, 0.0000, 0.0025 };
inline constexpr real kSmitsBlue[kSmitsN] = {
	1.0000, 1.0000, 0.8916, 0.3323, 0.0000, 0.0000, 0.0003, 0.0369, 0.0483, 0.0496 };

// Flat outside the measured range rather than extrapolated: the bases say
// nothing about the near-UV and near-IR tails, and a straight line there
// invents reflectance that is not in the data.
inline real smits_lookup(const real (&basis)[kSmitsN], real lambda) {
	real t = (lambda - kSmitsMin) / (kSmitsMax - kSmitsMin) * (kSmitsN - 1);
	if (t <= 0)            return basis[0];
	if (t >= kSmitsN - 1)  return basis[kSmitsN - 1];
	int  i = (int)t;
	real f = t - i;
	return basis[i] * (1 - f) + basis[i + 1] * f;
}

// Reflectance at one wavelength for an RGB albedo.
inline real rgb_to_reflectance(const color& rgb, real lambda) {
	real r = rgb.x(), g = rgb.y(), b = rgb.z();
	real out = 0;

	auto add = [&](real w, const real (&basis)[kSmitsN]) {
		out += w * smits_lookup(basis, lambda);
	};

	// The smallest component is how much white the colour contains; what is
	// left decomposes onto one secondary and one primary.
	if (r <= g && r <= b) {
		add(r, kSmitsWhite);
		if (g <= b) { add(g - r, kSmitsCyan);    add(b - g, kSmitsBlue);  }
		else        { add(b - r, kSmitsCyan);    add(g - b, kSmitsGreen); }
	} else if (g <= r && g <= b) {
		add(g, kSmitsWhite);
		if (r <= b) { add(r - g, kSmitsMagenta); add(b - r, kSmitsBlue);  }
		else        { add(b - g, kSmitsMagenta); add(r - b, kSmitsRed);   }
	} else {
		add(b, kSmitsWhite);
		if (r <= g) { add(r - b, kSmitsYellow);  add(g - r, kSmitsGreen); }
		else        { add(g - b, kSmitsYellow);  add(r - g, kSmitsRed);   }
	}
	return std::max((real)0, out);
}

// The three samples a path is carrying. A zero wavelength means the render is
// in RGB mode, and the triple passes through as the colour it already was.
inline color spectral_albedo(const color& rgb, const vec3& lambda) {
	if (lambda[0] <= 0) return rgb;
	return color(rgb_to_reflectance(rgb, lambda[0]),
				 rgb_to_reflectance(rgb, lambda[1]),
				 rgb_to_reflectance(rgb, lambda[2]));
}

// Emission is unbounded, and the bases are reflectances. Normalising by the
// brightest channel keeps the hue in the basis' range and puts the magnitude
// back afterwards -- right for a neutral lamp, approximate for a coloured one.
inline color spectral_emission(const color& rgb, const vec3& lambda) {
	if (lambda[0] <= 0) return rgb;
	real peak = rgb.max_component();
	if (!(peak > 0)) return color(0, 0, 0);
	return spectral_albedo(rgb / peak, lambda) * peak;
}
