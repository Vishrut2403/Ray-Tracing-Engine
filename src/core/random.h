#ifndef RANDOM_H
#define RANDOM_H
#pragma once

// CPU random functions — GPU will use cuRAND (see cuda/cuda_rand.cuh)
// None of these are marked HD because std::mt19937 is not available on device.

#include <algorithm>
#include <cmath>
#include <random>
#include "vec3.h"
#include "core/sampler.h"

inline double random_double() {
	if (g_sampler.active) return sampler_next();
	thread_local static std::mt19937 generator(std::random_device{}());
	thread_local static std::uniform_real_distribution<double> distribution(0.0, 1.0);
	return distribution(generator);
}

inline double random_double(double min, double max) {
	return min + (max - min) * random_double();
}

inline int random_int(int min, int max) {
	return static_cast<int>(random_double(min, max + 1));
}

// Every routine below draws a fixed number of samples, in a fixed order. The
// rejection loops these replace drew a variable number, which slid every later
// dimension of the sequence along by an amount that changed from path to path
// -- and a low-discrepancy sequence read at shifting dimensions is worth no
// more than white noise. The order matters too: each draw is a separate
// statement because the order of a function call's arguments is unspecified.

constexpr double kRandomPi = 3.1415926535897932385;

inline vec3 random_vec3() {
	double x = random_double(), y = random_double(), z = random_double();
	return vec3(x, y, z);
}

inline vec3 random_vec3(double min, double max) {
	double x = random_double(min,max), y = random_double(min,max);
	double z = random_double(min,max);
	return vec3(x, y, z);
}

inline vec3 random_unit_vector() {
	double z   = 1.0 - 2.0 * random_double();
	double r   = std::sqrt(std::max(0.0, 1.0 - z*z));
	double phi = 2.0 * kRandomPi * random_double();
	return vec3(r * std::cos(phi), r * std::sin(phi), z);
}

inline vec3 random_in_unit_sphere() {
	// Radius from the inverse cdf of r^3, which is what fills the ball evenly.
	double r = std::cbrt(random_double());
	return r * random_unit_vector();
}

// Shirley and Chiu's concentric mapping: equal-area, and it keeps the square's
// stratification instead of tearing it the way the polar mapping does.
inline vec3 random_in_unit_disk() {
	double a = 2.0 * random_double() - 1.0;
	double b = 2.0 * random_double() - 1.0;
	if (a == 0.0 && b == 0.0) return vec3(0, 0, 0);

	double r, phi;
	if (std::abs(a) > std::abs(b)) {
		r = a; phi = (kRandomPi / 4.0) * (b / a);
	} else {
		r = b; phi = kRandomPi / 2.0 - (kRandomPi / 4.0) * (a / b);
	}
	return vec3(r * std::cos(phi), r * std::sin(phi), 0);
}

inline vec3 random_cosine_direction() {
	auto r1 = random_double(), r2 = random_double();
	auto phi = 2 * kRandomPi * r1;
	return vec3(cos(phi)*sqrt(r2), sin(phi)*sqrt(r2), sqrt(1-r2));
}

#endif