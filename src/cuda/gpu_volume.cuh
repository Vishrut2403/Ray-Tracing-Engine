#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/cuda_rand.cuh"

#ifndef GPU_PI
#define GPU_PI 3.1415926535897932385
#endif

__device__ inline float hg_phase(float cos_theta, float g) {
	float g2  = g * g;
	float den = 1.0f + g2 - 2.0f * g * cos_theta;
	return (1.0f - g2) / (4.0f * (float)GPU_PI * den * sqrtf(den) + 1e-7f);
}

__device__ inline vec3 hg_sample(const vec3& wo, float g, curandState* rng) {
	float cos_theta;

	if (fabsf(g) < 1e-4f) {
		cos_theta = 1.0f - 2.0f * (float)rand_double(rng);
	} else {
		float xi  = (float)rand_double(rng);
		float sqr = (1.0f - g*g) / (1.0f - g + 2.0f*g*xi);
		cos_theta = (1.0f + g*g - sqr*sqr) / (2.0f * g);
	}

	float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta*cos_theta));
	float phi       = 2.0f * (float)GPU_PI * (float)rand_double(rng);

	vec3 w = unit_vector(wo);
	vec3 a = (fabsf(w.x()) > 0.9f) ? vec3(0,1,0) : vec3(1,0,0);
	vec3 v = unit_vector(cross(w, a));
	vec3 u = cross(w, v);

	return unit_vector(
		u * (sin_theta * cosf(phi)) +
		v * (sin_theta * sinf(phi)) +
		w * cos_theta
	);
}

__device__ inline vec3 transmittance(const GpuMedium& med, double d) {
	return vec3(
		exp(-med.sigma_t.x() * d),
		exp(-med.sigma_t.y() * d),
		exp(-med.sigma_t.z() * d)
	);
}

// Overlap of [0, t_max] along r with the medium box, as [t0, t1].
__device__ inline bool medium_span(const GpuMedium& med, const ray& r,
									double t_max, double& t0, double& t1) {
	t0 = 0.0; t1 = t_max;
	for (int a = 0; a < 3; ++a) {
		double d = r.direction()[a];
		double lo = med.bmin[a], hi = med.bmax[a];
		if (fabs(d) < 1e-12) {
			if (r.origin()[a] < lo || r.origin()[a] > hi) return false;
			continue;
		}
		double inv = 1.0 / d;
		double ta = (lo - r.origin()[a]) * inv;
		double tb = (hi - r.origin()[a]) * inv;
		if (ta > tb) { double s = ta; ta = tb; tb = s; }
		t0 = fmax(t0, ta);
		t1 = fmin(t1, tb);
		if (t1 <= t0) return false;
	}
	return true;
}

// Transmittance along the part of the segment actually inside the medium.
__device__ inline vec3 transmittance_seg(const GpuMedium& med, const ray& r,
										  double t_max) {
	if (!med.active) return vec3(1,1,1);
	double t0, t1;
	if (!medium_span(med, r, t_max, t0, t1)) return vec3(1,1,1);
	return transmittance(med, (t1 - t0) * r.direction().length());
}

__device__ inline double sample_free_flight(const GpuMedium& med,
											 curandState* rng) {
	float sigma_t_avg = (float)(
		med.sigma_t.x() + med.sigma_t.y() + med.sigma_t.z()) / 3.0f;
	if (sigma_t_avg < 1e-8f) return 1e30;
	float xi = fmaxf((float)rand_double(rng), 1e-7f);
	return (double)(-logf(xi) / sigma_t_avg);
}

struct MediumSample {
	bool  scattered;
	vec3  pos;
	vec3  wi;
	vec3  weight;
	double t_scatter;
};

__device__ inline MediumSample sample_medium(
	const GpuMedium& med,
	const ray&       r,
	double           t_surface,
	curandState*     rng)
{
	MediumSample ms;
	ms.scattered  = false;
	ms.weight     = vec3(1,1,1);
	ms.t_scatter  = t_surface;

	if (!med.active) return ms;

	double t0, t1;
	if (!medium_span(med, r, t_surface, t0, t1)) return ms;

	// Camera rays are not unit length, so ray parameters are not world
	// distances; free flight is sampled in distance and converted here.
	double len    = r.direction().length();
	if (len < 1e-12) return ms;
	double t_free = t0 + sample_free_flight(med, rng) / len;

	if (t_free < t1) {
		ms.scattered = true;
		ms.t_scatter = t_free;
		ms.pos       = r.at(t_free);
		ms.wi        = hg_sample(r.direction(), med.g, rng);

		// sample_free_flight never scatters at sigma_t ~ 0, so no guard here:
		// an epsilon in the denominator would bias the albedo.
		double sigma_t_avg =
			(med.sigma_t.x() + med.sigma_t.y() + med.sigma_t.z()) / 3.0;
		ms.weight = med.sigma_s / sigma_t_avg;
	} else {
		ms.weight = vec3(1,1,1);
	}

	return ms;
}