#pragma once

#include <curand_kernel.h>
#include "core/vec3.h"
#include "core/sampler.h"

// static: a non-static __global__ defined in a header multiply-defines as
// soon as a second .cu includes it.
__global__ static void cuda_rand_init(curandState* states, unsigned long long seed, int n) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= n) return;
	curand_init(seed, id, 0, &states[id]);
}

// The device half of the sampler. It walks the same Owen-scrambled
// (0,2)-sequence as the CPU -- the arithmetic is shared, in core/sampler.h --
// and keeps curand alongside it for the estimators that have not been keyed
// onto the sequence yet. Only curand's state is persistent; the sequence needs
// nothing but the pixel and sample index, so it lives on the stack.
struct GpuSampler {
	curandState rng;
	uint32_t    pixel       = 0;
	uint32_t    index       = 0;
	uint32_t    dim         = 0;
	real        pending     = 0;
	bool        has_pending = false;
	// False leaves every draw on curand, which is what BDPT and ReSTIR do.
	bool        active      = false;
};

__device__ inline void gpu_sampler_begin(GpuSampler& s, uint32_t pixel,
										 uint32_t index) {
	s.active      = true;
	s.pixel       = sampler_hash(pixel + 1u);
	s.index       = index;
	s.dim         = 0;
	s.has_pending = false;
}

// Mirrors sampler_next() on the host, dimension for dimension.
__device__ inline real rand_double(GpuSampler* s) {
	if (!s->active) return curand_uniform_double(&s->rng);

	if (s->has_pending) {
		s->has_pending = false;
		++s->dim;
		return s->pending;
	}

	uint32_t pair = s->dim >> 1;
	uint32_t idx  = sampler_owen(s->index, sampler_mix(s->pixel, pair + 0x51u));
	uint32_t sx   = sampler_mix(s->pixel, pair * 2u + 1u);
	uint32_t sy   = sampler_mix(s->pixel, pair * 2u + 2u);

	// owen reverses its argument first and reversal is its own inverse, so
	// owen(reverse(idx)) folds to reverse(permute(idx)).
	real u = sampler_unit(sampler_reverse_bits(sampler_lk_permute(idx, sx)));
	real v = sampler_unit(sampler_owen(sampler_sobol2(idx, 0u), sy));

	s->pending     = v;
	s->has_pending = true;
	++s->dim;
	return u;
}

// These match core/random.h: a fixed number of dimensions in a fixed order. A
// rejection loop would draw a varying number and slide every later dimension
// along, which is what costs the sequence its stratification.

__device__ inline vec3 rand_unit_vector(GpuSampler* s) {
	real z   = (real)1 - (real)2 * rand_double(s);
	real r   = sqrt(fmax((real)0, (real)1 - z*z));
	real phi = (real)2 * (real)3.1415926535897932385 * rand_double(s);
	return vec3(r * cos(phi), r * sin(phi), z);
}

__device__ inline vec3 rand_in_unit_sphere(GpuSampler* s) {
	// Radius from the inverse cdf of r^3, which is what fills the ball evenly.
	real r = cbrt(rand_double(s));
	return r * rand_unit_vector(s);
}

// Shirley and Chiu's concentric mapping: equal-area, and it keeps the square's
// stratification instead of tearing it the way the polar mapping does.
__device__ inline vec3 rand_in_unit_disk(GpuSampler* s) {
	real a = (real)2 * rand_double(s) - (real)1;
	real b = (real)2 * rand_double(s) - (real)1;
	if (a == 0 && b == 0) return vec3(0, 0, 0);

	const real pi_4 = (real)0.78539816339744831;
	real r, phi;
	if (fabs(a) > fabs(b)) { r = a; phi = pi_4 * (b / a); }
	else                   { r = b; phi = (real)2*pi_4 - pi_4 * (a / b); }
	return vec3(r * cos(phi), r * sin(phi), 0);
}

__device__ inline vec3 rand_cosine_direction(GpuSampler* s) {
	real r1  = rand_double(s);
	real r2  = rand_double(s);
	real phi = (real)2 * (real)3.1415926535897932385 * r1;
	return vec3(cos(phi)*sqrt(r2), sin(phi)*sqrt(r2), sqrt((real)1-r2));
}
