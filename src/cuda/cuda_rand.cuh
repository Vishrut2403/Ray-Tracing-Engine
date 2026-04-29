#pragma once

#include <curand_kernel.h>
#include "core/vec3.h"

__global__ void cuda_rand_init(curandState* states, unsigned long long seed, int n) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= n) return;
	curand_init(seed, id, 0, &states[id]);
}

__device__ inline double rand_double(curandState* state) {
	return curand_uniform_double(state);
}

__device__ inline vec3 rand_in_unit_sphere(curandState* state) {
	while (true) {
		vec3 p(
			2.0*curand_uniform_double(state)-1.0,
			2.0*curand_uniform_double(state)-1.0,
			2.0*curand_uniform_double(state)-1.0
		);
		if (p.length_squared() < 1.0) return p;
	}
}

__device__ inline vec3 rand_unit_vector(curandState* state) {
	return unit_vector(rand_in_unit_sphere(state));
}

__device__ inline vec3 rand_cosine_direction(curandState* state) {
	double r1  = curand_uniform_double(state);
	double r2  = curand_uniform_double(state);
	double phi = 2.0 * 3.1415926535897932385 * r1;
	return vec3(cos(phi)*sqrt(r2), sin(phi)*sqrt(r2), sqrt(1.0-r2));
}
