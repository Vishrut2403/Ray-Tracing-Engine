#pragma once
#include "core/vec3.h"
#include "cuda/cuda_rand.cuh"

struct GpuEnvMap {
	float*  d_pixels       = nullptr;
	float*  d_marginal_cdf = nullptr;
	float*  d_cond_cdf     = nullptr;
	float*  d_pixel_pdf    = nullptr; 
	int     width          = 0;
	int     height         = 0;
	bool    valid          = false;

	void free_device() {
		if (d_pixels)       { cudaFree(d_pixels);       d_pixels       = nullptr; }
		if (d_marginal_cdf) { cudaFree(d_marginal_cdf); d_marginal_cdf = nullptr; }
		if (d_cond_cdf)     { cudaFree(d_cond_cdf);     d_cond_cdf     = nullptr; }
		if (d_pixel_pdf)    { cudaFree(d_pixel_pdf);    d_pixel_pdf    = nullptr; }
		valid = false;
	}
};

__device__ inline int gpu_cdf_sample(const float* cdf, int n, float u) {
	int lo = 0, hi = n;
	while (lo < hi) {
		int mid = (lo + hi) / 2;
		if (cdf[mid] < u) lo = mid + 1;
		else              hi = mid;
	}
	return lo > 0 ? lo - 1 : 0;
}

__device__ inline vec3 gpu_env_sample(const GpuEnvMap& env, curandState* rng,
									   float& out_pdf) {
	float r1 = rand_double(rng);
	float r2 = rand_double(rng);

	int W = env.width, H = env.height;

	int row = gpu_cdf_sample(env.d_marginal_cdf, H, r1);
	row = min(max(row, 0), H - 1);

	const float* cond = env.d_cond_cdf + row * (W + 1);
	int col = gpu_cdf_sample(cond, W, r2);
	col = min(max(col, 0), W - 1);

	real u     = (col + 0.5) / W;
	real v     = (row + 0.5) / H;
	// Must invert gpu_env_Le: row 0 is the zenith, u = 0.5 is phi = 0.
	real phi   = 2.0 * 3.14159265358979 * (u - 0.5);
	real theta = 3.14159265358979 * v;

	real sin_theta = sin(theta);
	real cos_theta = cos(theta);

	vec3 dir(sin_theta * cos(phi),
			 cos_theta,
			 sin_theta * sin(phi));

	// pixel_pdf is a per-pixel mass; * W*H makes it a density over (u,v).
	float p = env.d_pixel_pdf[row * W + col];
	real n = (real)W * H;
	out_pdf = (sin_theta > 1e-8f)
			  ? (float)(p * n / (sin_theta * 2.0 * 3.14159265358979 * 3.14159265358979))
			  : 0.0f;

	return dir;
}

__device__ inline vec3 gpu_env_Le(const GpuEnvMap& env, const vec3& dir) {
	if (!env.valid) return vec3(0,0,0);
	vec3 d = unit_vector(dir);

	real u = 0.5 + atan2(d.z(), d.x()) / (2.0 * 3.14159265358979);
	real v = 0.5 + asin(rmax(-1.0, rmin(1.0, (real)d.y()))) / 3.14159265358979;
	v = 1.0 - v; 

	int i = min((int)(u * env.width),  env.width  - 1);
	int j = min((int)(v * env.height), env.height - 1);
	i = max(i, 0); j = max(j, 0);

	const float* px = env.d_pixels + (j * env.width + i) * 3;
	return vec3(px[0], px[1], px[2]);
}

__device__ inline float gpu_env_pdf(const GpuEnvMap& env, const vec3& dir) {
	if (!env.valid) return 0.0f;
	vec3 d = unit_vector(dir);

	real u = 0.5 + atan2(d.z(), d.x()) / (2.0 * 3.14159265358979);
	real v = 0.5 + asin(rmax(-1.0, rmin(1.0, (real)d.y()))) / 3.14159265358979;
	v = 1.0 - v;

	int i = min((int)(u * env.width),  env.width  - 1);
	int j = min((int)(v * env.height), env.height - 1);
	i = max(i, 0); j = max(j, 0);

	real sin_theta = sqrt(1.0 - (real)d.y() * (real)d.y());
	if (sin_theta < 1e-8) return 0.0f;

	float p = env.d_pixel_pdf[j * env.width + i];
	real n = (real)env.width * env.height;
	return (float)(p * n / (sin_theta * 2.0 * 3.14159265358979 * 3.14159265358979));
}