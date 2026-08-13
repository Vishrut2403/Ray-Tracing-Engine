#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/interval.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/gpu_hit.cuh"
#include "cuda/gpu_material.cuh"
#include "cuda/cuda_rand.cuh"

struct PathSample {
	vec3  x_s;
	vec3  n_s;
	vec3  L_o;
	vec3  x_v;
	float pdf;
	bool  valid;
};

struct GIReservoir {
	PathSample y;
	float      w_sum;
	int        M;
	float      W;

	__device__ void init() {
		y.valid = false;
		y.pdf   = 0.0f;
		y.L_o   = vec3(0,0,0);
		w_sum   = 0.0f;
		M       = 0;
		W       = 0.0f;
	}

	__device__ bool update(const PathSample& x, float w, curandState* rng) {
		w_sum += w;
		M     += 1;
		if (rand_double(rng) < (double)(w / (w_sum + 1e-30f))) {
			y = x;
			return true;
		}
		return false;
	}

	__device__ bool merge(const GIReservoir& other, float p_hat,
						  curandState* rng) {
		int M_new = M + other.M;
		bool sel  = update(other.y, p_hat * other.W * (float)other.M, rng);
		M = M_new;
		return sel;
	}

	__device__ void cap_M(int max_M) {
		if (M > max_M) {
			w_sum *= (float)max_M / (float)M;
			M      = max_M;
			// W stays the same — it's w_sum/(M*p_hat) and both scale together
		}
	}

	__device__ void finalize(float p_hat) {
		if (p_hat > 1e-10f && M > 0) {
			W = (w_sum / (float)M) / p_hat;
			W = fminf(W, 1.0f);  // ← clamp to 1.0, not 5.0 — prevents overweighting
			// Do NOT touch w_sum here — leave it as the true accumulated sum
			// so temporal reuse next frame merges correctly
		} else {
			W     = 0.0f;
		}
	}
};

// GI p_hat
__device__ inline float gi_p_hat(
	const PathSample&   ps,
	const GBufferPixel& gbuf,
	const GpuMaterial*  materials)
{
	if (!ps.valid || !gbuf.valid) return 0.0f;

	vec3  d    = ps.x_s - gbuf.pos;
	float dist = (float)d.length();
	if (dist < 1e-6f) return 0.0f;

	vec3  wi    = d / dist;
	float cos_i = fmaxf(0.0f, (float)dot(gbuf.normal, wi));
	if (cos_i < 1e-6f) return 0.0f;

	float lum = 0.2126f*(float)ps.L_o.x()
			  + 0.7152f*(float)ps.L_o.y()
			  + 0.0722f*(float)ps.L_o.z();

	float brdf_w = 1.0f;
	if (materials) {
		vec3 f   = gpu_f_dir(materials[gbuf.mat_id], gbuf.wo, wi, gbuf.normal);
		brdf_w   = fmaxf(0.0f, (float)(f.x() + f.y() + f.z()) / 3.0f);
	}

	return fmaxf(0.0f, lum * cos_i * brdf_w);
}

// Reconnection Jacobian
__device__ inline float reconnection_jacobian(
	const PathSample& ps,
	const vec3&       x_v_new)
{
	vec3  d_orig    = ps.x_s - ps.x_v;
	float dist_orig = (float)d_orig.length();
	if (dist_orig < 1e-6f) return 0.0f;
	float cos_orig  = fmaxf(1e-6f, (float)fabs(dot(ps.n_s,
							d_orig / dist_orig)));

	vec3  d_new    = ps.x_s - x_v_new;
	float dist_new = (float)d_new.length();
	if (dist_new < 1e-6f) return 0.0f;
	float cos_new  = fmaxf(1e-6f, (float)fabs(dot(ps.n_s,
							d_new / dist_new)));

	float jacobian = (cos_new * dist_orig * dist_orig)
				   / (cos_orig * dist_new * dist_new + 1e-10f);

	return fminf(jacobian, 5.0f);   // ← tighter clamp than before
}