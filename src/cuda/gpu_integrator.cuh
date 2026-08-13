#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/interval.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/gpu_hit.cuh"
#include "cuda/gpu_material.cuh"
#include "cuda/gpu_env_map.cuh"
#include "cuda/gpu_triangle.h"
#include "cuda/cuda_rand.cuh"
#include "cuda/gpu_camera.cuh"
#include "cuda/gpu_bdpt.cuh"

#define RAY_OFFSET 0.02

__device__ inline double gpu_power_heuristic(double a, double b) {
	double a2 = a*a, b2 = b*b;
	return a2 / (a2 + b2 + 1e-12);
}

__device__ inline double gpu_light_pdf(const GpuHittable* hittables,
										const int* light_ids, int n_lights,
										const vec3& origin, const vec3& dir) {
	if (n_lights == 0) return 0.0;
	double sum = 0.0, w = 1.0 / n_lights;
	for (int i = 0; i < n_lights; ++i) {
		const GpuHittable& lh = hittables[light_ids[i]];
		GpuHitRecord tmp;
		ray test_ray(origin, dir, 0.0);
		ray local = (lh.has_rotation || lh.translate_offset.length_squared() > 0.0)
					? apply_inverse_transform(test_ray, lh) : test_ray;
		if (hit_xz_rect(lh, local, interval(RAY_OFFSET, 1e30), tmp)) {
			double area  = (double)(lh.a1-lh.a0) * (double)(lh.b1-lh.b0);
			double dist2 = tmp.t * tmp.t * dir.length_squared();
			double cos_t = fabs(dot(tmp.normal, unit_vector(dir)));
			if (cos_t > 1e-8) sum += w * dist2 / (cos_t * area);
		}
	}
	return sum;
}

__device__ inline vec3 gpu_light_random(const GpuHittable* hittables,
										 const int* light_ids, int n_lights,
										 const vec3& origin, curandState* rng) {
	if (n_lights == 0) return vec3(1,0,0);
	int idx = min((int)(rand_double(rng) * n_lights), n_lights-1);
	const GpuHittable& lh = hittables[light_ids[idx]];
	double rx = (double)lh.a0 + rand_double(rng) * (double)(lh.a1-lh.a0);
	double rz = (double)lh.b0 + rand_double(rng) * (double)(lh.b1-lh.b0);
	vec3 pt(rx, (double)lh.k, rz);
	return pt + lh.translate_offset - origin;
}

#define MESH_ARGS tris, tri_bvh, tri_root, n_tris

__device__ vec3 gpu_Li(
	const ray& initial_ray,
	const vec3& background,
	const GpuHittable* hittables, int n_hittables,
	const GpuMaterial* materials,
	const int* light_ids, int n_lights,
	int max_depth,
	curandState* rng,
	const GpuTriangle*   tris     = nullptr,
	const GpuTriBVHNode* tri_bvh  = nullptr,
	int                  tri_root = 0,
	int                  n_tris   = 0,
	const GpuEnvMap*     env_map  = nullptr
) {
	ray   r    = initial_ray;
	vec3  L    (0,0,0);
	vec3  beta (1,1,1);
	bool  specular_bounce = false;
	double last_bsdf_pdf  = 0.0;
	vec3  prev_p          = initial_ray.origin();

	for (int depth = 0; depth < max_depth; ++depth) {
		GpuHitRecord rec;
		if (!hit_scene(hittables, n_hittables, r, interval(RAY_OFFSET, 1e30), rec,
					   MESH_ARGS)) {
			vec3 Le = (env_map && env_map->valid)
					  ? gpu_env_Le(*env_map, r.direction())
					  : background;

			if (specular_bounce || n_lights == 0 || depth == 0) {
				L = L + beta * Le;
			} else {
				// rec is unwritten on a miss; use the previous vertex.
				double lp = (env_map && env_map->valid)
							? (double)gpu_env_pdf(*env_map, r.direction())
							: gpu_light_pdf(hittables, light_ids, n_lights,
											prev_p, r.direction());
				L = L + beta * Le * gpu_power_heuristic(last_bsdf_pdf, lp);
			}
			break;
		}

		const GpuMaterial& mat = materials[rec.mat_id];
		vec3 emitted = gpu_emitted(mat, rec.front_face);
		if (emitted.length_squared() > 0.0) {
			if (specular_bounce || depth == 0) {
				L = L + beta * emitted;
			} else {
				// MIS partner is NEE's density at the PREVIOUS vertex. From rec.p
				// (already on the light) it re-intersects nothing and returns 0,
				// giving this hit full weight on top of NEE.
				double lp = gpu_light_pdf(hittables, light_ids, n_lights,
										   prev_p, r.direction());
				L = L + beta * emitted * gpu_power_heuristic(last_bsdf_pdf, lp);
			}
		}

		if (!specular_bounce && n_lights > 0) {
			vec3   to_light  = gpu_light_random(hittables, light_ids, n_lights,
												 rec.p, rng);
			double distance  = to_light.length();
			vec3   wi        = to_light / distance;
			double light_pdf = gpu_light_pdf(hittables, light_ids, n_lights,
											  rec.p, wi);

			if (light_pdf > 0.0) {
				GpuHitRecord shadow_rec;
				bool occluded = hit_scene(hittables, n_hittables,
										  ray(rec.p, wi, r.time()),
										  interval(RAY_OFFSET, distance - RAY_OFFSET),
										  shadow_rec, MESH_ARGS);
				if (!occluded) {
					GpuHitRecord light_rec;
					if (hit_scene(hittables, n_hittables,
								  ray(rec.p, wi, r.time()),
								  interval(RAY_OFFSET, 1e30), light_rec,
								  MESH_ARGS)) {
						vec3 Le = gpu_emitted(materials[light_rec.mat_id],
											  light_rec.front_face);
						if (Le.length_squared() > 0.0) {
							vec3   f        = gpu_f(mat, wi, rec.normal);
							double bsdf_pdf = gpu_pdf(mat, wi, rec.normal);
							double weight   = gpu_power_heuristic(light_pdf, bsdf_pdf);
							double cos_t    = fabs(dot(rec.normal, wi));
							L = L + beta * f * Le * (cos_t * weight / light_pdf);
						}
					}
				}
			}
		}

		if (!specular_bounce && env_map && env_map->valid) {
			float env_pdf_val;
			vec3  wi = gpu_env_sample(*env_map, rng, env_pdf_val);
			if (env_pdf_val > 0.0f) {
				GpuHitRecord shadow_rec;
				bool occluded = hit_scene(hittables, n_hittables,
										  ray(rec.p, wi, r.time()),
										  interval(RAY_OFFSET, 1e30),
										  shadow_rec, MESH_ARGS);
				if (!occluded) {
					vec3   Le       = gpu_env_Le(*env_map, wi);
					vec3   f        = gpu_f(mat, wi, rec.normal);
					double bsdf_pdf = gpu_pdf(mat, wi, rec.normal);
					double weight   = gpu_power_heuristic((double)env_pdf_val, bsdf_pdf);
					double cos_t    = fabs(dot(rec.normal, wi));
					L = L + beta * f * Le * (cos_t * weight / (double)env_pdf_val);
				}
			}
		}

		GpuBSDFSample bs = gpu_sample(mat, r, rec, rng);
		if (bs.pdf <= 0.0) break;

		last_bsdf_pdf = bs.pdf;
		if (bs.is_delta) {
			beta = beta * bs.f;
			specular_bounce = true;
		} else {
			double cos_t = fabs(dot(rec.normal, bs.wi));
			beta = beta * bs.f * (cos_t / bs.pdf);
			specular_bounce = false;
		}

		if (depth >= 3) {
			double survival = fmax(0.05, fmin(0.95, beta.max_component()));
			if (rand_double(rng) > survival) break;
			beta = beta / survival;
		}

		prev_p = rec.p;
		r = ray(rec.p, bs.wi, r.time());
	}
	return L;
}

#undef MESH_ARGS

// GpuCamera and gpu_get_ray live in gpu_camera.cuh.

__global__ void accumulate_kernel(
	float* d_accum, int width, int height, int batch_spp, int max_depth,
	GpuCamera cam, vec3 background,
	const GpuHittable*   hittables,  int n_hittables,
	const GpuMaterial*   materials,
	const int*           light_ids,  int n_lights,
	curandState*         rand_states,
	const GpuTriangle*   tris,
	const GpuTriBVHNode* tri_bvh,
	int                  tri_root,
	int                  n_tris,
	const GpuEnvMap*     env_map
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int         id  = y * width + x;
	curandState rng = rand_states[id];

	vec3 pixel(0,0,0);
	for (int s = 0; s < batch_spp; ++s) {
		double u = (x + rand_double(&rng)) / (width  - 1);
		double v = (y + rand_double(&rng)) / (height - 1);
		ray r    = gpu_get_ray(cam, u, v, &rng);
		pixel    = pixel + gpu_Li(r, background,
								   hittables, n_hittables, materials,
								   light_ids, n_lights, max_depth, &rng,
								   tris, tri_bvh, tri_root, n_tris,
								   env_map);
	}
	d_accum[id*3+0] += (float)pixel.x();
	d_accum[id*3+1] += (float)pixel.y();
	d_accum[id*3+2] += (float)pixel.z();
	rand_states[id]  = rng;
}

// Firefly clamp, matching the ReSTIR shade kernel.
__device__ inline vec3 bdpt_clamp(vec3 c) {
	double lum = 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
	if (!isfinite(lum) || lum < 0.0) return vec3(0,0,0);
	if (lum > 50.0) return c * (50.0 / lum);
	return c;
}

__global__ void accumulate_bdpt_kernel(
	float* d_accum, int width, int height, int max_depth,
	GpuCamera cam,
	const GpuHittable*   hittables,  int n_hittables,
	const GpuMaterial*   materials,
	const int*           light_ids,  int n_lights,
	curandState*         rand_states,
	const GpuTriangle*   tris,
	const GpuTriBVHNode* tri_bvh,
	int                  tri_root,
	int                  n_tris
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int         id  = y * width + x;
	curandState rng = rand_states[id];

	GpuCamAux aux = gpu_make_cam_aux(cam, width, height);

	GpuBDPTScene sc;
	sc.hittables   = hittables;
	sc.n_hittables = n_hittables;
	sc.materials   = materials;
	sc.light_ids   = light_ids;
	sc.n_lights    = n_lights;
	sc.tris        = tris;
	sc.tri_bvh     = tri_bvh;
	sc.tri_root    = tri_root;
	sc.n_tris      = n_tris;

	GpuPathVertex camv  [MAX_BDPT_VERTS];
	GpuPathVertex lightv[MAX_BDPT_VERTS];

	// Every split must be representable or the MIS weights lose energy.
	if (max_depth > MAX_BDPT_DEPTH) max_depth = MAX_BDPT_DEPTH;

	int cam_max = min(max_depth + 2, MAX_BDPT_VERTS);
	int lit_max = min(max_depth + 2, MAX_BDPT_VERTS);

	double u = (x + rand_double(&rng)) / (width  - 1);
	double v = (y + rand_double(&rng)) / (height - 1);
	ray    cam_ray = gpu_get_ray(cam, u, v, &rng);

	int nt = gpu_generate_camera_subpath(sc, aux, cam, cam_ray, cam_max,
										 camv, &rng);
	int ns = gpu_generate_light_subpath(sc, lit_max, lightv, &rng);

	// One light subpath per thread (= per pixel); the splat spreads over the
	// whole film, hence 1/(W*H).
	double inv_light_paths = 1.0 / ((double)width * (double)height);

	vec3 L(0,0,0);

	for (int t = 1; t <= nt; ++t) {
		for (int s = 0; s <= ns; ++s) {
			int depth = t + s - 2;
			if ((s == 1 && t == 1) || depth < 0 || depth > max_depth) continue;

			double rx = 0.0, ry = 0.0;
			vec3 c = gpu_connect_bdpt(sc, aux, cam, camv, t, lightv, s,
									  &rng, rx, ry);
			if (!(c.length_squared() > 0.0)) continue;

			if (t == 1) {
				int px = (int)(rx + 0.5);
				int py = (int)(ry + 0.5);
				if (px >= 0 && px < width && py >= 0 && py < height) {
					// Clamp after the 1/(W*H): a t == 1 contribution is carried
					// undivided and is W*H times larger.
					vec3 sp  = bdpt_clamp(c * inv_light_paths);
					int  sid = py * width + px;
					atomicAdd(&d_accum[sid*3+0], (float)sp.x());
					atomicAdd(&d_accum[sid*3+1], (float)sp.y());
					atomicAdd(&d_accum[sid*3+2], (float)sp.z());
				}
			} else {
				L = L + bdpt_clamp(c);
			}
		}
	}

	L = bdpt_clamp(L);

	// Other threads splat here concurrently.
	atomicAdd(&d_accum[id*3+0], (float)L.x());
	atomicAdd(&d_accum[id*3+1], (float)L.y());
	atomicAdd(&d_accum[id*3+2], (float)L.z());
	rand_states[id] = rng;
}

#undef RAY_OFFSET
