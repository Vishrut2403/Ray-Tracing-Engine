#pragma once

#include "cuda/gpu_restir.cuh"
#include "cuda/gpu_restir_gi.cuh"
#include "cuda/gpu_integrator.cuh"
#include "cuda/gpu_volume.cuh"

#define RAY_OFFSET        0.02
#define GI_TEMPORAL_M_CAP 20

__global__ void restir_gi_initial_kernel(
	int W, int H,
	const GpuHittable*   hittables,  int n_hittables,
	const GpuMaterial*   materials,
	const int*           light_ids,  int n_lights,
	const GpuTriangle*   tris,
	const GpuTriBVHNode* tri_bvh,
	int tri_root, int n_tris,
	const GBufferPixel*  gbuffer,
	GIReservoir*         gi_reservoirs,
	curandState*         rng_states,
	const GpuEnvMap*     env_map,
	vec3                 background
) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	int idx = y*W + x;
	curandState rng = rng_states[idx];

	GIReservoir& res = gi_reservoirs[idx];
	res.init();

	const GBufferPixel& gbuf = gbuffer[idx];
	if (!gbuf.valid) { rng_states[idx] = rng; return; }

	const GpuMaterial& mat = materials[gbuf.mat_id];

	GpuHitRecord gbuf_rec{};
	gbuf_rec.p          = gbuf.pos;
	gbuf_rec.normal     = gbuf.normal;
	gbuf_rec.mat_id     = gbuf.mat_id;
	gbuf_rec.front_face = true;
	gbuf_rec.t          = (double)gbuf.depth;

	GpuBSDFSample bs = gpu_sample(mat,
		ray(gbuf.pos - gbuf.wo*(double)RAY_OFFSET, -gbuf.wo, 0.0),
		gbuf_rec, &rng);

	if (bs.pdf <= 0.0) { rng_states[idx] = rng; return; }

	GpuHitRecord irec;
	ray indirect(gbuf.pos, bs.wi, 0.0);
	bool hit = hit_scene(hittables, n_hittables,
						 indirect, interval(RAY_OFFSET, 1e30),
						 irec, tris, tri_bvh, tri_root, n_tris);

	PathSample ps;
	ps.x_v   = gbuf.pos;
	ps.pdf   = (float)bs.pdf;
	ps.valid = false;

	if (hit) {
		vec3 L_o(0,0,0);
		vec3 Le = gpu_emitted(materials[irec.mat_id], irec.front_face);
		if (Le.length_squared() > 0.0) {
			// ReSTIR DI already integrates the whole area-light term here, with
			// no MIS partner, so counting this bounce too double-counts direct
			// lighting. With no area lights, DI contributes nothing — keep it.
			if (n_lights == 0) L_o = Le;
		} else if (n_lights > 0) {
			vec3 to_l = gpu_light_random(hittables, light_ids,
										  n_lights, irec.p, &rng);
			double dl = to_l.length();
			if (dl > 1e-6) {
				vec3   wl   = to_l / dl;
				double lpdf = gpu_light_pdf(hittables, light_ids,
											 n_lights, irec.p, wl);
				if (lpdf > 0.0) {
					GpuHitRecord sr;
					bool occ = hit_scene(hittables, n_hittables,
						ray(irec.p, wl, 0.0),
						interval(RAY_OFFSET, dl - RAY_OFFSET),
						sr, tris, tri_bvh, tri_root, n_tris);
					if (!occ) {
						GpuHitRecord lr;
						if (hit_scene(hittables, n_hittables,
							ray(irec.p, wl, 0.0),
							interval(RAY_OFFSET, 1e30), lr,
							tris, tri_bvh, tri_root, n_tris)) {
							vec3 lLe = gpu_emitted(
								materials[lr.mat_id], lr.front_face);
							if (lLe.length_squared() > 0.0) {
								vec3   f2  = gpu_f_dir(materials[irec.mat_id],
													-bs.wi, wl, irec.normal);
								double ct2 = fabs(dot(irec.normal, wl));
								L_o = f2 * lLe * (ct2 / lpdf);
							}
						}
					}
				}
			}
		}
		ps.x_s   = irec.p;
		ps.n_s   = irec.normal;
		ps.L_o   = L_o;
		ps.valid = true;
	} else {
		vec3 Le = (env_map && env_map->valid)
				  ? gpu_env_Le(*env_map, bs.wi)
				  : background;
		ps.x_s   = gbuf.pos + bs.wi * 1e4;
		ps.n_s   = -bs.wi;
		ps.L_o   = Le;
		ps.valid = Le.length_squared() > 1e-6;
	}

	if (ps.valid) {
		float ph = gi_p_hat(ps, gbuf, materials);
		float w  = (bs.pdf > 1e-10f) ? ph / (float)bs.pdf : 0.0f;
		res.update(ps, w, &rng);
		res.finalize(ph);   // already caps W and renormalizes w_sum
	}

	rng_states[idx] = rng;
}

__global__ void restir_gi_temporal_kernel(
	int W, int H,
	const GpuMaterial*   materials,
	const GBufferPixel*  gbuffer_cur,
	const GBufferPixel*  gbuffer_prev,
	const GIReservoir*   gi_cur,
	const GIReservoir*   gi_prev,
	GIReservoir*         gi_out,
	curandState*         rng_states,
	bool                 has_prev_frame
) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	int idx = y*W + x;
	curandState rng = rng_states[idx];

	GIReservoir combined = gi_cur[idx];
	const GBufferPixel& gbuf = gbuffer_cur[idx];

	if (has_prev_frame && gbuf.valid) {
		const GBufferPixel& pbuf = gbuffer_prev[idx];
		const GIReservoir&  prev = gi_prev[idx];

		// Base coherence checks────────────
		bool valid = pbuf.valid
			&& prev.M > 0
			&& prev.y.valid
			&& prev.W > 0.0f
			&& (float)dot(gbuf.normal, pbuf.normal) > 0.9f
			&& fabs(gbuf.depth - pbuf.depth) < 0.1f * gbuf.depth;

		if (valid) {
			// Shift mapping: Can we reconnect previous path to current viewer?
			vec3  d_prev     = prev.y.x_s - pbuf.pos;  // original connection
			vec3  d_new      = prev.y.x_s - gbuf.pos;  // new connection attempt
			float dist_prev  = (float)d_prev.length();
			float dist_new   = (float)d_new.length();
			
			if (dist_prev > 1e-6f && dist_new > 1e-6f) {
				// Compute reconnection jacobian
				float cos_prev = fmaxf(1e-6f, 
					(float)fabs(dot(prev.y.n_s, d_prev / dist_prev)));
				float cos_new = fmaxf(1e-6f, 
					(float)fabs(dot(prev.y.n_s, d_new / dist_new)));
				
				float jacobian = (cos_new * dist_prev * dist_prev) 
							   / (cos_prev * dist_new * dist_new + 1e-10f);
				
				// Clamp jacobian to detect extreme geometry changes (likely disocclusion)
				jacobian = fminf(jacobian, 5.0f);
				jacobian = fmaxf(jacobian, 0.2f);  // ← reject small jacobians (shift too extreme)
				
				// If jacobian is too extreme, treat as disocclusion (random replay)
				bool is_disoccluded = jacobian < 0.3f || jacobian > 3.0f;
				
				if (!is_disoccluded) {
				// Reconnection shift: safe to reuse
					GIReservoir prev_capped = prev;
					prev_capped.cap_M(GI_TEMPORAL_M_CAP);

					// Don't multiply by jacobian here — it's already in the importance!
					// Just use the base p_hat; jacobian was only for validity check
					float ph = gi_p_hat(prev_capped.y, gbuf, materials);
					combined.merge(prev_capped, ph, &rng);
					// Note: Don't finalize here! Finalization happens after spatial reuse
				}
				// else: disoccluded — skip temporal, use only current frame's samples
			}
		}
	}

	gi_out[idx] = combined;
	rng_states[idx] = rng;
}

__global__ void restir_gi_spatial_kernel(
	int W, int H,
	const GpuHittable*   hittables,  int n_hittables,
	const GpuMaterial*   materials,
	const GpuTriangle*   tris,
	const GpuTriBVHNode* tri_bvh,
	int tri_root, int n_tris,
	const GBufferPixel*  gbuffer,
	const GIReservoir*   input_res,
	GIReservoir*         output_res,
	curandState*         rng_states,
	int K_neighbors,
	int spatial_radius
) {
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= W || y >= H) return;

	int idx = y*W + x;
	curandState rng = rng_states[idx];

	const GBufferPixel& gbuf = gbuffer[idx];
	output_res[idx] = input_res[idx];

	if (!gbuf.valid) { rng_states[idx] = rng; return; }

	GIReservoir combined = input_res[idx];

	for (int k = 0; k < K_neighbors; ++k) {
		int nx = x + (int)((rand_double(&rng)*2.0-1.0) * spatial_radius);
		int ny = y + (int)((rand_double(&rng)*2.0-1.0) * spatial_radius);
		nx = max(0, min(nx, W-1));
		ny = max(0, min(ny, H-1));
		int nidx = ny*W + nx;

		const GBufferPixel& nbuf = gbuffer[nidx];
		if (!nbuf.valid) continue;
		if ((float)dot(gbuf.normal, nbuf.normal) < 0.9f) continue;
		if (fabs(gbuf.depth - nbuf.depth) > 0.5f * gbuf.depth) continue;

		const GIReservoir& nr = input_res[nidx];
		if (!nr.y.valid || nr.W <= 0.0f || nr.M == 0) continue;

		if (!is_visible(gbuf.pos, nr.y.x_s,
						hittables, n_hittables,
						tris, tri_bvh, tri_root, n_tris)) continue;

		// Don't multiply by jacobian — it's for shift mapping validity only
		float ph = gi_p_hat(nr.y, gbuf, materials);
		combined.merge(nr, ph, &rng);
	}

	if (combined.M > 0 && combined.y.valid) {
		float ph = gi_p_hat(combined.y, gbuf, materials);
		combined.finalize(ph);   // caps W and renormalizes w_sum
	}

	output_res[idx] = combined;
	rng_states[idx] = rng;
}