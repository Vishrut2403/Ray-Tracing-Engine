#pragma once

#include "cuda/gpu_restir.cuh"
#include "cuda/gpu_integrator.cuh"
#include "cuda/gpu_volume.cuh"
#define RAY_OFFSET 0.02

#define TEMPORAL_M_CAP 20

__global__ void restir_initial_kernel(
    int W, int H,
    GpuCamera cam,
    const GpuHittable*   hittables,  int n_hittables,
    const GpuMaterial*   materials,
    const int*           light_ids,  int n_lights,
    const GpuTriangle*   tris,
    const GpuTriBVHNode* tri_bvh,
    int tri_root, int n_tris,
    GBufferPixel* gbuffer,
    Reservoir*    reservoirs,
    curandState*  rng_states,
    int M_candidates
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y*W + x;
    curandState rng = rng_states[idx];

    double u = (x + rand_double(&rng)) / (W-1);
    double v = (y + rand_double(&rng)) / (H-1);
    ray primary = gpu_get_ray(cam, u, v, &rng);

    GBufferPixel& gbuf = gbuffer[idx];
    gbuf.valid      = false;
    gbuf.is_emitter = false;
    gbuf.Le         = vec3(0,0,0);

    GpuHitRecord rec;
    if (hit_scene(hittables, n_hittables, primary,
                  interval(RAY_OFFSET, 1e30), rec,
                  tris, tri_bvh, tri_root, n_tris)) {
        vec3 Le = gpu_emitted(materials[rec.mat_id], rec.front_face);
        if (Le.length_squared() >= 1e-6) {
            gbuf.is_emitter = true;
            gbuf.Le         = Le;
        } else {
            gbuf.pos    = rec.p;
            gbuf.normal = rec.normal;
            gbuf.wo     = -unit_vector(primary.direction());
            gbuf.mat_id = rec.mat_id;
            gbuf.depth  = (float)rec.t;
            gbuf.valid  = true;
        }
    }

    Reservoir& res = reservoirs[idx];
    res.init();

    if (gbuf.valid && n_lights > 0) {
        for (int i = 0; i < M_candidates; ++i) {
            LightSample ls = sample_light(hittables, materials,
                                          light_ids, n_lights, &rng);
            float p_hat = eval_p_hat(ls, gbuf, materials);
            float w     = (ls.pdf > 1e-10f) ? p_hat / ls.pdf : 0.0f;
            res.update(ls, w, &rng);
        }
        if (res.M > 0 && res.y.light_id >= 0) {
            float p_hat = eval_p_hat(res.y, gbuf, materials);
            res.W = (p_hat > 1e-10f) ? (res.w_sum / (float)res.M) / p_hat : 0.0f;
        }
    }

    rng_states[idx] = rng;
}

__global__ void restir_temporal_kernel(
    int W, int H,
    const GpuMaterial*   materials,
    const GBufferPixel*  gbuffer_cur,
    const GBufferPixel*  gbuffer_prev,
    const Reservoir*     res_cur,
    const Reservoir*     res_prev,
    Reservoir*           res_out,
    curandState*         rng_states,
    bool                 has_prev_frame
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y*W + x;
    curandState rng = rng_states[idx];

    Reservoir combined = res_cur[idx];
    const GBufferPixel& gbuf = gbuffer_cur[idx];

    if (has_prev_frame && gbuf.valid) {
        int prev_idx = idx;

        const GBufferPixel& pbuf = gbuffer_prev[prev_idx];
        const Reservoir&    prev = res_prev[prev_idx];

        bool valid_temporal = pbuf.valid
            && prev.M > 0
            && prev.y.light_id >= 0
            && prev.W > 0.0f;

        if (valid_temporal)
            valid_temporal = (float)dot(gbuf.normal, pbuf.normal) > 0.9f;

        if (valid_temporal)
            valid_temporal = fabs(gbuf.depth - pbuf.depth) < 0.1f * gbuf.depth;

        if (valid_temporal) {
            Reservoir prev_capped = prev;
            prev_capped.cap_M(TEMPORAL_M_CAP);

            float ph = eval_p_hat(prev_capped.y, gbuf, materials);
            combined.merge(prev_capped, ph, &rng);

            if (combined.M > 0 && combined.y.light_id >= 0) {
                float ph2 = eval_p_hat(combined.y, gbuf, materials);
                combined.W = (ph2 > 1e-10f)
                             ? (combined.w_sum / (float)combined.M) / ph2
                             : 0.0f;
            }
        }
    }

    res_out[idx] = combined;
    rng_states[idx] = rng;
}

__global__ void restir_spatial_kernel(
    int W, int H,
    const GpuHittable*   hittables,  int n_hittables,
    const GpuMaterial*   materials,
    const int*           light_ids,  int n_lights,
    const GpuTriangle*   tris,
    const GpuTriBVHNode* tri_bvh,
    int tri_root, int n_tris,
    const GBufferPixel* gbuffer,
    const Reservoir*    input_res,
    Reservoir*          output_res,
    curandState*        rng_states,
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

    Reservoir combined = input_res[idx];

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

        const Reservoir& nr = input_res[nidx];
        if (nr.y.light_id < 0 || nr.W <= 0.0f) continue;

        if (!is_visible(gbuf.pos, nr.y.pos,
                        hittables, n_hittables,
                        tris, tri_bvh, tri_root, n_tris)) continue;

        float p_hat = eval_p_hat(nr.y, gbuf, materials);
        combined.merge(nr, p_hat, &rng);
    }

    if (combined.M > 0 && combined.y.light_id >= 0) {
        float p_hat = eval_p_hat(combined.y, gbuf, materials);
        combined.W = (p_hat > 1e-10f)
                     ? (combined.w_sum / (float)combined.M) / p_hat
                     : 0.0f;
    }

    output_res[idx] = combined;
    rng_states[idx] = rng;
}

__global__ void restir_shade_kernel(
    int W, int H,
    float* d_accum,
    GpuCamera cam,
    vec3 background,
    const GpuHittable*   hittables,  int n_hittables,
    const GpuMaterial*   materials,
    const int*           light_ids,  int n_lights,
    const GpuTriangle*   tris,
    const GpuTriBVHNode* tri_bvh,
    int tri_root, int n_tris,
    const GBufferPixel* gbuffer,
    const Reservoir*    reservoirs,
    curandState*        rng_states,
    int max_depth,
    const GpuEnvMap*    env_map,
    GpuMedium           medium     
) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int idx = y*W + x;
    curandState rng = rng_states[idx];

    vec3 L(0,0,0);
    const GBufferPixel& gbuf = gbuffer[idx];

    if (gbuf.is_emitter) {
        vec3 emit = gbuf.Le;
        if (medium.active)
            emit = emit * transmittance(medium, (double)gbuf.depth);
        d_accum[idx*3+0] += (float)emit.x();
        d_accum[idx*3+1] += (float)emit.y();
        d_accum[idx*3+2] += (float)emit.z();
        rng_states[idx] = rng;
        return;
    }

    if (!gbuf.valid) {
        double u = (x + 0.5) / (W-1);
        double v = (y + 0.5) / (H-1);
        ray r = gpu_get_ray(cam, u, v, &rng);
        if (env_map && env_map->valid)
            L = gpu_env_Le(*env_map, r.direction());
        else
            L = background;

    } else {
        const GpuMaterial& mat = materials[gbuf.mat_id];
        const Reservoir&   res = reservoirs[idx];

        vec3 tr_primary = medium.active
            ? transmittance(medium, (double)gbuf.depth)
            : vec3(1,1,1);

        if (res.y.light_id >= 0 && res.W > 0.0f) {
            vec3  d    = res.y.pos - gbuf.pos;
            float dist = (float)d.length();
            if (dist > 1e-6f) {
                vec3 wi = d / dist;
                if (is_visible(gbuf.pos, res.y.pos,
                               hittables, n_hittables,
                               tris, tri_bvh, tri_root, n_tris)) {
                    vec3  f     = gpu_f(mat, wi, gbuf.normal);
                    float cos_i = (float)fabs(dot(gbuf.normal, wi));
                    float cos_l = (float)fabs(dot(res.y.normal, -(wi)));
                    float G     = cos_i * cos_l / (dist * dist);

                    vec3 tr_shadow = medium.active
                        ? transmittance(medium, (double)dist)
                        : vec3(1,1,1);

                    L = L + tr_primary * tr_shadow
                          * f * res.y.Le * (double)(G * res.W);
                }
            }
        }

        GpuHitRecord gbuf_rec{};
        gbuf_rec.p          = gbuf.pos;
        gbuf_rec.normal     = gbuf.normal;
        gbuf_rec.mat_id     = gbuf.mat_id;
        gbuf_rec.front_face = true;
        gbuf_rec.u = gbuf_rec.v = 0.0;
        gbuf_rec.t = (double)gbuf.depth;

        GpuBSDFSample bs = gpu_sample(mat,
            ray(gbuf.pos - gbuf.wo*(double)RAY_OFFSET, gbuf.wo, 0.0),
            gbuf_rec, &rng);

        if (bs.pdf > 0.0) {
            double cos_t = fabs(dot(gbuf.normal, bs.wi));
            vec3 beta = bs.is_delta
                        ? bs.f
                        : bs.f * cos_t / bs.pdf;
            beta = beta * tr_primary;

            ray indirect_ray(gbuf.pos, bs.wi, 0.0);
            GpuHitRecord irec;
            bool hit_surface = hit_scene(hittables, n_hittables,
                                         indirect_ray,
                                         interval(RAY_OFFSET, 1e30), irec,
                                         tris, tri_bvh, tri_root, n_tris);
            double t_surface = hit_surface ? irec.t : 1e30;

            MediumSample ms = sample_medium(medium, indirect_ray,
                                            t_surface, &rng);
            beta = beta * ms.weight;

            if (ms.scattered) {
                if (n_lights > 0) {
                    vec3 to_l = gpu_light_random(hittables, light_ids,
                                                  n_lights, ms.pos, &rng);
                    double dl = to_l.length();
                    if (dl > 1e-6) {
                        vec3 wl = to_l / dl;
                        double lpdf = gpu_light_pdf(hittables, light_ids,
                                                     n_lights, ms.pos, wl);
                        if (lpdf > 0.0) {
                            GpuHitRecord sr;
                            bool occ = hit_scene(hittables, n_hittables,
                                ray(ms.pos, wl, 0.0),
                                interval(RAY_OFFSET, dl - RAY_OFFSET),
                                sr, tris, tri_bvh, tri_root, n_tris);
                            if (!occ) {
                                GpuHitRecord lr;
                                if (hit_scene(hittables, n_hittables,
                                    ray(ms.pos, wl, 0.0),
                                    interval(RAY_OFFSET, 1e30), lr,
                                    tris, tri_bvh, tri_root, n_tris)) {
                                    vec3 lLe = gpu_emitted(
                                        materials[lr.mat_id], lr.front_face);
                                    if (lLe.length_squared() > 0.0) {
                                        float cos_s = (float)dot(
                                            unit_vector(ms.wi),
                                            unit_vector(wl));
                                        float phase = hg_phase(cos_s, medium.g);
                                        vec3 tr_light = medium.active
                                            ? transmittance(medium, dl)
                                            : vec3(1,1,1);
                                        L = L + beta * lLe * tr_light
                                                  * (double)(phase / (float)lpdf);
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (hit_surface) {
                vec3 iLe = gpu_emitted(materials[irec.mat_id], irec.front_face);
                if (iLe.length_squared() > 0.0) {
                    L = L + beta * iLe;
                } else if (!bs.is_delta && n_lights > 0) {
                    vec3 to_l = gpu_light_random(hittables, light_ids,
                                                  n_lights, irec.p, &rng);
                    double dl = to_l.length();
                    if (dl > 1e-6) {
                        vec3 wl = to_l / dl;
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
                                        vec3   if_ = gpu_f(materials[irec.mat_id],
                                                            wl, irec.normal);
                                        double ct2 = fabs(dot(irec.normal, wl));
                                        vec3 tr_light = medium.active
                                            ? transmittance(medium, dl)
                                            : vec3(1,1,1);
                                        L = L + beta * if_ * lLe
                                                  * tr_light * (ct2/lpdf);
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (env_map && env_map->valid) {
                L = L + beta * gpu_env_Le(*env_map, bs.wi);
            } else {
                L = L + beta * background;
            }
        }
    }

    double lum = 0.2126*L.x() + 0.7152*L.y() + 0.0722*L.z();
    if (lum > 50.0) L = L * (50.0/lum);

    d_accum[idx*3+0] += (float)L.x();
    d_accum[idx*3+1] += (float)L.y();
    d_accum[idx*3+2] += (float)L.z();
    rng_states[idx] = rng;
}