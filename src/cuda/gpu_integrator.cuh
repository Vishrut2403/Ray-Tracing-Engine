#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/interval.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/gpu_hit.cuh"
#include "cuda/gpu_material.cuh"
#include "cuda/cuda_rand.cuh"

constexpr double GPU_PI = 3.1415926535897932385;

__device__ inline double gpu_power_heuristic(double a, double b) {
    double a2 = a*a, b2 = b*b;
    return a2 / (a2 + b2 + 1e-12);
}

__device__ inline double gpu_light_pdf(const GpuHittable* hittables,
                                        const int* light_ids, int n_lights,
                                        const vec3& origin, const vec3& dir) {
    if (n_lights == 0) return 0.0;
    double sum = 0.0;
    double w   = 1.0 / n_lights;
    for (int i = 0; i < n_lights; ++i) {
        const GpuHittable& lh = hittables[light_ids[i]];
        GpuHitRecord tmp;
        ray test_ray(origin, dir, 0.0);
        ray local = (lh.has_rotation || lh.translate_offset.length_squared() > 0.0)
                    ? apply_inverse_transform(test_ray, lh)
                    : test_ray;
        if (hit_xz_rect(lh, local, interval(0.001, 1e30), tmp)) {
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
    int idx = (int)(rand_double(rng) * n_lights);
    idx = min(idx, n_lights-1);
    const GpuHittable& lh = hittables[light_ids[idx]];
    double rx = (double)lh.a0 + rand_double(rng) * (double)(lh.a1-lh.a0);
    double rz = (double)lh.b0 + rand_double(rng) * (double)(lh.b1-lh.b0);
    vec3 pt(rx, (double)lh.k, rz);
    pt = pt + lh.translate_offset;
    return pt - origin;
}

__device__ vec3 gpu_Li(const ray& initial_ray,
                        const vec3& background,
                        const GpuHittable* hittables, int n_hittables,
                        const GpuMaterial* materials,
                        const int* light_ids, int n_lights,
                        int max_depth,
                        curandState* rng) {
    ray   r    = initial_ray;
    vec3  L    (0,0,0);
    vec3  beta (1,1,1);
    bool  specular_bounce = false;
    double last_bsdf_pdf  = 0.0;

    for (int depth = 0; depth < max_depth; ++depth) {

        GpuHitRecord rec;
        if (!hit_scene(hittables, n_hittables, r, interval(0.001, 1e30), rec)) {
            if (specular_bounce || n_lights == 0) {
                L = L + beta * background;
            } else {
                double lp = gpu_light_pdf(hittables, light_ids, n_lights,
                                           rec.p, r.direction());
                L = L + beta * background * gpu_power_heuristic(last_bsdf_pdf, lp);
            }
            break;
        }

        const GpuMaterial& mat = materials[rec.mat_id];

        vec3 emitted = gpu_emitted(mat, rec.front_face);
        if (emitted.length_squared() > 0.0) {
            if (specular_bounce) {
                L = L + beta * emitted;
            } else {
                double lp = gpu_light_pdf(hittables, light_ids, n_lights,
                                           rec.p, r.direction());
                L = L + beta * emitted * gpu_power_heuristic(last_bsdf_pdf, lp);
            }
        }

        if (!specular_bounce && n_lights > 0) {
            vec3   to_light  = gpu_light_random(hittables, light_ids, n_lights, rec.p, rng);
            double distance  = to_light.length();
            vec3   wi        = to_light / distance;
            double light_pdf = gpu_light_pdf(hittables, light_ids, n_lights, rec.p, wi);

            if (light_pdf > 0.0) {
                GpuHitRecord shadow_rec;
                bool occluded = hit_scene(hittables, n_hittables,
                                          ray(rec.p, wi, r.time()),
                                          interval(0.001, distance - 1e-4),
                                          shadow_rec);
                if (!occluded) {
                    GpuHitRecord light_rec;
                    if (hit_scene(hittables, n_hittables,
                                  ray(rec.p, wi, r.time()),
                                  interval(0.001, 1e30), light_rec)) {
                        vec3 Le = gpu_emitted(materials[light_rec.mat_id],
                                              light_rec.front_face);
                        if (Le.length_squared() > 0.0) {
                            vec3   f         = gpu_f(mat, wi, rec.normal);
                            double bsdf_pdf  = gpu_pdf(mat, wi, rec.normal);
                            double weight    = gpu_power_heuristic(light_pdf, bsdf_pdf);
                            double cos_theta = fabs(dot(rec.normal, wi));
                            L = L + beta * f * Le * (cos_theta * weight / light_pdf);
                        }
                    }
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

        r = ray(rec.p, bs.wi, r.time());
    }

    return L;
}

struct GpuCamera {
    vec3   origin;
    vec3   lower_left;
    vec3   horizontal;
    vec3   vertical;
    vec3   u, v, w;
    double lens_radius;
};

__device__ inline ray gpu_get_ray(const GpuCamera& cam, double s, double t,
                                   curandState* rng) {
    vec3 rd     = cam.lens_radius * rand_in_unit_sphere(rng);
    rd[2] = 0.0;
    vec3 offset = cam.u * rd.x() + cam.v * rd.y();
    return ray(
        cam.origin + offset,
        cam.lower_left + s*cam.horizontal + t*cam.vertical - cam.origin - offset,
        0.0
    );
}

__global__ void render_kernel(
    float*              framebuffer,
    int                 width,
    int                 height,
    int                 spp,
    int                 max_depth,
    GpuCamera           cam,
    vec3                background,
    const GpuHittable*  hittables,
    int                 n_hittables,
    const GpuMaterial*  materials,
    const int*          light_ids,
    int                 n_lights,
    curandState*        rand_states
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int         id  = y * width + x;
    curandState rng = rand_states[id];

    vec3 pixel(0,0,0);
    for (int s = 0; s < spp; ++s) {
        double u = (x + rand_double(&rng)) / (width  - 1);
        double v = (y + rand_double(&rng)) / (height - 1);
        ray r    = gpu_get_ray(cam, u, v, &rng);
        pixel    = pixel + gpu_Li(r, background,
                                   hittables, n_hittables,
                                   materials,
                                   light_ids, n_lights,
                                   max_depth, &rng);
    }

    pixel = pixel / (double)spp;

    framebuffer[id*3 + 0] = (float)pixel.x();
    framebuffer[id*3 + 1] = (float)pixel.y();
    framebuffer[id*3 + 2] = (float)pixel.z();

    rand_states[id] = rng;
}
