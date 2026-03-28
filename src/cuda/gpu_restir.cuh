#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/interval.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/gpu_hit.cuh"
#include "cuda/gpu_material.cuh"
#include "cuda/cuda_rand.cuh"

#ifndef RAY_OFFSET
#define RAY_OFFSET 0.02
#endif

struct LightSample {
    vec3  pos;
    vec3  normal;
    vec3  Le;
    float pdf;
    int   light_id;
};

struct Reservoir {
    LightSample y;
    float w_sum;
    int   M;
    float W;

    __device__ void init() {
        y.light_id = -1;
        y.pdf = 0.0f;
        y.Le  = vec3(0,0,0);
        w_sum = 0.0f;
        M     = 0;
        W     = 0.0f;
    }

    __device__ bool update(const LightSample& x, float w, curandState* rng) {
        w_sum += w;
        M     += 1;
        if (rand_double(rng) < (double)(w / (w_sum + 1e-30f))) {
            y = x;
            return true;
        }
        return false;
    }

    __device__ bool merge(const Reservoir& other, float p_hat, curandState* rng) {
        int M_new = M + other.M;
        bool sel = update(other.y, p_hat * other.W * (float)other.M, rng);
        M = M_new;
        return sel;
    }

    __device__ void cap_M(int max_M) {
        if (M > max_M) {
            w_sum *= (float)max_M / (float)M;
            M = max_M;
        }
    }
};

struct GBufferPixel {
    vec3  pos;
    vec3  normal;
    vec3  wo;
    vec3  Le;
    int   mat_id;
    float depth;
    bool  valid;
    bool  is_emitter;
};

__device__ inline LightSample sample_light(
    const GpuHittable* hittables,
    const GpuMaterial* materials,
    const int* light_ids, int n_lights,
    curandState* rng
) {
    LightSample s;
    s.light_id = -1;
    s.pdf      = 0.0f;
    s.Le       = vec3(0,0,0);
    s.pos      = vec3(0,0,0);
    s.normal   = vec3(0,1,0);

    if (n_lights == 0) return s;

    int li = min((int)(rand_double(rng) * n_lights), n_lights-1);
    const GpuHittable& lh = hittables[light_ids[li]];
    s.light_id = li;

    double rx = (double)lh.a0 + rand_double(rng)*(lh.a1-lh.a0);
    double rz = (double)lh.b0 + rand_double(rng)*(lh.b1-lh.b0);
    s.pos    = vec3(rx, (double)lh.k, rz) + lh.translate_offset;
    s.normal = lh.flip_face ? vec3(0,-1,0) : vec3(0,1,0);

    double area = (double)(lh.a1-lh.a0) * (double)(lh.b1-lh.b0);
    s.pdf = (float)(1.0 / (area * n_lights));

    s.Le = gpu_emitted(materials[lh.mat_id], true);
    return s;
}

__device__ inline float eval_p_hat(
    const LightSample& ls,
    const GBufferPixel& gbuf,
    const GpuMaterial* materials
) {
    if (ls.light_id < 0 || !gbuf.valid) return 0.0f;
    vec3  d    = ls.pos - gbuf.pos;
    float dist = (float)d.length();
    if (dist < 1e-6f) return 0.0f;

    vec3  wi    = d / dist;
    float cos_i = (float)fabs(dot(gbuf.normal, wi));
    float cos_l = (float)fabs(dot(ls.normal, -wi));
    float G     = cos_i * cos_l / (dist * dist);

    float lum = 0.2126f*(float)ls.Le.x()
              + 0.7152f*(float)ls.Le.y()
              + 0.0722f*(float)ls.Le.z();

    if (materials == nullptr)
        return fmaxf(0.0f, lum * G);

    vec3 f = gpu_f(materials[gbuf.mat_id], wi, gbuf.normal);
    float f_avg = (float)(f.x()+f.y()+f.z()) / 3.0f;
    return fmaxf(0.0f, lum * G * f_avg);
}

__device__ inline bool is_visible(
    const vec3& from, const vec3& to,
    const GpuHittable* hittables, int n_hittables,
    const GpuTriangle* tris, const GpuTriBVHNode* tri_bvh,
    int tri_root, int n_tris
) {
    vec3   d    = to - from;
    double dist = d.length();
    if (dist < 1e-6) return true;
    GpuHitRecord shadow;
    return !hit_scene(hittables, n_hittables,
                      ray(from, d/dist, 0.0),
                      interval(RAY_OFFSET, dist - RAY_OFFSET),
                      shadow, tris, tri_bvh, tri_root, n_tris);
}