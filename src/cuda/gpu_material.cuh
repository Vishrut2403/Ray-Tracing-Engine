#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/onb.h"
#include "cuda/gpu_scene.cuh"
#include "cuda/gpu_hit.cuh"
#include "cuda/cuda_rand.cuh"

struct GpuBSDFSample {
    vec3   wi;
    vec3   f;
    double pdf;
    bool   is_delta;
};

__device__ inline vec3 gpu_emitted(const GpuMaterial& mat, bool front_face) {
    if (mat.type == MatType::DIFFUSE_LIGHT && front_face)
        return mat.albedo;
    return vec3(0,0,0);
}

__device__ inline vec3 gpu_f(const GpuMaterial& mat, const vec3& wi,
                               const vec3& normal) {
    if (mat.type == MatType::LAMBERTIAN) {
        if (dot(normal, wi) <= 0.0) return vec3(0,0,0);
        return mat.albedo * (1.0 / 3.1415926535897932385);
    }
    return vec3(0,0,0);
}

__device__ inline double gpu_pdf(const GpuMaterial& mat, const vec3& wi,
                                  const vec3& normal) {
    if (mat.type == MatType::LAMBERTIAN) {
        double c = dot(normal, wi);
        return c > 0.0 ? c / 3.1415926535897932385 : 0.0;
    }
    return 0.0;
}

__device__ inline GpuBSDFSample gpu_sample(const GpuMaterial& mat,
                                            const ray& wo,
                                            const GpuHitRecord& rec,
                                            curandState* rng) {
    constexpr double pi = 3.1415926535897932385;

    if (mat.type == MatType::LAMBERTIAN) {
        onb uvw; uvw.build_from_w(rec.normal);
        vec3   wi      = uvw.local(rand_cosine_direction(rng));
        double cosine  = dot(rec.normal, wi);
        double pdf_val = cosine > 0.0 ? cosine / pi : 0.0;
        return { wi, mat.albedo / pi, pdf_val, false };
    }

    if (mat.type == MatType::METAL) {
        vec3 reflected = reflect(unit_vector(wo.direction()), rec.normal);
        reflected = unit_vector(reflected + (double)mat.fuzz * rand_in_unit_sphere(rng));
        return { reflected, mat.albedo, 1.0, true };
    }

    if (mat.type == MatType::DIELECTRIC) {
        double ratio = rec.front_face ? (1.0 / (double)mat.ir) : (double)mat.ir;
        vec3   ud    = unit_vector(wo.direction());
        double cos_t = fmin(dot(-ud, rec.normal), 1.0);
        double sin_t = sqrt(1.0 - cos_t*cos_t);
        vec3 dir = (ratio*sin_t > 1.0 || reflectance(cos_t, ratio) > rand_double(rng))
                   ? reflect(ud, rec.normal)
                   : refract(ud, rec.normal, ratio);
        return { dir, vec3(1,1,1), 1.0, true };
    }

    return { vec3(0,0,0), vec3(0,0,0), 0.0, false };
}
