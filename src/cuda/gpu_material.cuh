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

constexpr double GPU_PI = 3.1415926535897932385;

__device__ inline vec3 gpu_emitted(const GpuMaterial& mat, bool front_face) {
    if (mat.type == MatType::DIFFUSE_LIGHT && front_face) return mat.albedo;
    return vec3(0,0,0);
}

__device__ inline double ggx_D(double ndoth, double a) {
    double a2 = a*a, d = ndoth*ndoth*(a2-1.0)+1.0;
    return a2 / (GPU_PI * d * d);
}

__device__ inline double ggx_G1(double ndotv, double a) {
    double a2 = a*a;
    return 2.0*ndotv / (ndotv + sqrt(a2 + (1.0-a2)*ndotv*ndotv));
}

__device__ inline double ggx_G2(double ndotv, double ndotl, double a) {
    double a2 = a*a;
    double gv = ndotl * sqrt(a2 + (1.0-a2)*ndotv*ndotv);
    double gl = ndotv * sqrt(a2 + (1.0-a2)*ndotl*ndotl);
    return 2.0*ndotv*ndotl / (gv + gl + 1e-7);
}

__device__ inline vec3 ggx_F(const vec3& f0, double vdoth) {
    return f0 + (vec3(1,1,1)-f0) * pow(1.0-vdoth, 5.0);
}

__device__ inline vec3 ggx_eval(const GpuMaterial& mat,
                                 const vec3& v, const vec3& l, const vec3& n) {
    if (dot(n, v) <= 0.0 || dot(n, l) <= 0.0) return vec3(0,0,0);

    double a     = (double)mat.roughness * (double)mat.roughness;
    vec3   h     = unit_vector(v + l);
    double ndotv = fmax(dot(n,v), 1e-7);
    double ndotl = fmax(dot(n,l), 1e-7);
    double ndoth = fmax(dot(n,h), 1e-7);
    double vdoth = fmax(dot(v,h), 1e-7);

    vec3 f0    = vec3(0.04,0.04,0.04)*(1.0-(double)mat.metallic)
               + mat.albedo*(double)mat.metallic;
    vec3   Fval   = ggx_F(f0, vdoth);
    double Dval   = ggx_D(ndoth, a);
    double Gval   = ggx_G2(ndotv, ndotl, a);

    vec3 specular = Fval * (Dval * Gval / (4.0 * ndotv * ndotl));
    vec3 diffuse  = mat.albedo / GPU_PI * (1.0-(double)mat.metallic)
                  * (vec3(1,1,1)-Fval);

    return (specular + diffuse) * ndotl;
}

__device__ inline vec3 ggx_sample_vndf(const vec3& wo, double a, curandState* rng) {
    vec3 vh = unit_vector(vec3(a*wo.x(), a*wo.y(), wo.z()));

    double lensq = vh.x()*vh.x() + vh.y()*vh.y();
    vec3 t1 = lensq > 0.0
              ? vec3(-vh.y(), vh.x(), 0.0) / sqrt(lensq)
              : vec3(1,0,0);
    vec3 t2 = cross(vh, t1);

    double r   = sqrt(rand_double(rng));
    double phi = 2.0 * GPU_PI * rand_double(rng);
    double t   = r * cos(phi);
    double s   = r * sin(phi);
    double bz  = sqrt(fmax(0.0, 1.0 - t*t - s*s));

    vec3 nh = t*t1 + s*t2 + bz*vh;
    return unit_vector(vec3(a*nh.x(), a*nh.y(), fmax(0.0, nh.z())));
}

__device__ inline double ggx_vndf_pdf(const vec3& wo, const vec3& h, double a) {
    double ndotwo = fmax(wo.z(), 1e-7);
    double hdotwo = fmax(dot(h, wo), 1e-7);
    double ndoth  = fmax(h.z(),  1e-7);
    return ggx_D(ndoth, a) * ggx_G1(ndotwo, a) * hdotwo / (4.0 * ndotwo * hdotwo);
}

__device__ inline vec3 gpu_f(const GpuMaterial& mat,
                               const vec3& wi, const vec3& normal) {
    if (mat.type == MatType::LAMBERTIAN) {
        if (dot(normal, wi) <= 0.0) return vec3(0,0,0);
        return mat.albedo / GPU_PI;
    }
    if (mat.type == MatType::GGX) {
        if (dot(normal, wi) <= 0.0) return vec3(0,0,0);
        return ggx_eval(mat, normal, wi, normal);
    }
    return vec3(0,0,0);
}

__device__ inline double gpu_pdf(const GpuMaterial& mat,
                                  const vec3& wi, const vec3& normal) {
    if (mat.type == MatType::LAMBERTIAN) {
        double c = dot(normal, wi);
        return c > 0.0 ? c / GPU_PI : 0.0;
    }
    if (mat.type == MatType::GGX) {
        double c = dot(normal, wi);
        return c > 0.0 ? c / GPU_PI : 0.0;
    }
    return 0.0;
}

__device__ inline GpuBSDFSample gpu_sample(const GpuMaterial& mat,
                                            const ray& wo,
                                            const GpuHitRecord& rec,
                                            curandState* rng) {
    GpuBSDFSample bs{};

    if (mat.type == MatType::LAMBERTIAN) {
        onb uvw; uvw.build_from_w(rec.normal);
        vec3   wi      = uvw.local(rand_cosine_direction(rng));
        double cosine  = dot(rec.normal, wi);
        double pdf_val = cosine > 0.0 ? cosine / GPU_PI : 0.0;
        return { wi, mat.albedo / GPU_PI, pdf_val, false };
    }

    if (mat.type == MatType::GGX) {
        double alpha = (double)mat.roughness * (double)mat.roughness;
        onb uvw; uvw.build_from_w(rec.normal);

        vec3 wo_l = -unit_vector(vec3(
            dot(wo.direction(), uvw.u()),
            dot(wo.direction(), uvw.v()),
            dot(wo.direction(), uvw.w())
        ));

        if (wo_l.z() <= 0.0) return bs;

        vec3 h_l = ggx_sample_vndf(wo_l, alpha, rng);
        vec3 h   = unit_vector(uvw.local(h_l));
        vec3 wi  = unit_vector(reflect(-unit_vector(wo.direction()), h));

        if (dot(wi, rec.normal) <= 0.0) return bs;

        double p = ggx_vndf_pdf(wo_l, h_l, alpha);
        if (p <= 0.0) return bs;

        vec3 v = -unit_vector(wo.direction());
        vec3 f = ggx_eval(mat, v, wi, rec.normal);
        return { wi, f, p, false };
    }

    if (mat.type == MatType::METAL) {
        vec3 r = reflect(unit_vector(wo.direction()), rec.normal);
        r = unit_vector(r + (double)mat.fuzz * rand_in_unit_sphere(rng));
        return { r, mat.albedo, 1.0, true };
    }

    if (mat.type == MatType::DIELECTRIC) {
        double ratio = rec.front_face ? (1.0/(double)mat.ir) : (double)mat.ir;
        vec3   ud    = unit_vector(wo.direction());
        double cos_t = fmin(dot(-ud, rec.normal), 1.0);
        double sin_t = sqrt(1.0 - cos_t*cos_t);
        vec3 dir = (ratio*sin_t > 1.0 || reflectance(cos_t, ratio) > rand_double(rng))
                   ? reflect(ud, rec.normal)
                   : refract(ud, rec.normal, ratio);
        return { dir, vec3(1,1,1), 1.0, true };
    }

    return bs;
}