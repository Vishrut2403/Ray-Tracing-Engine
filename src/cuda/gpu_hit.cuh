#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/interval.h"
#include "cuda/gpu_scene.cuh"

struct GpuHitRecord {
    vec3   p;
    vec3   normal;
    double t;
    double u, v;
    int    mat_id;
    bool   front_face;
};

__device__ inline void set_face_normal(GpuHitRecord& rec,
                                        const ray& r,
                                        const vec3& outward) {
    rec.front_face = dot(r.direction(), outward) < 0.0;
    rec.normal     = rec.front_face ? outward : -outward;
}

__device__ inline ray apply_inverse_transform(const ray& r, const GpuHittable& h) {
    vec3 orig = r.origin() - h.translate_offset;
    vec3 dir  = r.direction();
    if (h.has_rotation) {
        vec3 o2, d2;
        o2[0] =  h.cos_theta * orig[0] - h.sin_theta * orig[2];
        o2[1] =  orig[1];
        o2[2] =  h.sin_theta * orig[0] + h.cos_theta * orig[2];
        d2[0] =  h.cos_theta * dir[0]  - h.sin_theta * dir[2];
        d2[1] =  dir[1];
        d2[2] =  h.sin_theta * dir[0]  + h.cos_theta * dir[2];
        return ray(o2, d2, r.time());
    }
    return ray(orig, dir, r.time());
}

__device__ inline void apply_forward_transform(GpuHitRecord& rec,
                                                const ray& world_r,
                                                const GpuHittable& h) {
    vec3 p = rec.p, n = rec.normal;
    if (h.has_rotation) {
        vec3 p2, n2;
        p2[0] =  h.cos_theta * p[0] + h.sin_theta * p[2];
        p2[1] =  p[1];
        p2[2] = -h.sin_theta * p[0] + h.cos_theta * p[2];
        n2[0] =  h.cos_theta * n[0] + h.sin_theta * n[2];
        n2[1] =  n[1];
        n2[2] = -h.sin_theta * n[0] + h.cos_theta * n[2];
        p = p2; n = n2;
    }
    rec.p = p + h.translate_offset;
    set_face_normal(rec, world_r, n);
}

__device__ inline bool hit_sphere(const GpuHittable& h, const ray& r,
                                   interval ray_t, GpuHitRecord& rec) {
    vec3   oc   = r.origin() - h.center;
    double a    = r.direction().length_squared();
    double hb   = dot(oc, r.direction());
    double c    = oc.length_squared() - (double)h.radius * h.radius;
    double disc = hb*hb - a*c;
    if (disc < 0.0) return false;
    double sqrtd = sqrt(disc);
    double root  = (-hb - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (-hb + sqrtd) / a;
        if (!ray_t.surrounds(root)) return false;
    }
    rec.t = root;
    rec.p = r.at(rec.t);
    set_face_normal(rec, r, (rec.p - h.center) / (double)h.radius);
    rec.mat_id = h.mat_id;
    return true;
}

__device__ inline bool hit_xz_rect(const GpuHittable& h, const ray& r,
                                    interval ray_t, GpuHitRecord& rec) {
    double t = ((double)h.k - r.origin().y()) / r.direction().y();
    if (!ray_t.surrounds(t)) return false;
    double x = r.origin().x() + t * r.direction().x();
    double z = r.origin().z() + t * r.direction().z();
    if (x < h.a0 || x > h.a1 || z < h.b0 || z > h.b1) return false;
    rec.t = t; rec.u = (x-h.a0)/(h.a1-h.a0); rec.v = (z-h.b0)/(h.b1-h.b0);
    rec.p = r.at(t); rec.mat_id = h.mat_id;
    vec3 outward(0,1,0); if (h.flip_face) outward = -outward;
    set_face_normal(rec, r, outward);
    return true;
}

__device__ inline bool hit_xy_rect(const GpuHittable& h, const ray& r,
                                    interval ray_t, GpuHitRecord& rec) {
    double t = ((double)h.k - r.origin().z()) / r.direction().z();
    if (!ray_t.surrounds(t)) return false;
    double x = r.origin().x() + t * r.direction().x();
    double y = r.origin().y() + t * r.direction().y();
    if (x < h.a0 || x > h.a1 || y < h.b0 || y > h.b1) return false;
    rec.t = t; rec.u = (x-h.a0)/(h.a1-h.a0); rec.v = (y-h.b0)/(h.b1-h.b0);
    rec.p = r.at(t); rec.mat_id = h.mat_id;
    vec3 outward(0,0,1); if (h.flip_face) outward = -outward;
    set_face_normal(rec, r, outward);
    return true;
}

__device__ inline bool hit_yz_rect(const GpuHittable& h, const ray& r,
                                    interval ray_t, GpuHitRecord& rec) {
    double t = ((double)h.k - r.origin().x()) / r.direction().x();
    if (!ray_t.surrounds(t)) return false;
    double y = r.origin().y() + t * r.direction().y();
    double z = r.origin().z() + t * r.direction().z();
    if (y < h.a0 || y > h.a1 || z < h.b0 || z > h.b1) return false;
    rec.t = t; rec.u = (y-h.a0)/(h.a1-h.a0); rec.v = (z-h.b0)/(h.b1-h.b0);
    rec.p = r.at(t); rec.mat_id = h.mat_id;
    vec3 outward(1,0,0); if (h.flip_face) outward = -outward;
    set_face_normal(rec, r, outward);
    return true;
}

__device__ inline bool hit_triangle(const GpuTriangle& tri, const ray& r,
                                     interval ray_t, GpuHitRecord& rec) {
    const double eps = 1e-8;
    vec3 e1 = tri.v1 - tri.v0;
    vec3 e2 = tri.v2 - tri.v0;
    vec3 h  = cross(r.direction(), e2);
    double a = dot(e1, h);
    if (fabs(a) < eps) return false;

    double f = 1.0 / a;
    vec3   s = r.origin() - tri.v0;
    double u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) return false;

    vec3   q = cross(s, e1);
    double v = f * dot(r.direction(), q);
    if (v < 0.0 || u + v > 1.0) return false;

    double t = f * dot(e2, q);
    if (!ray_t.surrounds(t)) return false;

    double w = 1.0 - u - v;
    vec3 smooth_n = unit_vector(tri.n0*w + tri.n1*u + tri.n2*v);
    if (dot(smooth_n, r.direction()) > 0.0) smooth_n = -smooth_n;
    rec.t      = t;
    rec.p      = r.at(t);
    rec.u      = u;
    rec.v      = v;
    rec.mat_id = tri.mat_id;
    rec.normal = smooth_n;
    rec.front_face = true;
    return true;
}

__device__ inline bool hit_aabb(const float* mn, const float* mx,
                                  const ray& r, float t_min, float t_max) {
    for (int i = 0; i < 3; ++i) {
        float inv = 1.0f / (float)r.direction()[i];
        float t0  = ((float)mn[i] - (float)r.origin()[i]) * inv;
        float t1  = ((float)mx[i] - (float)r.origin()[i]) * inv;
        if (inv < 0.0f) { float tmp=t0; t0=t1; t1=tmp; }
        t_min = fmaxf(t_min, t0);
        t_max = fminf(t_max, t1);
        if (t_max <= t_min) return false;
    }
    return true;
}

__device__ inline bool hit_tri_bvh(
    const GpuTriBVHNode* bvh, int root,
    const GpuTriangle* tris,
    const ray& r, interval ray_t,
    GpuHitRecord& rec
) {
    int stack[64];
    int stack_top = 0;
    stack[stack_top++] = root;

    bool hit_any = false;
    double closest = ray_t.max;

    while (stack_top > 0) {
        int node_idx = stack[--stack_top];
        if (node_idx < 0) continue;

        const GpuTriBVHNode& node = bvh[node_idx];

        if (!hit_aabb(node.aabb_min, node.aabb_max, r,
                      (float)ray_t.min, (float)closest))
            continue;

        bool is_leaf = (node.left == node.right);

        if (is_leaf) {
            int tri_idx = -(node.left + 1);
            if (tri_idx >= 0) {
                GpuHitRecord tmp;
                if (hit_triangle(tris[tri_idx], r, interval(ray_t.min, closest), tmp)) {
                    hit_any = true;
                    closest = tmp.t;
                    rec     = tmp;
                }
            }
        } else {
            if (stack_top < 62) {
                stack[stack_top++] = node.right;
                stack[stack_top++] = node.left;
            }
        }
    }
    return hit_any;
}

__device__ inline bool hit_scene(
    const GpuHittable* hittables, int n_hittables,
    const ray& r, interval ray_t,
    GpuHitRecord& rec,
    const GpuTriangle*   tris     = nullptr,
    const GpuTriBVHNode* tri_bvh  = nullptr,
    int                  tri_root = 0,
    int                  n_tris   = 0
) {
    GpuHitRecord tmp;
    bool   hit_any = false;
    double closest = ray_t.max;

    for (int i = 0; i < n_hittables; ++i) {
        const GpuHittable& h = hittables[i];
        ray local_r = (h.has_rotation || h.translate_offset.length_squared() > 0.0)
                      ? apply_inverse_transform(r, h) : r;

        bool hit = false;
        switch (h.type) {
            case HitType::SPHERE:   hit = hit_sphere  (h, local_r, interval(ray_t.min, closest), tmp); break;
            case HitType::XZ_RECT: hit = hit_xz_rect (h, local_r, interval(ray_t.min, closest), tmp); break;
            case HitType::XY_RECT: hit = hit_xy_rect (h, local_r, interval(ray_t.min, closest), tmp); break;
            case HitType::YZ_RECT: hit = hit_yz_rect (h, local_r, interval(ray_t.min, closest), tmp); break;
        }
        if (hit) {
            hit_any = true; closest = tmp.t; rec = tmp;
            if (h.has_rotation || h.translate_offset.length_squared() > 0.0)
                apply_forward_transform(rec, r, h);
        }
    }

    if (tris && tri_bvh && n_tris > 0) {
        if (hit_tri_bvh(tri_bvh, tri_root, tris, r,
                        interval(ray_t.min, closest), tmp)) {
            hit_any = true; closest = tmp.t; rec = tmp;
        }
    }

    return hit_any;
}