#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/interval.h"
#include "core/onb.h"

// ── Material ─────────────────────────────────────────────────────────────────

enum class MatType : int {
    LAMBERTIAN = 0,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT,
    ISOTROPIC
};

struct GpuMaterial {
    MatType type;
    vec3    albedo;     // lambertian / metal / isotropic / light color
    float   fuzz;       // metal only
    float   ir;         // dielectric only
};

// ── Hittable ─────────────────────────────────────────────────────────────────

enum class HitType : int {
    SPHERE = 0,
    XZ_RECT,
    XY_RECT,
    YZ_RECT
};

struct GpuHittable {
    HitType type;
    int     mat_id;     // index into GpuScene::materials[]
    bool    flip_face;  // absorbs flip_face wrapper

    // Sphere
    vec3   center;
    float  radius;

    // Rect (xz / xy / yz)
    float  a0, a1;      // first axis range
    float  b0, b1;      // second axis range
    float  k;           // fixed axis value

    // Transform baked in at build time — translate offset + rotate_y
    vec3   translate_offset;
    float  sin_theta;
    float  cos_theta;
    bool   has_rotation;
};

// ── BVH node (linear, for GPU traversal) ─────────────────────────────────────

struct GpuBVHNode {
    vec3  aabb_min;
    vec3  aabb_max;
    int   left;         // index into nodes[] or -(prim_id+1) if leaf
    int   right;
};

// ── Full scene ────────────────────────────────────────────────────────────────

struct GpuScene {
    // Host-side flat arrays (built by SceneUploader)
    GpuMaterial*  materials   = nullptr;
    GpuHittable*  hittables   = nullptr;
    GpuBVHNode*   bvh_nodes   = nullptr;
    int*          light_ids   = nullptr;   // indices of emissive hittables

    int n_materials  = 0;
    int n_hittables  = 0;
    int n_bvh_nodes  = 0;
    int n_lights     = 0;

    // Device-side mirrors (cudaMalloc'd)
    GpuMaterial*  d_materials  = nullptr;
    GpuHittable*  d_hittables  = nullptr;
    GpuBVHNode*   d_bvh_nodes  = nullptr;
    int*          d_light_ids  = nullptr;

    void free_device() {
        if (d_materials) { cudaFree(d_materials); d_materials = nullptr; }
        if (d_hittables) { cudaFree(d_hittables); d_hittables = nullptr; }
        if (d_bvh_nodes) { cudaFree(d_bvh_nodes); d_bvh_nodes = nullptr; }
        if (d_light_ids) { cudaFree(d_light_ids); d_light_ids = nullptr; }
    }
};
