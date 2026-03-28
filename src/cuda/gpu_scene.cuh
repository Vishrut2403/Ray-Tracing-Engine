#pragma once
#include "core/vec3.h"
#include "core/ray.h"
#include "core/interval.h"
#include "core/onb.h"
#include "cuda/gpu_triangle.h"

enum class MatType : int {
    LAMBERTIAN = 0,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT,
    ISOTROPIC,
    GGX
};

struct GpuMaterial {
    MatType type;
    vec3    albedo;
    float   fuzz;
    float   ir;
    float   roughness;
    float   metallic;
};

enum class HitType : int {
    SPHERE = 0,
    XZ_RECT,
    XY_RECT,
    YZ_RECT
};

struct GpuHittable {
    HitType type;
    int     mat_id;
    bool    flip_face;
    vec3    center;
    float   radius;
    float   a0, a1;
    float   b0, b1;
    float   k;
    vec3    translate_offset;
    float   sin_theta;
    float   cos_theta;
    bool    has_rotation;
};

struct GpuBVHNode {
    vec3  aabb_min;
    vec3  aabb_max;
    int   left;
    int   right;
};

struct GpuEnvMap;


struct GpuMedium {
    vec3  sigma_s;   
    vec3  sigma_a;   
    vec3  sigma_t;   
    float g;
    bool  active;    
};

struct GpuScene {
    GpuMaterial*    d_materials    = nullptr;
    GpuHittable*    d_hittables    = nullptr;
    GpuBVHNode*     d_bvh_nodes    = nullptr;
    int*            d_light_ids    = nullptr;
    int n_materials  = 0;
    int n_hittables  = 0;
    int n_bvh_nodes  = 0;
    int n_lights     = 0;

    GpuTriangle*    d_triangles    = nullptr;
    GpuTriBVHNode*  d_tri_bvh      = nullptr;
    int n_triangles  = 0;
    int n_tri_bvh    = 0;
    int tri_bvh_root = 0;

    GpuEnvMap*      d_env_map      = nullptr;
    bool            has_env_map    = false;

    GpuMedium       medium         = {{}, {}, {}, 0.0f, false};

    void free_device();
};