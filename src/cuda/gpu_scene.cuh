#pragma once

#include "core/vec3.h"
#include "core/ray.h"
#include "core/interval.h"
#include "core/onb.h"

enum class MatType : int {
    LAMBERTIAN = 0,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT,
    ISOTROPIC
};

struct GpuMaterial {
    MatType type;
    vec3    albedo; 
    float   fuzz;  
    float   ir;
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

    vec3   center;
    float  radius;

    float  a0, a1;    
    float  b0, b1;   
    float  k;     

    vec3   translate_offset;
    float  sin_theta;
    float  cos_theta;
    bool   has_rotation;
};

struct GpuBVHNode {
    vec3  aabb_min;
    vec3  aabb_max;
    int   left;   
    int   right;
};

struct GpuScene {
    GpuMaterial*  materials   = nullptr;
    GpuHittable*  hittables   = nullptr;
    GpuBVHNode*   bvh_nodes   = nullptr;
    int*          light_ids   = nullptr;

    int n_materials  = 0;
    int n_hittables  = 0;
    int n_bvh_nodes  = 0;
    int n_lights     = 0;

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
