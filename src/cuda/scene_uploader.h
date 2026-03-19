#pragma once

#include "cuda/gpu_scene.cuh"
#include <vector>
#include <cstdio>

// Builds and uploads a GpuScene from the hardcoded Cornell box.
// This mirrors build_cornell_volume_scene() but produces flat GPU structs
// instead of shared_ptr hittables.
//
// Strategy: CPU scene and GPU scene are built in parallel.
// CPU renderer keeps working; GPU scene is uploaded alongside it.

class SceneUploader {
public:
    static GpuScene build_and_upload() {
        std::vector<GpuMaterial> mats;
        std::vector<GpuHittable> hits;
        std::vector<int>         light_ids;

        // ── Materials ─────────────────────────────────────────────────────
        int red_id   = add_lambertian(mats, vec3(0.65f, 0.05f, 0.05f));
        int white_id = add_lambertian(mats, vec3(0.73f, 0.73f, 0.73f));
        int green_id = add_lambertian(mats, vec3(0.12f, 0.45f, 0.15f));
        int light_id = add_light    (mats, vec3(60.f,  60.f,  60.f ));

        // ── Walls ─────────────────────────────────────────────────────────
        // Left wall (green) — yz_rect at x=555, facing inward
        add_yz_rect(hits, 0,555, 0,555, 555, green_id, true);

        // Right wall (red) — yz_rect at x=0, facing inward
        add_yz_rect(hits, 0,555, 0,555, 0,   red_id,   false);

        // Floor — xz_rect at y=0
        add_xz_rect(hits, 0,555, 0,555, 0,   white_id, false);

        // Ceiling — xz_rect at y=555, flipped
        add_xz_rect(hits, 0,555, 0,555, 555, white_id, true);

        // Back wall — xy_rect at z=555, flipped
        add_xy_rect(hits, 0,555, 0,555, 555, white_id, true);

        // ── Ceiling light ─────────────────────────────────────────────────
        int lid = (int)hits.size();
        add_xz_rect(hits, 213,343, 227,332, 554, light_id, true);
        light_ids.push_back(lid);

        // ── Tall box (rotate_y 15°, translate 265,0,295) ──────────────────
        add_box(hits, mats,
                vec3(0,0,0), vec3(165,330,165), white_id,
                15.f, vec3(265,0,295));

        // ── Short box (rotate_y -18°, translate 130,0,65) ─────────────────
        add_box(hits, mats,
                vec3(0,0,0), vec3(165,165,165), white_id,
                -18.f, vec3(130,0,65));

        // ── Upload to device ──────────────────────────────────────────────
        GpuScene scene;
        scene.n_materials = (int)mats.size();
        scene.n_hittables = (int)hits.size();
        scene.n_lights    = (int)light_ids.size();
        scene.n_bvh_nodes = 0; // BVH built in M3

        upload(mats,      &scene.d_materials, scene.n_materials);
        upload(hits,      &scene.d_hittables, scene.n_hittables);
        upload(light_ids, &scene.d_light_ids, scene.n_lights);

        printf("[SceneUploader] materials=%d  hittables=%d  lights=%d\n",
               scene.n_materials, scene.n_hittables, scene.n_lights);

        return scene;
    }

private:
    // ── Material helpers ──────────────────────────────────────────────────

    static int add_lambertian(std::vector<GpuMaterial>& m, vec3 albedo) {
        GpuMaterial mat{}; mat.type = MatType::LAMBERTIAN; mat.albedo = albedo;
        m.push_back(mat); return (int)m.size()-1;
    }
    static int add_light(std::vector<GpuMaterial>& m, vec3 emit) {
        GpuMaterial mat{}; mat.type = MatType::DIFFUSE_LIGHT; mat.albedo = emit;
        m.push_back(mat); return (int)m.size()-1;
    }

    // ── Geometry helpers ──────────────────────────────────────────────────

    static void add_xz_rect(std::vector<GpuHittable>& h,
                             float x0, float x1, float z0, float z1, float k,
                             int mat_id, bool flip) {
        GpuHittable g{};
        g.type=HitType::XZ_RECT; g.mat_id=mat_id; g.flip_face=flip;
        g.a0=x0; g.a1=x1; g.b0=z0; g.b1=z1; g.k=k;
        h.push_back(g);
    }
    static void add_xy_rect(std::vector<GpuHittable>& h,
                             float x0, float x1, float y0, float y1, float k,
                             int mat_id, bool flip) {
        GpuHittable g{};
        g.type=HitType::XY_RECT; g.mat_id=mat_id; g.flip_face=flip;
        g.a0=x0; g.a1=x1; g.b0=y0; g.b1=y1; g.k=k;
        h.push_back(g);
    }
    static void add_yz_rect(std::vector<GpuHittable>& h,
                             float y0, float y1, float z0, float z1, float k,
                             int mat_id, bool flip) {
        GpuHittable g{};
        g.type=HitType::YZ_RECT; g.mat_id=mat_id; g.flip_face=flip;
        g.a0=y0; g.a1=y1; g.b0=z0; g.b1=z1; g.k=k;
        h.push_back(g);
    }

    // Bakes rotate_y + translate into 6 rect faces of a box
    static void add_box(std::vector<GpuHittable>& h,
                        std::vector<GpuMaterial>& /*m*/,
                        vec3 p0, vec3 p1, int mat_id,
                        float angle_deg, vec3 offset) {
        float rad = angle_deg * 3.1415926535f / 180.f;
        float sin_t = sinf(rad), cos_t = cosf(rad);

        // 6 faces — same as box.h, transforms baked per-face
        auto make_face = [&](HitType t,
                              float a0, float a1,
                              float b0, float b1,
                              float k, bool flip) {
            GpuHittable g{};
            g.type = t; g.mat_id = mat_id; g.flip_face = flip;
            g.a0=a0; g.a1=a1; g.b0=b0; g.b1=b1; g.k=k;
            g.translate_offset = offset;
            g.sin_theta = sin_t; g.cos_theta = cos_t;
            g.has_rotation = (angle_deg != 0.f);
            h.push_back(g);
        };

        make_face(HitType::XY_RECT, p0.x(),p1.x(), p0.y(),p1.y(), p1.z(), false);
        make_face(HitType::XY_RECT, p0.x(),p1.x(), p0.y(),p1.y(), p0.z(), true );
        make_face(HitType::XZ_RECT, p0.x(),p1.x(), p0.z(),p1.z(), p1.y(), false);
        make_face(HitType::XZ_RECT, p0.x(),p1.x(), p0.z(),p1.z(), p0.y(), true );
        make_face(HitType::YZ_RECT, p0.y(),p1.y(), p0.z(),p1.z(), p1.x(), false);
        make_face(HitType::YZ_RECT, p0.y(),p1.y(), p0.z(),p1.z(), p0.x(), true );
    }

    // ── Upload helper ─────────────────────────────────────────────────────

    template<typename T>
    static void upload(const std::vector<T>& src, T** dst, int n) {
        cudaMalloc(dst, n * sizeof(T));
        cudaMemcpy(*dst, src.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    }
};
