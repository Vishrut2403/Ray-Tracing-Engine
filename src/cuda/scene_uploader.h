#pragma once

#include "cuda/gpu_scene.cuh"
#include <vector>
#include <cstdio>
#include <cstring>

class SceneUploader {
public:
    static GpuScene build_and_upload(const std::string& scene_name = "cornell") {
        if (scene_name == "ggx")
            return build_ggx(scene_name);
        return build_cornell(scene_name);
    }

private:
    // ── Material helpers ──────────────────────────────────────────────────────

    static int add_lambertian(std::vector<GpuMaterial>& m, vec3 albedo) {
        GpuMaterial mat{}; mat.type = MatType::LAMBERTIAN; mat.albedo = albedo;
        m.push_back(mat); return (int)m.size()-1;
    }
    static int add_light(std::vector<GpuMaterial>& m, vec3 emit) {
        GpuMaterial mat{}; mat.type = MatType::DIFFUSE_LIGHT; mat.albedo = emit;
        m.push_back(mat); return (int)m.size()-1;
    }
    static int add_ggx(std::vector<GpuMaterial>& m, vec3 albedo,
                        float roughness, float metallic) {
        GpuMaterial mat{};
        mat.type      = MatType::GGX;
        mat.albedo    = albedo;
        mat.roughness = roughness;
        mat.metallic  = metallic;
        m.push_back(mat); return (int)m.size()-1;
    }

    // ── Geometry helpers ──────────────────────────────────────────────────────

    static void add_xz_rect(std::vector<GpuHittable>& h,
                             float x0, float x1, float z0, float z1,
                             float k, int mat_id, bool flip) {
        GpuHittable g{}; g.type=HitType::XZ_RECT; g.mat_id=mat_id;
        g.flip_face=flip; g.a0=x0; g.a1=x1; g.b0=z0; g.b1=z1; g.k=k;
        h.push_back(g);
    }
    static void add_xy_rect(std::vector<GpuHittable>& h,
                             float x0, float x1, float y0, float y1,
                             float k, int mat_id, bool flip) {
        GpuHittable g{}; g.type=HitType::XY_RECT; g.mat_id=mat_id;
        g.flip_face=flip; g.a0=x0; g.a1=x1; g.b0=y0; g.b1=y1; g.k=k;
        h.push_back(g);
    }
    static void add_yz_rect(std::vector<GpuHittable>& h,
                             float y0, float y1, float z0, float z1,
                             float k, int mat_id, bool flip) {
        GpuHittable g{}; g.type=HitType::YZ_RECT; g.mat_id=mat_id;
        g.flip_face=flip; g.a0=y0; g.a1=y1; g.b0=z0; g.b1=z1; g.k=k;
        h.push_back(g);
    }
    static void add_sphere(std::vector<GpuHittable>& h,
                            vec3 center, float radius, int mat_id) {
        GpuHittable g{}; g.type=HitType::SPHERE; g.mat_id=mat_id;
        g.center=center; g.radius=radius;
        h.push_back(g);
    }
    static void add_box(std::vector<GpuHittable>& h,
                        vec3 p0, vec3 p1, int mat_id,
                        float angle_deg, vec3 offset) {
        float rad = angle_deg * 3.1415926535f / 180.f;
        float sin_t = sinf(rad), cos_t = cosf(rad);
        auto face = [&](HitType t, float a0,float a1,float b0,float b1,
                        float k, bool flip) {
            GpuHittable g{}; g.type=t; g.mat_id=mat_id; g.flip_face=flip;
            g.a0=a0; g.a1=a1; g.b0=b0; g.b1=b1; g.k=k;
            g.translate_offset=offset; g.sin_theta=sin_t; g.cos_theta=cos_t;
            g.has_rotation=(angle_deg!=0.f); h.push_back(g);
        };
        face(HitType::XY_RECT, p0.x(),p1.x(), p0.y(),p1.y(), p1.z(), false);
        face(HitType::XY_RECT, p0.x(),p1.x(), p0.y(),p1.y(), p0.z(), true );
        face(HitType::XZ_RECT, p0.x(),p1.x(), p0.z(),p1.z(), p1.y(), false);
        face(HitType::XZ_RECT, p0.x(),p1.x(), p0.z(),p1.z(), p0.y(), true );
        face(HitType::YZ_RECT, p0.y(),p1.y(), p0.z(),p1.z(), p1.x(), false);
        face(HitType::YZ_RECT, p0.y(),p1.y(), p0.z(),p1.z(), p0.x(), true );
    }

    template<typename T>
    static void upload(const std::vector<T>& src, T** dst, int n) {
        cudaMalloc(dst, n * sizeof(T));
        cudaMemcpy(*dst, src.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    }

    static GpuScene finalise(std::vector<GpuMaterial>& mats,
                              std::vector<GpuHittable>& hits,
                              std::vector<int>& lids,
                              const std::string& name) {
        GpuScene scene;
        scene.n_materials = (int)mats.size();
        scene.n_hittables = (int)hits.size();
        scene.n_lights    = (int)lids.size();
        upload(mats, &scene.d_materials, scene.n_materials);
        upload(hits, &scene.d_hittables, scene.n_hittables);
        upload(lids, &scene.d_light_ids, scene.n_lights);
        printf("[SceneUploader:%s] mats=%d  hittables=%d  lights=%d\n",
               name.c_str(), scene.n_materials, scene.n_hittables, scene.n_lights);
        return scene;
    }

    // ── Cornell box ───────────────────────────────────────────────────────────

    static GpuScene build_cornell(const std::string& name) {
        std::vector<GpuMaterial> mats;
        std::vector<GpuHittable> hits;
        std::vector<int>         lids;

        int red_id   = add_lambertian(mats, vec3(0.65f,0.05f,0.05f));
        int white_id = add_lambertian(mats, vec3(0.73f,0.73f,0.73f));
        int green_id = add_lambertian(mats, vec3(0.12f,0.45f,0.15f));
        int light_id = add_light    (mats, vec3(60.f, 60.f, 60.f ));

        add_yz_rect(hits, 0,555, 0,555, 555, green_id, true );
        add_yz_rect(hits, 0,555, 0,555, 0,   red_id,   false);
        add_xz_rect(hits, 0,555, 0,555, 0,   white_id, false);
        add_xz_rect(hits, 0,555, 0,555, 555, white_id, true );
        add_xy_rect(hits, 0,555, 0,555, 555, white_id, true );

        int lid = (int)hits.size();
        add_xz_rect(hits, 213,343, 227,332, 554, light_id, true);
        lids.push_back(lid);

        add_box(hits, mats, vec3(0,0,0), vec3(165,330,165), white_id,  15.f, vec3(265,0,295));
        add_box(hits, mats, vec3(0,0,0), vec3(165,165,165), white_id, -18.f, vec3(130,0,65 ));

        return finalise(mats, hits, lids, name);
    }

    // avoid add_box overload conflict
    static void add_box(std::vector<GpuHittable>& h,
                        std::vector<GpuMaterial>&,
                        vec3 p0, vec3 p1, int mat_id,
                        float angle_deg, vec3 offset) {
        add_box(h, p0, p1, mat_id, angle_deg, offset);
    }

    // ── GGX roughness scene ───────────────────────────────────────────────────

    static GpuScene build_ggx(const std::string& name) {
        std::vector<GpuMaterial> mats;
        std::vector<GpuHittable> hits;
        std::vector<int>         lids;

        int floor_id = add_lambertian(mats, vec3(0.4f, 0.4f, 0.4f));
        int light_id = add_light(mats, vec3(12.f, 12.f, 12.f));

        // Ceiling light
        int lid = (int)hits.size();
        add_xz_rect(hits, -6, 6, -2, 2, 5.0f, light_id, true);
        lids.push_back(lid);

        // Floor
        add_xz_rect(hits, -8, 8, -4, 4, -1.0f, floor_id, false);

        float roughness_steps[] = { 0.025f, 0.25f, 0.5f, 0.75f, 1.0f };

        for (int i = 0; i < 5; ++i) {
            float r = roughness_steps[i];
            float x = (i - 2) * 2.5f;

            // Metallic gold
            int m_id = add_ggx(mats, vec3(1.0f, 0.76f, 0.33f), r, 1.0f);
            add_sphere(hits, vec3(x, 1.1f, 0), 0.9f, m_id);

            // Dielectric blue
            int d_id = add_ggx(mats, vec3(0.05f, 0.2f, 0.8f), r, 0.0f);
            add_sphere(hits, vec3(x, -0.55f, 0), 0.9f, d_id);
        }

        return finalise(mats, hits, lids, name);
    }
};