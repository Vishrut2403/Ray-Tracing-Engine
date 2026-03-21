#pragma once

#include "hittables/hittable.h"
#include "hittables/hittable_list.h"
#include "core/rtweekend.h"
#include "acceleration/bvh.h"
#include "external/tiny_obj_loader.h"

#include <string>
#include <vector>
#include <iostream>

class triangle : public hittable {
public:
    point3 v0, v1, v2;
    vec3   normal;
    std::shared_ptr<material> mat;

    triangle(const point3& a, const point3& b, const point3& c,
             std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c), mat(m)
    {
        normal = unit_vector(cross(b - a, c - a));
    }

    virtual bool hit(const ray& r, const interval& ray_t,
                     hit_record& rec) const override {
        const double eps = 1e-8;
        vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;
        vec3 h  = cross(r.direction(), e2);
        double a = dot(e1, h);

        if (std::abs(a) < eps) return false;

        double f = 1.0 / a;
        vec3   s = r.origin() - v0;
        double u = f * dot(s, h);
        if (u < 0.0 || u > 1.0) return false;

        vec3   q = cross(s, e1);
        double v = f * dot(r.direction(), q);
        if (v < 0.0 || u + v > 1.0) return false;

        double t = f * dot(e2, q);
        if (!ray_t.surrounds(t)) return false;

        rec.t       = t;
        rec.p       = r.at(t);
        rec.mat_ptr = mat;
        rec.u       = u;
        rec.v       = v;
        rec.set_face_normal(r, normal);
        return true;
    }

    virtual bool bounding_box(double, double, aabb& out) const override {
        point3 lo(fmin(fmin(v0.x(),v1.x()),v2.x()) - 1e-4,
                  fmin(fmin(v0.y(),v1.y()),v2.y()) - 1e-4,
                  fmin(fmin(v0.z(),v1.z()),v2.z()) - 1e-4);
        point3 hi(fmax(fmax(v0.x(),v1.x()),v2.x()) + 1e-4,
                  fmax(fmax(v0.y(),v1.y()),v2.y()) + 1e-4,
                  fmax(fmax(v0.z(),v1.z()),v2.z()) + 1e-4);
        out = aabb(lo, hi);
        return true;
    }
};

inline std::shared_ptr<hittable> load_obj(
    const std::string& path,
    std::shared_ptr<material> mat,
    double scale    = 1.0,
    vec3   offset   = vec3(0,0,0)
) {
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials,
                                &warn, &err, path.c_str());

    if (!warn.empty()) std::cerr << "[OBJ] " << warn << "\n";
    if (!err.empty())  std::cerr << "[OBJ] " << err  << "\n";
    if (!ok) {
        std::cerr << "[OBJ] failed to load: " << path << "\n";
        return nullptr;
    }

    hittable_list tris;
    size_t tri_count = 0;

    for (const auto& shape : shapes) {
        size_t idx_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) { idx_offset += fv; continue; }

            point3 verts[3];
            for (int v = 0; v < 3; ++v) {
                tinyobj::index_t i = shape.mesh.indices[idx_offset + v];
                verts[v] = point3(
                    attrib.vertices[3*i.vertex_index + 0] * scale + offset.x(),
                    attrib.vertices[3*i.vertex_index + 1] * scale + offset.y(),
                    attrib.vertices[3*i.vertex_index + 2] * scale + offset.z()
                );
            }

            tris.add(std::make_shared<triangle>(verts[0], verts[1], verts[2], mat));
            ++tri_count;
            idx_offset += fv;
        }
    }

    std::cerr << "[OBJ] loaded " << tri_count << " triangles from " << path << "\n";

    return std::make_shared<bvh_node>(
        tris.objects, 0, tris.objects.size(), 0.0, 1.0);
}