#ifndef HITTABLE_H
#define HITTABLE_H
#include <memory>
#include "ray.h"
#include "aabb.h"

class material;

struct hit_record {
    point3 p;
    vec3 normal;
    double t;
    bool front_face;
    double u;
    double v;
    std::shared_ptr<material> mat_ptr;

    void set_face_normal(const ray& r, const vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face
                 ? outward_normal
                 : -outward_normal;
    }
};

class hittable {
public:
    virtual bool hit(
        const ray& r,
        double t_min,
        double t_max,
        hit_record& rec
    ) const = 0;

    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const = 0;
};

#endif