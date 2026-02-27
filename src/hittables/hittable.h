#ifndef HITTABLE_H
#define HITTABLE_H
#include <memory>
#include "ray.h"
#include "aabb.h"
#include "interval.h"

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
        const interval& ray_t,
        hit_record& rec
    ) const = 0;

    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const = 0;

    virtual double pdf_value(const point3&, const vec3&) const {
        return 0.0;
    }

    virtual vec3 random(const point3&) const {
        return vec3(1,0,0);
    }
};



#endif