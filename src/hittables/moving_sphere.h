#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include <memory>
#include "rtweekend.h"
#include "hittable.h"
#include "material.h"

class moving_sphere : public hittable {
public:
    moving_sphere() {}

    moving_sphere(
        point3 cen0,
        point3 cen1,
        double _time0,
        double _time1,
        double r,
        std::shared_ptr<material> m
    ) : center0(cen0),
        center1(cen1),
        time0(_time0),
        time1(_time1),
        radius(r),
        mat_ptr(m)
    {}

    point3 center(double time) const {
        return center0 +
               ((time - time0) / (time1 - time0)) *
               (center1 - center0);
    }

    virtual bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const override {

        vec3 oc = r.origin() - center(r.time());
        auto a = r.direction().length_squared();
        auto half_b = dot(oc, r.direction());
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = half_b * half_b - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = sqrt(discriminant);

        // Find nearest root in interval
        auto root = (-half_b - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (-half_b + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);

        vec3 outward_normal =
            (rec.p - center(r.time())) / radius;

        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;

        return true;
    }

    virtual bool bounding_box(
        double _time0,
        double _time1,
        aabb& output_box
    ) const override {

        aabb box0(
            center(_time0) - vec3(radius, radius, radius),
            center(_time0) + vec3(radius, radius, radius)
        );

        aabb box1(
            center(_time1) - vec3(radius, radius, radius),
            center(_time1) + vec3(radius, radius, radius)
        );

        output_box = surrounding_box(box0, box1);

        return true;
    }

public:
    point3 center0, center1;
    double time0, time1;
    double radius;
    std::shared_ptr<material> mat_ptr;
};

#endif