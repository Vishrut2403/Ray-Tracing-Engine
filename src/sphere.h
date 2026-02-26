#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable {
public:
    point3 center;
    double radius;
    std::shared_ptr<material> mat_ptr;

    sphere(point3 cen, double r, std::shared_ptr<material> m)
        : center(cen), radius(r), mat_ptr(m) {}

    virtual bool hit(
        const ray& r,
        double t_min,
        double t_max,
        hit_record& rec
    ) const override {

        vec3 oc = r.origin() - center;

        auto a = dot(r.direction(), r.direction());
        auto half_b = dot(oc, r.direction());
        auto c = dot(oc, oc) - radius*radius;

        auto discriminant = half_b*half_b - a*c;
        if (discriminant < 0) return false;

        auto sqrtd = std::sqrt(discriminant);

        auto root = (-half_b - sqrtd) / a;
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || root > t_max)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        rec.normal = (rec.p - center) / radius;
        rec.mat_ptr = mat_ptr;

        return true;
    }
};

#endif