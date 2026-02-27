#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "hittable.h"
#include "material.h"
#include "isotropic.h"
#include "interval.h"
#include "random.h"

class constant_medium : public hittable {
public:
    std::shared_ptr<hittable> boundary;
    std::shared_ptr<material> phase_function;
    double neg_inv_density;

    constant_medium(std::shared_ptr<hittable> b,
                    double density,
                    std::shared_ptr<texture> tex)
        : boundary(b),
          neg_inv_density(-1.0 / density),
          phase_function(std::make_shared<isotropic>(tex)) {}

    constant_medium(std::shared_ptr<hittable> b,
                    double density,
                    const color& c)
        : boundary(b),
          neg_inv_density(-1.0 / density),
          phase_function(std::make_shared<isotropic>(c)) {}

    virtual bool hit(const ray& r,
                     const interval& ray_t,
                     hit_record& rec) const override {

        hit_record rec1, rec2;

        // Find first boundary intersection
        if (!boundary->hit(r, interval(-infinity, infinity), rec1))
            return false;

        // Find second intersection
        if (!boundary->hit(r,
                           interval(rec1.t + 0.0001, infinity),
                           rec2))
            return false;

        double t0 = rec1.t;
        double t1 = rec2.t;

        // Clamp to ray interval
        if (t0 < ray_t.min) t0 = ray_t.min;
        if (t1 > ray_t.max) t1 = ray_t.max;

        if (t0 >= t1)
            return false;

        if (t0 < 0)
            t0 = 0;

        const auto ray_length = r.direction().length();
        const auto distance_inside_boundary =
            (t1 - t0) * ray_length;

        const auto hit_distance =
            neg_inv_density * log(random_double());

        if (hit_distance > distance_inside_boundary)
            return false;

        rec.t = t0 + hit_distance / ray_length;
        rec.p = r.at(rec.t);

        rec.normal = vec3(1, 0, 0); // arbitrary
        rec.front_face = true;
        rec.mat_ptr = phase_function;

        return true;
    }

    virtual bool bounding_box(double time0,
                              double time1,
                              aabb& output_box) const override {
        return boundary->bounding_box(time0, time1, output_box);
    }
};

#endif