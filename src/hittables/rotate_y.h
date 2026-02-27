#pragma once

#include <memory>
#include <cmath>
#include "hittable.h"
#include "rtweekend.h"

class rotate_y : public hittable {
public:
    rotate_y(std::shared_ptr<hittable> p, double angle)
        : ptr(p) {

        auto radians = degrees_to_radians(angle);
        sin_theta = sin(radians);
        cos_theta = cos(radians);

        hasbox = ptr->bounding_box(0, 1, bbox);

point3 min( infinity,  infinity,  infinity);
point3 max(-infinity, -infinity, -infinity);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {

                auto x = i ? bbox.x.max : bbox.x.min;
                auto y = j ? bbox.y.max : bbox.y.min;
                auto z = k ? bbox.z.max : bbox.z.min;

                auto newx =  cos_theta*x + sin_theta*z;
                auto newz = -sin_theta*x + cos_theta*z;

                vec3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
    }

    virtual bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const override {

        // Rotate ray into object space
        auto origin = r.origin();
        auto direction = r.direction();

        origin[0] =  cos_theta*r.origin()[0]
                   - sin_theta*r.origin()[2];

        origin[2] =  sin_theta*r.origin()[0]
                   + cos_theta*r.origin()[2];

        direction[0] =  cos_theta*r.direction()[0]
                      - sin_theta*r.direction()[2];

        direction[2] =  sin_theta*r.direction()[0]
                      + cos_theta*r.direction()[2];

        ray rotated_r(origin, direction, r.time());

        if (!ptr->hit(rotated_r, ray_t, rec))
            return false;

        // Rotate hit point and normal back to world space
        auto p = rec.p;
        auto normal = rec.normal;

        p[0] =  cos_theta*rec.p[0]
              + sin_theta*rec.p[2];

        p[2] = -sin_theta*rec.p[0]
              + cos_theta*rec.p[2];

        normal[0] =  cos_theta*rec.normal[0]
                   + sin_theta*rec.normal[2];

        normal[2] = -sin_theta*rec.normal[0]
                   + cos_theta*rec.normal[2];

        rec.p = p;

        // IMPORTANT:
        // set_face_normal must use the original ray
        rec.set_face_normal(r, normal);

        return true;
    }

    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const override {

        output_box = bbox;
        return hasbox;
    }

private:
    std::shared_ptr<hittable> ptr;
    double sin_theta;
    double cos_theta;
    bool hasbox;
    aabb bbox;
};