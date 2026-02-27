#pragma once

#include <memory>
#include "hittable.h"
#include "aabb.h"
#include "rtweekend.h"

class xy_rect : public hittable {
public:
    xy_rect() {}

    xy_rect(
        double _x0, double _x1,
        double _y0, double _y1,
        double _k,
        std::shared_ptr<material> mat
    )
        : x0(_x0), x1(_x1),
          y0(_y0), y1(_y1),
          k(_k),
          mp(mat) {}

    virtual bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const override {

        auto t = (k - r.origin().z())
                 / r.direction().z();

        if (!ray_t.surrounds(t))
            return false;

        auto x = r.origin().x()
               + t * r.direction().x();

        auto y = r.origin().y()
               + t * r.direction().y();

        if (x < x0 || x > x1 ||
            y < y0 || y > y1)
            return false;

        rec.u = (x - x0) / (x1 - x0);
        rec.v = (y - y0) / (y1 - y0);

        rec.t = t;
        rec.p = r.at(t);

        vec3 outward_normal(0, 0, 1);
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mp;

        return true;
    }

    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const override {

        output_box = aabb(
            point3(x0, y0, k - 0.0001),
            point3(x1, y1, k + 0.0001)
        );

        return true;
    }

    virtual double pdf_value(
        const point3& origin,
        const vec3& direction
    ) const override {

        hit_record rec;

        if (!this->hit(
                ray(origin, direction),
                interval(0.001, infinity),
                rec))
            return 0;

        double area = (x1 - x0) * (y1 - y0);

        double distance_squared =
            rec.t * rec.t *
            direction.length_squared();

        double cosine =
            fabs(dot(direction, rec.normal)
                 / direction.length());

        return distance_squared / (cosine * area);
    }

    virtual vec3 random(
        const point3& origin
    ) const override {

        auto random_point = point3(
            random_double(x0, x1),
            random_double(y0, y1),
            k
        );

        return random_point - origin;
    }

private:
    std::shared_ptr<material> mp;
    double x0, x1, y0, y1, k;
};