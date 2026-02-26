#pragma once

#include "hittable.h"
#include "aabb.h"

class yz_rect : public hittable {
public:
    yz_rect() {}

    yz_rect(
        double _y0, double _y1,
        double _z0, double _z1,
        double _k,
        shared_ptr<material> mat
    )
        : y0(_y0), y1(_y1),
          z0(_z0), z1(_z1),
          k(_k),
          mp(mat) {}

    virtual bool hit(
        const ray& r,
        double t0,
        double t1,
        hit_record& rec
    ) const override {

        auto t = (k - r.origin().x())
                 / r.direction().x();

        if (t < t0 || t > t1)
            return false;

        auto y = r.origin().y()
               + t * r.direction().y();

        auto z = r.origin().z()
               + t * r.direction().z();

        if (y < y0 || y > y1 ||
            z < z0 || z > z1)
            return false;

        rec.u = (y - y0) / (y1 - y0);
        rec.v = (z - z0) / (z1 - z0);

        rec.t = t;
        rec.p = r.at(t);

        auto outward_normal = vec3(1,0,0);
        rec.set_face_normal(r, outward_normal);

        rec.mat_ptr = mp;

        return true;
    }

    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const override {

        // Add small thickness in x direction
        output_box = aabb(
            point3(k - 0.0001, y0, z0),
            point3(k + 0.0001, y1, z1)
        );

        return true;
    }

private:
    shared_ptr<material> mp;
    double y0, y1;
    double z0, z1;
    double k;
};