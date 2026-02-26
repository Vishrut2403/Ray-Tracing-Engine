#pragma once

#include "hittable.h"

class translate : public hittable {
public:
    translate(shared_ptr<hittable> p,
              const vec3& displacement)
        : ptr(p), offset(displacement) {}

    virtual bool hit(
        const ray& r,
        double t_min,
        double t_max,
        hit_record& rec
    ) const override {

        ray moved_r(
            r.origin() - offset,
            r.direction(),
            r.time()
        );

        if (!ptr->hit(moved_r, t_min, t_max, rec))
            return false;

        rec.p += offset;
        rec.set_face_normal(moved_r, rec.normal);

        return true;
    }

    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const override {

        if (!ptr->bounding_box(time0, time1, output_box))
            return false;

        output_box = aabb(
            output_box.min() + offset,
            output_box.max() + offset
        );

        return true;
    }

private:
    shared_ptr<hittable> ptr;
    vec3 offset;
};