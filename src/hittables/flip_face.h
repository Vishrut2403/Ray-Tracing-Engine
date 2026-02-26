#pragma once

#include "hittable.h"

class flip_face : public hittable {
public:
    flip_face(std::shared_ptr<hittable> p)
        : ptr(p) {}

    virtual bool hit(
        const ray& r,
        double t_min,
        double t_max,
        hit_record& rec
    ) const override {

        if (!ptr->hit(r, t_min, t_max, rec))
            return false;

        rec.front_face = !rec.front_face;
        return true;
    }

    virtual bool bounding_box(
        double time0,
        double time1,
        aabb& output_box
    ) const override {

        return ptr->bounding_box(time0, time1, output_box);
    }

public:
    std::shared_ptr<hittable> ptr;
};