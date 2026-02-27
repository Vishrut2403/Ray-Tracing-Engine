#pragma once

#include <memory>
#include "hittable.h"

class flip_face : public hittable {
public:
    flip_face(std::shared_ptr<hittable> p)
        : ptr(p) {}

    virtual bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const override {

        if (!ptr->hit(r, ray_t, rec))
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