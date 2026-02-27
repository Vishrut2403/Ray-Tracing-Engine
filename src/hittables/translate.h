#pragma once

#include <memory>
#include "hittable.h"

class translate : public hittable {
public:
    translate(std::shared_ptr<hittable> p,
              const vec3& displacement)
        : ptr(p), offset(displacement) {}

    virtual bool hit(
        const ray& r,
        const interval& ray_t,
        hit_record& rec
    ) const override {

        // Move ray into object space
        ray moved_r(
            r.origin() - offset,
            r.direction(),
            r.time()
        );

        if (!ptr->hit(moved_r, ray_t, rec))
            return false;

        // Move intersection point back to world space
        rec.p += offset;

        // IMPORTANT:
        // set_face_normal must use the moved ray
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
            interval(output_box.x.min + offset.x(),
                    output_box.x.max + offset.x()),
            interval(output_box.y.min + offset.y(),
                    output_box.y.max + offset.y()),
            interval(output_box.z.min + offset.z(),
                    output_box.z.max + offset.z())
        );

        return true;
    }
private:
    std::shared_ptr<hittable> ptr;
    vec3 offset;
};