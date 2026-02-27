#ifndef AABB_H
#define AABB_H

#include "rtweekend.h"
#include "interval.h"

class aabb {
public:
    interval x, y, z;

    aabb() {}

    aabb(const interval& ix,
         const interval& iy,
         const interval& iz)
        : x(ix), y(iy), z(iz) {}

    aabb(const point3& a,
         const point3& b)
        : x(interval(fmin(a.x(), b.x()),
                     fmax(a.x(), b.x()))),
          y(interval(fmin(a.y(), b.y()),
                     fmax(a.y(), b.y()))),
          z(interval(fmin(a.z(), b.z()),
                     fmax(a.z(), b.z()))) {}

    const interval& axis_interval(int n) const {
        if (n == 0) return x;
        if (n == 1) return y;
        return z;
    }

    bool hit(const ray& r,
             interval ray_t) const {

        for (int axis = 0; axis < 3; axis++) {

            const interval& ax = axis_interval(axis);

            auto invD = 1.0 / r.direction()[axis];

            auto t0 = (ax.min - r.origin()[axis]) * invD;
            auto t1 = (ax.max - r.origin()[axis]) * invD;

            if (invD < 0.0)
                std::swap(t0, t1);

            if (t0 > ray_t.min) ray_t.min = t0;
            if (t1 < ray_t.max) ray_t.max = t1;

            if (ray_t.max <= ray_t.min)
                return false;
        }

        return true;
    }
};

inline aabb surrounding_box(const aabb& box0,
                            const aabb& box1) {

    interval x(
        fmin(box0.x.min, box1.x.min),
        fmax(box0.x.max, box1.x.max)
    );

    interval y(
        fmin(box0.y.min, box1.y.min),
        fmax(box0.y.max, box1.y.max)
    );

    interval z(
        fmin(box0.z.min, box1.z.min),
        fmax(box0.z.max, box1.z.max)
    );

    return aabb(x, y, z);   
}

#endif