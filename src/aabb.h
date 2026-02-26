#ifndef AABB_H
#define AABB_H

#include "rtweekend.h"

class aabb {
public:
    aabb() {}

    aabb(const point3& a, const point3& b)
        : minimum(a), maximum(b) {}

    point3 min() const { return minimum; }
    point3 max() const { return maximum; }

    bool hit(const ray& r, double t_min, double t_max) const {
        for (int axis = 0; axis < 3; axis++) {

            auto invD = 1.0 / r.direction()[axis];
            auto t0 = (minimum[axis] - r.origin()[axis]) * invD;
            auto t1 = (maximum[axis] - r.origin()[axis]) * invD;

            if (invD < 0.0)
                std::swap(t0, t1);

            t_min = t0 > t_min ? t0 : t_min;
            t_max = t1 < t_max ? t1 : t_max;

            if (t_max <= t_min)
                return false;
        }
        return true;
    }

public:
    point3 minimum;
    point3 maximum;
};

#endif