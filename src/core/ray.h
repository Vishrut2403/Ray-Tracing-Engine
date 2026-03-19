#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
public:
    HD ray() {}
    HD ray(const point3& origin, const vec3& direction, double time = 0.0)
        : orig(origin), dir(direction), tm(time) {}

    HD point3 origin()    const { return orig; }
    HD vec3   direction() const { return dir;  }
    HD double time()      const { return tm;   }
    HD point3 at(double t) const { return orig + t*dir; }

private:
    point3 orig;
    vec3   dir;
    double tm;
};

#endif