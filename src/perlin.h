#ifndef PERLIN_H
#define PERLIN_H

#include "rtweekend.h"

class perlin {
public:
    perlin();

    double noise(const point3& p) const;

private:
    static const int point_count = 256;

    vec3* ranvec;
    int* perm_x;
    int* perm_y;
    int* perm_z;
};

#endif