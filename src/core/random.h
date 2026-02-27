#ifndef RANDOM_H
#define RANDOM_H

#include <cstdlib>
#include <cmath>

#include "vec3.h"

//////////////////////////////////////////////////////////////
// Basic Random Numbers
//////////////////////////////////////////////////////////////

inline double random_double() {
    return rand() / (RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}

inline int random_int(int min, int max) {
    return static_cast<int>(random_double(min, max + 1));
}

//////////////////////////////////////////////////////////////
// Random Vectors
//////////////////////////////////////////////////////////////

inline vec3 random_vec3() {
    return vec3(random_double(),
                random_double(),
                random_double());
}

inline vec3 random_vec3(double min, double max) {
    return vec3(random_double(min, max),
                random_double(min, max),
                random_double(min, max));
}

//////////////////////////////////////////////////////////////
// Unit Sphere
//////////////////////////////////////////////////////////////

inline vec3 random_in_unit_sphere() {
    while (true) {
        auto p = random_vec3(-1,1);
        if (p.length_squared() >= 1)
            continue;
        return p;
    }
}

inline vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

//////////////////////////////////////////////////////////////
// Disk (DOF)
//////////////////////////////////////////////////////////////

inline vec3 random_in_unit_disk() {
    while (true) {
        vec3 p(random_double(-1,1),
               random_double(-1,1),
               0);

        if (p.length_squared() >= 1)
            continue;

        return p;
    }
}

//////////////////////////////////////////////////////////////
// Cosine PDF
//////////////////////////////////////////////////////////////

inline vec3 random_cosine_direction() {
    constexpr double local_pi = 3.1415926535897932385;

    auto r1 = random_double();
    auto r2 = random_double();

    auto phi = 2 * local_pi * r1;

    auto x = cos(phi) * sqrt(r2);
    auto y = sin(phi) * sqrt(r2);
    auto z = sqrt(1 - r2);

    return vec3(x, y, z);
}
#endif