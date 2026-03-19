#ifndef RANDOM_H
#define RANDOM_H
#pragma once

// CPU random functions — GPU will use cuRAND (see cuda/cuda_rand.cuh)
// None of these are marked HD because std::mt19937 is not available on device.

#include <cmath>
#include <random>
#include "vec3.h"

inline double random_double() {
    thread_local static std::mt19937 generator(std::random_device{}());
    thread_local static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

inline double random_double(double min, double max) {
    return min + (max - min) * random_double();
}

inline int random_int(int min, int max) {
    return static_cast<int>(random_double(min, max + 1));
}

inline vec3 random_vec3() {
    return vec3(random_double(), random_double(), random_double());
}

inline vec3 random_vec3(double min, double max) {
    return vec3(random_double(min,max), random_double(min,max), random_double(min,max));
}

inline vec3 random_in_unit_sphere() {
    while (true) {
        auto p = random_vec3(-1, 1);
        if (p.length_squared() < 1) return p;
    }
}

inline vec3 random_unit_vector() { return unit_vector(random_in_unit_sphere()); }

inline vec3 random_in_unit_disk() {
    while (true) {
        vec3 p(random_double(-1,1), random_double(-1,1), 0);
        if (p.length_squared() < 1) return p;
    }
}

inline vec3 random_cosine_direction() {
    constexpr double local_pi = 3.1415926535897932385;
    auto r1 = random_double(), r2 = random_double();
    auto phi = 2 * local_pi * r1;
    return vec3(cos(phi)*sqrt(r2), sin(phi)*sqrt(r2), sqrt(1-r2));
}

#endif