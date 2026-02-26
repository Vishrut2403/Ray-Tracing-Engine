#ifndef PERLIN_H
#define PERLIN_H

#include "rtweekend.h"

class perlin {
public:
    perlin() {

        ranvec = new vec3[point_count];
        for (int i = 0; i < point_count; i++) {
            ranvec[i] = unit_vector(vec3::random(-1,1));
        }

        perm_x = perlin_generate_perm();
        perm_y = perlin_generate_perm();
        perm_z = perlin_generate_perm();
    }

    ~perlin() {
        delete[] ranvec;
        delete[] perm_x;
        delete[] perm_y;
        delete[] perm_z;
    }

    double noise(const point3& p) const;

private:
    static const int point_count = 256;

    vec3* ranvec;
    int* perm_x;
    int* perm_y;
    int* perm_z;

    static int* perlin_generate_perm() {
        auto p = new int[point_count];

        for (int i = 0; i < point_count; i++)
            p[i] = i;

        permute(p, point_count);

        return p;
    }

    static void permute(int* p, int n) {
        for (int i = n-1; i > 0; i--) {
            int target = random_int(0, i);
            std::swap(p[i], p[target]);
        }
    }
};

#endif