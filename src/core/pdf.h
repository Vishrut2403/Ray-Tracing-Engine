#ifndef PDF_H
#define PDF_H

#include "rtweekend.h"

class pdf {
public:
    virtual ~pdf() = default;

    virtual double value(const vec3& direction) const = 0;
    virtual vec3 generate() const = 0;
};

#endif