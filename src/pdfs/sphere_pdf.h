#ifndef SPHERE_PDF_H
#define SPHERE_PDF_H

#include "pdf.h"
#include "random.h"
#include "rtweekend.h"

class sphere_pdf : public pdf {
public:
    virtual double value(const vec3& direction) const override {
        return 1.0 / (4.0 * pi);
    }

    virtual vec3 generate() const override {
        return random_unit_vector();
    }
};

#endif