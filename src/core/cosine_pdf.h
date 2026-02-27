#ifndef COSINE_PDF_H
#define COSINE_PDF_H

#include "pdf.h"
#include "onb.h"

class cosine_pdf : public pdf {
public:
    cosine_pdf(const vec3& w) {
        uvw.build_from_w(w);
    }

    virtual double value(const vec3& direction) const override {
        auto cosine = dot(unit_vector(direction), uvw.w());
        return (cosine <= 0) ? 0 : cosine / pi;
    }

    virtual vec3 generate() const override {
        return uvw.local(random_cosine_direction());
    }

private:
    onb uvw;
};

#endif