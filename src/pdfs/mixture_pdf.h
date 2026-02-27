#ifndef MIXTURE_PDF_H
#define MIXTURE_PDF_H

#include "pdf.h"
#include <memory>

using std::shared_ptr;

class mixture_pdf : public pdf {
public:
    mixture_pdf(shared_ptr<pdf> p0, shared_ptr<pdf> p1)
        : p{p0, p1} {}

    double value(const vec3& direction) const override {
        return 0.5 * p[0]->value(direction)
             + 0.5 * p[1]->value(direction);
    }

    vec3 generate() const override {
        if (random_double() < 0.5)
            return p[0]->generate();
        else
            return p[1]->generate();
    }

private:
    shared_ptr<pdf> p[2];
};

#endif