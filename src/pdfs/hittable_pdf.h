#ifndef HITTABLE_PDF_H
#define HITTABLE_PDF_H

#include "pdf.h"
#include "hittable.h"

class hittable_pdf : public pdf {
public:
    hittable_pdf(shared_ptr<hittable> objects, const point3& origin)
        : objects(objects), origin(origin) {}

    double value(const vec3& direction) const override {
        return objects->pdf_value(origin, direction);
    }

    vec3 generate() const override {
        return objects->random(origin);
    }

private:
    shared_ptr<hittable> objects;
    point3 origin;
};

#endif