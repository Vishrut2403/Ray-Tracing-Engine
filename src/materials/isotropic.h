#ifndef ISOTROPIC_H
#define ISOTROPIC_H

#include "materials/material.h"
#include "core/random.h"

class isotropic : public material {
public:
    std::shared_ptr<texture> albedo;

    isotropic(const color& c)
        : albedo(std::make_shared<solid_color>(c)) {}

    isotropic(std::shared_ptr<texture> a)
        : albedo(a) {}

    virtual BSDFSample sample(
        const ray&,
        const hit_record& rec
    ) const override {

        vec3 wi = random_unit_vector();

        color rho = albedo->value(rec.u, rec.v, rec.p);

        color f_val = rho / (4 * pi);
        double pdf_val = 1.0 / (4 * pi);

        return { wi, f_val, pdf_val, false };
    }

    virtual color f(
        const ray&,
        const vec3&,
        const hit_record& rec
    ) const override {

        return albedo->value(rec.u, rec.v, rec.p)
               / (4 * pi);
    }

    virtual double pdf(
        const ray&,
        const vec3&,
        const hit_record&
    ) const override {

        return 1.0 / (4 * pi);
    }
};

#endif