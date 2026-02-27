#ifndef ISOTROPIC_H
#define ISOTROPIC_H

#include "material.h"
#include "random.h"
#include "sphere_pdf.h"

class isotropic : public material {
public:
    std::shared_ptr<texture> albedo;

    isotropic(const color& c)
        : albedo(std::make_shared<solid_color>(c)) {}

    isotropic(std::shared_ptr<texture> a)
        : albedo(a) {}

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        scatter_record& srec
    ) const override {

        srec.is_specular = false;
        srec.attenuation = albedo->value(rec.u, rec.v, rec.p);

        // uniform sphere sampling
        srec.is_specular = false;
        srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
        srec.pdf_ptr = std::make_shared<sphere_pdf>();
        return true;

        return true;
    }

    virtual double scattering_pdf(
        const ray& r_in,
        const hit_record& rec,
        const ray& scattered
    ) const override {
        return 1.0 / (4 * pi);
    }
};

#endif