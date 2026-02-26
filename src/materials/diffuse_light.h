#pragma once

#include "material.h"
#include "texture.h"

class diffuse_light : public material {
public:
    diffuse_light(shared_ptr<texture> a)
        : emit(a) {}

    diffuse_light(color c)
        : emit(make_shared<solid_color>(c)) {}

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        color& attenuation,
        ray& scattered
    ) const override {
        return false; // no scattering
    }

    virtual color emitted(
        double u,
        double v,
        const point3& p
    ) const override {
        return emit->value(u, v, p);
    }

private:
    shared_ptr<texture> emit;
};