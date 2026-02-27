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
        scatter_record& srec
    ) const override {
        return false;
    }

    virtual color emitted(
        const ray& r_in,
        const hit_record& rec,
        double u,
        double v,
        const point3& p
    ) const override {
        if (!rec.front_face)
            return color(0,0,0);
        return emit->value(u, v, p);
    }

private:
    shared_ptr<texture> emit;
};