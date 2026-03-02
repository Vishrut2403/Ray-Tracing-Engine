#ifndef DIFFUSE_LIGHT_H
#define DIFFUSE_LIGHT_H

#include "materials/material.h"

class diffuse_light : public material {
public:
    std::shared_ptr<texture> emit;

    diffuse_light(std::shared_ptr<texture> a)
        : emit(a) {}

    diffuse_light(color c)
        : emit(std::make_shared<solid_color>(c)) {}

    virtual color emitted(
        const ray&,
        const hit_record& rec,
        double u,
        double v,
        const point3& p
    ) const override {

        if (!rec.front_face)
            return color(0,0,0);

        return emit->value(u, v, p);
    }

    // Emitters do not scatter

    virtual BSDFSample sample(
        const ray&,
        const hit_record&
    ) const override {
        return { vec3(0,0,0), color(0,0,0), 0.0, false };
    }

    virtual color f(
        const ray&,
        const vec3&,
        const hit_record&
    ) const override {
        return color(0,0,0);
    }

    virtual double pdf(
        const ray&,
        const vec3&,
        const hit_record&
    ) const override {
        return 0.0;
    }
};

#endif