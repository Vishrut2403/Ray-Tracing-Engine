#ifndef MATERIAL_H
#define MATERIAL_H

#include "ray.h"
#include "hittable.h"

class material {
public:
    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        color& attenuation,
        ray& scattered
    ) const = 0;
};

class lambertian : public material {
public:
    color albedo;

    lambertian(const color& a) : albedo(a) {}

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        color& attenuation,
        ray& scattered
    ) const override {

        vec3 scatter_direction = rec.normal + random_in_unit_sphere();

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;

        return true;
    }
};

#endif