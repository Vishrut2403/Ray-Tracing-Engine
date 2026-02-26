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

class metal : public material {
public:
    color albedo;
    double fuzz;

    metal(const color& a, double f)
        : albedo(a), fuzz(f < 1 ? f : 1) {}

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        color& attenuation,
        ray& scattered
    ) const override {

        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);

        scattered = ray(
            rec.p,
            reflected + fuzz * random_in_unit_sphere()
        );

        attenuation = albedo;

        return (dot(scattered.direction(), rec.normal) > 0);
    }
};

class dielectric : public material {
public:
    double ir; 

    dielectric(double index_of_refraction)
        : ir(index_of_refraction) {}

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        color& attenuation,
        ray& scattered
    ) const override {

        attenuation = color(1.0, 1.0, 1.0);

        double refraction_ratio =
            (dot(r_in.direction(), rec.normal) > 0)
            ? ir
            : (1.0 / ir);

        vec3 unit_direction = unit_vector(r_in.direction());

        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract =
            refraction_ratio * sin_theta > 1.0;

        vec3 direction;

        if (cannot_refract ||
            reflectance(cos_theta, refraction_ratio) > random_double()) {

            direction = reflect(unit_direction, rec.normal);

        } else {

            direction = refract(
                unit_direction,
                rec.normal,
                refraction_ratio
            );
        }

        scattered = ray(rec.p, direction);

        return true;
    }
};

#endif