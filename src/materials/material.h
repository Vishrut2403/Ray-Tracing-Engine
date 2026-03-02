#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <cmath>

#include "core/rtweekend.h"
#include "core/ray.h"
#include "hittables/hittable.h"
#include "textures/texture.h"
#include "materials/bsdf_sample.h"

class material {
public:
    virtual ~material() = default;

    virtual color emitted(
        const ray&,
        const hit_record&,
        double,
        double,
        const point3&
    ) const {
        return color(0,0,0);
    }

    virtual BSDFSample sample(
        const ray& wo,
        const hit_record& rec
    ) const = 0;

    virtual color f(
        const ray& wo,
        const vec3& wi,
        const hit_record& rec
    ) const = 0;

    virtual double pdf(
        const ray& wo,
        const vec3& wi,
        const hit_record& rec
    ) const = 0;
};

class lambertian : public material {
public:
    std::shared_ptr<texture> albedo;

    lambertian(const color& a)
        : albedo(std::make_shared<solid_color>(a)) {}

    lambertian(std::shared_ptr<texture> a)
        : albedo(a) {}

    virtual BSDFSample sample(
        const ray& wo,
        const hit_record& rec
    ) const override {

        vec3 wi = random_unit_vector();
        if (dot(wi, rec.normal) < 0)
            wi = -wi;

        double cosine = dot(rec.normal, wi);
        double pdf_val = cosine > 0 ? cosine / pi : 0.0;

        color rho = albedo->value(rec.u, rec.v, rec.p);
        color f_val = rho / pi;

        return { wi, f_val, pdf_val, false };
    }

    virtual color f(
        const ray&,
        const vec3& wi,
        const hit_record& rec
    ) const override {

        if (dot(rec.normal, wi) <= 0)
            return color(0,0,0);

        return albedo->value(rec.u, rec.v, rec.p) / pi;
    }

    virtual double pdf(
        const ray&,
        const vec3& wi,
        const hit_record& rec
    ) const override {

        double cosine = dot(rec.normal, wi);
        return (cosine > 0) ? cosine / pi : 0.0;
    }
};

class metal : public material {
public:
    color albedo;
    double fuzz;

    metal(const color& a, double f)
        : albedo(a), fuzz(f < 1 ? f : 1) {}

    virtual BSDFSample sample(
        const ray& wo,
        const hit_record& rec
    ) const override {

        vec3 reflected =
            reflect(unit_vector(wo.direction()), rec.normal);

        reflected += fuzz * random_in_unit_sphere();
        reflected = unit_vector(reflected);

        return { reflected, albedo, 1.0, true };
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

class dielectric : public material {
public:
    double ir; 

    dielectric(double index_of_refraction)
        : ir(index_of_refraction) {}

    virtual BSDFSample sample(
        const ray& wo,
        const hit_record& rec
    ) const override {

        vec3 unit_direction =
            unit_vector(wo.direction());

        double refraction_ratio =
            rec.front_face ? (1.0 / ir) : ir;

        double cos_theta =
            fmin(dot(-unit_direction, rec.normal), 1.0);

        double sin_theta =
            std::sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract =
            refraction_ratio * sin_theta > 1.0;

        vec3 direction;

        if (cannot_refract ||
            reflectance(cos_theta, refraction_ratio)
                > random_double()) {

            direction =
                reflect(unit_direction, rec.normal);
        } else {

            direction =
                refract(unit_direction,
                        rec.normal,
                        refraction_ratio);
        }

        return { direction, color(1,1,1), 1.0, true };
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

private:
    static double reflectance(
        double cosine,
        double ref_idx
    ) {
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0 = r0*r0;
        return r0 + (1 - r0) *
               std::pow((1 - cosine), 5);
    }
};

#endif