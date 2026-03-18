#ifndef MATERIAL_H
#define MATERIAL_H

#include <memory>
#include <cmath>

#include "core/rtweekend.h"
#include "core/ray.h"
#include "core/onb.h"
#include "hittables/hittable.h"
#include "textures/texture.h"
#include "materials/bsdf_sample.h"

class material {
public:
    virtual ~material() = default;

    virtual color emitted(
        const ray&, const hit_record&,
        double, double, const point3&
    ) const { return color(0, 0, 0); }

    virtual BSDFSample sample(const ray& wo, const hit_record& rec) const = 0;
    virtual color      f      (const ray& wo, const vec3& wi, const hit_record& rec) const = 0;
    virtual double     pdf    (const ray& wo, const vec3& wi, const hit_record& rec) const = 0;
};

class lambertian : public material {
public:
    std::shared_ptr<texture> albedo;

    lambertian(const color& a) : albedo(std::make_shared<solid_color>(a)) {}
    lambertian(std::shared_ptr<texture> a) : albedo(a) {}

    virtual BSDFSample sample(const ray&, const hit_record& rec) const override {
        onb uvw;
        uvw.build_from_w(rec.normal);
        vec3   wi      = uvw.local(random_cosine_direction());
        double cosine  = dot(rec.normal, wi);
        double pdf_val = cosine > 0.0 ? cosine / pi : 0.0;
        return { wi, albedo->value(rec.u, rec.v, rec.p) / pi, pdf_val, false };
    }

    virtual color f(const ray&, const vec3& wi, const hit_record& rec) const override {
        if (dot(rec.normal, wi) <= 0.0) return color(0, 0, 0);
        return albedo->value(rec.u, rec.v, rec.p) / pi;
    }

    virtual double pdf(const ray&, const vec3& wi, const hit_record& rec) const override {
        double cosine = dot(rec.normal, wi);
        return cosine > 0.0 ? cosine / pi : 0.0;
    }
};

class metal : public material {
public:
    color  albedo;
    double fuzz;

    metal(const color& a, double f) : albedo(a), fuzz(f < 1.0 ? f : 1.0) {}

    virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
        vec3 reflected = unit_vector(
            reflect(unit_vector(wo.direction()), rec.normal)
            + fuzz * random_in_unit_sphere()
        );
        return { reflected, albedo, 1.0, true };
    }

    virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
    virtual double pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }
};

class dielectric : public material {
public:
    double ir;

    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
        vec3   ud    = unit_vector(wo.direction());
        double ratio = rec.front_face ? (1.0 / ir) : ir;
        double cos_t = fmin(dot(-ud, rec.normal), 1.0);
        double sin_t = std::sqrt(1.0 - cos_t * cos_t);

        vec3 dir = (ratio * sin_t > 1.0 || reflectance(cos_t, ratio) > random_double())
                   ? reflect(ud, rec.normal)
                   : refract(ud, rec.normal, ratio);

        return { dir, color(1, 1, 1), 1.0, true };
    }

    virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
    virtual double pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }

private:
    static double reflectance(double cosine, double ref_idx) {
        double r0 = (1.0 - ref_idx) / (1.0 + ref_idx); r0 *= r0;
        return r0 + (1.0 - r0) * std::pow(1.0 - cosine, 5.0);
    }
};

#endif