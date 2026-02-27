#ifndef MATERIAL_H
#define MATERIAL_H

#include "core/rtweekend.h"
#include "core/pdf.h"
#include "hittables/hittable.h"
#include "textures/texture.h"
#include "random.h"

class material {
public:
    virtual color emitted(
        const ray& r_in,
        const hit_record& rec,
        double u,
        double v,
        const point3& p
    ) const {
        return color(0,0,0);
    }

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        color& attenuation,
        ray& scattered,
        double& pdf
    ) const {
        return false;
    }

    virtual double scattering_pdf(
        const ray& r_in,
        const hit_record& rec,
        const ray& scattered
    ) const {
        return 0;
    }
};

//////////////////////////////////////////////////////////////
// Lambertian (Diffuse)
//////////////////////////////////////////////////////////////

class lambertian : public material {
public:
    std::shared_ptr<texture> albedo;

    lambertian(const color& a)
        : albedo(std::make_shared<solid_color>(a)) {}

    lambertian(std::shared_ptr<texture> a)
        : albedo(a) {}

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        color& attenuation,
        ray& scattered,
        double& pdf
    ) const override {

        vec3 scatter_direction =
            rec.normal + random_unit_vector();

        scattered = ray(rec.p,
                        unit_vector(scatter_direction),
                        r_in.time());

        attenuation =
            albedo->value(rec.u, rec.v, rec.p);

        pdf = dot(rec.normal,
                  scattered.direction()) / pi;

        return true;
    }

    virtual double scattering_pdf(
        const ray& r_in,
        const hit_record& rec,
        const ray& scattered
    ) const override {

        auto cosine =
            dot(rec.normal,
                unit_vector(scattered.direction()));

        return cosine < 0 ? 0 : cosine / pi;
    }
};

//////////////////////////////////////////////////////////////
// Metal (Specular Reflection — Delta Distribution)
//////////////////////////////////////////////////////////////

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
        ray& scattered,
        double& pdf
    ) const override {

        vec3 reflected =
            reflect(unit_vector(r_in.direction()),
                    rec.normal);

        scattered = ray(
            rec.p,
            reflected + fuzz * random_in_unit_sphere(),
            r_in.time()
        );

        attenuation = albedo;
        pdf = 1.0; // delta distribution

        return (dot(scattered.direction(),
                    rec.normal) > 0);
    }
};

//////////////////////////////////////////////////////////////
// Dielectric (Glass — Delta Distribution)
//////////////////////////////////////////////////////////////

class dielectric : public material {
public:
    double ir; // Index of refraction

    dielectric(double index_of_refraction)
        : ir(index_of_refraction) {}

    virtual bool scatter(
        const ray& r_in,
        const hit_record& rec,
        color& attenuation,
        ray& scattered,
        double& pdf
    ) const override {

        attenuation = color(1.0, 1.0, 1.0);
        pdf = 1.0; // delta distribution

        double refraction_ratio =
            rec.front_face
            ? (1.0 / ir)
            : ir;

        vec3 unit_direction =
            unit_vector(r_in.direction());

        double cos_theta =
            fmin(dot(-unit_direction,
                     rec.normal), 1.0);

        double sin_theta =
            sqrt(1.0 - cos_theta*cos_theta);

        bool cannot_refract =
            refraction_ratio * sin_theta > 1.0;

        vec3 direction;

        if (cannot_refract ||
            reflectance(cos_theta,
                        refraction_ratio) > random_double()) {

            direction =
                reflect(unit_direction,
                        rec.normal);
        } else {

            direction =
                refract(unit_direction,
                        rec.normal,
                        refraction_ratio);
        }

        scattered =
            ray(rec.p,
                direction,
                r_in.time());

        return true;
    }

private:
    static double reflectance(
        double cosine,
        double ref_idx
    ) {
        auto r0 = (1 - ref_idx) /
                  (1 + ref_idx);
        r0 = r0*r0;
        return r0 +
               (1 - r0) *
               pow((1 - cosine), 5);
    }
};

#endif