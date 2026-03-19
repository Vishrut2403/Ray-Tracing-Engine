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
    virtual color      f  (const ray& wo, const vec3& wi, const hit_record& rec) const = 0;
    virtual double     pdf(const ray& wo, const vec3& wi, const hit_record& rec) const = 0;
};

// ─────────────────────────────────────────────────────────────────────────────

class lambertian : public material {
public:
    std::shared_ptr<texture> albedo;

    lambertian(const color& a) : albedo(std::make_shared<solid_color>(a)) {}
    lambertian(std::shared_ptr<texture> a) : albedo(a) {}

    virtual BSDFSample sample(const ray&, const hit_record& rec) const override {
        onb uvw; uvw.build_from_w(rec.normal);
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
        double c = dot(rec.normal, wi);
        return c > 0.0 ? c / pi : 0.0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────

class metal : public material {
public:
    color  albedo;
    double fuzz;

    metal(const color& a, double f) : albedo(a), fuzz(f < 1.0 ? f : 1.0) {}

    virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
        vec3 r = unit_vector(reflect(unit_vector(wo.direction()), rec.normal)
                             + fuzz * random_in_unit_sphere());
        return { r, albedo, 1.0, true };
    }

    virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
    virtual double pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }
};

// ─────────────────────────────────────────────────────────────────────────────

class dielectric : public material {
public:
    double ir;

    dielectric(double index_of_refraction) : ir(index_of_refraction) {}

    virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
        vec3   ud    = unit_vector(wo.direction());
        double ratio = rec.front_face ? (1.0 / ir) : ir;
        double cos_t = fmin(dot(-ud, rec.normal), 1.0);
        double sin_t = std::sqrt(1.0 - cos_t * cos_t);
        vec3 dir = (ratio * sin_t > 1.0 || schlick(cos_t, ratio) > random_double())
                   ? reflect(ud, rec.normal)
                   : refract(ud, rec.normal, ratio);
        return { dir, color(1, 1, 1), 1.0, true };
    }

    virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
    virtual double pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }

private:
    static double schlick(double cos, double ri) {
        double r0 = (1.0 - ri) / (1.0 + ri); r0 *= r0;
        return r0 + (1.0 - r0) * std::pow(1.0 - cos, 5.0);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// GGX microfacet BRDF (Cook-Torrance)
//
// f = D(h) * G(wo,wi) * F(wo,h) / (4 |wo·n| |wi·n|)
//   + diffuse lobe (lambertian, energy-conserving via Fresnel)
//
// Sampling: visible normal distribution (VNDF, Heitz 2018)
// ─────────────────────────────────────────────────────────────────────────────

class ggx : public material {
public:
    color  base_color;
    double roughness; // perceptual — remapped to alpha = r^2 internally
    double metallic;  // 0 = dielectric, 1 = conductor

    ggx(const color& base, double r, double m = 0.0)
        : base_color(base),
          roughness(clamp(r, 0.001, 1.0)),
          metallic (clamp(m, 0.0,   1.0)) {}

    virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
        double alpha = roughness * roughness;
        onb uvw; uvw.build_from_w(rec.normal);

        // Transform wo to local (shading) space, flip to outgoing convention
        vec3 wo_l = -unit_vector(vec3(
            dot(wo.direction(), uvw.u()),
            dot(wo.direction(), uvw.v()),
            dot(wo.direction(), uvw.w())
        ));

        vec3 h_l = sample_vndf(wo_l, alpha);
        vec3 h   = unit_vector(uvw.local(h_l));
        vec3 wi  = unit_vector(reflect(-unit_vector(wo.direction()), h));

        if (dot(wi, rec.normal) <= 0.0)
            return { wi, color(0,0,0), 0.0, false };

        double p = vndf_pdf(wo_l, h_l, alpha);
        if (p <= 0.0) return { wi, color(0,0,0), 0.0, false };

        return { wi, eval(-unit_vector(wo.direction()), wi, rec.normal), p, false };
    }

    virtual color f(const ray& wo, const vec3& wi, const hit_record& rec) const override {
        if (dot(rec.normal, wi) <= 0.0) return color(0,0,0);
        return eval(-unit_vector(wo.direction()), wi, rec.normal);
    }

    virtual double pdf(const ray& wo, const vec3& wi, const hit_record& rec) const override {
        if (dot(rec.normal, wi) <= 0.0) return 0.0;
        double alpha = roughness * roughness;
        onb uvw; uvw.build_from_w(rec.normal);

        vec3 v = -unit_vector(wo.direction());
        vec3 wo_l = vec3(dot(v, uvw.u()), dot(v, uvw.v()), dot(v, uvw.w()));
        vec3 h    = unit_vector(v + wi);
        vec3 h_l  = vec3(dot(h, uvw.u()), dot(h, uvw.v()), dot(h, uvw.w()));

        return vndf_pdf(wo_l, h_l, alpha);
    }

private:
    // GGX NDF
    static double D(double ndoth, double a) {
        double a2 = a*a, d = ndoth*ndoth*(a2-1.0)+1.0;
        return a2 / (pi * d * d);
    }

    // Smith height-correlated G2
    static double G2(double ndotv, double ndotl, double a) {
        double a2 = a*a;
        double gv = ndotl * std::sqrt(a2 + (1.0-a2)*ndotv*ndotv);
        double gl = ndotv * std::sqrt(a2 + (1.0-a2)*ndotl*ndotl);
        return 2.0*ndotv*ndotl / (gv + gl + 1e-7);
    }

    // Smith G1 (used in VNDF pdf)
    static double G1(double ndotv, double a) {
        double a2 = a*a;
        return 2.0*ndotv / (ndotv + std::sqrt(a2 + (1.0-a2)*ndotv*ndotv));
    }

    // Schlick Fresnel — F0 blends between dielectric (0.04) and metal (base_color)
    color F(double vdoth) const {
        color f0 = color(0.04,0.04,0.04)*(1.0-metallic) + base_color*metallic;
        return f0 + (color(1,1,1)-f0) * std::pow(1.0-vdoth, 5.0);
    }

    // Full Cook-Torrance evaluation including diffuse lobe
    color eval(const vec3& v, const vec3& l, const vec3& n) const {
        double a      = roughness * roughness;
        vec3   h      = unit_vector(v + l);
        double ndotv  = std::max(dot(n,v), 1e-7);
        double ndotl  = std::max(dot(n,l), 1e-7);
        double ndoth  = std::max(dot(n,h), 1e-7);
        double vdoth  = std::max(dot(v,h), 1e-7);

        color  Fval   = F(vdoth);
        double Dval   = D(ndoth, a);
        double Gval   = G2(ndotv, ndotl, a);

        color specular = Fval * (Dval * Gval / (4.0 * ndotv * ndotl));
        color diffuse  = base_color / pi * (1.0 - metallic) * (color(1,1,1) - Fval);

        return (specular + diffuse) * ndotl;
    }

    // Visible normal distribution sampling (Heitz 2018)
    static vec3 sample_vndf(const vec3& wo, double a) {
        vec3 vh = unit_vector(vec3(a*wo.x(), a*wo.y(), wo.z()));

        double lensq = vh.x()*vh.x() + vh.y()*vh.y();
        vec3 t1 = lensq > 0.0
                  ? vec3(-vh.y(), vh.x(), 0.0) / std::sqrt(lensq)
                  : vec3(1,0,0);
        vec3 t2 = cross(vh, t1);

        double r  = std::sqrt(random_double());
        double phi = 2.0 * pi * random_double();
        double t  = r * std::cos(phi);
        double s  = r * std::sin(phi);
        double bz = std::sqrt(std::max(0.0, 1.0 - t*t - s*s));

        vec3 nh = t*t1 + s*t2 + bz*vh;
        return unit_vector(vec3(a*nh.x(), a*nh.y(), std::max(0.0, nh.z())));
    }

    // VNDF pdf in local space
    static double vndf_pdf(const vec3& wo, const vec3& h, double a) {
        double ndotwo = std::max(wo.z(), 1e-7);
        double hdotwo = std::max(dot(h, wo), 1e-7);
        double ndoth  = std::max(h.z(), 1e-7);
        return D(ndoth, a) * G1(ndotwo, a) * hdotwo / (4.0 * ndotwo * hdotwo);
    }
};

// ─────────────────────────────────────────────────────────────────────────────

class diffuse_light : public material {
public:
    std::shared_ptr<texture> emit;

    diffuse_light(std::shared_ptr<texture> a) : emit(a) {}
    diffuse_light(color c) : emit(std::make_shared<solid_color>(c)) {}

    virtual color emitted(const ray&, const hit_record& rec,
                          double u, double v, const point3& p) const override {
        if (!rec.front_face) return color(0,0,0);
        return emit->value(u, v, p);
    }

    virtual BSDFSample sample(const ray&, const hit_record&) const override {
        return { vec3(0,0,0), color(0,0,0), 0.0, false };
    }
    virtual color  f  (const ray&, const vec3&, const hit_record&) const override { return color(0,0,0); }
    virtual double pdf(const ray&, const vec3&, const hit_record&) const override { return 0.0; }
};

// ─────────────────────────────────────────────────────────────────────────────

class isotropic : public material {
public:
    std::shared_ptr<texture> albedo;

    isotropic(const color& c) : albedo(std::make_shared<solid_color>(c)) {}
    isotropic(std::shared_ptr<texture> a) : albedo(a) {}

    virtual BSDFSample sample(const ray&, const hit_record& rec) const override {
        vec3 wi  = random_unit_vector();
        color rho = albedo->value(rec.u, rec.v, rec.p);
        return { wi, rho / (4.0*pi), 1.0/(4.0*pi), false };
    }

    virtual color f(const ray&, const vec3&, const hit_record& rec) const override {
        return albedo->value(rec.u, rec.v, rec.p) / (4.0 * pi);
    }

    virtual double pdf(const ray&, const vec3&, const hit_record&) const override {
        return 1.0 / (4.0 * pi);
    }
};

#endif