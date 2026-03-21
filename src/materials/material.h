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

// ─────────────────────────────────────────────────────────────────────────────
// Rough dielectric (microfacet transmission + reflection)
//
// Uses the same GGX NDF and Smith G2 as the ggx conductor class.
// At each sample, Fresnel decides whether to reflect or refract.
// Reflection: VNDF-sampled GGX specular lobe (same as ggx class)
// Transmission: GGX microfacet BTDF via Snell's law refraction
//
// Reference: Walter et al. 2007 "Microfacet Models for Refraction"
// ─────────────────────────────────────────────────────────────────────────────

class rough_dielectric : public material {
public:
    color  tint;         // colour tint of the transmitted light (1,1,1 = clear)
    double roughness;    // perceptual roughness (remapped to alpha = r^2)
    double ior;          // index of refraction (glass ~1.5, water ~1.33)

    rough_dielectric(const color& t, double r, double ior = 1.5)
        : tint(t),
          roughness(clamp(r, 0.001, 1.0)),
          ior(ior) {}

    virtual BSDFSample sample(const ray& wo, const hit_record& rec) const override {
        double alpha  = roughness * roughness;
        double eta    = rec.front_face ? (1.0 / ior) : ior;
        vec3   v      = -unit_vector(wo.direction());

        onb uvw; uvw.build_from_w(rec.normal);

        // Transform v to local space
        vec3 v_l = vec3(dot(v, uvw.u()), dot(v, uvw.v()), dot(v, uvw.w()));

        // Sample microfacet normal from VNDF
        vec3 h_l = sample_vndf(v_l, alpha);
        vec3 h   = unit_vector(uvw.local(h_l));

        // Make sure h is on the same side as v
        if (dot(h, v) < 0.0) h = -h;

        double vdoth   = clamp(dot(v, h), 0.0, 1.0);
        double F_val   = schlick(vdoth, eta);

        if (random_double() < F_val) {
            // ── Reflection lobe ──────────────────────────────────────────────
            vec3 wi = unit_vector(reflect(-v, h));
            if (dot(wi, rec.normal) <= 0.0)
                return { wi, color(0,0,0), 0.0, false };

            double p = F_val * vndf_pdf(v_l, h_l, alpha);
            if (p <= 0.0) return { wi, color(0,0,0), 0.0, false };

            // Cook-Torrance specular BRDF value
            double ndotv = std::max(dot(rec.normal, v),  1e-7);
            double ndotl = std::max(dot(rec.normal, wi), 1e-7);
            double ndoth = std::max(dot(rec.normal, h),  1e-7);
            double Dval  = D(ndoth, alpha);
            double Gval  = G2(ndotv, ndotl, alpha);
            color  f_val = color(F_val,F_val,F_val) * (Dval * Gval / (4.0*ndotv*ndotl));

            return { wi, f_val * ndotl / p, p, false };

        } else {
            // ── Transmission lobe ────────────────────────────────────────────
            vec3 wi_try;
            bool tir = !refract_microfacet(-v, h, eta, wi_try);
            if (tir) {
                // Total internal reflection — fall back to specular
                vec3 wi = unit_vector(reflect(-v, h));
                return { wi, color(F_val,F_val,F_val), 1.0, true };
            }
            vec3 wi = unit_vector(wi_try);

            if (dot(wi, rec.normal) >= 0.0)
                return { wi, color(0,0,0), 0.0, false }; // wrong side

            // Walter 2007 BTDF pdf
            double p = (1.0 - F_val) * vndf_pdf_transmission(v_l, h_l, wi, h, eta, alpha);
            if (p <= 0.0) return { wi, color(0,0,0), 0.0, false };

            double ndotv  = std::max( dot(rec.normal, v),   1e-7);
            double ndotl  = std::max(-dot(rec.normal, wi),  1e-7);
            double ndoth  = std::max( dot(rec.normal, h),   1e-7);
            double idoth  = std::max(-dot(wi, h),           1e-7);
            double Dval   = D(ndoth, alpha);
            double Gval   = G2(ndotv, ndotl, alpha);

            // Walter BTDF
            double denom  = (vdoth + eta * idoth);
            double f_val  = (1.0 - F_val) * Dval * Gval * vdoth * idoth
                            / (ndotv * denom * denom);

            return { wi, tint * f_val * ndotl / p, p, false };
        }
    }

    virtual color f(const ray& wo, const vec3& wi,
                    const hit_record& rec) const override {
        // Not used directly (we importance sample), return 0
        return color(0,0,0);
    }

    virtual double pdf(const ray& wo, const vec3& wi,
                       const hit_record& rec) const override {
        return 0.0;
    }

private:
    // Schlick Fresnel for dielectric
    static double schlick(double cos_theta, double eta) {
        double r0 = (1.0 - eta) / (1.0 + eta); r0 *= r0;
        return r0 + (1.0 - r0) * std::pow(1.0 - cos_theta, 5.0);
    }

    // GGX NDF (same as ggx class)
    static double D(double ndoth, double a) {
        double a2 = a*a, d = ndoth*ndoth*(a2-1.0)+1.0;
        return a2 / (pi * d * d);
    }

    // Smith G2 (same as ggx class)
    static double G2(double ndotv, double ndotl, double a) {
        double a2 = a*a;
        double gv = ndotl * std::sqrt(a2 + (1.0-a2)*ndotv*ndotv);
        double gl = ndotv * std::sqrt(a2 + (1.0-a2)*ndotl*ndotl);
        return 2.0*ndotv*ndotl / (gv + gl + 1e-7);
    }

    static double G1(double ndotv, double a) {
        double a2 = a*a;
        return 2.0*ndotv / (ndotv + std::sqrt(a2 + (1.0-a2)*ndotv*ndotv));
    }

    // Refract with microfacet normal h instead of geometric normal
    static bool refract_microfacet(const vec3& v, const vec3& h,
                                    double eta, vec3& refracted) {
        double cos_i = dot(v, h);
        double sin2_t = eta*eta * (1.0 - cos_i*cos_i);
        if (sin2_t >= 1.0) return false; // TIR
        refracted = (eta * cos_i - std::sqrt(1.0 - sin2_t)) * h - eta * v;
        return true;
    }

    // VNDF sampling (same as ggx class)
    static vec3 sample_vndf(const vec3& wo, double a) {
        vec3 vh = unit_vector(vec3(a*wo.x(), a*wo.y(), wo.z()));
        double lensq = vh.x()*vh.x() + vh.y()*vh.y();
        vec3 t1 = lensq > 0.0
                  ? vec3(-vh.y(), vh.x(), 0.0) / std::sqrt(lensq)
                  : vec3(1,0,0);
        vec3 t2 = cross(vh, t1);
        double r  = std::sqrt(random_double());
        double phi = 2.0*pi*random_double();
        double t  = r*std::cos(phi), s = r*std::sin(phi);
        double bz = std::sqrt(std::max(0.0, 1.0-t*t-s*s));
        vec3 nh = t*t1 + s*t2 + bz*vh;
        return unit_vector(vec3(a*nh.x(), a*nh.y(), std::max(0.0, nh.z())));
    }

    static double vndf_pdf(const vec3& wo, const vec3& h, double a) {
        double ndotwo = std::max(wo.z(), 1e-7);
        double hdotwo = std::max(dot(h, wo), 1e-7);
        double ndoth  = std::max(h.z(), 1e-7);
        return D(ndoth, a) * G1(ndotwo, a) * hdotwo / (4.0*ndotwo*hdotwo);
    }

    static double vndf_pdf_transmission(const vec3& v_l, const vec3& h_l,
                                         const vec3& wi, const vec3& h,
                                         double eta, double a) {
        double vdoth = std::max(dot(vec3(v_l.x(),v_l.y(),v_l.z()), h_l), 1e-7);
        double idoth = std::max(-dot(wi, h), 1e-7);
        double denom = vdoth + eta * idoth;
        double jacobian = eta*eta * idoth / (denom*denom);
        double ndoth = std::max(h_l.z(), 1e-7);
        double ndov  = std::max(v_l.z(), 1e-7);
        return D(ndoth, a) * G1(ndov, a) * vdoth / (4.0*ndov*vdoth) * jacobian;
    }
};

#endif