#include "render/integrator.h"
#include "materials/material.h"
#include <algorithm>

static inline double power_heuristic(double pdf_a, double pdf_b) {
    double a2 = pdf_a * pdf_a, b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-12);
}

color Li(
    const ray& initial_ray,
    const color& background,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<hittable_list>& lights,
    int max_depth
) {
    ray   r    = initial_ray;
    color L    (0, 0, 0);
    color beta (1, 1, 1);
    bool  specular_bounce = false;
    double last_bsdf_pdf  = 0.0;

    for (int depth = 0; depth < max_depth; ++depth) {

        hit_record rec;

        if (!world->hit(r, interval(0.001, infinity), rec)) {
            if (specular_bounce || !(lights && !lights->objects.empty())) {
                L += beta * background;
            } else {
                double lp = lights->pdf_value(rec.p, r.direction());
                L += beta * background * power_heuristic(last_bsdf_pdf, lp);
            }
            break;
        }

        color emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
        if (emitted.length_squared() > 0.0) {
            if (specular_bounce) {
                L += beta * emitted;
            } else {
                double lp = (lights && !lights->objects.empty())
                            ? lights->pdf_value(rec.p, r.direction()) : 0.0;
                L += beta * emitted * power_heuristic(last_bsdf_pdf, lp);
            }
        }

        if (!specular_bounce && lights && !lights->objects.empty()) {
            vec3   to_light  = lights->random(rec.p);
            double distance  = to_light.length();
            vec3   wi        = to_light / distance;
            double light_pdf = lights->pdf_value(rec.p, wi);

            if (light_pdf > 0.0) {
                hit_record shadow_rec;
                bool occluded = world->hit(
                    ray(rec.p, wi, r.time()),
                    interval(0.001, distance - 1e-4),
                    shadow_rec
                );

                if (!occluded) {
                    hit_record light_rec;
                    if (world->hit(ray(rec.p, wi, r.time()),
                                   interval(0.001, infinity), light_rec)) {
                        color Le = light_rec.mat_ptr->emitted(
                            ray(rec.p, wi, r.time()),
                            light_rec, light_rec.u, light_rec.v, light_rec.p
                        );
                        if (Le.length_squared() > 0.0) {
                            double bsdf_pdf  = rec.mat_ptr->pdf(r, wi, rec);
                            double cos_theta = std::abs(dot(rec.normal, wi));
                            L += beta * rec.mat_ptr->f(r, wi, rec)
                                 * Le * cos_theta
                                 * power_heuristic(light_pdf, bsdf_pdf)
                                 / light_pdf;
                        }
                    }
                }
            }
        }

        BSDFSample bs = rec.mat_ptr->sample(r, rec);
        if (bs.pdf <= 0.0) break;

        last_bsdf_pdf = bs.pdf;

        if (bs.is_delta) {
            beta *= bs.f;
            specular_bounce = true;
        } else {
            beta *= bs.f * std::abs(dot(rec.normal, bs.wi)) / bs.pdf;
            specular_bounce = false;
        }

        if (depth >= 3) {
            double survival = std::clamp(beta.max_component(), 0.05, 0.95);
            if (random_double() > survival) break;
            beta /= survival;
        }

        r = ray(rec.p, bs.wi, r.time());
    }

    return L;
}