#include "render/integrator.h"
#include "materials/material.h"
#include <algorithm>

static inline double power_heuristic(double pdf_a, double pdf_b)
{
    double a2 = pdf_a * pdf_a;
    double b2 = pdf_b * pdf_b;
    return a2 / (a2 + b2 + 1e-12);
}

color Li(
    const ray& initial_ray,
    const color& background,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<hittable_list>& lights,
    int max_depth
) {
    ray r = initial_ray;

    color L(0,0,0);
    color beta(1,1,1);

    bool specular_bounce = true;

    for (int depth = 0; depth < max_depth; ++depth) {

        hit_record rec;

        if (!world->hit(r, interval(0.001, infinity), rec)) {
            L += beta * background;
            break;
        }

        color emitted =
            rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

        if (specular_bounce || depth == 0)
            L += beta * emitted;

        if (!specular_bounce &&
            lights &&
            !lights->objects.empty()) {

            vec3 to_light = lights->random(rec.p);

            double distance2 = to_light.length_squared();
            double distance = std::sqrt(distance2);

            vec3 wi = to_light / distance;

            double light_pdf =
                lights->pdf_value(rec.p, wi);

            if (light_pdf > 0.0) {

                ray shadow_ray(rec.p, wi, r.time());

                hit_record shadow_hit;

                if (!world->hit(
                        shadow_ray,
                        interval(0.001, distance - 1e-4),
                        shadow_hit)) {

                    ray light_ray(rec.p, wi, r.time());

                    hit_record light_hit;

                    if (world->hit(light_ray,
                                   interval(0.001, infinity),
                                   light_hit)) {

                        color Le =
                            light_hit.mat_ptr->emitted(
                                light_ray,
                                light_hit,
                                light_hit.u,
                                light_hit.v,
                                light_hit.p
                            );

                        if (Le.length_squared() > 0.0) {

                            color f =
                                rec.mat_ptr->f(r, wi, rec);

                            double bsdf_pdf =
                                rec.mat_ptr->pdf(r, wi, rec);

                            double weight =
                                power_heuristic(light_pdf,
                                                bsdf_pdf);

                            double cos_theta =
                                std::abs(dot(rec.normal, wi));

                            L += beta *
                                 f *
                                 Le *
                                 cos_theta *
                                 weight /
                                 light_pdf;
                        }
                    }
                }
            }
        }

        BSDFSample bs =
            rec.mat_ptr->sample(r, rec);

        if (bs.pdf <= 0.0)
            break;

        if (bs.is_delta) {

            beta *= bs.f;
            specular_bounce = true;

        } else {

            double cos_theta =
                std::abs(dot(rec.normal, bs.wi));

            beta *= bs.f * cos_theta / bs.pdf;
            specular_bounce = false;
        }

        if (depth >= 3) {

            double max_comp = beta.max_component();

            double survival_prob = std::clamp(max_comp, 0.05, 0.95);

            if (random_double() > survival_prob)
                break;

            beta /= survival_prob;
        }

        // Spawn next ray
        r = ray(rec.p, bs.wi, r.time());
    }

    return L;
}