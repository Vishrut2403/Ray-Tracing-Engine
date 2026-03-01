#include "path_integrator.h"
#include "rtweekend.h"
#include "material.h"
#include "hittable_pdf.h"
#include "mixture_pdf.h"

color PathIntegrator::Li(
    const ray& r,
    const color& background,
    const hittable& world,
    const std::shared_ptr<hittable>& lights,
    int max_depth
) const
{
    ray current_ray = r;
    color L(0,0,0);
    color beta(1,1,1);

    for (int depth = 0; depth < max_depth; ++depth) {

        hit_record rec;

        if (!world.hit(current_ray, interval(0.001, infinity), rec)) {
            L += beta * background;
            break;
        }

        // Emission
        L += beta * rec.mat_ptr->emitted(
                current_ray, rec, rec.u, rec.v, rec.p);

        scatter_record srec;

        if (!rec.mat_ptr->scatter(current_ray, rec, srec))
            break;

        if (srec.is_specular) {
            beta *= srec.attenuation;
            current_ray = srec.specular_ray;
            continue;
        }

        // Russian roulette
        if (depth >= 5) {
            double luminance =
                0.2126 * beta.x() +
                0.7152 * beta.y() +
                0.0722 * beta.z();

            double survival_prob = std::min(0.95, luminance);

            if (random_double() > survival_prob)
                break;

            beta /= survival_prob;
        }

        // ⚠ TEMP: still using shared_ptr (next step removes it)
        auto light_pdf =
            std::make_shared<hittable_pdf>(lights, rec.p);

        mixture_pdf mixed_pdf(light_pdf, srec.pdf_ptr);

        vec3 new_direction = mixed_pdf.generate();
        double pdf_val = mixed_pdf.value(new_direction);

        if (pdf_val <= 1e-8)
            break;

        ray scattered(rec.p, new_direction, current_ray.time());

        double scattering_pdf =
            rec.mat_ptr->scattering_pdf(
                current_ray, rec, scattered);

        beta *= srec.attenuation *
                scattering_pdf / pdf_val;

        current_ray = scattered;
    }

    return L;
}