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
    int depth
) const
{
    hit_record rec;

    if (depth <= 0)
        return color(0,0,0);

    if (!world.hit(r, interval(0.001, infinity), rec))
        return background;

    color emitted =
        rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

    scatter_record srec;

    if (!rec.mat_ptr->scatter(r, rec, srec))
        return emitted;

    if (srec.is_specular) {
        return emitted +
               srec.attenuation *
               Li(
                   srec.specular_ray,
                   background,
                   world,
                   lights,
                   depth - 1
               );
    }

    if (depth >= 5) {

        double luminance =
            0.2126 * srec.attenuation.x() +
            0.7152 * srec.attenuation.y() +
            0.0722 * srec.attenuation.z();

        double survival_prob = std::min(0.95, luminance);

        if (random_double() > survival_prob)
            return emitted;

        srec.attenuation /= survival_prob;
    }

    auto light_pdf =
        std::make_shared<hittable_pdf>(lights, rec.p);

    mixture_pdf mixed_pdf(light_pdf, srec.pdf_ptr);

    ray scattered(
        rec.p,
        mixed_pdf.generate(),
        r.time()
    );

    double pdf_val =
        mixed_pdf.value(scattered.direction());

    if (pdf_val <= 1e-8)
        return emitted;

    double scattering_pdf =
        rec.mat_ptr->scattering_pdf(
            r, rec, scattered);

    return emitted +
           srec.attenuation *
           scattering_pdf *
           Li(
               scattered,
               background,
               world,
               lights,
               depth - 1
           ) / pdf_val;
}