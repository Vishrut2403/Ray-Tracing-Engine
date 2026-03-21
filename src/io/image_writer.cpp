#include "image_writer.h"
#include <fstream>
#include <cmath>
#include "core/rtweekend.h"

static inline double aces_filmic(double x) {
    const double a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x*(a*x+b)) / (x*(c*x+d)+e), 0.0, 1.0);
}

static constexpr double EXPOSURE = 1.0;

static inline double gamma_encode(double x) {
    return std::pow(clamp(x, 0.0, 1.0), 1.0 / 2.2);
}

static constexpr double FIREFLY_CLAMP = 20.0;

static inline color firefly_clamp(color c) {
    double lum = 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
    if (lum > FIREFLY_CLAMP && lum > 0.0)
        c = c * (FIREFLY_CLAMP / lum);
    return c;
}

void ImageWriter::write_ppm(
    const std::string& path,
    const Framebuffer& fb,
    int /*samples_per_pixel*/
) {
    std::ofstream out(path);
    int width  = fb.get_width();
    int height = fb.get_height();
    out << "P3\n" << width << " " << height << "\n255\n";

    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {
            color c = fb.get(i, j);

            c = firefly_clamp(c);

            c = c * EXPOSURE;

            double r = aces_filmic(c.x());
            double g = aces_filmic(c.y());
            double b = aces_filmic(c.z());

            r = gamma_encode(r);
            g = gamma_encode(g);
            b = gamma_encode(b);

            int ir = static_cast<int>(256 * clamp(r, 0.0, 0.999));
            int ig = static_cast<int>(256 * clamp(g, 0.0, 0.999));
            int ib = static_cast<int>(256 * clamp(b, 0.0, 0.999));
            out << ir << " " << ig << " " << ib << "\n";
        }
    }
    out.close();
}