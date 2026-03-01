#include "image_writer.h"

#include <fstream>
#include <cmath>

#include "core/rtweekend.h"

void ImageWriter::write_ppm(
    const std::string& path,
    const Framebuffer& fb,
    int /*samples_per_pixel*/   // no longer used
) {
    std::ofstream out(path);

    int width  = fb.get_width();
    int height = fb.get_height();

    out << "P3\n"
        << width << " " << height << "\n255\n";

    for (int j = height - 1; j >= 0; --j) {
        for (int i = 0; i < width; ++i) {

            // Already averaged in renderer (linear space)
            color pixel_color = fb.get(i, j);

            // Gamma correction (gamma 2.0)
            double r = std::sqrt(pixel_color.x());
            double g = std::sqrt(pixel_color.y());
            double b = std::sqrt(pixel_color.z());

            // Clamp after gamma
            int ir = static_cast<int>(256 * clamp(r, 0.0, 0.999));
            int ig = static_cast<int>(256 * clamp(g, 0.0, 0.999));
            int ib = static_cast<int>(256 * clamp(b, 0.0, 0.999));

            out << ir << " " << ig << " " << ib << "\n";
        }
    }

    out.close();
}