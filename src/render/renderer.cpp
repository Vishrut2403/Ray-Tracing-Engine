#include "render/renderer.h"

#include <omp.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <mutex>

#include "render/integrator.h"
#include "render/tile.h"

Renderer::Renderer(int spp, int depth, int tile_size)
    : samples_per_pixel(spp),
      max_depth(depth),
      tile_size(tile_size)
{}

void Renderer::render(
    const Scene& scene,
    Framebuffer& fb,
    const camera& cam,
    const color& background
)
{
    auto world  = scene.world;
    auto lights = scene.lights;

    const int width  = fb.get_width();
    const int height = fb.get_height();

    int tiles_done = 0;

    auto tiles = generate_tiles(width, height, tile_size);

    // Center-priority ordering (optional but nice UX)
    float cx = width  * 0.5f;
    float cy = height * 0.5f;

    std::sort(tiles.begin(), tiles.end(),
        [cx, cy](const Tile& a, const Tile& b)
    {
        float ax = (a.x0 + a.x1) * 0.5f;
        float ay = (a.y0 + a.y1) * 0.5f;
        float bx = (b.x0 + b.x1) * 0.5f;
        float by = (b.y0 + b.y1) * 0.5f;

        float da = (ax - cx)*(ax - cx) + (ay - cy)*(ay - cy);
        float db = (bx - cx)*(bx - cx) + (by - cy)*(by - cy);

        return da < db;
    });

#pragma omp parallel for schedule(dynamic)
    for (size_t t = 0; t < tiles.size(); ++t)
    {
        const Tile& tile = tiles[t];

        const int tile_w = tile.x1 - tile.x0;
        const int tile_h = tile.y1 - tile.y0;

        // Local tile accumulation buffer
        std::vector<color> tile_buffer(tile_w * tile_h);

        // ---- Render Tile ----
        for (int j = tile.y0; j < tile.y1; ++j)
        {
            for (int i = tile.x0; i < tile.x1; ++i)
            {
                color pixel_color(0, 0, 0);

                for (int s = 0; s < samples_per_pixel; ++s)
                {
                    double u = (i + random_double()) / (width - 1);
                    double v = (j + random_double()) / (height - 1);

                    ray r = cam.get_ray(u, v);

                    pixel_color += Li(
                        r,
                        background,
                        world,
                        lights,
                        max_depth
                    );
                }

                pixel_color /= double(samples_per_pixel);
                
                int local_x = i - tile.x0;
                int local_y = j - tile.y0;

                tile_buffer[local_y * tile_w + local_x] = pixel_color;
            }
        }

        // ---- Commit Tile To Framebuffer (lock once) ----
        {
            std::lock_guard<std::mutex> lock(fb.mtx);

            for (int j = tile.y0; j < tile.y1; ++j)
            {
                for (int i = tile.x0; i < tile.x1; ++i)
                {
                    int local_x = i - tile.x0;
                    int local_y = j - tile.y0;

                    fb.set(i, j, tile_buffer[local_y * tile_w + local_x]);
                }
            }
        }

        // ---- Progress Reporting ----
#pragma omp critical
        {
            ++tiles_done;
            std::cerr << "\rTiles completed: "
                      << tiles_done << " / "
                      << tiles.size()
                      << std::flush;
        }
    }

    std::cerr << "\nRendering finished.\n";
}