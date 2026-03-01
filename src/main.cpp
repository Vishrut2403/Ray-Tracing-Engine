#include "app/render_config.h"
#include "app/cli.h"
#include "scenes/scene_factory.h"
#include "core/camera_factory.h"
#include "render/renderer.h"
#include "render/framebuffer.h"
#include "io/image_writer.h"
#include "viewer/preview_window.h"

#include <thread>
#include <atomic>
#include <mutex>

int main(int argc, char** argv)
{
    RenderConfig config = parse_cli(argc, argv);

    Scene scene = SceneFactory::build(config.feature);
    camera cam = CameraFactory::build(config);

    Framebuffer fb(config.width, config.height);

    Renderer renderer(
        config.samples,
        config.max_depth,
        config.tile_size
    );

    PreviewWindow preview(config.width, config.height);

    std::atomic<bool> render_done = false;

    std::thread render_thread([&]() {
        renderer.render(scene, fb, cam, config.background);
        render_done = true;
    });

    // Keep preview alive until user closes it
    while (!preview.should_close())
    {
        preview.poll_events();

        {
            std::lock_guard<std::mutex> lock(fb.mtx);
            preview.update(fb.raw_data(), 1.0f / config.samples);
        }
    }

    render_thread.join();

    // Write final image after window closes
    ImageWriter::write_ppm(
        config.output_path,
        fb,
        config.samples
    );

    return 0;
}