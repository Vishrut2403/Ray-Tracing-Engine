#include "app/render_config.h"
#include "app/cli.h"
#include "scenes/scene_factory.h"
#include "core/camera_factory.h"
#include "render/renderer.h"
#include "render/framebuffer.h"
#include "io/image_writer.h"
#include "viewer/preview_window.h"
#include "cuda/cuda_renderer.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>

static bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return true;
    return false;
}

static bool get_flag_value(int argc, char** argv,
                            const char* flag, const char* value) {
    for (int i = 1; i < argc - 1; ++i)
        if (strcmp(argv[i], flag) == 0 && strcmp(argv[i+1], value) == 0)
            return true;
    return false;
}

int main(int argc, char** argv)
{
    RenderConfig config = parse_cli(argc, argv);
    bool use_gpu     = get_flag_value(argc, argv, "--device", "gpu");
    bool no_preview  = has_flag(argc, argv, "--no-preview");

    if (config.feature == "furnace")
        apply_furnace_preset(config);

    Scene  scene = SceneFactory::build(config.feature);
    camera cam   = CameraFactory::build(config);
    Framebuffer fb(config.width, config.height);

    if (use_gpu) {
        if (no_preview) {
            cuda_render(scene, fb, cam, config.background,
                        config.samples, config.max_depth, nullptr);
        } else {
            PreviewWindow preview(config.width, config.height);
            std::thread render_thread([&]() {
                cuda_render(scene, fb, cam, config.background,
                            config.samples, config.max_depth, &preview);
            });
            while (!preview.should_close()) {
                preview.poll_events();
                std::lock_guard<std::mutex> lock(fb.mtx);
                preview.update(fb.raw_data(), 1.0f);
            }
            render_thread.join();
        }
    } else {
        if (no_preview) {
            Renderer renderer(config.samples, config.max_depth, config.tile_size);
            renderer.render(scene, fb, cam, config.background);
        } else {
            Renderer renderer(config.samples, config.max_depth, config.tile_size);
            PreviewWindow preview(config.width, config.height);
            std::atomic<bool> render_done = false;
            std::thread render_thread([&]() {
                renderer.render(scene, fb, cam, config.background);
                render_done = true;
            });
            while (!preview.should_close()) {
                preview.poll_events();
                std::lock_guard<std::mutex> lock(fb.mtx);
                preview.update(fb.raw_data(), 1.0f);
            }
            render_thread.join();
        }
    }

    ImageWriter::write_ppm(config.output_path, fb, config.samples);
    return 0;
}