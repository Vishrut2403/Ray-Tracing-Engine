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
#include <string>
#include <cstring>

static bool has_flag(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], flag) == 0) return true;
    return false;
}

int main(int argc, char** argv)
{
    RenderConfig config = parse_cli(argc, argv);
    bool use_gpu = has_flag(argc, argv, "--device") &&
                   [&]{ for(int i=1;i<argc-1;i++) if(!strcmp(argv[i],"--device")) return !strcmp(argv[i+1],"gpu"); return false; }();

    if (config.feature == "furnace")
        apply_furnace_preset(config);

    Scene  scene = SceneFactory::build(config.feature);
    camera cam   = CameraFactory::build(config);
    Framebuffer fb(config.width, config.height);

    if (use_gpu) {
        printf("[main] using CUDA renderer\n");
        cuda_render(scene, fb, cam, config.background,
                    config.samples, config.max_depth);
        ImageWriter::write_ppm(config.output_path, fb, config.samples);
        return 0;
    }

    // CPU path
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
        preview.update(fb.raw_data(), 1.0f / config.samples);
    }

    render_thread.join();
    ImageWriter::write_ppm(config.output_path, fb, config.samples);
    return 0;
}