#include "cuda/cuda_renderer.h"
#include "cuda/gpu_integrator.cuh"
#include "cuda/scene_uploader.h"
#include "cuda/cuda_rand.cuh"
#include "render/framebuffer.h"
#include "core/camera.h"
#include "scenes/cornell_scene.h"

#include <cstdio>

static GpuCamera make_gpu_camera(const camera& cam) {
    GpuCamera gc;
    gc.origin      = cam.get_origin();
    gc.lower_left  = cam.get_lower_left();
    gc.horizontal  = cam.get_horizontal();
    gc.vertical    = cam.get_vertical();
    gc.u           = cam.get_u();
    gc.v           = cam.get_v();
    gc.lens_radius = cam.get_lens_radius();
    return gc;
}

void cuda_render(const Scene& scene,
                 Framebuffer& fb,
                 const camera& cam,
                 const color& background,
                 int spp,
                 int max_depth) {

    const int W = fb.get_width();
    const int H = fb.get_height();
    const int N = W * H;

    GpuScene gpu_scene = SceneUploader::build_and_upload();

    curandState* d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));
    {
        int t = 128, b = (N + t - 1) / t;
        cuda_rand_init<<<b, t>>>(d_states, 1234ULL, N);
        cudaDeviceSynchronize();
    }

    float* d_fb;
    cudaMalloc(&d_fb, N * 3 * sizeof(float));

    GpuCamera gpu_cam = make_gpu_camera(cam);

    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16);

    printf("[CUDA] launching %dx%d  spp=%d  depth=%d\n", W, H, spp, max_depth);

    render_kernel<<<blocks, threads>>>(
        d_fb, W, H, spp, max_depth,
        gpu_cam, background,
        gpu_scene.d_hittables, gpu_scene.n_hittables,
        gpu_scene.d_materials,
        gpu_scene.d_light_ids, gpu_scene.n_lights,
        d_states
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("[CUDA] error: %s\n", cudaGetErrorString(err));
    else
        printf("[CUDA] done\n");

    float* h_fb = new float[N * 3];
    cudaMemcpy(h_fb, d_fb, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            int id = y * W + x;
            fb.set(x, y, color(h_fb[id*3], h_fb[id*3+1], h_fb[id*3+2]));
        }

    delete[] h_fb;
    cudaFree(d_fb);
    cudaFree(d_states);
    gpu_scene.free_device();
}
