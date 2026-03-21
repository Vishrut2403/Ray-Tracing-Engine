#include "cuda/cuda_renderer.h"
#include "cuda/gpu_integrator.cuh"
#include "cuda/gpu_env_map.cuh"
#include "cuda/gpu_env_uploader.h"
#include "cuda/gpu_mesh_uploader.h"
#include "cuda/scene_uploader.h"
#include "cuda/cuda_rand.cuh"
#include "render/framebuffer.h"
#include "core/camera.h"
#include "scenes/cornell_scene.h"

#include <cstdio>
#include <mutex>

void GpuScene::free_device() {
    if (d_materials) { cudaFree(d_materials); d_materials = nullptr; }
    if (d_hittables) { cudaFree(d_hittables); d_hittables = nullptr; }
    if (d_bvh_nodes) { cudaFree(d_bvh_nodes); d_bvh_nodes = nullptr; }
    if (d_light_ids) { cudaFree(d_light_ids); d_light_ids = nullptr; }
}

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
                 int spp, int max_depth,
                 PreviewWindow* preview,
                 const std::string& scene_name) {

    const int W = fb.get_width();
    const int H = fb.get_height();
    const int N = W * H;

    GpuMesh gpu_mesh;
    GpuScene gpu_scene;

    if (scene_name == "bunny") {
        gpu_scene = SceneUploader::build_bunny_scene(scene_name, gpu_mesh);
    } else {
        gpu_scene = SceneUploader::build_and_upload(scene_name);
    }

    GpuEnvMap* d_env_map = nullptr;
    if (scene.env) {
        d_env_map = upload_env_map(*scene.env);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    curandState* d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));
    {
        int t = 128, b = (N+t-1)/t;
        cuda_rand_init<<<b,t,0,stream>>>(d_states, 1234ULL, N);
        cudaStreamSynchronize(stream);
    }

    float* d_accum;
    cudaMalloc(&d_accum, N * 3 * sizeof(float));
    cudaMemsetAsync(d_accum, 0, N * 3 * sizeof(float), stream);

    float* h_accum;
    cudaMallocHost(&h_accum, N * 3 * sizeof(float));

    GpuCamera gpu_cam = make_gpu_camera(cam);
    dim3 threads(16, 16);
    dim3 blocks((W+15)/16, (H+15)/16);

    const int BATCH         = 32;
    const int PREVIEW_EVERY = 4;
    int samples_done = 0, batch_count = 0;

    printf("[CUDA] %dx%d  spp=%d  depth=%d  scene=%s  mesh=%s  env=%s\n",
           W, H, spp, max_depth, scene_name.c_str(),
           gpu_scene.n_triangles > 0 ? "yes" : "no",
           d_env_map ? "yes" : "no");

    while (samples_done < spp) {
        int batch = (samples_done + BATCH <= spp) ? BATCH : (spp - samples_done);

        accumulate_kernel<<<blocks, threads, 0, stream>>>(
            d_accum, W, H, batch, max_depth, gpu_cam, background,
            gpu_scene.d_hittables, gpu_scene.n_hittables,
            gpu_scene.d_materials,
            gpu_scene.d_light_ids, gpu_scene.n_lights,
            d_states,
            gpu_scene.d_triangles,
            gpu_scene.d_tri_bvh,
            gpu_scene.tri_bvh_root,
            gpu_scene.n_triangles,
            d_env_map
        );
        samples_done += batch;
        ++batch_count;

        bool do_preview = preview &&
                          (batch_count % PREVIEW_EVERY == 0 || samples_done == spp);

        if (do_preview) {
            cudaMemcpyAsync(h_accum, d_accum, N*3*sizeof(float),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            float inv = 1.0f / samples_done;
            {
                std::lock_guard<std::mutex> lock(fb.mtx);
                fb.set_bulk(h_accum, inv);
            }
            preview->poll_events();
            {
                std::lock_guard<std::mutex> lock(fb.mtx);
                preview->update(fb.raw_data(), 1.0f);
            }
        } else {
            cudaStreamSynchronize(stream);
        }

        printf("\r[CUDA] %d / %d spp", samples_done, spp);
        fflush(stdout);
    }

    cudaMemcpyAsync(h_accum, d_accum, N*3*sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    {
        std::lock_guard<std::mutex> lock(fb.mtx);
        fb.set_bulk(h_accum, 1.0f / spp);
    }

    printf("\n[CUDA] done\n");

    cudaFreeHost(h_accum);
    cudaFree(d_accum);
    cudaFree(d_states);
    cudaStreamDestroy(stream);
    gpu_scene.free_device();
    gpu_mesh.free_device();

    if (d_env_map) {
        GpuEnvMap host_env;
        cudaMemcpy(&host_env, d_env_map, sizeof(GpuEnvMap), cudaMemcpyDeviceToHost);
        host_env.free_device();
        cudaFree(d_env_map);
    }
}