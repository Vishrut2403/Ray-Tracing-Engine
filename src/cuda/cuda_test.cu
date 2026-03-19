#include "core/vec3.h"
#include "cuda/cuda_rand.cuh"
#include "cuda/gpu_scene.cuh"
#include "cuda/scene_uploader.h"
#include <cstdio>

// ── Smoke test: cuRAND ────────────────────────────────────────────────────────

__global__ void smoke_kernel(curandState* states, float* out, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;
    out[id] = (float)rand_unit_vector(&states[id]).length();
}

static void smoke_test_rand() {
    const int N = 256, T = 64, B = (N+T-1)/T;
    curandState* ds; float* dout;
    cudaMalloc(&ds, N*sizeof(curandState));
    cudaMalloc(&dout, N*sizeof(float));
    cuda_rand_init<<<B,T>>>(ds, 42ULL, N);
    smoke_kernel<<<B,T>>>(ds, dout, N);
    float h[N]; cudaMemcpy(h, dout, N*sizeof(float), cudaMemcpyDeviceToHost);
    float sum = 0; for (int i=0;i<N;i++) sum+=h[i];
    float avg = sum/N;
    printf("[M1 cuRAND]  avg unit length = %.6f  %s\n",
           avg, (avg>0.99f&&avg<1.01f) ? "PASS" : "FAIL");
    cudaFree(ds); cudaFree(dout);
}

// ── M2 test: verify scene uploaded correctly ──────────────────────────────────

__global__ void verify_scene(GpuMaterial* mats, GpuHittable* hits,
                              int n_mats, int n_hits, int* out) {
    // Thread 0 does a simple sanity check on device-side data
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int ok = 1;

    // First material should be lambertian (red wall)
    if (mats[0].type != MatType::LAMBERTIAN) ok = 0;

    // All hittables should have valid mat_id
    for (int i = 0; i < n_hits; i++)
        if (hits[i].mat_id < 0 || hits[i].mat_id >= n_mats) ok = 0;

    *out = ok;
}

static void smoke_test_scene() {
    GpuScene scene = SceneUploader::build_and_upload();

    int* d_ok; cudaMalloc(&d_ok, sizeof(int));
    verify_scene<<<1,1>>>(scene.d_materials, scene.d_hittables,
                           scene.n_materials, scene.n_hittables, d_ok);

    int h_ok = 0;
    cudaMemcpy(&h_ok, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
    printf("[M2 scene]   materials=%d  hittables=%d  lights=%d  verify=%s\n",
           scene.n_materials, scene.n_hittables, scene.n_lights,
           h_ok ? "PASS" : "FAIL");

    cudaFree(d_ok);
    scene.free_device();
}

// ── Entry point called from main.cpp ─────────────────────────────────────────

void run_cuda_smoke_test() {
    smoke_test_rand();
    smoke_test_scene();
}
