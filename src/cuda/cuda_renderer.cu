#include "cuda/cuda_renderer.h"
#include "cuda/gpu_integrator.cuh"
#include "cuda/gpu_restir.cuh"
#include "cuda/gpu_restir_kernels.cuh"
#include "cuda/gpu_restir_gi_kernels.cuh"
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
#include <string>


void GpuScene::free_device() {
	if (d_materials) { cudaFree(d_materials); d_materials = nullptr; }
	if (d_hittables) { cudaFree(d_hittables); d_hittables = nullptr; }
	if (d_bvh_nodes) { cudaFree(d_bvh_nodes); d_bvh_nodes = nullptr; }
	if (d_light_ids) { cudaFree(d_light_ids); d_light_ids = nullptr; }
}

bool cuda_supports_scene(const std::string& name) {
	return SceneUploader::supports(name);
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
				 const std::string& scene_name,
				 GpuIntegrator integrator) {
					
	const int W = fb.get_width();
	const int H = fb.get_height();
	const int N = W * H;

	GpuMesh  gpu_mesh;
	GpuScene gpu_scene;
	if (scene_name == "bunny")
		gpu_scene = SceneUploader::build_bunny_scene(scene_name, gpu_mesh);
	else
		gpu_scene = SceneUploader::build_and_upload(scene_name);

	GpuEnvMap* d_env_map = nullptr;
	if (scene.env) d_env_map = upload_env_map(*scene.env);

	// Debug: confirm medium state
	printf("[Medium] active=%d sigma_t=(%.4f,%.4f,%.4f) g=%.2f\n",
		gpu_scene.medium.active,
		(float)gpu_scene.medium.sigma_t.x(),
		(float)gpu_scene.medium.sigma_t.y(),
		(float)gpu_scene.medium.sigma_t.z(),
		gpu_scene.medium.g);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	// ── RNG ───────────────────────────────────────────────────────────────────
	curandState* d_states;
	cudaMalloc(&d_states, N * sizeof(curandState));
	{
		int t=128, b=(N+t-1)/t;
		cuda_rand_init<<<b,t,0,stream>>>(d_states, 1234ULL, N);
		cudaStreamSynchronize(stream);
	}

	// ── Accumulation ──────────────────────────────────────────────────────────
	float* d_accum;
	cudaMalloc(&d_accum, N * 3 * sizeof(float));
	cudaMemsetAsync(d_accum, 0, N*3*sizeof(float), stream);

	// ── ReSTIR DI buffers ─────────────────────────────────────────────────────
	GBufferPixel* d_gbuffer_cur;
	GBufferPixel* d_gbuffer_prev;
	Reservoir*    d_res_initial;
	Reservoir*    d_res_temporal;
	Reservoir*    d_res_spatial;
	Reservoir*    d_res_prev;

	cudaMalloc(&d_gbuffer_cur,  N * sizeof(GBufferPixel));
	cudaMalloc(&d_gbuffer_prev, N * sizeof(GBufferPixel));
	cudaMalloc(&d_res_initial,  N * sizeof(Reservoir));
	cudaMalloc(&d_res_temporal, N * sizeof(Reservoir));
	cudaMalloc(&d_res_spatial,  N * sizeof(Reservoir));
	cudaMalloc(&d_res_prev,     N * sizeof(Reservoir));

	cudaMemsetAsync(d_gbuffer_prev, 0, N * sizeof(GBufferPixel), stream);
	cudaMemsetAsync(d_res_prev,     0, N * sizeof(Reservoir),    stream);

	// ── ReSTIR GI buffers ─────────────────────────────────────────────────────
	GIReservoir* d_gi_initial;
	GIReservoir* d_gi_temporal;
	GIReservoir* d_gi_spatial;
	GIReservoir* d_gi_prev;

	cudaMalloc(&d_gi_initial,  N * sizeof(GIReservoir));
	cudaMalloc(&d_gi_temporal, N * sizeof(GIReservoir));
	cudaMalloc(&d_gi_spatial,  N * sizeof(GIReservoir));
	cudaMalloc(&d_gi_prev,     N * sizeof(GIReservoir));

	cudaMemsetAsync(d_gi_prev, 0, N * sizeof(GIReservoir), stream);

	float* h_accum;
	cudaMallocHost(&h_accum, N * 3 * sizeof(float));

	GpuCamera gpu_cam = make_gpu_camera(cam);
	dim3 threads(16,16);
	dim3 blocks((W+15)/16, (H+15)/16);

	if (gpu_scene.use_bdpt) {
		printf("[CUDA/BDPT] %dx%d spp=%d\n", W, H, spp);

		float* h_accum;
		cudaMallocHost(&h_accum, N * 3 * sizeof(float));

		for (int s = 0; s < spp; ++s) {
			accumulate_bdpt_kernel<<<blocks,threads,0,stream>>>(
				d_accum, W, H, max_depth,
				gpu_cam,
				gpu_scene.d_hittables, gpu_scene.n_hittables,
				gpu_scene.d_materials,
				gpu_scene.d_light_ids, gpu_scene.n_lights,
				d_states,
				gpu_scene.d_triangles, gpu_scene.d_tri_bvh,
				gpu_scene.tri_bvh_root, gpu_scene.n_triangles
			);
			cudaStreamSynchronize(stream);

			if (preview && (s % 4 == 0 || s == spp-1)) {
				cudaMemcpyAsync(h_accum, d_accum, N*3*sizeof(float),
								cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
				float inv = 1.0f / (s+1);
				{
					std::lock_guard<std::mutex> lock(fb.mtx);
					fb.set_bulk(h_accum, inv);
				}
				preview->poll_events();
				{
					std::lock_guard<std::mutex> lock(fb.mtx);
					preview->update(fb.raw_data());
				}
			}
		}

		cudaMemcpyAsync(h_accum, d_accum, N*3*sizeof(float),
						cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		{
			std::lock_guard<std::mutex> lock(fb.mtx);
			fb.set_bulk(h_accum, 1.0f/spp);
		}

		printf("[CUDA/BDPT] done\n");

		cudaFreeHost(h_accum);
		cudaFree(d_accum);
		cudaFree(d_states);
		cudaStreamDestroy(stream);
		gpu_scene.free_device();
		if (d_env_map) {
			GpuEnvMap host_env;
			cudaMemcpy(&host_env, d_env_map, sizeof(GpuEnvMap),
					   cudaMemcpyDeviceToHost);
			host_env.free_device();
			cudaFree(d_env_map);
		}
		return;
	}

	// ReSTIR shades from a G-buffer, so primary rays never sample the medium
	// and a participating-medium scene comes out far too dark. Fall back rather
	// than return a wrong image.
	if (integrator == GpuIntegrator::RESTIR && gpu_scene.medium.active) {
		printf("[note] ReSTIR does not support participating media; "
			   "using the path tracer\n");
		integrator = GpuIntegrator::PATH_TRACER;
	}

	// ── Path tracer (default) ─────────────────────────────────────────────────
	if (integrator == GpuIntegrator::PATH_TRACER) {
		printf("[CUDA/PT] %dx%d spp=%d depth=%d\n", W, H, spp, max_depth);
		for (int s = 0; s < spp; ++s) {
			accumulate_kernel<<<blocks,threads,0,stream>>>(
				d_accum, W, H, 1, max_depth, gpu_cam,
				vec3(background.x(), background.y(), background.z()),
				gpu_scene.d_hittables, gpu_scene.n_hittables,
				gpu_scene.d_materials,
				gpu_scene.d_light_ids, gpu_scene.n_lights,
				d_states,
				gpu_scene.d_triangles, gpu_scene.d_tri_bvh,
				gpu_scene.tri_bvh_root, gpu_scene.n_triangles,
				d_env_map, gpu_scene.medium);
			cudaStreamSynchronize(stream);

			if (preview && (s % 4 == 0 || s == spp-1)) {
				cudaMemcpyAsync(h_accum, d_accum, N*3*sizeof(float),
								cudaMemcpyDeviceToHost, stream);
				cudaStreamSynchronize(stream);
				{
					std::lock_guard<std::mutex> lock(fb.mtx);
					fb.set_bulk(h_accum, 1.0f/(s+1));
				}
				preview->poll_events();
				{
					std::lock_guard<std::mutex> lock(fb.mtx);
					preview->update(fb.raw_data());
				}
			}
		}
		cudaMemcpyAsync(h_accum, d_accum, N*3*sizeof(float),
						cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		{
			std::lock_guard<std::mutex> lock(fb.mtx);
			fb.set_bulk(h_accum, 1.0f/spp);
		}
		printf("[CUDA/PT] done\n");

		cudaFreeHost(h_accum);
		cudaFree(d_accum); cudaFree(d_states);
		cudaFree(d_gbuffer_cur); cudaFree(d_gbuffer_prev);
		cudaFree(d_res_initial); cudaFree(d_res_temporal);
		cudaFree(d_res_spatial); cudaFree(d_res_prev);
		cudaFree(d_gi_initial); cudaFree(d_gi_temporal);
		cudaFree(d_gi_spatial); cudaFree(d_gi_prev);
		cudaStreamDestroy(stream);
		gpu_scene.free_device();
		gpu_mesh.free_device();
		if (d_env_map) {
			GpuEnvMap host_env;
			cudaMemcpy(&host_env, d_env_map, sizeof(GpuEnvMap),
					   cudaMemcpyDeviceToHost);
			host_env.free_device();
			cudaFree(d_env_map);
		}
		return;
	}

	const int M_CANDIDATES   = 32;
	const int K_NEIGHBORS    = 5;
	const int SPATIAL_RADIUS = 30;
	const int GI_K_NEIGHBORS = 1;
	const int GI_SPATIAL_RADIUS = 5;
	const int PREVIEW_EVERY  = 4;

	printf("[CUDA/ReSTIR+GI] %dx%d spp=%d M=%d K=%d r=%d\n",
		   W, H, spp, M_CANDIDATES, K_NEIGHBORS, SPATIAL_RADIUS);

	for (int s = 0; s < spp; ++s) {
		bool has_prev = (s > 0);

		// ── DI Pass 0: G-buffer + initial RIS ────────────────────────────────
		restir_initial_kernel<<<blocks,threads,0,stream>>>(
			W, H, gpu_cam,
			gpu_scene.d_hittables, gpu_scene.n_hittables,
			gpu_scene.d_materials,
			gpu_scene.d_light_ids, gpu_scene.n_lights,
			gpu_scene.d_triangles, gpu_scene.d_tri_bvh,
			gpu_scene.tri_bvh_root, gpu_scene.n_triangles,
			d_gbuffer_cur, d_res_initial, d_states,
			M_CANDIDATES
		);

		// ── DI Pass 1: Temporal reuse ─────────────────────────────────────────
		restir_temporal_kernel<<<blocks,threads,0,stream>>>(
			W, H,
			gpu_scene.d_materials,
			d_gbuffer_cur, d_gbuffer_prev,
			d_res_initial, d_res_prev,
			d_res_temporal,
			d_states, has_prev
		);

		// ── DI Pass 2: Spatial reuse ──────────────────────────────────────────
		restir_spatial_kernel<<<blocks,threads,0,stream>>>(
			W, H,
			gpu_scene.d_hittables, gpu_scene.n_hittables,
			gpu_scene.d_materials,
			gpu_scene.d_light_ids, gpu_scene.n_lights,
			gpu_scene.d_triangles, gpu_scene.d_tri_bvh,
			gpu_scene.tri_bvh_root, gpu_scene.n_triangles,
			d_gbuffer_cur, d_res_temporal, d_res_spatial,
			d_states, K_NEIGHBORS, SPATIAL_RADIUS
		);

		// ── GI Pass 0: Initial path sampling ──────────────────────────────────
		restir_gi_initial_kernel<<<blocks,threads,0,stream>>>(
			W, H,
			gpu_scene.d_hittables, gpu_scene.n_hittables,
			gpu_scene.d_materials,
			gpu_scene.d_light_ids, gpu_scene.n_lights,
			gpu_scene.d_triangles, gpu_scene.d_tri_bvh,
			gpu_scene.tri_bvh_root, gpu_scene.n_triangles,
			d_gbuffer_cur, d_gi_initial, d_states,
			d_env_map, background
		);

		// ── GI Pass 1: Temporal reuse with reconnection shift ───────────────
		// DISABLED for offline rendering - temporal reuse primarily benefits real-time (1 spp/frame)
		// At 64+ spp, spatial reuse alone provides sufficient quality
		// For now: copy initial directly to temporal, skip temporal merge logic
		cudaMemcpyAsync(d_gi_temporal, d_gi_initial, N * sizeof(GIReservoir),
						cudaMemcpyDeviceToDevice, stream);

		// ── GI Pass 2: Spatial reuse ─────────────────────────────────────────
		// Reconnection shift with jacobian bounds checking
		// K=1 neighbor, r=5 radius to smooth noise without destroying detail
		restir_gi_spatial_kernel<<<blocks,threads,0,stream>>>(
			W, H,
			gpu_scene.d_hittables, gpu_scene.n_hittables,
			gpu_scene.d_materials,
			gpu_scene.d_triangles, gpu_scene.d_tri_bvh,
			gpu_scene.tri_bvh_root, gpu_scene.n_triangles,
			d_gbuffer_cur, d_gi_temporal, d_gi_spatial,
			d_states, GI_K_NEIGHBORS, GI_SPATIAL_RADIUS
		);

		// Note: d_gi_prev buffer remains allocated but unused
		// Kept for future temporal reuse implementation (hybrid shift mapping)

		// ── Pass: Shade (DI + GI + volumetrics) ──────────────────────────────
		restir_shade_kernel<<<blocks,threads,0,stream>>>(
			W, H, d_accum, background,
			gpu_scene.d_hittables, gpu_scene.n_hittables,
			gpu_scene.d_materials,
			gpu_scene.d_light_ids, gpu_scene.n_lights,
			gpu_scene.d_triangles, gpu_scene.d_tri_bvh,
			gpu_scene.tri_bvh_root, gpu_scene.n_triangles,
			d_gbuffer_cur, d_res_spatial,
			d_states, max_depth, d_env_map,
			d_gi_spatial,        // ← GI reservoir
			true,
			gpu_scene.medium
		);

		cudaStreamSynchronize(stream);

		// ── Swaps ─────────────────────────────────────────────────────────────
		// G-buffer
		GBufferPixel* tmp_g = d_gbuffer_prev;
		d_gbuffer_prev      = d_gbuffer_cur;
		d_gbuffer_cur       = tmp_g;

		// DI reservoir
		Reservoir* tmp_r = d_res_prev;
		d_res_prev       = d_res_spatial;
		d_res_spatial    = tmp_r;

		// GI reservoir
		GIReservoir* tmp_gi = d_gi_prev;
		d_gi_prev           = d_gi_spatial;
		d_gi_spatial        = tmp_gi;

		if (preview && (s % PREVIEW_EVERY == 0 || s == spp-1)) {
			cudaMemcpyAsync(h_accum, d_accum, N*3*sizeof(float),
							cudaMemcpyDeviceToHost, stream);
			cudaStreamSynchronize(stream);
			float inv = 1.0f / (s+1);
			{
				std::lock_guard<std::mutex> lock(fb.mtx);
				fb.set_bulk(h_accum, inv);
			}
			preview->poll_events();
			{
				std::lock_guard<std::mutex> lock(fb.mtx);
				preview->update(fb.raw_data());
			}
		}
	}

	cudaMemcpyAsync(h_accum, d_accum, N*3*sizeof(float),
					cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	{
		std::lock_guard<std::mutex> lock(fb.mtx);
		fb.set_bulk(h_accum, 1.0f/spp);
	}

	printf("\n[CUDA/ReSTIR+GI] done\n");

	// ── Free ──────────────────────────────────────────────────────────────────
	cudaFreeHost(h_accum);
	cudaFree(d_accum);
	cudaFree(d_states);
	cudaFree(d_gbuffer_cur);
	cudaFree(d_gbuffer_prev);
	cudaFree(d_res_initial);
	cudaFree(d_res_temporal);
	cudaFree(d_res_spatial);
	cudaFree(d_res_prev);
	cudaFree(d_gi_initial);
	cudaFree(d_gi_temporal);
	cudaFree(d_gi_spatial);
	cudaFree(d_gi_prev);
	cudaStreamDestroy(stream);
	gpu_scene.free_device();
	gpu_mesh.free_device();

	if (d_env_map) {
		GpuEnvMap host_env;
		cudaMemcpy(&host_env, d_env_map, sizeof(GpuEnvMap),
				   cudaMemcpyDeviceToHost);
		host_env.free_device();
		cudaFree(d_env_map);
	}
}