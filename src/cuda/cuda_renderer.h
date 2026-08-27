#pragma once

#include "render/framebuffer.h"
#include "core/camera.h"
#include "scenes/cornell_scene.h"
#include "core/vec3.h"
#include <atomic>
#include <string>

bool cuda_supports_scene(const std::string& name);

// Everything device-side that depends only on the scene: geometry, materials,
// the mesh BVH and the environment map. Uploading it is most of the wall clock
// on a mesh scene, so a viewport restarting a render on every camera nudge has
// to hold on to one of these rather than pay for it again each time.
struct CudaScene;
CudaScene* cuda_scene_upload(const Scene& scene, const std::string& scene_name);
void       cuda_scene_free(CudaScene* prepared);

// PATH_TRACER (gpu_Li) is the default: validated against the CPU and BDPT to
// within 1.5%. RESTIR is the fast approximate mode — its GI is a single-bounce
// biased estimator. BDPT is selected by the scene regardless.
enum class GpuIntegrator { PATH_TRACER, RESTIR };

void cuda_render(const Scene& scene,
				 Framebuffer& fb,
				 const camera& cam,
				 const color& background,
				 int spp,
				 int max_depth,
				 // Publish the running average to `fb` for the preview. The
				 // window belongs to the main thread and is not reachable here.
				 bool stage_frames = false,
				 const std::string& scene_name = "cornell",
				 GpuIntegrator integrator = GpuIntegrator::PATH_TRACER,
				 // Checked between samples. A cancelled render leaves the mean
				 // of the samples that did finish.
				 const std::atomic<bool>* cancel = nullptr,
				 // Reused rather than uploaded when supplied, and left alone
				 // afterwards; null means upload and free one for this render.
				 CudaScene* prepared = nullptr);