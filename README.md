# Ray Tracing Engine

A physically-based CPU/CUDA path tracer written in C++, originally following *Ray Tracing: The Next Week* by Peter Shirley, then significantly extended with production rendering techniques — MIS, NEE, a full BSDF interface, and a CUDA GPU backend achieving a 13.4× speedup over the CPU renderer.

---

## Renders

| Cornell Box — GPU (1024spp) | High Quality — GPU (4096spp) |
|:---:|:---:|
| ![gpu](results/cornell_gpu.png) | ![hq](results/cornell_hq.png) |

| Furnace Test — Energy Conservation |
|:---:|
| ![furnace](results/furnace.png) |

---

## Platform Support

> **This project runs on Linux only.**
> macOS is not supported — NVIDIA dropped CUDA support on macOS after 2019,
> and the GPU backend is a core part of this project.
> Windows has not been tested.

| Platform | Status |
|:---|:---:|
| Linux (Arch / CachyOS) | ✓ fully supported |
| macOS | ✗ not supported (no CUDA) |
| Windows | untested |

---

## Features

### Core Rendering
- Iterative path tracing integrator — no recursion, no stack overflow
- Physically-based throughput accumulation (`beta`)
- Next Event Estimation (direct lighting via shadow rays)
- Multiple Importance Sampling with symmetric power heuristic
- Russian roulette path termination after depth 3
- Solid-angle PDF conversion for area lights

### Materials
- Lambertian diffuse with cosine-weighted importance sampling
- Metal with configurable roughness fuzz
- Dielectric glass with Schlick Fresnel approximation
- Diffuse area lights
- Isotropic participating media (constant density volumes)

### Acceleration
- BVH with axis-aligned bounding boxes
- Center-priority tile ordering for progressive refinement
- OpenMP multithreading across tiles

### CUDA Backend
- Full path tracing kernel (`__global__` `Li()`) — one thread per pixel
- Per-thread `curandState` for independent random streams
- Flat scene representation — tagged unions replace virtual dispatch
- Baked `rotate_y` + `translate` transforms per hittable
- Batch accumulation with `cudaMemcpyAsync` and dedicated stream
- Progressive preview via OpenGL between batches

### Infrastructure
- Live OpenGL preview window with gamma-correct display
- PPM image output
- Dual backend: `--device cpu` / `--device gpu`
- Headless mode: `--no-preview`
- Full CLI control over all render parameters
- Furnace test scene for energy conservation validation

---

## Benchmark

| Scene | Backend | Resolution | SPP | Time | Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|
| Cornell Box | CPU (OpenMP) | 600×600 | 400 | 264s | 1× |
| Cornell Box | GPU (CUDA) | 600×600 | 400 | 19.7s | **13.4×** |
| Cornell Box | GPU (CUDA) | 800×800 | 1024 | 86s | — |
| Cornell HQ | GPU (CUDA) | 1200×1200 | 4096 | 780s | — |

**Hardware:**
- CPU: AMD Ryzen 7 5800 (8 cores / 16 threads)
- GPU: NVIDIA RTX 3060 6GB Laptop — CUDA 13.1, sm\_86

---

## Requirements

### Dependencies
```bash
sudo pacman -S cmake gcc openmp glfw
```

### CUDA Toolkit (~3GB)
```bash
sudo pacman -S cuda
```

Add to `~/.config/fish/config.fish`:
```fish
set -x PATH /opt/cuda/bin $PATH
set -x LD_LIBRARY_PATH /opt/cuda/lib64 $LD_LIBRARY_PATH
```

Or `~/.bashrc` / `~/.zshrc`:
```bash
export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
nvcc --version
nvidia-smi
```

> This project targets **sm_86** (RTX 30xx). Change `CMAKE_CUDA_ARCHITECTURES`
> in `CMakeLists.txt` to match your GPU.
> Find your compute capability at: https://developer.nvidia.com/cuda-gpus

---

## Build

```bash
git clone https://github.com/Vishrut2403/Ray-Tracing-Engine
cd Ray-Tracing-Engine
cmake -S . -B build
cmake --build build
```

---

## Usage

```bash
# CPU render with live preview
./build/render output.ppm cornell --spp 400 --width 800 --height 800

# GPU render with live preview
./build/render output.ppm cornell --spp 400 --width 800 --height 800 --device gpu

# Headless GPU render (fastest)
./build/render output.ppm cornell --spp 1024 --width 1200 --height 1200 --device gpu --no-preview

# Energy conservation validation
./build/render furnace.ppm furnace --no-preview
```

**All CLI flags:**

| Flag | Default | Description |
|:---|:---:|:---|
| `--width N` | 400 | Image width in pixels |
| `--height N` | 400 | Image height in pixels |
| `--spp N` | 64 | Samples per pixel |
| `--depth N` | 10 | Maximum ray bounce depth |
| `--tile N` | 32 | Tile size (CPU only) |
| `--device cpu/gpu` | cpu | Render backend |
| `--no-preview` | off | Disable OpenGL preview window |

---

## Validation

Energy conservation verified via furnace test — a single lambertian sphere (albedo = 0.5) rendered in a uniform white environment with no area lights. Under correct energy conservation every pixel must converge to linear 0.5 regardless of the number of bounces.

- Expected: linear 0.5 → gamma-corrected PPM value ≈ 181
- Measured: linear avg = 0.507, error = 0.007
- Pass threshold: < 0.01 ✓

---

## Origin and Extensions

This project started as an implementation of *Ray Tracing: The Next Week* (Peter Shirley), covering BVH acceleration, procedural textures, motion blur, constant-density volumes, and the Cornell box scene.

Extended beyond the book:

| What changed | Why |
|:---|:---|
| Recursive scatter → iterative `Li()` | No stack overflow, GPU-portable |
| Scatter/attenuation → `f / sample / pdf / emitted` BSDF interface | Correct MIS requires separate eval and sample |
| Single-bounce direct → NEE + MIS power heuristic | Lower variance, correct weighting |
| `shared_ptr` scene → flat tagged-union arrays | Required for CUDA device code |
| CPU-only → dual CPU/CUDA backend | 13.4× speedup on RTX 3060 |
| Energy conservation validation added | Catches PDF/throughput bugs before they show visually |

---

## Roadmap

- [ ] GGX microfacet BRDF with Smith masking-shadowing
- [ ] HDR environment map with 2D CDF importance sampling
- [ ] Rough dielectric (microfacet transmission)
- [ ] EXR output via tinyexr
- [ ] OBJ / glTF scene loading
- [ ] OptiX backend (hardware RT cores)
- [ ] Bidirectional path tracing (BDPT)
- [ ] Light trees for many-light scenes
- [ ] Spectral rendering (hero wavelength)