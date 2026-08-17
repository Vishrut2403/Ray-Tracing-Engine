# Ray Tracing Engine

A physically-based CPU/CUDA renderer written in C++, originally following *Ray Tracing: The Next Week* by Peter Shirley and since rebuilt around production techniques: a GGX microfacet BSDF with energy compensation, MIS/NEE, four independent integrators (path tracing, BDPT, progressive photon mapping, ReSTIR DI+GI), HDR environment lighting, glTF meshes, and a CUDA backend that runs the same shading model as the CPU.

Correctness is the point of the project. Every integrator is cross-checked against the others and against both backends by a 295-check test suite.

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

### Integrators
- **Path tracing** — iterative, no recursion; NEE plus BSDF sampling combined with the MIS power heuristic; Russian roulette after depth 3. CPU and GPU.
- **Bidirectional path tracing** — Veach-style, full MIS over all connection strategies including light tracing (`t == 1`) splats. CPU and GPU.
- **Progressive photon mapping** — for specular-to-diffuse transport the other integrators sample poorly. CPU only.
- **ReSTIR DI + GI** — spatiotemporal reservoir resampling from a G-buffer. GPU only, and approximate: its GI is a biased single-bounce estimator, so it is the fast preview mode rather than the reference.

Pick one with `--integrator`; scenes carry a sensible default (`caustics` defaults to BDPT).

### Materials
- **GGX microfacet** metal/dielectric with VNDF sampling, Smith height-correlated masking-shadowing, and Kulla–Conty multiple-scattering energy compensation — Smith GGX drops ~65% of the second-bounce energy at roughness 1, which this adds back
- **Rough dielectric** microfacet transmission (Walter et al. 2007)
- Lambertian diffuse, smooth metal, smooth dielectric with Schlick Fresnel
- Diffuse area lights
- Homogeneous participating media (Henyey–Greenstein phase function), and a subsurface material

### Lighting and Geometry
- HDR environment maps with 2D CDF importance sampling
- glTF and OBJ mesh loading, with a triangle BVH
- Perlin noise, image, and checker textures
- Spheres, boxes, axis-aligned rects, moving spheres, transforms, constant-density media

### Sampling and Precision
- Owen-scrambled (0,2)-sequence low-discrepancy sampler on the CPU (the GPU still uses cuRAND white noise)
- Single precision throughout by default — FP64 runs at 1/64 rate on consumer NVIDIA parts, and single precision is ~4× faster on the GPU here at no measurable cost in accuracy. Build with `-DRT_DOUBLE=ON` for a double-precision reference.

### Acceleration and Infrastructure
- BVH over both hittables and triangles
- OpenMP across center-priority tiles, for progressive refinement from the middle out
- Intel Open Image Denoise via `--denoise`
- PPM output, or linear HDR OpenEXR — the writer is chosen by the output extension
- Live OpenGL preview that applies the same tonemap as the file writer, so what you see is what gets saved
- ACES filmic tonemap with a firefly clamp at luminance 20

---

## Benchmark

256×256, 64 spp, depth 10, `--no-preview`, path tracer. Wall clock end to end — scene build, render, and image write. GPU best of 3, CPU best of 2.

| Scene | CPU | GPU | Speedup |
|:---|---:|---:|---:|
| furnace | 1.15s | 0.20s | 5.8× |
| ggx | 2.00s | 0.24s | 8.4× |
| hdr | 2.08s | 0.31s | 6.7× |
| glass | 2.50s | 0.28s | 8.8× |
| closed_furnace | 4.10s | — | CPU only |
| sss | 5.55s | 0.30s | 18.4× |
| cornell | 5.83s | 0.31s | 18.7× |
| bunny | 10.61s | 5.22s | 2.0× |
| volume | 11.49s | 0.42s | 27.7× |
| helmet | 14.54s | — | CPU only |
| caustics | 31.07s | 1.22s | 25.4× |

`bunny`'s 2× is misleading: its GPU render takes 0.20s and the remaining 5.0s is glTF parsing, BVH construction, and upload — single-threaded CPU work that happens before a ray is cast. Same for `helmet`. Excluding scene build, cornell is 0.12s and caustics 1.03s on the GPU.

The preview window costs ~0.34s of one-time GL context creation plus roughly 5–8% of a CPU render; on the GPU it is free.

**Hardware:**
- CPU: AMD Ryzen 7 5800H (8 cores / 16 threads)
- GPU: NVIDIA RTX 3060 6GB Laptop — CUDA 13.1, sm\_86

CPU timings on this laptop vary ±10% with thermal state; treat differences under ~15% as noise.

---

## Requirements

### Dependencies
```bash
sudo pacman -S cmake gcc openmp glfw openimagedenoise
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

Double-precision reference build:

```bash
cmake -S . -B build-double -DRT_DOUBLE=ON
cmake --build build-double
```

---

## Usage

Arguments are positional: `render <output> [scene] [flags]`. The output extension picks the writer — `.exr` gives linear HDR, anything else is tonemapped PPM.

```bash
# CPU render with live preview
./build/render output.ppm cornell --spp 400 --width 800 --height 800

# GPU render with live preview
./build/render output.ppm cornell --spp 400 --width 800 --height 800 --device gpu

# Headless GPU render (fastest)
./build/render output.ppm cornell --spp 1024 --width 1200 --height 1200 --device gpu --no-preview

# Caustics through bidirectional path tracing
./build/render caustics.ppm caustics --integrator bdpt --no-preview

# Photon mapping, with denoising
./build/render ppm.ppm ppm --integrator ppm --denoise --no-preview

# Linear HDR output, for grading elsewhere
./build/render output.exr cornell --spp 512 --device gpu --no-preview

# Energy conservation validation
./build/render furnace.ppm furnace --no-preview
```

**Flags:**

| Flag | Short | Default | Description |
|:---|:---:|:---:|:---|
| `--width N` | `-w` | 400 | Image width in pixels |
| `--height N` | `-h` | 400 | Image height in pixels |
| `--spp N` | `-s` | 64 | Samples per pixel |
| `--depth N` | `-d` | 10 | Maximum ray bounce depth |
| `--tile N` | `-t` | 32 | Tile size (CPU only) |
| `--device cpu\|gpu` | | cpu | Render backend |
| `--integrator pt\|bdpt\|ppm\|restir` | | per scene | Override the scene's integrator |
| `--denoise` | | off | Run Open Image Denoise on the result |
| `--no-preview` | | off | Headless; no OpenGL window |

**Scenes:** `cornell`, `furnace`, `closed_furnace`, `ggx`, `hdr`, `bunny`, `glass`, `caustics`, `helmet`, `ppm`, `sss`, `volume`.

The GPU implements `cornell`, `furnace`, `ggx`, `hdr`, `bunny`, `glass`, `caustics`, `volume`, and `sss`; anything else falls back to the CPU with a note. Progressive photon mapping is CPU-only.

---

## Validation

```bash
./build/tests
```

295 checks covering:

- **Analytic BSDF identities** — white furnace (`E[f·cos/pdf]` matches the known albedo and never exceeds 1), Helmholtz reciprocity, PDF normalization, and the `f·cos/pdf = G2/G1` cancellation for VNDF sampling
- **Energy conservation** — the closed furnace, where `L = Le/(1-rho)` must converge to exactly 1 across the whole image, catches throughput and PDF bugs that the open furnace misses
- **Cross-backend agreement** — CPU and GPU means must agree on `cornell`, `glass`, and `ggx`
- **Cross-integrator agreement** — PT against BDPT, and PPM against BDPT; three independent estimators of the same integral disagreeing means at least one is wrong
- **Photon mapping invariants** — the result must not depend on the iteration count
- **Denoiser** — output finite, non-negative, and energy-preserving

Bounds are self-calibrating: Monte Carlo checks derive their tolerance from the estimator's own standard error rather than a fixed margin, so they neither pass trivially nor fail at random. Numeric tolerances scale off `std::numeric_limits<real>::epsilon()`, so the suite is meaningful in both the float and double builds.

---

## Origin and Extensions

This project started as an implementation of *Ray Tracing: The Next Week* (Peter Shirley), covering BVH acceleration, procedural textures, motion blur, constant-density volumes, and the Cornell box scene.

Extended beyond the book:

| What changed | Why |
|:---|:---|
| Recursive scatter → iterative `Li()` | No stack overflow, GPU-portable |
| Scatter/attenuation → `f / sample / pdf / emitted` BSDF interface | Correct MIS requires separate eval and sample |
| Single-bounce direct → NEE + MIS power heuristic | Lower variance, correct weighting |
| Schlick-only dielectrics → GGX microfacet with Kulla–Conty compensation | The book's materials lose energy and cannot represent rough metal |
| `shared_ptr` scene → flat tagged-union arrays | Required for CUDA device code |
| CPU-only → dual CPU/CUDA backend | Up to 27× here, with identical shading on both |
| One integrator → PT, BDPT, PPM, ReSTIR | Caustics and SDS paths need estimators the book's approach cannot reach |
| Double → single precision, with a double build kept as reference | 4× on the GPU; the double build is how the float build is checked |
| White noise → Owen-scrambled (0,2)-sequence | Stratification the RNG cannot give |
| Ad-hoc eyeballing → 295-check suite | Catches PDF and throughput bugs before they show visually |

---

## Roadmap

- [x] GGX microfacet BRDF with Smith masking-shadowing
- [x] HDR environment map with 2D CDF importance sampling
- [x] Rough dielectric (microfacet transmission)
- [x] OBJ / glTF scene loading
- [x] Bidirectional path tracing (BDPT)
- [x] Progressive photon mapping
- [x] ReSTIR direct and indirect lighting
- [x] EXR output via tinyexr
- [ ] Port the low-discrepancy sampler to the GPU (still cuRAND white noise there)
- [ ] Fixed per-bounce sampler dimension allocation, replacing the global counter
- [ ] Profile and cut the CPU hot path (virtual dispatch through `shared_ptr` per intersection)
- [ ] OptiX backend (hardware RT cores)
- [ ] Light trees for many-light scenes
- [ ] Spectral rendering (hero wavelength)
