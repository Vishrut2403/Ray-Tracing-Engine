#!/usr/bin/env python3
"""Regenerate the README's benchmark table.

Wall clock of the whole process, so the totals include scene build and image
write, not just the sampling loop. The render columns subtract each backend's
own --spp 1 time, which is the only way to separate the loop from a fixed cost
that dominates at these sample counts.

    python3 tools/benchmark.py > table.md

Run it on a cool machine and in one pass: a heat-soaked laptop gives figures
15-35% slower on both backends, so rows are only comparable within a pass.
"""

import argparse
import os
import subprocess
import time

# Ordered by CPU cost, which is how the table reads.
SCENES = ["caustics", "volume", "helmet", "bunny", "cornell", "sss",
          "closed_furnace", "glass", "hdr", "ggx", "furnace"]

# Everything else falls back to the CPU with a note, which would time the
# wrong backend.
GPU_SCENES = {"cornell", "furnace", "ggx", "hdr", "bunny", "glass",
              "caustics", "volume", "sss"}

OUT = "/tmp/rt_benchmark.ppm"


def run(scene, spp, device, width, depth, repeats):
    best = float("inf")
    for _ in range(repeats):
        start = time.time()
        subprocess.run(["./build/render", OUT, scene,
                        "--spp", str(spp), "--width", str(width),
                        "--height", str(width), "--depth", str(depth),
                        "--device", device, "--no-preview"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        best = min(best, time.time() - start)
    return best


def fmt(seconds):
    return f"{seconds:.2f}s" if seconds >= 0.01 else "<0.01s"


def speedup(cpu, gpu):
    return f"{cpu/gpu:.1f}×" if gpu > 0 else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--spp", type=int, default=64)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--cpu-repeats", type=int, default=2)
    ap.add_argument("--gpu-repeats", type=int, default=3)
    args = ap.parse_args()

    if not os.path.exists("./build/render"):
        raise SystemExit("build/render not found -- run cmake --build build")

    print(f"| Scene | CPU | GPU | Speedup | CPU render | GPU render "
          f"| Render speedup |")
    print("|:---|---:|---:|---:|---:|---:|---:|")

    for scene in SCENES:
        cpu  = run(scene, args.spp, "cpu", args.width, args.depth, args.cpu_repeats)
        cpu1 = run(scene, 1,        "cpu", args.width, args.depth, args.cpu_repeats)
        cpu_render = max(cpu - cpu1, 0.0)

        if scene not in GPU_SCENES:
            print(f"| {scene} | {fmt(cpu)} | — | CPU only "
                  f"| {fmt(cpu_render)} | — | — |")
            continue

        gpu  = run(scene, args.spp, "gpu", args.width, args.depth, args.gpu_repeats)
        gpu1 = run(scene, 1,        "gpu", args.width, args.depth, args.gpu_repeats)
        gpu_render = max(gpu - gpu1, 0.0)

        # Below the resolution of this method: the subtraction is then noise.
        rs = speedup(cpu_render, gpu_render) if gpu_render >= 0.01 else "—"
        print(f"| {scene} | {fmt(cpu)} | {fmt(gpu)} | {speedup(cpu, gpu)} "
              f"| {fmt(cpu_render)} | {fmt(gpu_render)} | {rs} |")


if __name__ == "__main__":
    main()
