#!/usr/bin/env python3
"""Measure render variance against a converged reference.

relMSE is mean((est - ref)^2 / (ref^2 + 0.01)) over every pixel and channel --
the Rousselle form, where the epsilon keeps dark pixels from dominating. It is
comparable across scenes in a way plain MSE is not, which is what makes it
usable for judging a sampling change.

    ./build/render ref.exr cornell --spp 4096 --no-preview
    ./build/render a.exr   cornell --spp 32   --no-preview
    python3 tools/relmse.py ref.exr a.exr

A single relMSE number is itself one draw of a random quantity, so two of them
cannot be compared by eye. --compare resamples pixels to put a confidence
interval on the ratio; the resample is paired, using the same pixels for both
renders, because they share a reference and a scene and their errors are
correlated. Read it as: the change is real only if the interval excludes 1.

    python3 tools/relmse.py --compare ref.exr before.exr after.exr
"""

import sys
import numpy as np
import OpenEXR


def read(path):
    with OpenEXR.File(path) as f:
        px = f.channels()
        for key in ("RGB", "RGBA"):
            if key in px:
                return np.asarray(px[key].pixels, dtype=np.float64)[..., :3]
        return np.stack([np.asarray(px[c].pixels, dtype=np.float64)
                         for c in ("R", "G", "B")], axis=-1)


def per_pixel(ref, est):
    return np.mean((est - ref) ** 2 / (ref ** 2 + 0.01), axis=-1).ravel()


def compare(ref_path, a_path, b_path, draws=4000, seed=12345):
    ref = read(ref_path)
    a, b = per_pixel(ref, read(a_path)), per_pixel(ref, read(b_path))

    rng = np.random.default_rng(seed)
    ratios = np.empty(draws)
    for k in range(draws):
        i = rng.integers(0, a.size, a.size)
        ratios[k] = b[i].mean() / a[i].mean()

    lo, mid, hi = np.percentile(ratios, [2.5, 50, 97.5])
    verdict = "significant" if hi < 1.0 or lo > 1.0 else "not significant"
    print(f"{a.mean():.5f} -> {b.mean():.5f}")
    print(f"ratio {mid:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   {verdict}")


def main(argv):
    if len(argv) >= 2 and argv[0] == "--compare":
        if len(argv) != 4:
            sys.exit("usage: relmse.py --compare <ref> <before> <after>")
        compare(argv[1], argv[2], argv[3])
        return

    if len(argv) < 2:
        sys.exit(__doc__)

    ref = read(argv[0])
    for path in argv[1:]:
        print(f"{path.split('/')[-1]:32s} relMSE {per_pixel(ref, read(path)).mean():.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])
