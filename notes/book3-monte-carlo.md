# Book 3 – Monte Carlo Integration

## Rendering as Integration

The renderer solves:

L_o = ∫_Ω f_r L_i cosθ dω

This integral is approximated via Monte Carlo sampling.

---

## Estimator Used In Engine

L ≈ f_r * Li * cosθ / p(ω)

Implemented inside ray_color().

---

## Importance Sampling

Using cosine-weighted sampling reduces variance because:

f_r cosθ / p(ω) simplifies to albedo.

This dramatically improves convergence.

---

## Russian Roulette Termination

Instead of fixed path depth truncation, paths are terminated
probabilistically after several bounces.

If survival probability is s:

Contribution is scaled by 1/s to preserve unbiasedness.

This removes truncation bias and matches the formulation
presented in Book 3.