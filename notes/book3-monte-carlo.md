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

---

# Book 3 – Geometry as a Probability Distribution

## Motivation

In Book 2, rays randomly bounced and sometimes hit light sources.
This produced high variance and noisy renders.

Book 3 introduces **explicit light sampling** by treating geometry as a probability distribution.

---

## Extending the `hittable` Interface

Added two new virtual functions:

```cpp
virtual double pdf_value(const point3& origin,
                         const vec3& direction) const;

virtual vec3 random(const point3& origin) const;
```

---

## `pdf_value(origin, direction)`

Returns the probability density (in solid angle measure) of sampling `direction`
toward this object from `origin`.

This requires converting:

**Area PDF → Solid Angle PDF**

\[
p(\omega) = \frac{\text{distance}^2}{\cos(\theta) \cdot \text{area}}
\]

---

## `random(origin)`

Generates a random direction from `origin`
toward a randomly chosen point on the object.

Used for:

- Direct light sampling  
- Multiple Importance Sampling (MIS)

---

## Why This Matters

Instead of hoping a BRDF sample hits a light,
we now directly sample lights.

This dramatically reduces variance.

---

## Integration with MIS

We combine:

- Light sampling PDF  
- BRDF sampling PDF  

\[
p_{\text{mix}} = \frac{1}{2} p_{\text{light}} + \frac{1}{2} p_{\text{brdf}}
\]

---

This is the foundation of efficient Monte Carlo path tracing.

---

## Rectangle Light PDF (Solid Angle Conversion)

When sampling a rectangle light, we first sample uniformly over its surface:

\[
p_A = \frac{1}{Area}
\]

However, rendering integrates over solid angle, not area.

Using the geometric relationship:

\[
d\omega = \frac{\cos\theta \, dA}{distance^2}
\]

We convert area density into solid angle density:

\[
p_\omega = \frac{distance^2}{\cos\theta \cdot Area}
\]

### Implementation

```cpp
double area = (x1 - x0) * (z1 - z0);

double distance_squared =
    rec.t * rec.t *
    direction.length_squared();

double cosine =
    fabs(dot(direction, rec.normal)
         / direction.length());

return distance_squared / (cosine * area);