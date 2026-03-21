#include "render/integrator.h"
#include "materials/material.h"
#include "lights/env_light.h"
#include "bdpt/path_vertex.h"
#include <algorithm>
#include <cmath>
#include <vector>

static inline bool is_valid(const color& c) {
    return std::isfinite(c.x()) && std::isfinite(c.y()) && std::isfinite(c.z())
        && c.x() >= 0.0 && c.y() >= 0.0 && c.z() >= 0.0;
}

static inline color safe(const color& c) {
    if (!is_valid(c)) return color(0,0,0);
    double lum = 0.2126*c.x() + 0.7152*c.y() + 0.0722*c.z();
    if (lum > 50.0) return c * (50.0 / lum);
    return c;
}

static inline double power_heuristic(double pdf_a, double pdf_b) {
    double a2 = pdf_a*pdf_a, b2 = pdf_b*pdf_b;
    return a2 / (a2 + b2 + 1e-12);
}

static bool visible(const point3& p, const point3& q,
                    const std::shared_ptr<hittable>& world) {
    vec3   d    = q - p;
    double dist = d.length();
    if (dist < 1e-6) return true;
    hit_record shadow;
    return !world->hit(ray(p, d/dist, 0.0),
                       interval(1e-4, dist - 1e-4), shadow);
}

int trace_camera_path(
    const ray& r,
    const color& initial_beta,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<env_light>& env,
    int max_depth,
    std::vector<PathVertex>& path
) {
    path.clear();
    path.push_back(PathVertex::make_camera(r.origin()));

    ray   cur = r;
    color beta = initial_beta;
    double pdf_fwd = 1.0;

    for (int depth = 0; depth < max_depth; ++depth) {
        hit_record rec;
        if (!world->hit(cur, interval(1e-4, infinity), rec)) break;

        PathVertex v = PathVertex::from_hit(rec, beta, pdf_fwd);
        path.push_back(v);

        color Le = rec.mat_ptr->emitted(cur, rec, rec.u, rec.v, rec.p);
        if (Le.length_squared() > 0.0) break;

        BSDFSample bs = rec.mat_ptr->sample(cur, rec);
        if (bs.pdf <= 1e-10 || !is_valid(bs.f)) break;

        double cos_t = std::abs(dot(rec.normal, bs.wi));
        color new_beta = beta * bs.f * cos_t / bs.pdf;
        if (!is_valid(new_beta)) break;
        beta = new_beta;

        path.back().delta = bs.is_delta;
        pdf_fwd = bs.pdf;

        if (depth >= 3) {
            double q = std::clamp(beta.max_component(), 0.05, 0.95);
            if (random_double() > q) break;
            beta /= q;
        }
        cur = ray(rec.p, bs.wi, cur.time());
    }
    return (int)path.size();
}

int trace_light_path(
    const std::shared_ptr<hittable_list>& lights,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<env_light>& env,
    int max_depth,
    std::vector<PathVertex>& path
) {
    path.clear();
    if (!lights || lights->objects.empty()) return 0;

    int n_lights = (int)lights->objects.size();
    int li       = std::min((int)(random_double()*n_lights), n_lights-1);
    auto& light  = lights->objects[li];
    double sel_pdf = 1.0 / n_lights;

    point3 origin(278, 100, 278);
    vec3   to_l  = light->random(origin);
    double dist  = to_l.length();
    if (dist < 1e-6) return 0;
    vec3 wi = to_l / dist;

    hit_record lrec;
    if (!world->hit(ray(origin, wi, 0.0), interval(0.001, dist+1.0), lrec)) return 0;

    color Le = lrec.mat_ptr->emitted(ray(origin,wi,0.0), lrec,
                                      lrec.u, lrec.v, lrec.p);
    if (Le.length_squared() < 1e-10) return 0;

    double pdf_area = light->pdf_value(origin, wi) * sel_pdf;
    if (pdf_area <= 1e-10) return 0;

    path.push_back(PathVertex::make_light(lrec.p, lrec.normal, Le, pdf_area));

    onb uvw; uvw.build_from_w(lrec.normal);
    vec3 wo = uvw.local(random_cosine_direction());
    double cos_emit = std::max(dot(lrec.normal, wo), 0.0);
    if (cos_emit < 1e-6) return (int)path.size();

    double pdf_emit = cos_emit / pi;
    color beta = Le * cos_emit / (pdf_area * pdf_emit);
    if (!is_valid(beta)) return (int)path.size();

    ray cur(lrec.p, wo, 0.0);

    for (int depth = 0; depth < max_depth-1; ++depth) {
        hit_record rec;
        if (!world->hit(cur, interval(1e-4, infinity), rec)) break;

        PathVertex v = PathVertex::from_hit(rec, beta, pdf_emit);
        path.push_back(v);

        BSDFSample bs = rec.mat_ptr->sample(cur, rec);
        if (bs.pdf <= 1e-10 || !is_valid(bs.f)) break;

        double cos_t = std::abs(dot(rec.normal, bs.wi));
        color new_beta = beta * bs.f * cos_t / bs.pdf;
        if (!is_valid(new_beta)) break;
        beta = new_beta;

        path.back().delta = bs.is_delta;

        if (depth >= 2) {
            double q = std::clamp(beta.max_component(), 0.05, 0.95);
            if (random_double() > q) break;
            beta /= q;
        }
        cur = ray(rec.p, bs.wi, 0.0);
        pdf_emit = bs.pdf;
    }
    return (int)path.size();
}

color connect(
    const std::vector<PathVertex>& cp, int t,
    const std::vector<PathVertex>& lp, int s,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<env_light>& env,
    const color& background
) {
    if (t <= 0 || t > (int)cp.size()) return color(0,0,0);
    const PathVertex& qv = cp[t-1];

    if (s == 0) {
        if (!qv.mat) return color(0,0,0);
        hit_record dummy; dummy.p=qv.p; dummy.normal=qv.n;
        dummy.front_face=true; dummy.mat_ptr=qv.mat;
        ray dr(qv.p-qv.n*0.001, qv.n, 0.0);
        color Le = qv.mat->emitted(dr, dummy, qv.u, qv.v, qv.p);
        return safe(qv.beta * Le);
    }

    if (s <= 0 || s > (int)lp.size()) return color(0,0,0);
    const PathVertex& pv = lp[s-1];

    if (!qv.is_connectable() || !pv.is_connectable()) return color(0,0,0);
    if (!visible(qv.p, pv.p, world)) return color(0,0,0);

    vec3   d     = pv.p - qv.p;
    double dist2 = d.length_squared();
    if (dist2 < 1e-8) return color(0,0,0);

    vec3 d_hat = d / std::sqrt(dist2);

    if (s == 1) {
        if (!pv.is_light()) return color(0,0,0);
        double cos_q = std::abs(dot(qv.n, d_hat));
        double cos_p = std::abs(dot(pv.n, -d_hat));
        ray dummy_ray(qv.p - d_hat*0.001, d_hat, 0.0);
        hit_record qrec; qrec.p=qv.p; qrec.normal=qv.n;
        qrec.front_face=true; qrec.mat_ptr=qv.mat;
        color f_q = qv.mat ? qv.mat->f(dummy_ray, d_hat, qrec) : color(0,0,0);
        color contrib = qv.beta * f_q * pv.beta * cos_q * cos_p / dist2;
        return safe(contrib);
    }

    vec3 wo_q = (t >= 2) ? unit_vector(qv.p - cp[t-2].p) : d_hat;
    vec3 wo_p = (s >= 2) ? unit_vector(pv.p - lp[s-2].p) : -d_hat;

    color f_q = qv.f(wo_q,  d_hat);
    color f_p = pv.f(wo_p, -d_hat);

    double cos_q = std::abs(dot(qv.n,  d_hat));
    double cos_p = std::abs(dot(pv.n, -d_hat));
    double G     = cos_q * cos_p / dist2;

    if (G < 1e-10) return color(0,0,0);

    color contrib = qv.beta * f_q * G * f_p * pv.beta;
    return safe(contrib);
}

double mis_weight(
    const std::vector<PathVertex>& cp, int t,
    const std::vector<PathVertex>& lp, int s,
    const PathVertex&
) {
    int n = s + t;
    int valid = 0;
    for (int tt = 1; tt <= n-1 && tt <= (int)cp.size(); ++tt) {
        int ss = n - tt;
        if (ss < 0 || ss > (int)lp.size()) continue;
        bool c_ok = (tt == 1) || !cp[tt-1].delta;
        bool l_ok = (ss == 0) || !lp[ss-1].delta;
        if (c_ok && l_ok) ++valid;
    }
    if (!cp.empty() && !cp.back().delta) ++valid;
    return valid > 0 ? 1.0 / valid : 1.0;
}

color bdpt_Li(
    const ray& camera_ray,
    const color& background,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<hittable_list>& lights,
    const std::shared_ptr<env_light>& env,
    int max_depth
) {
    std::vector<PathVertex> cp, lp;
    cp.reserve(max_depth+2);
    lp.reserve(max_depth+2);

    trace_camera_path(camera_ray, color(1,1,1), world, env, max_depth, cp);
    trace_light_path(lights, world, env, max_depth, lp);

    color L(0,0,0);

    for (int t = 1; t <= (int)cp.size(); ++t) {
        for (int s = 0; s <= (int)lp.size(); ++s) {
            int depth = s + t - 2;
            if (depth < 0 || depth > max_depth) continue;

            color contrib = connect(cp, t, lp, s, world, env, background);
            if (!is_valid(contrib) || contrib.length_squared() < 1e-20) continue;

            double w = mis_weight(cp, t, lp, s,
                                   s > 0 ? lp[s-1] : cp[t-1]);
            L += contrib * w;
        }
    }

    return safe(L);
}

color Li(
    const ray& initial_ray,
    const color& background,
    const std::shared_ptr<hittable>& world,
    const std::shared_ptr<hittable_list>& lights,
    int max_depth,
    const std::shared_ptr<env_light>& env
) {
    ray   r    = initial_ray;
    color L    (0,0,0);
    color beta (1,1,1);
    bool  specular_bounce = false;
    double last_bsdf_pdf  = 0.0;

    for (int depth = 0; depth < max_depth; ++depth) {
        hit_record rec;
        if (!world->hit(r, interval(0.001, infinity), rec)) {
            color Le = env ? env->Le(r.direction()) : background;
            if (specular_bounce || !(lights && !lights->objects.empty()))
                L += beta * Le;
            else {
                double lp = lights->pdf_value(rec.p, r.direction());
                L += beta * Le * power_heuristic(last_bsdf_pdf, lp);
            }
            break;
        }

        color emitted = rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);
        if (emitted.length_squared() > 0.0) {
            if (specular_bounce) L += beta * emitted;
            else {
                double lp = (lights && !lights->objects.empty())
                            ? lights->pdf_value(rec.p, r.direction()) : 0.0;
                L += beta * emitted * power_heuristic(last_bsdf_pdf, lp);
            }
        }

        if (!specular_bounce && lights && !lights->objects.empty()) {
            vec3   to_light  = lights->random(rec.p);
            double distance  = to_light.length();
            vec3   wi        = to_light / distance;
            double light_pdf = lights->pdf_value(rec.p, wi);
            if (light_pdf > 0.0) {
                hit_record sr;
                bool occ = world->hit(ray(rec.p, wi, r.time()),
                                      interval(0.001, distance-1e-4), sr);
                if (!occ) {
                    hit_record lr;
                    if (world->hit(ray(rec.p,wi,r.time()),
                                   interval(0.001,infinity),lr)) {
                        color Le2 = lr.mat_ptr->emitted(
                            ray(rec.p,wi,r.time()),lr,lr.u,lr.v,lr.p);
                        if (Le2.length_squared() > 0.0) {
                            color  f   = rec.mat_ptr->f(r, wi, rec);
                            double bp  = rec.mat_ptr->pdf(r, wi, rec);
                            double wt  = power_heuristic(light_pdf, bp);
                            double ct  = std::abs(dot(rec.normal, wi));
                            L += beta * f * Le2 * ct * wt / light_pdf;
                        }
                    }
                }
            }
        }

        BSDFSample bs = rec.mat_ptr->sample(r, rec);
        if (bs.pdf <= 0.0) break;
        last_bsdf_pdf = bs.pdf;
        if (bs.is_delta) { beta *= bs.f; specular_bounce = true; }
        else { beta *= bs.f * std::abs(dot(rec.normal,bs.wi)) / bs.pdf; specular_bounce = false; }
        if (depth >= 3) {
            double s = std::clamp(beta.max_component(), 0.05, 0.95);
            if (random_double() > s) break;
            beta /= s;
        }
        r = ray(rec.p, bs.wi, r.time());
    }
    return L;
}