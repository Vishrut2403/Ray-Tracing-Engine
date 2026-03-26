#pragma once

#include "hittables/hittable.h"
#include "hittables/hittable_list.h"
#include "materials/material.h"
#include "textures/image_texture.h"
#include "textures/texture.h"
#include "acceleration/bvh.h"
#include "core/rtweekend.h"
#include "external/tiny_gltf.h"

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <functional>

struct GltfUV { double u, v; };

class gltf_triangle : public hittable {
public:
    point3  v0, v1, v2;
    vec3    n0, n1, n2;
    GltfUV  uv0, uv1, uv2;
    std::shared_ptr<material> mat;

    gltf_triangle(const point3& a,  const point3& b,  const point3& c,
                  const vec3&   na, const vec3&   nb, const vec3&   nc,
                  const GltfUV& ta, const GltfUV& tb, const GltfUV& tc,
                  std::shared_ptr<material> m)
        : v0(a), v1(b), v2(c)
        , n0(na), n1(nb), n2(nc)
        , uv0(ta), uv1(tb), uv2(tc)
        , mat(m) {}

    virtual bool hit(const ray& r, const interval& ray_t,
                        hit_record& rec) const override {
            const double eps = 1e-8;
            vec3 e1 = v1 - v0, e2 = v2 - v0;
            vec3 h  = cross(r.direction(), e2);
            double a = dot(e1, h);
            if (std::abs(a) < eps) return false;

            double f = 1.0 / a;
            vec3   s = r.origin() - v0;
            double u = f * dot(s, h);
            if (u < 0.0 || u > 1.0) return false;

            vec3   q = cross(s, e1);
            double v = f * dot(r.direction(), q);
            if (v < 0.0 || u + v > 1.0) return false;

            double t = f * dot(e2, q);
            if (!ray_t.surrounds(t)) return false;

            double w = 1.0 - u - v;

            // Interpolate smooth normal and UV
            vec3 smooth_n = unit_vector(n0*w + n1*u + n2*v);
            rec.u = uv0.u*w + uv1.u*u + uv2.u*v;
            rec.v = uv0.v*w + uv1.v*u + uv2.v*v;

            // Compute tangent from UV gradients (Lengyel 2001)
            double du1 = uv1.u - uv0.u, du2 = uv2.u - uv0.u;
            double dv1 = uv1.v - uv0.v, dv2 = uv2.v - uv0.v;
            double det = du1*dv2 - du2*dv1;

            vec3 tangent, bitangent;
            if (std::abs(det) > 1e-8) {
                double inv = 1.0 / det;
                tangent   = unit_vector((e1*dv2 - e2*dv1) * inv);
                bitangent = unit_vector((e2*du1 - e1*du2) * inv);
            } else {
                // Degenerate UV — build arbitrary TBN from normal
                onb uvw; uvw.build_from_w(smooth_n);
                tangent   = uvw.u();
                bitangent = uvw.v();
            }

            // Gram-Schmidt orthogonalise tangent against normal
            tangent   = unit_vector(tangent - smooth_n * dot(smooth_n, tangent));
            bitangent = cross(smooth_n, tangent);

            rec.t         = t;
            rec.p         = r.at(t);
            rec.mat_ptr   = mat;
            rec.tangent   = tangent;
            rec.bitangent = bitangent;
            rec.set_face_normal(r, smooth_n);
            return true;
        }

    virtual bool bounding_box(double, double, aabb& out) const override {
        point3 lo(fmin(fmin(v0.x(),v1.x()),v2.x()) - 1e-4,
                  fmin(fmin(v0.y(),v1.y()),v2.y()) - 1e-4,
                  fmin(fmin(v0.z(),v1.z()),v2.z()) - 1e-4);
        point3 hi(fmax(fmax(v0.x(),v1.x()),v2.x()) + 1e-4,
                  fmax(fmax(v0.y(),v1.y()),v2.y()) + 1e-4,
                  fmax(fmax(v0.z(),v1.z()),v2.z()) + 1e-4);
        out = aabb(lo, hi);
        return true;
    }
};

class gltf_ggx : public material {
public:
    std::shared_ptr<texture> albedo_tex;
    std::shared_ptr<texture> metal_rough_tex;
    std::shared_ptr<texture> emissive_tex;
    std::shared_ptr<texture> normal_tex;      // tangent-space normal map
    std::shared_ptr<texture> ao_tex;             // ambient occlusion (R channel)
    color  base_color_factor = color(1,1,1);
    double metallic_factor   = 1.0;
    double roughness_factor  = 1.0;
    color  emissive_factor   = color(0,0,0);
    double normal_scale      = 1.0;           // normal map strength

    virtual color emitted(const ray&, const hit_record& rec,
                           double u, double v, const point3& p) const override {
        if (!rec.front_face) return color(0,0,0);
        color e = emissive_factor;
        if (emissive_tex) e = e * emissive_tex->value(u, v, p);
        return e;
    }

    virtual BSDFSample sample(const ray& wo,
                               const hit_record& rec) const override {
        hit_record perturbed = perturb_normal(rec);
        ggx delegate(resolved_albedo(rec), resolved_roughness(rec),
                     resolved_metallic(rec));
        return delegate.sample(wo, perturbed);
    }

    virtual color f(const ray& wo, const vec3& wi,
                    const hit_record& rec) const override {
        hit_record perturbed = perturb_normal(rec);
        ggx delegate(resolved_albedo(rec), resolved_roughness(rec),
                     resolved_metallic(rec));
        return delegate.f(wo, wi, perturbed);
    }

    virtual double pdf(const ray& wo, const vec3& wi,
                       const hit_record& rec) const override {
        hit_record perturbed = perturb_normal(rec);
        ggx delegate(resolved_albedo(rec), resolved_roughness(rec),
                     resolved_metallic(rec));
        return delegate.pdf(wo, wi, perturbed);
    }

private:
    hit_record perturb_normal(const hit_record& rec) const {
        if (!normal_tex) return rec;

        color ns = normal_tex->value(rec.u, rec.v, rec.p);
        double nx = (ns.x() * 2.0 - 1.0) * normal_scale;
        double ny = (ns.y() * 2.0 - 1.0) * normal_scale;
        double nz =  ns.z() * 2.0 - 1.0; 

        vec3 n_world = unit_vector(
            rec.tangent   * nx +
            rec.bitangent * ny +
            rec.normal    * nz
        );

        if (dot(n_world, rec.normal) < 0.0) n_world = -n_world;

        hit_record perturbed = rec;
        perturbed.normal = n_world;
        return perturbed;
    }

    color resolved_albedo(const hit_record& rec) const {
        color alb = base_color_factor;
        if (albedo_tex) alb = alb * albedo_tex->value(rec.u, rec.v, rec.p);
        if (ao_tex) {
            double ao = ao_tex->value(rec.u, rec.v, rec.p).x();
            alb = alb * ao;
        }
        return alb;
    }
    double resolved_roughness(const hit_record& rec) const {
        double r = roughness_factor;
        if (metal_rough_tex) r *= metal_rough_tex->value(rec.u,rec.v,rec.p).y();
        return clamp(r, 0.001, 1.0);
    }
    double resolved_metallic(const hit_record& rec) const {
        double m = metallic_factor;
        if (metal_rough_tex) m *= metal_rough_tex->value(rec.u,rec.v,rec.p).z();
        return clamp(m, 0.0, 1.0);
    }
};

static std::vector<float> get_accessor_floats(
    const tinygltf::Model& model, int acc_idx, int n_comp
) {
    if (acc_idx < 0) return {};
    const auto& acc  = model.accessors[acc_idx];
    const auto& view = model.bufferViews[acc.bufferView];
    const auto& buf  = model.buffers[view.buffer];

    int comp_bytes = (int)tinygltf::GetComponentSizeInBytes(acc.componentType);
    int stride = view.byteStride > 0 ? (int)view.byteStride : n_comp * comp_bytes;

    std::vector<float> result;
    result.reserve(acc.count * n_comp);
    const unsigned char* base = buf.data.data() + view.byteOffset + acc.byteOffset;

    for (size_t i = 0; i < acc.count; ++i) {
        const unsigned char* ptr = base + i * stride;
        for (int c = 0; c < n_comp; ++c) {
            if (acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                float val; std::memcpy(&val, ptr + c*4, 4);
                result.push_back(val);
            } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                unsigned short val; std::memcpy(&val, ptr + c*2, 2);
                result.push_back(val / 65535.0f);
            } else {
                result.push_back(ptr[c] / 255.0f);
            }
        }
    }
    return result;
}

static std::vector<unsigned int> get_accessor_indices(
    const tinygltf::Model& model, int acc_idx
) {
    if (acc_idx < 0) return {};
    const auto& acc  = model.accessors[acc_idx];
    const auto& view = model.bufferViews[acc.bufferView];
    const auto& buf  = model.buffers[view.buffer];
    const unsigned char* base = buf.data.data() + view.byteOffset + acc.byteOffset;

    std::vector<unsigned int> idx;
    idx.reserve(acc.count);
    for (size_t i = 0; i < acc.count; ++i) {
        if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
            unsigned int v; std::memcpy(&v, base+i*4, 4); idx.push_back(v);
        } else if (acc.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
            unsigned short v; std::memcpy(&v, base+i*2, 2); idx.push_back(v);
        } else {
            idx.push_back(base[i]);
        }
    }
    return idx;
}

inline std::shared_ptr<hittable> load_gltf(
    const std::string& path,
    double scale  = 1.0,
    vec3   offset = vec3(0,0,0)
) {
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string warn, err;

    std::string base_dir =
        std::filesystem::path(path).parent_path().string() + "/";

    if (!loader.LoadASCIIFromFile(&model, &err, &warn, path)) {
        std::cerr << "[glTF] failed: " << err << "\n";
        return nullptr;
    }
    if (!warn.empty()) std::cerr << "[glTF] warn: " << warn << "\n";

    std::vector<std::shared_ptr<texture>> textures;
    for (const auto& img : model.images) {
        std::string p = base_dir + img.uri;
        textures.push_back(std::make_shared<image_texture>(p.c_str()));
        std::cerr << "[glTF] texture: " << img.uri << "\n";
    }

    auto get_tex = [&](int tex_idx) -> std::shared_ptr<texture> {
        if (tex_idx < 0 || tex_idx >= (int)model.textures.size()) return nullptr;
        int src = model.textures[tex_idx].source;
        if (src < 0 || src >= (int)textures.size()) return nullptr;
        return textures[src];
    };

    std::vector<std::shared_ptr<material>> mats;
    for (const auto& gm : model.materials) {
        auto m = std::make_shared<gltf_ggx>();
        const auto& pbr = gm.pbrMetallicRoughness;
        if (!pbr.baseColorFactor.empty())
            m->base_color_factor = color(pbr.baseColorFactor[0],
                                          pbr.baseColorFactor[1],
                                          pbr.baseColorFactor[2]);
        m->albedo_tex       = get_tex(pbr.baseColorTexture.index);
        m->metal_rough_tex  = get_tex(pbr.metallicRoughnessTexture.index);
        m->metallic_factor  = pbr.metallicFactor;
        m->roughness_factor = pbr.roughnessFactor;
        if (!gm.emissiveFactor.empty())
            m->emissive_factor = color(gm.emissiveFactor[0],
                                        gm.emissiveFactor[1],
                                        gm.emissiveFactor[2]);
        m->emissive_tex = get_tex(gm.emissiveTexture.index);
        m->ao_tex = get_tex(gm.occlusionTexture.index);
                if (!gm.normalTexture.extensions.empty() ||
                    gm.normalTexture.scale != 0.0)
                    m->normal_scale = gm.normalTexture.scale > 0.0
                                    ? gm.normalTexture.scale : 1.0;
        mats.push_back(m);
        std::cerr << "[glTF] material: " << gm.name << "\n";
    }

    auto fallback = std::make_shared<lambertian>(color(0.7,0.7,0.7));

    hittable_list tris;
    size_t tri_count = 0;

    std::function<void(int)> proc_node = [&](int ni) {
        const auto& node = model.nodes[ni];

        double ns = scale;
        vec3   no = offset;
        if (!node.translation.empty())
            no = no + vec3(node.translation[0]*scale,
                           node.translation[1]*scale,
                           node.translation[2]*scale);
        if (!node.scale.empty())
            ns *= node.scale[0];

        if (node.mesh >= 0) {
            const auto& mesh = model.meshes[node.mesh];
            for (const auto& prim : mesh.primitives) {
                if (prim.mode != TINYGLTF_MODE_TRIANGLES) continue;

                std::shared_ptr<material> mat = fallback;
                if (prim.material >= 0 && prim.material < (int)mats.size())
                    mat = mats[prim.material];

                auto pi = prim.attributes.find("POSITION");
                auto ni2 = prim.attributes.find("NORMAL");
                auto ui = prim.attributes.find("TEXCOORD_0");
                if (pi == prim.attributes.end()) continue;

                auto pos = get_accessor_floats(model, pi->second, 3);
                auto nor = (ni2 != prim.attributes.end())
                           ? get_accessor_floats(model, ni2->second, 3)
                           : std::vector<float>{};
                auto uvs = (ui != prim.attributes.end())
                           ? get_accessor_floats(model, ui->second, 2)
                           : std::vector<float>{};
                auto idx = get_accessor_indices(model, prim.indices);

                int nv = (int)pos.size() / 3;

                if (nor.empty()) {
                    nor.resize(nv*3, 0.0f);
                    for (size_t fi = 0; fi+2 < idx.size(); fi += 3) {
                        auto i0=idx[fi],i1=idx[fi+1],i2=idx[fi+2];
                        vec3 a(pos[i0*3],pos[i0*3+1],pos[i0*3+2]);
                        vec3 b(pos[i1*3],pos[i1*3+1],pos[i1*3+2]);
                        vec3 c(pos[i2*3],pos[i2*3+1],pos[i2*3+2]);
                        vec3 wn = cross(b-a,c-a);
                        for (auto v:{(int)i0,(int)i1,(int)i2}) {
                            nor[v*3+0]+=(float)wn.x();
                            nor[v*3+1]+=(float)wn.y();
                            nor[v*3+2]+=(float)wn.z();
                        }
                    }
                    for (int i=0;i<nv;++i) {
                        vec3 n(nor[i*3],nor[i*3+1],nor[i*3+2]);
                        double l=n.length();
                        if(l>1e-8){nor[i*3+0]/=(float)l;nor[i*3+1]/=(float)l;nor[i*3+2]/=(float)l;}
                    }
                }

                auto vp  = [&](unsigned int i) -> point3 {
                    return point3(pos[i*3]*ns+no.x(),
                                  pos[i*3+1]*ns+no.y(),
                                  pos[i*3+2]*ns+no.z());
                };
                auto vn  = [&](unsigned int i) -> vec3 {
                    return unit_vector(vec3(nor[i*3],nor[i*3+1],nor[i*3+2]));
                };
                auto vuv = [&](unsigned int i) -> GltfUV {
                    if(uvs.empty()) return {0,0};
                    return {uvs[i*2], 1.0-uvs[i*2+1]};
                };

                for (size_t fi=0;fi+2<idx.size();fi+=3) {
                    auto i0=idx[fi],i1=idx[fi+1],i2=idx[fi+2];
                    tris.add(std::make_shared<gltf_triangle>(
                        vp(i0),vp(i1),vp(i2),
                        vn(i0),vn(i1),vn(i2),
                        vuv(i0),vuv(i1),vuv(i2),
                        mat));
                    ++tri_count;
                }
            }
        }
        for (int ch : node.children) proc_node(ch);
    };

    int si = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (si < (int)model.scenes.size())
        for (int ni : model.scenes[si].nodes) proc_node(ni);

    std::cerr << "[glTF] " << tri_count << " triangles from " << path << "\n";

    return std::make_shared<bvh_node>(
        tris.objects, 0, tris.objects.size(), 0.0, 1.0);
}