#pragma once

#include "cuda/gpu_triangle.h"
#include "cuda/gpu_scene.cuh"
#include "geometry/mesh.h"
#include "external/tiny_obj_loader.h"

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <limits>

struct AABB3f {
    float mn[3], mx[3];
    AABB3f() {
        for (int i=0;i<3;++i) { mn[i]=1e30f; mx[i]=-1e30f; }
    }
    void expand(const float* p) {
        for (int i=0;i<3;++i) {
            mn[i] = fminf(mn[i], p[i]);
            mx[i] = fmaxf(mx[i], p[i]);
        }
    }
    void expand(const AABB3f& o) {
        for (int i=0;i<3;++i) {
            mn[i] = fminf(mn[i], o.mn[i]);
            mx[i] = fmaxf(mx[i], o.mx[i]);
        }
    }
    float centroid(int axis) const { return 0.5f*(mn[axis]+mx[axis]); }
    float surface_area() const {
        float dx=mx[0]-mn[0], dy=mx[1]-mn[1], dz=mx[2]-mn[2];
        return 2.0f*(dx*dy + dy*dz + dz*dx);
    }
};

inline AABB3f tri_aabb(const GpuTriangle& t) {
    AABB3f b;
    float v[3][3] = {
        {(float)t.v0.x(),(float)t.v0.y(),(float)t.v0.z()},
        {(float)t.v1.x(),(float)t.v1.y(),(float)t.v1.z()},
        {(float)t.v2.x(),(float)t.v2.y(),(float)t.v2.z()}
    };
    for (int i=0;i<3;++i) b.expand(v[i]);
    for (int i=0;i<3;++i) { b.mn[i]-=1e-4f; b.mx[i]+=1e-4f; }
    return b;
}

static int build_bvh_node(
    std::vector<GpuTriBVHNode>& nodes,
    std::vector<int>& indices,       
    const std::vector<GpuTriangle>& tris,
    int start, int end
) {
    GpuTriBVHNode node;

    AABB3f bounds;
    for (int i=start;i<end;++i) bounds.expand(tri_aabb(tris[indices[i]]));
    for (int k=0;k<3;++k) { node.aabb_min[k]=bounds.mn[k]; node.aabb_max[k]=bounds.mx[k]; }

    int count = end - start;

    if (count == 1) {
        node.left = node.right = -(indices[start] + 1);
        nodes.push_back(node);
        return (int)nodes.size() - 1;
    }

    int axis = 0;
    float ext = 0.0f;
    for (int k=0;k<3;++k) {
        float e = bounds.mx[k] - bounds.mn[k];
        if (e > ext) { ext = e; axis = k; }
    }

    std::sort(indices.begin()+start, indices.begin()+end,
        [&](int a, int b) {
            return tri_aabb(tris[a]).centroid(axis) <
                   tri_aabb(tris[b]).centroid(axis);
        });

    int mid = start + count / 2;

    int node_idx = (int)nodes.size();
    nodes.push_back(node);

    int left_idx  = build_bvh_node(nodes, indices, tris, start, mid);
    int right_idx = build_bvh_node(nodes, indices, tris, mid,   end);

    nodes[node_idx].left  = left_idx;
    nodes[node_idx].right = right_idx;
    for (int k=0;k<3;++k) {
        nodes[node_idx].aabb_min[k] = bounds.mn[k];
        nodes[node_idx].aabb_max[k] = bounds.mx[k];
    }

    return node_idx;
}

struct GpuMesh {
    GpuTriangle*    d_triangles  = nullptr;
    GpuTriBVHNode*  d_bvh        = nullptr;
    int             n_triangles  = 0;
    int             n_bvh_nodes  = 0;
    int             bvh_root     = 0;

    void free_device() {
        if (d_triangles) { cudaFree(d_triangles); d_triangles = nullptr; }
        if (d_bvh)       { cudaFree(d_bvh);       d_bvh       = nullptr; }
    }
};

inline GpuMesh upload_obj_mesh(
    const std::string& path,
    int mat_id,
    double scale  = 1.0,
    vec3   offset = vec3(0,0,0)
) {
    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials,
                                &warn, &err, path.c_str());
    if (!ok) {
        printf("[GpuMesh] failed to load %s: %s\n", path.c_str(), err.c_str());
        return {};
    }

    int nv = (int)attrib.vertices.size() / 3;

    std::vector<vec3> smooth_normals(nv, vec3(0,0,0));
    for (const auto& shape : shapes) {
        size_t idx_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) { idx_offset += fv; continue; }
            int i0 = shape.mesh.indices[idx_offset+0].vertex_index;
            int i1 = shape.mesh.indices[idx_offset+1].vertex_index;
            int i2 = shape.mesh.indices[idx_offset+2].vertex_index;
            auto vert = [&](int i) {
                return vec3(attrib.vertices[3*i]*scale   + offset.x(),
                            attrib.vertices[3*i+1]*scale + offset.y(),
                            attrib.vertices[3*i+2]*scale + offset.z());
            };
            vec3 wn = cross(vert(i1)-vert(i0), vert(i2)-vert(i0));
            smooth_normals[i0] = smooth_normals[i0] + wn;
            smooth_normals[i1] = smooth_normals[i1] + wn;
            smooth_normals[i2] = smooth_normals[i2] + wn;
            idx_offset += fv;
        }
    }
    for (auto& n : smooth_normals) {
        double len = n.length();
        n = (len > 1e-8) ? n/len : vec3(0,1,0);
    }

    std::vector<GpuTriangle> cpu_tris;
    for (const auto& shape : shapes) {
        size_t idx_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) { idx_offset += fv; continue; }
            int i0 = shape.mesh.indices[idx_offset+0].vertex_index;
            int i1 = shape.mesh.indices[idx_offset+1].vertex_index;
            int i2 = shape.mesh.indices[idx_offset+2].vertex_index;
            auto vert = [&](int i) -> vec3 {
                return vec3(attrib.vertices[3*i]*scale   + offset.x(),
                            attrib.vertices[3*i+1]*scale + offset.y(),
                            attrib.vertices[3*i+2]*scale + offset.z());
            };
            GpuTriangle t;
            t.v0 = vert(i0); t.v1 = vert(i1); t.v2 = vert(i2);
            t.n0 = smooth_normals[i0];
            t.n1 = smooth_normals[i1];
            t.n2 = smooth_normals[i2];
            t.mat_id = mat_id;
            cpu_tris.push_back(t);
            idx_offset += fv;
        }
    }

    printf("[GpuMesh] loaded %zu triangles from %s\n", cpu_tris.size(), path.c_str());

    std::vector<GpuTriBVHNode> bvh_nodes;
    bvh_nodes.reserve(cpu_tris.size() * 2);
    std::vector<int> indices(cpu_tris.size());
    for (int i=0;i<(int)cpu_tris.size();++i) indices[i]=i;

    int root = build_bvh_node(bvh_nodes, indices, cpu_tris, 0, (int)cpu_tris.size());
    printf("[GpuMesh] BVH: %zu nodes, root=%d\n", bvh_nodes.size(), root);

    GpuMesh mesh;
    mesh.n_triangles = (int)cpu_tris.size();
    mesh.n_bvh_nodes = (int)bvh_nodes.size();
    mesh.bvh_root    = root;

    cudaMalloc(&mesh.d_triangles, cpu_tris.size()  * sizeof(GpuTriangle));
    cudaMalloc(&mesh.d_bvh,       bvh_nodes.size() * sizeof(GpuTriBVHNode));
    cudaMemcpy(mesh.d_triangles, cpu_tris.data(),
               cpu_tris.size()  * sizeof(GpuTriangle),  cudaMemcpyHostToDevice);
    cudaMemcpy(mesh.d_bvh, bvh_nodes.data(),
               bvh_nodes.size() * sizeof(GpuTriBVHNode), cudaMemcpyHostToDevice);

    float mb = (cpu_tris.size()*sizeof(GpuTriangle) +
                bvh_nodes.size()*sizeof(GpuTriBVHNode)) / 1e6f;
    printf("[GpuMesh] uploaded %.1f MB to GPU\n", mb);

    return mesh;
}