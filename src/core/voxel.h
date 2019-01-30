//
// Created by wei on 17-10-21.
//

#ifndef CORE_VOXEL_H
#define CORE_VOXEL_H

#include "core/common.h"
#include "helper_math.h"

// Statistics typically reserved for Voxels
// float: *Laplacian* and *entropy* are intuitive statistics
// float: *duration* is time-interval that the voxel exists
struct __ALIGN__(4) Stat {
  float laplacian;
  float entropy;
  float duration;

  __host__ __device__
  void Clear() {
    laplacian = 0;
    entropy = 0;
    duration = 0;
  }
};

struct __ALIGN__(4) MeshUnit {
  // mesh
  int vertex_ptrs   [N_VERTEX];
  int vertex_mutexes[N_VERTEX];
  int triangle_ptrs [N_TRIANGLE];
  short mc_idx[2];

  __host__ __device__
  void ResetMutexes() {
#pragma unroll
    for (uint i = 0; i < N_VERTEX; i++)
    {
      vertex_mutexes[i] = FREE_PTR;
    }
  }

  __host__ __device__
  int GetVertex(int idx) {
    return vertex_ptrs[idx];
  }

  __host__ __device__
  void Clear() {
    for (uint i = 0; i < N_VERTEX; i++)
    {
      vertex_ptrs[i] = vertex_mutexes[i] = FREE_PTR;
    }

    for (uint i = 0; i < N_TRIANGLE; i++)
    {
      triangle_ptrs[i] = FREE_PTR;
    }

    mc_idx[0] = 0;
    mc_idx[1] = 0;
  }
};

struct __ALIGN__(4) PrimalDualVariables {
  bool   mask;
  float  sdf0, sdf_bar, inv_sigma2;
  float3 p;

  __host__ __device__
  void operator = (const PrimalDualVariables& pdv) {
    mask = pdv.mask;
    sdf0 = pdv.sdf0;
    sdf_bar = pdv.sdf_bar;
    p = pdv.p;
  }

  __host__ __device__
  void Clear() {
    mask = false;
    sdf0 = sdf_bar = 0;
    inv_sigma2 = 0;
    p = make_float3(0);
  }
};

struct __ALIGN__(4) Voxel {
  float  sdf;    // signed distance function, mu
  float  inv_sigma2; // sigma
  float  a, b;
  uchar3 color;  // color
  unsigned int num_updates;

  __host__ __device__
  void operator = (const Voxel& v) {
    sdf = v.sdf;
    inv_sigma2 = v.inv_sigma2;
    color = v.color;
    a = v.a;
    b = v.b;
  }

  __host__ __device__
  void Clear() {
    sdf = inv_sigma2 = 0.0f;
    color = make_uchar3(0, 0, 0);
    a = b = 0;
    num_updates = 0;
  }

  __host__ __device__
  void Update(const Voxel &delta) {
    float3 c_prev  = make_float3(color.x, color.y, color.z);
    float3 c_delta = make_float3(delta.color.x, delta.color.y, delta.color.z);
    float3 c_curr  = 0.5f * c_prev + 0.5f * c_delta;
    color = make_uchar3(c_curr.x + 0.5f, c_curr.y + 0.5f, c_curr.z + 0.5f);

    sdf = (sdf * inv_sigma2 + delta.sdf * delta.inv_sigma2) / (inv_sigma2 + delta.inv_sigma2);
    inv_sigma2 += delta.inv_sigma2;
  }
};

#endif // CORE_VOXEL_H
