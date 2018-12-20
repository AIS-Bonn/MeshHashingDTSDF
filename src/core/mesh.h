//
// Created by wei on 17-5-21.
//

#ifndef CORE_MESH_H
#define CORE_MESH_H

#include "core/common.h"
#include "core/params.h"
#include "core/vertex.h"
#include "core/triangle.h"

#include <helper_cuda.h>
#include <helper_math.h>

class Mesh {
public:
  __host__ Mesh() = default;
  // __host__ ~Mesh();

  __host__ void Alloc(const MeshParams &mesh_params);
  __host__ void Resize(const MeshParams &mesh_params);
  __host__ void Free();
  __host__ void Reset();

  const MeshParams& params() {
    return mesh_params_;
  }

  __device__ __host__ Vertex& vertex(uint i) {
    return vertices[i];
  }
  __device__ __host__ Triangle& triangle(uint i) {
    return triangles[i];
  }

  __host__ uint vertex_heap_count();
  __host__ uint triangle_heap_count();

private:
  bool is_allocated_on_gpu_ = false;
  uint*     vertex_heap_;
  uint*     vertex_heap_counter_;
  Vertex*   vertices;

  uint*     triangle_heap_;
  uint*     triangle_heap_counter_;
  Triangle* triangles;

public:
  __device__ uint AllocVertex();
  __device__ void FreeVertex(uint ptr);
  __device__ uint AllocTriangle();
  __device__ void FreeTriangle(uint ptr);

  /// Release is NOT always a FREE operation
  __device__ void ReleaseTriangleVertexReferences(Triangle &triangle);
  __device__ void AssignTriangleVertexReferences(Triangle &triangle, int3 vertex_ptrs);
  __device__ void ComputeTriangleNormal(Triangle& triangle);

  MeshParams mesh_params_;

};

#endif //VOXEL_HASHING_MESH_H
