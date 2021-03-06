#include "mesh.h"

#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include "params.h"
#include <glog/logging.h>

////////////////////
/// class Mesh
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void MeshResetVerticesKernel(uint *vertex_heap, Vertex *vertices, int max_vertex_count)
{
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < max_vertex_count)
  {
    vertex_heap[idx] = max_vertex_count - idx - 1;
    vertices[idx].Clear();
  }
}

__global__
void MeshResetTrianglesKernel(uint *triangle_heap, Triangle *triangles, int max_triangle_count)
{
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < max_triangle_count)
  {
    triangle_heap[idx] = max_triangle_count - idx - 1;
    triangles[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////
// Mesh::~Mesh() {
//Free();
//}

__host__
void Mesh::Alloc(const MeshParams &mesh_params)
{
  if (!is_allocated_on_gpu_)
  {
    checkCudaErrors(cudaMalloc(&vertex_heap_,
                               sizeof(uint) * mesh_params.max_vertex_count));
    checkCudaErrors(cudaMalloc(&vertex_heap_counter_, sizeof(int)));
    checkCudaErrors(cudaMalloc(&vertices,
                               sizeof(Vertex) * mesh_params.max_vertex_count));

    checkCudaErrors(cudaMalloc(&triangle_heap_,
                               sizeof(uint) * mesh_params.max_triangle_count));
    checkCudaErrors(cudaMalloc(&triangle_heap_counter_, sizeof(int)));
    checkCudaErrors(cudaMalloc(&triangles,
                               sizeof(Triangle) * mesh_params.max_triangle_count));
    is_allocated_on_gpu_ = true;
  }
}

void Mesh::Free()
{
  if (is_allocated_on_gpu_)
  {
    checkCudaErrors(cudaFree(vertex_heap_));
    checkCudaErrors(cudaFree(vertex_heap_counter_));
    checkCudaErrors(cudaFree(vertices));

    checkCudaErrors(cudaFree(triangle_heap_));
    checkCudaErrors(cudaFree(triangle_heap_counter_));
    checkCudaErrors(cudaFree(triangles));

    is_allocated_on_gpu_ = false;
  }
}

void Mesh::Resize(const MeshParams &mesh_params)
{
  mesh_params_ = mesh_params;
  if (is_allocated_on_gpu_)
  {
    Free();
  }
  Alloc(mesh_params);
  Reset();
}

void Mesh::Reset()
{
  uint val;

  val = mesh_params_.max_vertex_count - 1;
  checkCudaErrors(cudaMemcpy(vertex_heap_counter_,
                             &val,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));

  val = mesh_params_.max_triangle_count - 1;
  checkCudaErrors(cudaMemcpy(triangle_heap_counter_,
                             &val,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));

  {
    const int threads_per_block = 64;
    const dim3 grid_size((mesh_params_.max_vertex_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    MeshResetVerticesKernel << < grid_size, block_size >> > (vertex_heap_, vertices,
        mesh_params_.max_vertex_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const int threads_per_block = 64;
    const dim3 grid_size((mesh_params_.max_triangle_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    MeshResetTrianglesKernel << < grid_size, block_size >> > (triangle_heap_, triangles,
        mesh_params_.max_triangle_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

uint Mesh::vertex_heap_count()
{
  uint vertex_heap_count;
  checkCudaErrors(cudaMemcpy(&vertex_heap_count,
                             vertex_heap_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return vertex_heap_count;
}

uint Mesh::triangle_heap_count()
{
  uint triangle_heap_count;
  checkCudaErrors(cudaMemcpy(&triangle_heap_count,
                             triangle_heap_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return triangle_heap_count;
}

__device__
int Mesh::AllocVertex()
{
  int addr = atomicSub(&vertex_heap_counter_[0], 1);
  if (addr < MEMORY_LIMIT)
  {
    printf("vertex heap: %d\n", addr, vertex_heap_[addr]);
    if (addr < 0)
    {
      atomicAdd(&vertex_heap_counter_[0], 1);
      return FREE_PTR;
    }
  }
  return vertex_heap_[addr];
}

__device__
void Mesh::FreeVertex(uint ptr)
{
  vertices[ptr].Clear();
  uint addr = atomicAdd(&vertex_heap_counter_[0], 1);
  vertex_heap_[addr + 1] = ptr;
}

__device__
int Mesh::AllocTriangle()
{
  int addr = atomicSub(&triangle_heap_counter_[0], 1);
  if (addr < MEMORY_LIMIT)
  {
    printf("triangle heap: %d\n", addr);
    if (addr < 0)
    {
      atomicAdd(&triangle_heap_counter_[0], 1);
      return FREE_PTR;
    }
  }
  triangle(triangle_heap_[addr]).Clear();
  return triangle_heap_[addr];
}

__device__
void Mesh::FreeTriangle(uint ptr)
{
  ReleaseTriangleVertexReferences(triangle(ptr));
  uint addr = atomicAdd(&triangle_heap_counter_[0], 1);
  triangle_heap_[addr + 1] = ptr;
}

/**
 * Releases the references to the current set of vertex pointers.
 * This is NOT equal to a free operation!
 * @param triangle
 */
__device__
void Mesh::ReleaseTriangleVertexReferences(Triangle &triangle)
{
  int3 vertex_ptrs = triangle.vertex_ptrs;
  if (vertex_ptrs.x >= 0)
    atomicSub(&vertices[vertex_ptrs.x].ref_count, 1);
  if (vertex_ptrs.y >= 0)
    atomicSub(&vertices[vertex_ptrs.y].ref_count, 1);
  if (vertex_ptrs.z >= 0)
    atomicSub(&vertices[vertex_ptrs.z].ref_count, 1);
  triangle.Clear();
}

__device__
void Mesh::AssignTriangleVertexReferences(Triangle &triangle, int3 vertex_ptrs)
{
  triangle.vertex_ptrs = vertex_ptrs;
  atomicAdd(&vertices[vertex_ptrs.x].ref_count, 1);
  atomicAdd(&vertices[vertex_ptrs.y].ref_count, 1);
  atomicAdd(&vertices[vertex_ptrs.z].ref_count, 1);
}

__device__
void Mesh::ComputeTriangleNormal(Triangle &triangle)
{
  int3 vertex_ptrs = triangle.vertex_ptrs;
  float3 p0 = vertices[vertex_ptrs.x].pos;
  float3 p1 = vertices[vertex_ptrs.y].pos;
  float3 p2 = vertices[vertex_ptrs.z].pos;
  float3 n = normalize(cross(p2 - p0, p1 - p0));
  vertices[vertex_ptrs.x].normal = n;
  vertices[vertex_ptrs.y].normal = n;
  vertices[vertex_ptrs.z].normal = n;
}
