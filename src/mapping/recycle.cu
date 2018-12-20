//
// Created by wei on 17-10-22.
//

#include "mapping/recycle.h"

////////////////////
/// Device code
////////////////////
#include "core/common.h"
#include "core/entry_array.h"
#include "core/block_array.h"
#include "helper_math.h"

__global__
void StarveOccupiedBlocksKernel(
    EntryArray candidate_entries,
    BlockArray blocks
)
{
  const uint idx = blockIdx.x;
  const HashEntry &entry = candidate_entries[idx];
  for (size_t i = 0; i < 6; i++)
  {
    if (not blocks.HasVoxelArray(entry.ptr, i))
      continue;
    Voxel &voxel = blocks.GetVoxelArray(entry.ptr, i).voxels[threadIdx.x];
    float inv_sigma2 = voxel.inv_sigma2;
    inv_sigma2 = fmaxf(0, inv_sigma2 - 1.0f);
    voxel.inv_sigma2 = inv_sigma2;
  }
}

/**
 * Identify impossible blocks.
 *
 * Accumulates information over all voxels of the block.
 * If the minimum sdf values is too far away from surface or the maximum sdf weight (inv_sigma2) is too small:
 * flag block for cleanup.
 *
 * @param candidate_entries
 * @param blocks
 * @param geometry_helper
 */
__global__
void CollectGarbageBlockArrayKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    GeometryHelper geometry_helper
)
{
  const uint bIdx = blockIdx.x;
  const uint tIdx = threadIdx.x;
  const HashEntry &entry = candidate_entries[bIdx];

  __shared__ float shared_min_sdf[BLOCK_SIZE / 2];
  __shared__ float shared_max_inv_sigma2[BLOCK_SIZE / 2];

  // 1) Initialize shared memory: find min SDF and max inv_sigma2 for two voxels
  float min_sdf_thread = PINF;
  float max_inv_sigma2_thread = 0;
  for (size_t i = 0; i < 6; i++)
  {
    if (not blocks.HasVoxelArray(entry.ptr, i))
      continue;
    const Voxel &v0 = blocks.GetVoxelArray(entry.ptr, i).voxels[2 * tIdx + 0];
    const Voxel &v1 = blocks.GetVoxelArray(entry.ptr, i).voxels[2 * tIdx + 1];

    float sdf0 = v0.sdf, sdf1 = v1.sdf;
    if (v0.inv_sigma2 < EPSILON) sdf0 = PINF;
    if (v1.inv_sigma2 < EPSILON) sdf1 = PINF;

    min_sdf_thread = fminf(min_sdf_thread, fminf(fabsf(sdf0), fabsf(sdf1)));
    max_inv_sigma2_thread = fmaxf(max_inv_sigma2_thread, fmaxf(v0.inv_sigma2, v1.inv_sigma2));
  }
  shared_min_sdf[tIdx] = min_sdf_thread;
  shared_max_inv_sigma2[tIdx] = max_inv_sigma2_thread;
  __syncthreads();

  // 2) reduction operation: find min SDf and max inv_sigma2 over all voxels
#pragma unroll 1
  for (uint stride = blockDim.x / 2; stride > 0; stride >>= 1)
  { // inverse counting to use a minimal number of Warps (physical groups of threads)
    if (tIdx < stride)
    {
      shared_min_sdf[tIdx] = fminf(shared_min_sdf[tIdx + stride],
                                          shared_min_sdf[tIdx]);
      shared_max_inv_sigma2[tIdx] = fmaxf(shared_max_inv_sigma2[tIdx + stride],
                                                 shared_max_inv_sigma2[tIdx]);
    }
    __syncthreads();
  }

  // 3) Finally decide whether to flag the current block
  if (tIdx == 0)
  {
    float min_sdf = shared_min_sdf[tIdx];
    float max_inv_sigma2 = shared_max_inv_sigma2[tIdx];

    // TODO(wei): check this weird reference
    float t = geometry_helper.truncate_distance(5.0f);

    // TODO(wei): add || valid_triangles == 0 when memory leak is dealt with
    candidate_entries.flag(bIdx) |=
        (min_sdf >= t || max_inv_sigma2 < EPSILON) ? (uchar) 1 : (uchar) 0;
  }
}


/**
 * Check whether there are no outer surfaces -> reduce life count -> recycle if 0
 */
__global__
void CollectLowSurfelBlocksKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    int processing_block_count
)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= processing_block_count)
  {
    return;
  }
  const HashEntry &entry = candidate_entries[idx];

  candidate_entries.flag(idx) = 0;
  if (blocks[entry.ptr].inner_surfel_count < 10
      && blocks[entry.ptr].boundary_surfel_count == 0)
  {
    blocks[entry.ptr].life_count_down--;
  } else
  {
    blocks[entry.ptr].life_count_down = BLOCK_LIFE;
  }
  __syncthreads();

  const int3 offsets[6] = {
      {0,  0,  1},
      {0,  0,  -1},
      {0,  1,  0},
      {0,  -1, 0},
      {1,  0,  0},
      {-1, 0,  0}
  };
  // Check neighboring blocks in cross pattern. If all have life count = 0 -> flag for removal
  if (blocks[entry.ptr].life_count_down <= 0)
  {
    for (int j = 0; j < 6; ++j)
    {
      HashEntry query_entry = hash_table.GetEntry(entry.pos + offsets[j]);
      if (query_entry.ptr == FREE_ENTRY) continue;
      if (blocks[query_entry.ptr].life_count_down != 0)
        return;
    }
    candidate_entries.flag(idx) |= 1;
  }
}

/// !!! Their mesh not recycled
__global__
void RecycleGarbageTrianglesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    HashTable hash_table
)
{
  const uint idx = blockIdx.x;
  if (candidate_entries.flag(idx) == 0) return;

  const HashEntry &entry = candidate_entries[idx];
  const uint local_idx = threadIdx.x;  //inside an SDF block
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[local_idx];

  for (int i = 0; i < N_TRIANGLE; ++i)
  {
    int triangle_ptr = mesh_unit.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) continue;

    // Clear ref_count of its pointed vertices
    mesh.ReleaseTriangleVertexReferences(mesh.triangle(triangle_ptr));
    mesh.FreeTriangle(triangle_ptr);
    mesh_unit.triangle_ptrs[i] = FREE_PTR;
  }
}

__global__
void RecycleGarbageVerticesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    HashTable hash_table
)
{
  if (candidate_entries.flag(blockIdx.x) == 0) return;
  const HashEntry &entry = candidate_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  MeshUnit &cube = blocks[entry.ptr].mesh_units[local_idx];

  __shared__ int valid_vertex_count;
  if (threadIdx.x == 0) valid_vertex_count = 0;
  __syncthreads();

#pragma unroll 1
  for (int i = 0; i < N_VERTEX; ++i)
  {
    if (cube.vertex_ptrs[i] != FREE_PTR)
    {
      if (mesh.vertex(cube.vertex_ptrs[i]).ref_count <= 0)
      {
        mesh.FreeVertex(cube.vertex_ptrs[i]);
        cube.vertex_ptrs[i] = FREE_PTR;
        cube.vertex_mutexes[i] = FREE_ENTRY;
      } else
      {
        atomicAdd(&valid_vertex_count, 1);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 && valid_vertex_count == 0)
  {
    if (hash_table.FreeEntry(entry.pos))
    {
      blocks.FreeBlock(entry.ptr);
    }
  }
}

void StarveOccupiedBlockArray(
    EntryArray &candidate_entries,
    BlockArray &blocks
)
{
  const uint threads_per_block = BLOCK_SIZE;

  uint processing_block_count = candidate_entries.count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  StarveOccupiedBlocksKernel << < grid_size, block_size >> > (candidate_entries, blocks);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void CollectGarbageBlockArray(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    GeometryHelper &geometry_helper
)
{
  const uint threads_per_block = BLOCK_SIZE / 2;

  uint processing_block_count = candidate_entries.count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  CollectGarbageBlockArrayKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void CollectLowSurfelBlocks(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    HashTable &hash_table,
    GeometryHelper &geometry_helper
)
{
  uint processing_block_count = candidate_entries.count();
  if (processing_block_count <= 0)
    return;

  const int threads_per_block = 64;
  const dim3 grid_size((processing_block_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  CollectLowSurfelBlocksKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          hash_table,
          geometry_helper,
          processing_block_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

// TODO(wei): Check vertex / triangles in detail
void RecycleGarbageBlockArray(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh,
    HashTable &hash_table
)
{
  const uint threads_per_block = BLOCK_SIZE;

  uint processing_block_count = candidate_entries.count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  RecycleGarbageTrianglesKernel << < grid_size, block_size >> > (
      candidate_entries, blocks, mesh, hash_table);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleGarbageVerticesKernel << < grid_size, block_size >> > (
      candidate_entries, blocks, mesh, hash_table);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

