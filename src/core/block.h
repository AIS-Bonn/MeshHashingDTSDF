//
// Created by wei on 17-5-21.
//

#ifndef CORE_BLOCK_H
#define CORE_BLOCK_H

#include "core/common.h"
#include "core/voxel.h"

#include <helper_math.h>

#define BLOCK_LIFE 3

// Typically Block is a 8x8x8 voxel array
struct __ALIGN__(8) VoxelArray
{
  Voxel voxels[BLOCK_SIZE];

  __host__ __device__
  void Clear()
  {
#ifdef __CUDA_ARCH__ // __CUDA_ARCH__ is only defined for __device__
#pragma unroll 8
#endif
    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
      voxels[i].Clear();
    }
  }
};

// Typically Block is a 8x8x8 voxel array
struct __ALIGN__(8) Block
{
  int life_count_down;

  /** Counts the number of surfaces on the boundary of the block */
  int boundary_surfel_count;

  /** Counts the number of surfaces inside (not on the boundary) the block */
  int inner_surfel_count;

  MeshUnit mesh_units[BLOCK_SIZE];
  VoxelArray *voxel_arrays[6];
  int voxel_array_ptrs[6];
  int voxel_array_mutexes[6];

  __host__ __device__
  void Clear()
  {
    inner_surfel_count = 0;
    boundary_surfel_count = 0;
    life_count_down = BLOCK_LIFE;

#ifdef __CUDA_ARCH__
#pragma unroll 6
#endif
    for (uint i = 0; i < 6; ++i)
    {
      voxel_arrays[i] = nullptr;
      voxel_array_ptrs[i] = FREE_PTR;
      voxel_array_mutexes[i] = FREE_PTR;
    }
#ifdef __CUDA_ARCH__
#pragma unroll 4 // Should be equal to BLOCK_SIZE (macros don't work here)
#endif
    for (int i = 0; i < BLOCK_SIZE; ++i)
    {
      mesh_units[i].Clear();
    }
  }
};

#endif // CORE_BLOCK_H
