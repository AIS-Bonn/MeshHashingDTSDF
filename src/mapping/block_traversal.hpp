#pragma once

#include "core/common.h"
#include "extern/cuda/helper_math.h"
#include <host_defines.h>
#include <vector_types.h>

struct BlockTraversal
{
  /**
   * @param origin Ray origin
   * @param dir Ray direction
   * @param truncation_distance Truncation distance
   * @param block_size Size of the blocks to traverse
   * @param round_to_nearest Whether to round the indices to the nearest responsible index (used for Voxels)
   */
  __device__ __host__
  BlockTraversal(const float3 &origin,
                 const float3 &dir,
                 const float truncation_distance,
                 const float block_size,
                 const bool round_to_nearest = true
  )
      : origin(origin), direction(normalize(dir)), truncation_distance(truncation_distance),
        block_size(block_size),
        round_to_nearest(round_to_nearest),
        step_size(make_int3(dir.x > 0 ? 1 : -1,
                            dir.y > 0 ? 1 : -1,
                            dir.z > 0 ? 1 : -1)),
        tDelta(fabs(block_size / normalize(dir)))
  {
    if (length(direction) == 0)
    {
      printf("ERROR: direction of block traversal must not be 0!\n");
    }

    float3 val = WorldToBlockf(origin);
    float3 inner_block_offset = make_float3((val.x - floor(val.x)),
                                            (val.y - floor(val.y)),
                                            (val.z - floor(val.z)))
                                * block_size;

    // Initialize with distance along ray to first x/y/z block borders
    tMax = fabs(make_float3(
        direction.x > 0 ?
        (block_size - inner_block_offset.x) / direction.x :
        direction.x == 0 ? PINF : inner_block_offset.x / direction.x,

        direction.y > 0 ?
        (block_size - inner_block_offset.y) / direction.y :
        direction.y == 0 ? PINF : inner_block_offset.y / direction.y,

        direction.z > 0 ?
        (block_size - inner_block_offset.z) / direction.z :
        direction.z == 0 ? PINF : inner_block_offset.z / direction.z
    ));
    next_block = WorldToBlocki(origin);
    distance = 0;
  }

  __host__ __device__
  inline float3 WorldToBlockf(const float3 world_pos)
  {
    return world_pos / block_size;
  }

  __host__ __device__
  inline int3 WorldToBlocki(const float3 world_pos)
  {
    const float3 p = WorldToBlockf(world_pos);
    // FIXME: This only finds the nearest responsible voxel, not working for blocks
    if (round_to_nearest)
    {
      return make_int3(p + make_float3(sign(p)) * 0.5f);
    }
    int3 idx = make_int3(p);
    if (p.x < 0) idx.x -= 1;
    if (p.y < 0) idx.y -= 1;
    if (p.z < 0) idx.z -= 1;
    return idx;
  }


  __device__ __host__
  bool HasNextBlock()
  {
    return distance < truncation_distance;
  }

  __device__ __host__
  int3 GetNextBlock()
  {
    int3 current_block = next_block;

    // Distance along the ray to next block
    distance = fminf(fminf(tMax.x, tMax.y), tMax.z);

    if (tMax.x < tMax.y)
    {
      if (tMax.x < tMax.z)
      {
        next_block.x += step_size.x;
        tMax.x += tDelta.x;
      } else
      {
        next_block.z += step_size.z;
        tMax.z += tDelta.z;
      }
    } else
    {
      if (tMax.y < tMax.z)
      {
        next_block.y += step_size.y;
        tMax.y += tDelta.y;
      } else
      {
        next_block.z += step_size.z;
        tMax.z += tDelta.z;
      }
    }
    return current_block;
  }

  const float3 origin;
  const float3 direction;
  const float truncation_distance;
  const float block_size;
  const bool round_to_nearest;

  const int3 step_size;

  /** Distance along the ray to cover one block size in x/y/z direction, respectively */
  const float3 tDelta;

  /** Distance along the ray of the next boundary crossing in x/y/z direction, respectively */
  float3 tMax{};

  /** Traversed distance along the ray */
  float distance;

  /** Next block (integer coordinates) */
  int3 next_block{};
};
