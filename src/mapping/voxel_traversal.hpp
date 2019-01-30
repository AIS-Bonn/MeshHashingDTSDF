#pragma once

#include "geometry/geometry_helper.h"
#include <host_defines.h>
#include <vector_types.h>

struct VoxelTraversal
{
  __device__ __host__
  VoxelTraversal(const float3 &origin,
                 const float3 &dir,
                 const float truncation_distance,
                 GeometryHelper &geometry_helper
  )
      : origin(origin), direction(normalize(dir)), truncation_distance(truncation_distance),
        step_size(make_int3(dir.x > 0 ? 1 : -1,
                            dir.y > 0 ? 1 : -1,
                            dir.z > 0 ? 1 : -1)),
        tDelta(fabs(geometry_helper.voxel_size / normalize(dir)))
  {
    if (length(direction) == 0)
    {
      printf("ERROR: direction of voxel traversal must not be 0!\n");
    }

    float3 val = geometry_helper.WorldToVoxelf(origin);
    float3 inner_voxel_offset = make_float3((val.x - floor(val.x)),
                                            (val.y - floor(val.y)),
                                            (val.z - floor(val.z)))
                                * geometry_helper.voxel_size;

    // Initialize with distance along ray to first x/y/z voxel borders
    tMax = fabs(make_float3(
        direction.x > 0 ?
        (geometry_helper.voxel_size - inner_voxel_offset.x) / direction.x :
        direction.x == 0 ? PINF : inner_voxel_offset.x / direction.x,

        direction.y > 0 ?
        (geometry_helper.voxel_size - inner_voxel_offset.y) / direction.y :
        direction.y == 0 ? PINF : inner_voxel_offset.y / direction.y,

        direction.z > 0 ?
        (geometry_helper.voxel_size - inner_voxel_offset.z) / direction.z :
        direction.z == 0 ? PINF : inner_voxel_offset.z / direction.z
    ));
    next_voxel = geometry_helper.WorldToVoxeli(origin);
    distance = 0;
  }

  __device__ __host__
  bool HasNextVoxel()
  {
    return distance < truncation_distance;
  }

  __device__ __host__
  int3 GetNextVoxel()
  {
    int3 current_voxel = next_voxel;

    // Distance along the ray to next voxel
    distance = fminf(fminf(tMax.x, tMax.y), tMax.z);

    if (tMax.x < tMax.y)
    {
      if (tMax.x < tMax.z)
      {
        next_voxel.x += step_size.x;
        tMax.x += tDelta.x;
      } else
      {
        next_voxel.z += step_size.z;
        tMax.z += tDelta.z;
      }
    } else
    {
      if (tMax.y < tMax.z)
      {
        next_voxel.y += step_size.y;
        tMax.y += tDelta.y;
      } else
      {
        next_voxel.z += step_size.z;
        tMax.z += tDelta.z;
      }
    }
    return current_voxel;
  }

  /** Ray origin */
  const float3 origin;
  /** Ray direction */
  const float3 direction;
  /** Truncation distance */
  const float truncation_distance;

  const int3 step_size;

  /** Distance along the ray to cover one voxel size in x/y/z direction, respectively */
  const float3 tDelta;

  /** Distance along the ray of the next boundary crossing in x/y/z direction, respectively */
  float3 tMax;

  /** Traversed distance along the ray */
  float distance;

  /** Next voxel (integer coordinates) */
  int3 next_voxel;
};
