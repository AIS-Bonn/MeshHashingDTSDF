#pragma once

#include <device_launch_parameters.h>
#include <extern/cuda/matrix.h>

__host__ __device__
inline bool IsValidNormal(const float3 &normal)
{
  return normal.x == normal.x and (normal.x != 0 or normal.y != 0 or normal.z != 0);
}

__host__ __device__
inline bool IsValidNormal(const float4 &normal)
{
  return normal.x == normal.x and (normal.x != 0 or normal.y != 0 or normal.z != 0);
}

__device__
inline bool IsValidDepth(const float depth)
{
  return depth != MINF and depth != 0.0f;
}


/**
 * Interpolate the surface offset between two voxel vertices given their SDF values.
 * The offset denotes the distance from v1 to the iso surface in [0, 1] (-> voxel side length)
 * @param v1 SDF value of corner 1
 * @param v2 SDF value of corner 2
 * @param isolevel Surface iso level
 * @return Vertex offset between two voxel vertices
 */
__device__
inline float InterpolateSurfaceOffset(const float &v1, const float &v2,
                                             const float &isolevel)
{
  if (fabs(v1 - isolevel) < 0.008) return 0;
  if (fabs(v2 - isolevel) < 0.008) return 1;
  return (isolevel - v1) / (v2 - v1);
}

/**
 * Interpolate vertex position on voxel edge for the given SDF values.
 * @param p1 Voxel corner 1
 * @param p2 Voxel corner 2
 * @param v1 SDF value of corner 1
 * @param v2 SDF value of corner 2
 * @param isolevel Surface iso level
 * @return Vertex position on voxel edge
 */
__device__
inline float3 VertexIntersection(const float3 &p1, const float3 p2,
                                 const float &v1, const float &v2,
                                 const float &isolevel)
{
  float mu = InterpolateSurfaceOffset(v1, v2, isolevel);

  float3 p = make_float3(p1.x + mu * (p2.x - p1.x),
                         p1.y + mu * (p2.y - p1.y),
                         p1.z + mu * (p2.z - p1.z));
  return p;
}
