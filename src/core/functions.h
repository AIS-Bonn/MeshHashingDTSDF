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

