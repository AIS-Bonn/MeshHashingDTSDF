#pragma once

#include <helper_math.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector_types.h>
#include "common.h"

enum class TSDFDirection : std::uint8_t
{
  UP = 0,
  DOWN,
  LEFT,
  RIGHT,
  FORWARD,
  BACKWARD
};


__constant__ static TSDFDirection TSDFDirectionNeighbors[6][4] = {
    {TSDFDirection::LEFT, TSDFDirection::RIGHT, TSDFDirection::FORWARD, TSDFDirection::BACKWARD},
    {TSDFDirection::LEFT, TSDFDirection::RIGHT, TSDFDirection::FORWARD, TSDFDirection::BACKWARD},
    {TSDFDirection::UP,   TSDFDirection::DOWN,  TSDFDirection::FORWARD, TSDFDirection::BACKWARD},
    {TSDFDirection::UP,   TSDFDirection::DOWN,  TSDFDirection::FORWARD, TSDFDirection::BACKWARD},
    {TSDFDirection::UP,   TSDFDirection::DOWN,  TSDFDirection::LEFT,    TSDFDirection::RIGHT},
    {TSDFDirection::UP,   TSDFDirection::DOWN,  TSDFDirection::LEFT,    TSDFDirection::RIGHT},
};


__host__ __device__
inline const char *TSDFDirectionToString(const TSDFDirection direction)
{
  switch (direction)
  {
    case TSDFDirection::UP:
      return "UP";
    case TSDFDirection::DOWN:
      return "DOWN";
    case TSDFDirection::RIGHT:
      return "RIGHT";
    case TSDFDirection::FORWARD:
      return "FORWARD";
    case TSDFDirection::LEFT:
      return "LEFT";
    case TSDFDirection::BACKWARD:
      return "BACKWARD";
  }
}

/**
 * Computes the compliance of the given normal with each directions.
 *
 * The weight is computed by a dot product, so each weight is in [-1, 1]. Negative values mean non-compliance
 * and a value of 1 is full compliance.
 * @param normal
 * @param weights
 */
__host__ __device__
inline void ComputeDirectionWeights(const float4 &normal, float weights[N_DIRECTIONS])
{
  const static float3 normal_directions[N_DIRECTIONS] = {
      {0,  1, 0},  // Up
      {0,  -1,  0}, // Down
      {1,  0,  0},  // Left
      {-1, 0,  0},  // Right
      {0,  0,  -1},  // Forward
      {0,  0,  1},  // Backward
  };
  float3 vector_ = make_float3(normal);
  for (size_t i = 0; i < 3; i++)
  {
    weights[2 * i] = dot(vector_, normal_directions[2 * i]);
    weights[2 * i + 1] = -weights[2 * i]; // opposite direction -> negative value
  }
}

__host__ __device__
inline TSDFDirection VectorToTSDFDirection(const float4 &vector)
{
  float weights[N_DIRECTIONS];
  ComputeDirectionWeights(vector, weights);
  float max_weight = -2;
  int max_direction = -1;
  for (int i = 0; i < N_DIRECTIONS; i++)
  {
    if (weights[i] > max_weight)
    {
      max_weight = weights[i];
      max_direction = i;
    }
  }
  if (max_direction >= 0)
    return TSDFDirection(max_direction);
  return TSDFDirection::FORWARD;
}

__device__
short FilterMCIndexDirection(const short mc_index, const TSDFDirection direction, const float sdf[8]);

/**
 * Check, whether the given MC index is compatible to the direction.
 *
 * @param mc_index MC index
 * @param direction Direction
 * @param sdf SDF values of voxel corners
 * @return
 */
__device__
bool IsMCIndexDirectionCompatible(const short mc_index, const TSDFDirection direction, const float sdf[8]);
