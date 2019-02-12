#pragma once

#include <helper_math.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector_types.h>
#include "common.h"


__device__
const static float direction_weight_threshold = 0.3826834323650898f; // approx of sin(pi/8)

enum class TSDFDirection : std::uint8_t
{
  UP = 0,
  DOWN,
  LEFT,
  RIGHT,
  FORWARD,
  BACKWARD
};


__host__ __device__
const char *TSDFDirectionToString(TSDFDirection direction);

/**
 * Computes the compliance of the given normal with each directions.
 *
 * The weight is computed by a dot product, so each weight is in [-1, 1]. Negative values mean non-compliance
 * and a value of 1 is full compliance.
 * @param normal
 * @param weights
 */
__host__ __device__
void ComputeDirectionWeights(const float4 &normal, float weights[N_DIRECTIONS]);

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
