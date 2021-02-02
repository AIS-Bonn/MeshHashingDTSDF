#pragma once

#include <helper_math.h>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector_types.h>
#include "common.h"


__device__
const static float direction_weight_threshold = 0.3826834323650898f; // approx of sin(pi/8)
const static float direction_angle_threshold = 0.7 * M_PI_2;

enum class TSDFDirection : std::uint8_t
{
  UP = 0,   // Y_POS
  DOWN,     // Y_NEG
  LEFT,     // X_POS
  RIGHT,    // X_NEG
  FORWARD,  // Z_NEG
  BACKWARD  // Z_POS
};

__device__
const static float3 TSDFDirectionVectors[N_DIRECTIONS] = {
    {0,  1,  0},  // Y_POS
    {0,  -1, 0},  // Y_NEG
    {1,  0,  0},  // X_POS
    {-1, 0,  0},  // X_NEG
    {0,  0,  -1}, // Z_NEG
    {0,  0,  1},  // Z_POS
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
__device__
void ComputeDirectionWeights(const float3 &normal, float weights[N_DIRECTIONS]);

__device__
float DirectionWeight(float angle);

__device__
float DirectionAngle(const float3& normal, size_t direction);

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
