#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <vector_types.h>

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
const char *TSDFDirectionToString(TSDFDirection direction)
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

__host__ __device__
TSDFDirection VectorToTSDFDirection(const float4 &vector)
{
  // Tait-Bryan angles
  float heading = std::atan2(vector.x, vector.z);
  float elevation = std::sin(vector.y);
  if (elevation > M_PI_4)
  {
    return TSDFDirection::UP;
  } else if (elevation < -M_PI_4)
  {
    return TSDFDirection::DOWN;
  } else if ((heading >= 0 and heading < M_PI_4) or (heading >= 7 * M_PI_4 and heading <= 2 * M_PI))
  {
    return TSDFDirection::RIGHT;
  } else if (heading >= M_PI_4 and heading < 3 * M_PI_4)
  {
    return TSDFDirection::FORWARD;
  } else if (heading >= 3 * M_PI_4 and heading < 5 * M_PI_4)
  {
    return TSDFDirection::LEFT;
  } else if (heading >= 5 * M_PI_4 and heading < 7 * M_PI_4)
  {
    return TSDFDirection::BACKWARD;
  }

}
