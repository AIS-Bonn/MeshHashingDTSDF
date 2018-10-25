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


__constant__ static TSDFDirection DirectionalNeighbors[6][4] = {
    {TSDFDirection::LEFT, TSDFDirection::RIGHT, TSDFDirection::FORWARD, TSDFDirection::BACKWARD},
    {TSDFDirection::LEFT, TSDFDirection::RIGHT, TSDFDirection::FORWARD, TSDFDirection::BACKWARD},
    {TSDFDirection::UP, TSDFDirection::DOWN, TSDFDirection::FORWARD, TSDFDirection::BACKWARD},
    {TSDFDirection::UP, TSDFDirection::DOWN, TSDFDirection::FORWARD, TSDFDirection::BACKWARD},
    {TSDFDirection::UP, TSDFDirection::DOWN, TSDFDirection::LEFT, TSDFDirection::RIGHT},
    {TSDFDirection::UP, TSDFDirection::DOWN, TSDFDirection::LEFT, TSDFDirection::RIGHT},
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

__host__ __device__
inline TSDFDirection VectorToTSDFDirection(const float4 &vector)
{
  // Tait-Bryan angles
  float heading = std::atan2(vector.z, vector.x);
  float elevation = std::sin(vector.y);
  if (heading < 0)
    heading += 2 * M_PI;
  if (elevation < - M_PI_4)
  {
    return TSDFDirection::DOWN;
  } else if (elevation > M_PI_4)
  {
    return TSDFDirection::UP;
  } else if ((heading >= 0 and heading < M_PI_4) or (heading >= 7 * M_PI_4 and heading <= 2 * M_PI))
  {
    return TSDFDirection::LEFT;
  } else if (heading >= M_PI_4 and heading < 3 * M_PI_4)
  {
    return TSDFDirection::BACKWARD;
  } else if (heading >= 3 * M_PI_4 and heading < 5 * M_PI_4)
  {
    return TSDFDirection::RIGHT;
  } else if (heading >= 5 * M_PI_4 and heading < 7 * M_PI_4)
  {
    return TSDFDirection::FORWARD;
  }

}
