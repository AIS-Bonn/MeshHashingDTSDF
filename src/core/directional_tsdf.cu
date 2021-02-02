#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "meshing/mc_tables.h"

__device__
short FilterMCIndexDirection(const short mc_index, const TSDFDirection direction, const float sdf[8])
{
  if (mc_index <= 0 or mc_index == 255)
    return mc_index;

  short new_index = 0;
  for (int component = 0; component < 4 and kIndexDecomposition[mc_index][component] != -1; component++)
  {
    const short part_idx = kIndexDecomposition[mc_index][component];
    if (not IsMCIndexDirectionCompatible(part_idx, direction, sdf))
      continue;
    new_index |= part_idx;
  }

  if (new_index == 0)
  { // If 0 after filtering -> invalidate, so it doesn't affect other directions during later filtering process
    new_index = -1;
  }
  return new_index;
}


__device__
bool IsMCIndexDirectionCompatible(const short mc_index, const TSDFDirection direction, const float sdf[8])
{
  // Table containing for each direction:
  // 4 opposite edge pairs, each of which is checked individually.
  const static size_t view_direction_edges_to_check[6][8] = {
      {0, 4, 1, 5, 2,  6,  3,  7},  // Y_POS
      {4, 0, 5, 1, 6,  2,  7,  3},  // Y_NEG
      {1, 3, 5, 7, 9,  8,  10, 11}, // X_POS
      {3, 1, 7, 5, 8,  9,  11, 10}, // X_NEG
      {2, 0, 6, 4, 10, 9,  11, 8},  // Z_NEG
      {0, 2, 4, 6, 8,  11, 9,  10}  // Z_POS
  };
  if (kIndexDirectionCompatibility[mc_index][static_cast<size_t>(direction)] == 0)
    return false;
  if (kIndexDirectionCompatibility[mc_index][static_cast<size_t>(direction)] == 2)
  {
    for (int e = 0; e < 4; e++)
    {
      const size_t edge_idx = view_direction_edges_to_check[static_cast<size_t>(direction)][2 * e];
      const size_t opposite_edge_idx = view_direction_edges_to_check[static_cast<size_t>(direction)][2 * e + 1];
      int2 edge = kEdgeEndpointVertices[edge_idx];
      int2 opposite_edge = kEdgeEndpointVertices[opposite_edge_idx];

      int2 endpoint_values;
      endpoint_values.x = (mc_index & (1 << edge.x)) > 0;
      endpoint_values.y = (mc_index & (1 << edge.y)) > 0;

      // If edge has NO zero-crossing -> continue
      if (endpoint_values.x + endpoint_values.y != 1)
        continue;

      // Swap vertex indices, s.t. first endpoint is behind the surface
      if (endpoint_values.y == 1)
      {
        int tmp;
        tmp = edge.x;
        edge.x = edge.y;
        edge.y = tmp;
        tmp = opposite_edge.x;
        opposite_edge.x = opposite_edge.y;
        opposite_edge.y = tmp;
      }

      float offset = InterpolateSurfaceOffset(sdf[edge.x], sdf[edge.y], 0);
      float opposite_offset = InterpolateSurfaceOffset(sdf[opposite_edge.x], sdf[opposite_edge.y], 0);

      // If interpolated surface more than 90 degrees from view direction vector -> discard
      if (offset > opposite_offset)
      {
        return false;
      }
//      if (fabs(opposite_offset - offset) < 0.5)
//      {
//        return false;
//      }
    }
  }
  return true;
}

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
    default:
      return "ERROR/UNKNOWN";
  }
}

__device__
float DirectionAngle(const float3& normal, size_t direction)
{
  float angleCos = dot(normal, TSDFDirectionVectors[direction]);
  angleCos = fmaxf(fminf(angleCos, 1), -1);
  return acos(angleCos);
}

__device__
float DirectionWeight(float angle)
{
  float width = direction_angle_threshold;

  if (width <= M_PI_4 + 1e-6)
  {
    return 1 - fminf(angle / width, 1);
  }

  width /= M_PI_2;
  angle /= M_PI_2;
  return 1 - fminf((fmaxf(angle, 1 - width) - (1 - width)) / (2 * width - 1), 1);
}

__device__
void ComputeDirectionWeights(const float3 &normal, float *weights)
{
  for (size_t i = 0; i < 3; i++)
  {
    float angle = DirectionAngle(normal, 2 * i);
    weights[2 * i] = DirectionWeight(angle);
    weights[2 * i + 1] = DirectionWeight(M_PI - angle); // opposite direction -> negative value
  }
}
