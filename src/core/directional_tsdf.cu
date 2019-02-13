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
      {0, 4, 1, 5, 2,  6,  3,  7},
      {4, 0, 5, 1, 6,  2,  7,  3},
      {1, 3, 5, 7, 9,  8,  10, 11},
      {3, 1, 7, 5, 8,  9,  11, 10},
      {2, 0, 6, 4, 10, 9,  11, 8},
      {0, 2, 4, 6, 8,  11, 9,  10}
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
  }
}

void ComputeDirectionWeights(const float3 &normal, float *weights)
{
  const static float3 normal_directions[N_DIRECTIONS] = {
      {0,  1,  0},  // Up
      {0,  -1, 0}, // Down
      {1,  0,  0},  // Left
      {-1, 0,  0},  // Right
      {0,  0,  -1},  // Forward
      {0,  0,  1},  // Backward
  };
  for (size_t i = 0; i < 3; i++)
  {
    weights[2 * i] = dot(normal, normal_directions[2 * i]);
    weights[2 * i + 1] = -weights[2 * i]; // opposite direction -> negative value
  }
}
