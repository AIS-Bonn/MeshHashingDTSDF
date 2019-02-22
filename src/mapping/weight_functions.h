#pragma once

/**
 * Compute weight based on the measurement depth (higher depth -> more error -> lower weight)
 * @param normalized_depth
 * @return
 */
__device__
inline float weight_depth(const float normalized_depth)
{
  return 1.0f - normalized_depth;
//  float w_depth = sensor_params.min_depth_range / normalized_depth;
//  w_depth = w_depth * w_depth;
//  w_depth *= (0.0012 + 0.0019 * (sensor_params.min_depth_range - 0.4) * (sensor_params.min_depth_range - 0.4));
//  w_depth /= (0.0012 + 0.0019 * (normalized_depth - 0.4) * (normalized_depth - 0.4));
}


/**
 * Compute weight for correlation of voxel to the measured surface point
 * (Further away voxels are likely to be less correlated)
 * @param surface_point_world
 * @param voxel_pos_world
 * @param truncation_distance
 * @return
 */
__device__
inline float weight_voxel_correlation(const float3 surface_point_world,
                                      const float3 voxel_pos_world,
                                      const float truncation_distance)
{
  return length(surface_point_world - voxel_pos_world) / truncation_distance;
}

/**
 * Compute weight based on surface normal in the camera frame.
 * (Steep angle -> more error -> lower weight)
 * @param normal_camera
 * @return
 */
__device__
inline float weight_normal_angle(const float3 normal_camera)
{
  return 2 - normal_camera.x + normal_camera.y;
}

/**
 * Compute weight based on fusion direction and surface normal in world frame.
 * @param direction
 * @param normal_world
 * @return
 */
__device__
inline float weight_direction_compliance(size_t direction, const float3 normal_world)
{
  return dot(normal_world, TSDFDirectionVectors[direction]);
}
