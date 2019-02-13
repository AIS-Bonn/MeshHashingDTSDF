#include <device_launch_parameters.h>
#include <util/timer.h>

#include "core/block_array.h"
#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "mapping/allocate.h"
#include "mapping/update_simple.h"
#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/spatial_query.h"
#include "util/debugging.hpp"

////////////////////
/// Device code
////////////////////

__global__
void UpdateBlocksSimpleKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    bool enable_point_to_plane,
    GeometryHelper geometry_helper
)
{
  const size_t voxel_array_idx = 0;

  //TODO check if we should load this in shared memory (candidate_entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

  Voxel &this_voxel = blocks.GetVoxelArray(entry.ptr, voxel_array_idx).voxels[local_idx];
  /// 2. Project to camera
  float3 voxel_world_pos = geometry_helper.VoxelToWorld(voxel_pos);
  float3 voxel_camera_pos = cTw * voxel_world_pos;
  uint2 image_pos = make_uint2(
      geometry_helper.CameraProjectToImagei(voxel_camera_pos,
                                            sensor_params.fx, sensor_params.fy,
                                            sensor_params.cx, sensor_params.cy));
  if (image_pos.x >= sensor_params.width
      || image_pos.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (not IsValidDepth(depth) or depth >= geometry_helper.sdf_upper_bound)
    return;
  float sdf;
  if (enable_point_to_plane)
  { // Use point-to-plane metric (Bylow2013 "Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions")
    float3 normal = make_float3(tex2D<float4>(sensor_data.normal_texture, image_pos.x, image_pos.y));
    if (not IsValidNormal(normal))
    { // No normal value for this coordinate
      return;
    }

    float3 surface_point = GeometryHelper::ImageReprojectToCamera(image_pos.x, image_pos.y, depth,
                                                                  sensor_params.fx, sensor_params.fy,
                                                                  sensor_params.cx, sensor_params.cy);
    sdf = dot(surface_point - voxel_camera_pos, -normal);
  } else
  { // Use point-to-point metric
    sdf = depth - voxel_camera_pos.z;
  }
  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float inv_sigma2 = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth),
                           1.0f);
  float truncation = geometry_helper.truncate_distance(depth);
  if (depth - voxel_camera_pos.z <= -truncation)
    return;
  if (sdf >= 0.0f)
  {
    sdf = fminf(truncation, sdf);
  } else
  {
    sdf = fmaxf(-truncation, sdf);
  }

  /// 5. Update
  Voxel delta;
  delta.sdf = sdf;
  delta.inv_sigma2 = inv_sigma2;

  if (sensor_data.color_data)
  {
    float4 color = tex2D<float4>(sensor_data.color_texture, image_pos.x, image_pos.y);
    delta.color = make_uchar3(255 * color.x, 255 * color.y, 255 * color.z);
  } else
  {
    delta.color = make_uchar3(0, 255, 0);
  }
  this_voxel.Update(delta);
}


__global__
void UpdateBlocksSimpleKernelDirectional(
    EntryArray candidate_entries,
    BlockArray blocks,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    float4x4 wTc,
    bool enable_point_to_plane,
    GeometryHelper geometry_helper
)
{

  //TODO check if we should load this in shared memory (entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

  /// 2. Project to camera
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);
  float3 camera_pos = cTw * world_pos;
  uint2 image_pos = make_uint2(
      geometry_helper.CameraProjectToImagei(camera_pos,
                                            sensor_params.fx, sensor_params.fy,
                                            sensor_params.cx, sensor_params.cy));
  if (image_pos.x >= sensor_params.width
      || image_pos.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float4 normal = tex2D<float4>(sensor_data.normal_texture, image_pos.x, image_pos.y);
  normal.w = 0;
  if (not IsValidNormal(normal))
  { // No normal value for this coordinate
    return;
  }

  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (not IsValidDepth(depth) or depth >= geometry_helper.sdf_upper_bound)
    return;
  float sdf;
  if (enable_point_to_plane)
  { // Use point-to-plane metric (Bylow2013 "Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions")
    float3 normal_ = make_float3(normal);

    float3 surface_point = GeometryHelper::ImageReprojectToCamera(image_pos.x, image_pos.y, depth,
                                                                  sensor_params.fx, sensor_params.fy,
                                                                  sensor_params.cx, sensor_params.cy);
    sdf = dot(surface_point - camera_pos, -normal_);
  } else
  { // Use point-to-point metric
    sdf = depth - camera_pos.z;
  }
  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float inv_sigma2 = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth),
                           1.0f);
  float truncation = geometry_helper.truncate_distance(depth);
  if (depth - camera_pos.z <= -truncation)
    return;
  if (sdf >= 0.0f)
  {
    sdf = fminf(truncation, sdf);
  } else
  {
    sdf = fmaxf(-truncation, sdf);
  }

  /// 4. Find TSDF direction and Update
  float4 normal_world = wTc * normal;

  float weights[N_DIRECTIONS];
  ComputeDirectionWeights(normal_world, weights);
  for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
  {
    if (weights[direction] < direction_weight_threshold)
      continue;

    Voxel &voxel = blocks.GetVoxelArray(entry.ptr, direction).voxels[local_idx];
    Voxel delta;
    delta.sdf = sdf;
    delta.inv_sigma2 = inv_sigma2 * weights[direction]; // additionally weight by normal-direction-compliance

    if (sensor_data.color_data)
    {
      float4 color = tex2D<float4>(sensor_data.color_texture, image_pos.x, image_pos.y);
      delta.color = make_uchar3(255 * color.x, 255 * color.y, 255 * color.z);
    } else
    {
      delta.color = make_uchar3(0, 255, 0);
    }
    voxel.Update(delta);
  }
}

double UpdateBlocksSimple(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Sensor &sensor,
    const RuntimeParams &runtime_params,
    HashTable &hash_table,
    GeometryHelper &geometry_helper
)
{

  Timer timer;
  timer.Tick();

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return timer.Tock();

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);

  if (runtime_params.enable_directional_sdf)
  {
    UpdateBlocksSimpleKernelDirectional << < grid_size, block_size >> > (
        candidate_entries,
            blocks,
            sensor.data(),
            sensor.sensor_params(),
            sensor.cTw(),
            sensor.wTc(),
            runtime_params.enable_point_to_plane,
            geometry_helper);

  } else
  {
    UpdateBlocksSimpleKernel << < grid_size, block_size >> > (
        candidate_entries,
            blocks,
            sensor.data(),
            sensor.sensor_params(),
            sensor.cTw(),
            runtime_params.enable_point_to_plane,
            geometry_helper);

  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  return timer.Tock();
}
