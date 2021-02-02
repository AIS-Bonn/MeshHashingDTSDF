#include <device_launch_parameters.h>
#include <util/timer.h>

#include "core/block_array.h"
#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "engine/main_engine.h"
#include "mapping/allocate.h"
#include "mapping/update_simple.h"
#include "mapping/weight_functions.h"
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
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    float4x4 wTc,
    BlockArray blocks,
    GeometryHelper geometry_helper,
    bool enable_point_to_plane
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
  float3 voxel_pos_world = geometry_helper.VoxelToWorld(voxel_pos);
  float3 voxel_pos_camera = cTw * voxel_pos_world;
  uint2 image_pos = make_uint2(
      geometry_helper.CameraProjectToImagei(voxel_pos_camera,
                                            sensor_params.fx, sensor_params.fy,
                                            sensor_params.cx, sensor_params.cy));
  if (image_pos.x >= sensor_params.width
      || image_pos.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  float3 normal_camera = make_float3(tex2D<float4>(sensor_data.normal_texture, image_pos.x, image_pos.y));
  if (not IsValidDepth(depth) or depth >= geometry_helper.sdf_upper_bound or not IsValidNormal(normal_camera))
    return;
  float3 surface_point_camera = GeometryHelper::ImageReprojectToCamera(image_pos.x, image_pos.y, depth,
                                                                       sensor_params.fx, sensor_params.fy,
                                                                       sensor_params.cx, sensor_params.cy);
  float sdf;
  if (enable_point_to_plane)
  { // Use point-to-plane metric (Bylow2013 "Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions")

    sdf = dot(surface_point_camera - voxel_pos_camera, -normal_camera);
  } else
  { // Use point-to-point metric
    sdf = depth - voxel_pos_camera.z;
  }
  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float truncation_distance = geometry_helper.truncate_distance(depth);
  float3 surface_point_world = make_float3(wTc * make_float4(surface_point_camera, 1));

//  float weight = fmaxf(1e8 * powf(geometry_helper.voxel_size, 3) *
  float weight = fmaxf(20 * 1.4732e+05f * powf(geometry_helper.voxel_size, 1.7013f) *
                       geometry_helper.weight_sample *
                       weight_depth(normalized_depth) *
//                       weight_voxel_correlation(surface_point_world, voxel_pos_world, truncation_distance) *
                       weight_normal_angle(normal_camera),
                       1.0f);
  if (depth - voxel_pos_camera.z <= -truncation_distance)
    return;
  if (sdf >= 0.0f)
  {
    sdf = fminf(truncation_distance, sdf);
  } else
  {
    sdf = fmaxf(-truncation_distance, sdf);
  }

  /// 5. Update
  Voxel delta;
  delta.sdf = sdf;
  delta.inv_sigma2 = weight;

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
void UpdateBlocksSimpleDirectionalKernel(
    EntryArray candidate_entries,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    float4x4 wTc,
    BlockArray blocks,
    GeometryHelper geometry_helper,
    bool enable_point_to_plane
)
{

  //TODO check if we should load this in shared memory (entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

  /// 2. Project to camera
  float3 voxel_pos_world = geometry_helper.VoxelToWorld(voxel_pos);
  float3 voxel_pos_camera = cTw * voxel_pos_world;
  uint2 voxel_pos_image = make_uint2(
      geometry_helper.CameraProjectToImagei(voxel_pos_camera,
                                            sensor_params.fx, sensor_params.fy,
                                            sensor_params.cx, sensor_params.cy));
  if (voxel_pos_image.x >= sensor_params.width
      || voxel_pos_image.y >= sensor_params.height)
    return;

  /// 3. Find correspondent depth observation
  float4 normal_camera = tex2D<float4>(sensor_data.normal_texture, voxel_pos_image.x, voxel_pos_image.y);
  normal_camera.w = 0;
  if (not IsValidNormal(normal_camera))
  { // No normal value for this coordinate
    return;
  }

  float depth = tex2D<float>(sensor_data.depth_texture, voxel_pos_image.x, voxel_pos_image.y);
  if (not IsValidDepth(depth) or depth >= geometry_helper.sdf_upper_bound)
    return;
  float3 surface_point_camera = GeometryHelper::ImageReprojectToCamera(voxel_pos_image.x, voxel_pos_image.y, depth,
                                                                       sensor_params.fx, sensor_params.fy,
                                                                       sensor_params.cx, sensor_params.cy);
  float sdf;
  if (enable_point_to_plane)
  { // Use point-to-plane metric (Bylow2013 "Real-Time Camera Tracking and 3D Reconstruction Using Signed Distance Functions")
    float3 normal_ = make_float3(normal_camera);
    sdf = dot(surface_point_camera - voxel_pos_camera, -normal_);
  } else
  { // Use point-to-point metric
    sdf = depth - voxel_pos_camera.z;
  }
  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float truncation_distance = geometry_helper.truncate_distance(depth);
  float3 surface_point_world = make_float3(wTc * make_float4(surface_point_camera, 1));
  float weight = fmaxf(20 * 1.4732e+05f * powf(geometry_helper.voxel_size, 1.7013f) *
                       geometry_helper.weight_sample *
                       weight_depth(normalized_depth) *
//                       weight_voxel_correlation(surface_point_world, voxel_pos_world, truncation_distance) *
                       weight_normal_angle(make_float3(normal_camera)),
                       1.0f);
  if (depth - voxel_pos_camera.z <= -truncation_distance)
    return;
  if (sdf >= 0.0f)
  {
    sdf = fminf(truncation_distance, sdf);
  } else
  {
    sdf = fmaxf(-truncation_distance, sdf);
  }

  /// 4. Find TSDF direction and Update
  float3 normal_world = make_float3(wTc * normal_camera);

  float weights[N_DIRECTIONS];
  ComputeDirectionWeights(normal_world, weights);
  for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
  {
    if (weights[direction] <= 0)
      continue;

    Voxel &voxel = blocks.GetVoxelArray(entry.ptr, direction).voxels[local_idx];
    Voxel delta;
    delta.sdf = sdf;
    delta.inv_sigma2 = weight * weight_direction_compliance(direction, normal_world);

    if (sensor_data.color_data)
    {
      float4 color = tex2D<float4>(sensor_data.color_texture, voxel_pos_image.x, voxel_pos_image.y);
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
    Sensor &sensor,
    MainEngine &main_engine
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

  if (main_engine.runtime_params().enable_directional_sdf)
  {
    UpdateBlocksSimpleDirectionalKernel << < grid_size, block_size >> > (
        candidate_entries,
            sensor.data(),
            sensor.sensor_params(),
            sensor.cTw(),
            sensor.wTc(),
            main_engine.blocks(),
            main_engine.geometry_helper(),
            main_engine.runtime_params().enable_point_to_plane);

  } else
  {
    UpdateBlocksSimpleKernel << < grid_size, block_size >> > (
        candidate_entries,
            sensor.data(),
            sensor.sensor_params(),
            sensor.cTw(),
            sensor.wTc(),
            main_engine.blocks(),
            main_engine.geometry_helper(),
            main_engine.runtime_params().enable_point_to_plane);

  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  return timer.Tock();
}
