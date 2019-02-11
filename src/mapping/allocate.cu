//
// Created by wei on 17-10-22.
//

#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "mapping/allocate.h"
#include "mapping/block_traversal.hpp"
#include "util/timer.h"

/**
 *
 * @param hash_table
 * @param sensor_data
 * @param sensor_params
 * @param w_T_c
 * @param geometry_helper
 * @param candidate_entries
 * @param allocate_along_normal Determines wheter allocation is done along the view ray or in normal direction
 */
__global__
void AllocBlockArrayKernel(HashTable hash_table,
                           SensorData sensor_data,
                           SensorParams sensor_params,
                           float4x4 w_T_c,
                           GeometryHelper geometry_helper,
                           EntryArray candidate_entries,
                           bool allocate_along_normal)
{

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sensor_params.width || y >= sensor_params.height)
    return;

  /// 1. Get observed data
  float depth = tex2D<float>(sensor_data.depth_texture, x, y);
  if (not IsValidDepth(depth) or depth >= geometry_helper.sdf_upper_bound)
    return;

  float truncation = geometry_helper.truncate_distance(depth);

  /// 2. Set range where blocks are allocated
  float3 world_pos_start;
  float3 world_pos_end;
  float3 world_ray_dir;
  if (allocate_along_normal)
  {
    float4 normal_camera = tex2D<float4>(sensor_data.normal_texture, x, y);
    normal_camera.w = 0;
    if (not IsValidNormal(normal_camera))
      return;

    float3 point_camera_pos = geometry_helper.ImageReprojectToCamera(x, y, depth,
                                                                     sensor_params.fx, sensor_params.fy,
                                                                     sensor_params.cx, sensor_params.cy);
    float3 point_world_pos = w_T_c * point_camera_pos;

    world_ray_dir = make_float3(w_T_c * normal_camera);

    world_pos_start = point_world_pos - world_ray_dir * truncation;
  } else
  {
    float near_depth = fminf(geometry_helper.sdf_upper_bound, depth - truncation);
    float far_depth = fminf(geometry_helper.sdf_upper_bound, depth + truncation);
    if (near_depth >= far_depth) return;

    float3 camera_pos_near = geometry_helper.ImageReprojectToCamera(x, y, near_depth,
                                                                    sensor_params.fx, sensor_params.fy,
                                                                    sensor_params.cx, sensor_params.cy);
    float3 camera_pos_far = geometry_helper.ImageReprojectToCamera(x, y, far_depth,
                                                                   sensor_params.fx, sensor_params.fy,
                                                                   sensor_params.cx, sensor_params.cy);
    world_pos_start = w_T_c * camera_pos_near;
    world_pos_end = w_T_c * camera_pos_far;
    world_ray_dir = normalize(world_pos_end - world_pos_start);
  }

  /// 3. Traverse all blocks in truncation range and allocate VoxelArray, if necessary
  BlockTraversal block_traversal(
      world_pos_start,
      world_ray_dir,
      2 * truncation, // 2 * truncation, because it covers both positive and negative range
      geometry_helper.voxel_size * BLOCK_SIDE_LENGTH,
      false // switch of rounding to nearest corner, allocate CONTAINING block
  );
  while (block_traversal.HasNextBlock())
  {
    int3 block_idx = block_traversal.GetNextBlock();
    hash_table.AllocEntry(block_idx);

    // Flag the corresponding hash entry
    int entry_idx = hash_table.GetEntryIndex(block_idx);
    if (entry_idx >= 0)
    {
      candidate_entries.flag(entry_idx) |= 1;
    }
  }
}

double AllocBlockArray(
    HashTable &hash_table,
    Sensor &sensor,
    RuntimeParams &runtime_params,
    GeometryHelper &geometry_helper,
    EntryArray candidate_entries
)
{
  Timer timer;
  timer.Tick();
  hash_table.ResetMutexes();

  const uint threads_per_block = 8;
  const dim3 grid_size((sensor.sensor_params().width + threads_per_block - 1)
                       / threads_per_block,
                       (sensor.sensor_params().height + threads_per_block - 1)
                       / threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  AllocBlockArrayKernel << < grid_size, block_size >> > (
      hash_table,
          sensor.data(),
          sensor.sensor_params(), sensor.wTc(),
          geometry_helper,
          candidate_entries,
          runtime_params.raycasting_mode == 1
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  return timer.Tock();
}

__global__
void AllocateVoxelArrayKernel(
    EntryArray candidate_entries,
    uint num_entries,
    BlockArray blocks
)
{
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= num_entries)
  {
    return;
  }
  const HashEntry &entry = candidate_entries[idx];

  blocks.AllocateVoxelArrayWithMutex(entry.ptr, 0);
}

__global__
void AllocateVoxelArrayKernelDirectional(
    EntryArray candidate_entries,
    uint num_entries,
    BlockArray blocks,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    float4x4 wTc,
    GeometryHelper geometry_helper
)
{
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= num_entries)
  {
    return;
  }
  const HashEntry &entry = candidate_entries[idx];

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);

  int allocate_directions[6] = {0};
  // For each voxel check which direction normal is pointing -> allocate VoxelArrays accordingly
  for (uint voxel_idx = 0; voxel_idx < BLOCK_SIZE; voxel_idx++)
  {
    int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(voxel_idx));
    float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);
    float3 camera_pos = cTw * world_pos;
    uint2 image_pos = make_uint2(
        geometry_helper.CameraProjectToImagei(camera_pos,
                                              sensor_params.fx, sensor_params.fy,
                                              sensor_params.cx, sensor_params.cy));
    if (image_pos.x >= sensor_params.width or image_pos.y >= sensor_params.height)
      continue;

    float4 normal = tex2D<float4>(sensor_data.normal_texture, image_pos.x, image_pos.y);
    if (not IsValidNormal(normal))
    { // No normal value for this coordinate (NaN or only 0s)
      continue;
    }
    normal.w = 1;

    float4x4 wTcRotOnly = wTc;
    wTcRotOnly.m14 = 0;
    wTcRotOnly.m24 = 0;
    wTcRotOnly.m34 = 0;
    float4 normal_world = wTcRotOnly * normal;

    float weights[N_DIRECTIONS];
    ComputeDirectionWeights(normal_world, weights);

    for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
    {
      allocate_directions[direction] |= (weights[direction] >= direction_weight_threshold);
    }
  }

  for (uint i = 0; i < 6; i++)
  {
    if (allocate_directions[i])
    {
      blocks.AllocateVoxelArrayWithMutex(entry.ptr, i);
    }
  }
}


__global__
void AllocateVoxelArrayRaycastingKernel(
    HashTable hash_table,
    BlockArray blocks,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 wTc,
    GeometryHelper geometry_helper,
    bool allocate_along_normal,
    bool allocate_directional
)
{
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= sensor_params.width || y >= sensor_params.height)
    return;

  /// TODO(wei): change it here
  /// 1. Get observed data
  float depth = tex2D<float>(sensor_data.depth_texture, x, y);
  if (not IsValidDepth(depth) or depth >= geometry_helper.sdf_upper_bound)
    return;

  float truncation = geometry_helper.truncate_distance(depth);

  float4 normal_camera = tex2D<float4>(sensor_data.normal_texture, x, y);
  normal_camera.w = 0;
  float4 normal_world = wTc * normal_camera;
  float3 point_camera_pos = geometry_helper.ImageReprojectToCamera(x, y, depth,
                                                                   sensor_params.fx, sensor_params.fy,
                                                                   sensor_params.cx, sensor_params.cy);

  /// 2. Find traversal parameters
  float3 start_world_pos;
  float3 direction_ray_world;
  if (allocate_along_normal)
  {
    if (not IsValidNormal(normal_camera))
      return;

    float3 point_world_pos = wTc * point_camera_pos;
    direction_ray_world = make_float3(normal_world);
    start_world_pos = point_world_pos - direction_ray_world * truncation;
  } else
  {
    float near_depth = fminf(geometry_helper.sdf_upper_bound, depth - truncation);
    float3 camera_pos_near = geometry_helper.ImageReprojectToCamera(x, y, near_depth,
                                                                    sensor_params.fx, sensor_params.fy,
                                                                    sensor_params.cx, sensor_params.cy);
    start_world_pos = wTc * camera_pos_near;
    direction_ray_world = normalize(point_camera_pos - start_world_pos);
  }

  /// 3. Traverse all blocks in truncation range and allocate VoxelArray, if necessary
  float direction_weights[N_DIRECTIONS];
  ComputeDirectionWeights(normal_world, direction_weights);

  BlockTraversal block_traversal(
      start_world_pos,
      direction_ray_world,
      2 * truncation, // 2 * truncation, because it covers both positive and negative range
      geometry_helper.voxel_size * BLOCK_SIDE_LENGTH);
  while (block_traversal.HasNextBlock())
  {
    int3 block_idx = block_traversal.GetNextBlock();
    HashEntry entry = hash_table.GetEntry(block_idx);

    if (entry.ptr < 0)
    {
      // An uninitialized block might occur due to a hash-collision during allocation:
      // When two block ids are supposed to be initialized within the same cycle and
      // both are hashed to the same bucket, only one will be initialized (per-bucket mutex!)
      continue;
    }

    if (allocate_directional)
    {
      for (uint i = 0; i < N_DIRECTIONS; i++)
      {
        if (direction_weights[i] > direction_weight_threshold)
        {
          blocks.AllocateVoxelArrayWithMutex(entry.ptr, i);
        }
      }
    } else
    {
      blocks.AllocateVoxelArrayWithMutex(entry.ptr, 0);
    }
  }
}
