//
// Created by wei on 17-10-22.
//

#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "mapping/allocate.h"
#include "mapping/block_traversal.hpp"
#include "util/timer.h"
#include "allocate.h"


/**
 * @param hash_table
 * @param sensor_data
 * @param sensor_params
 * @param wTc
 * @param geometry_helper
 * @param candidate_entries
 * @param allocate_along_normal Determines whether allocation is done along the view ray or in normal direction
 */
__global__
void AllocBlockArrayKernel(HashTable hash_table,
                           SensorData sensor_data,
                           SensorParams sensor_params,
                           float4x4 wTc,
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
  float4 normal_camera = tex2D<float4>(sensor_data.normal_texture, x, y);
  normal_camera.w = 0;

  if (not IsValidDepth(depth) or depth >= geometry_helper.sdf_upper_bound or not IsValidNormal(normal_camera))
    return;

  float truncation = geometry_helper.truncate_distance(depth);

  /// 2. Set range where blocks are allocated
  float3 world_pos_start;
  float3 world_pos_end;
  float3 world_ray_dir;
  float3 normal_world = make_float3(wTc * normal_camera);
  if (allocate_along_normal)
  {

    float3 point_camera_pos = geometry_helper.ImageReprojectToCamera(x, y, depth,
                                                                     sensor_params.fx, sensor_params.fy,
                                                                     sensor_params.cx, sensor_params.cy);
    float3 point_world_pos = wTc * point_camera_pos;

    world_ray_dir = normal_world;

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
    world_pos_start = wTc * camera_pos_near;
    world_pos_end = wTc * camera_pos_far;
    world_ray_dir = normalize(world_pos_end - world_pos_start);
  }

  /// 3. Traverse all blocks in truncation range and allocate VoxelArray, if necessary

  float directional_weights[N_DIRECTIONS];
  ComputeDirectionWeights(normal_world, directional_weights);

  BlockTraversal voxel_traversal(
      world_pos_start,
      world_ray_dir,
      2 * truncation, // 2 * truncation, because it covers both positive and negative range
      geometry_helper.voxel_size
  );
  while (voxel_traversal.HasNextBlock())
  {
    int3 voxel_idx = voxel_traversal.GetNextBlock();
    int3 block_idx = geometry_helper.VoxelToBlock(voxel_idx);
    hash_table.AllocEntry(block_idx);

    // Flag the corresponding hash entry
    int entry_idx = hash_table.GetEntryIndex(block_idx);
    if (entry_idx >= 0)
    {
      // set flag to binary mask indicating which directions are affected (for allocating VoxelArrays in the next step)
      for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
      {
        if (directional_weights[direction] > direction_weight_threshold)
        {
          candidate_entries.flag(entry_idx) |= (1 << direction);
        }
      }
    }
  }
}

/**
 *
 * @param candidate_entries
 * @param num_entries
 * @param blocks
 * @param enable_directional Whether to perform directional allocation. Otherwise voxel array 0 is allocated for all blocks.
 */
__global__
void AllocateVoxelArrayKernel(
    EntryArray candidate_entries,
    uint num_entries,
    BlockArray blocks,
    bool enable_directional = false
)
{
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= num_entries)
  {
    return;
  }
  const HashEntry &entry = candidate_entries[idx];

  if (enable_directional)
  {
    for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
    {
      if (entry.direction_flags & (1 << direction))
      {
        blocks.AllocateVoxelArrayWithMutex(entry.ptr, direction);
      }
    }

  } else
  {
    blocks.AllocateVoxelArrayWithMutex(entry.ptr, 0);
  }
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
    float3 normal_world = make_float3(wTcRotOnly * normal);

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

double AllocVoxelArray(
    EntryArray candidate_entries,
    BlockArray blocks,
    Sensor &sensor,
    GeometryHelper &geometry_helper,
    RuntimeParams &runtime_params
)
{
  Timer timer;
  timer.Tick();

  const dim3 grid_size(static_cast<unsigned int>(
                           std::ceil(candidate_entries.count() / static_cast<double>(CUDA_THREADS_PER_BLOCK))));
  const dim3 block_size(CUDA_THREADS_PER_BLOCK);
  if (runtime_params.enable_directional_sdf and runtime_params.update_type == UPDATE_TYPE_VOXEL_PROJECTION)
  {
    AllocateVoxelArrayKernelDirectional << < grid_size, block_size >> > (
        candidate_entries,
            candidate_entries.count(),
            blocks,
            sensor.data(),
            sensor.sensor_params(),
            sensor.cTw(),
            sensor.wTc(),
            geometry_helper
    );
  } else
  {
    AllocateVoxelArrayKernel << < grid_size, block_size >> > (
        candidate_entries,
            candidate_entries.count(),
            blocks,
            runtime_params.enable_directional_sdf
    );
  }

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return timer.Tock();
}
