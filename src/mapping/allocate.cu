//
// Created by wei on 17-10-22.
//

#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "engine/main_engine.h"
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
void AllocBlockArrayKernel(
    EntryArray candidate_entries,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 wTc,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    RuntimeParams runtime_params)
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

  float3 point_camera_pos = geometry_helper.ImageReprojectToCamera(x, y, depth,
                                                                   sensor_params.fx, sensor_params.fy,
                                                                   sensor_params.cx, sensor_params.cy);
  float3 point_world_pos = wTc * point_camera_pos;
  float3 normal_world = make_float3(wTc * normal_camera);

  /// 2. Set range where blocks are allocated
  float3 ray_direction_before;
  float3 ray_direction_behind;
  if (runtime_params.raycasting_mode == RAY_DIRECTION_CAMERA)
  {
    float3 camera_world_pos = make_float3(wTc * make_float4(0, 0, 0, 1));
    ray_direction_before = ray_direction_behind = normalize(point_world_pos - camera_world_pos);
  }
  if (runtime_params.raycasting_mode == RAY_DIRECTION_POS_CAMERA_NEG_NORMAL)
  {
    float3 camera_world_pos = make_float3(wTc * make_float4(0, 0, 0, 1));
    ray_direction_before = normalize(point_world_pos - camera_world_pos);
    ray_direction_behind = -normal_world;
  } else // (runtime_params.raycasting_mode == RAY_DIRECTION_NORMAL)
  {
    ray_direction_behind = ray_direction_before = -normal_world;
  }


  /// 3. Traverse all blocks in truncation range and allocate VoxelArray, if necessary

  float directional_weights[N_DIRECTIONS];
  ComputeDirectionWeights(normal_world, directional_weights);

  BlockTraversal voxel_traversal_before(
      point_world_pos - truncation * ray_direction_before,
      ray_direction_before,
      truncation,
      geometry_helper.voxel_size);
  BlockTraversal voxel_traversal_behind(
      point_world_pos,
      ray_direction_behind,
      truncation,
      geometry_helper.voxel_size);
  if (voxel_traversal_behind.HasNextBlock()) voxel_traversal_behind.GetNextBlock(); // Skip first voxel to prevent duplicate fusion
  while (voxel_traversal_before.HasNextBlock() or voxel_traversal_behind.HasNextBlock())
  {
    int3 voxel_idx;
    if (voxel_traversal_before.HasNextBlock())
      voxel_idx = voxel_traversal_before.GetNextBlock();
    else
      voxel_idx = voxel_traversal_behind.GetNextBlock();

    int3 block_idx = geometry_helper.VoxelToBlock(voxel_idx);
    hash_table.AllocEntry(block_idx);

    // Flag the corresponding hash entry
    int entry_idx = hash_table.GetEntryIndex(block_idx);
    if (entry_idx >= 0)
    {
      // set flag to binary mask indicating which directions are affected (for allocating VoxelArrays in the next step)
      for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
      {
        if (directional_weights[direction] > 0)
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
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    float4x4 wTc,
    BlockArray blocks,
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
      allocate_directions[direction] |= (weights[direction] > 0);
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
    EntryArray candidate_entries,
    Sensor &sensor,
    MainEngine &main_engine
)
{
  Timer timer;
  timer.Tick();
  main_engine.hash_table().ResetMutexes();

  const uint threads_per_block = 8;
  const dim3 grid_size((sensor.sensor_params().width + threads_per_block - 1)
                       / threads_per_block,
                       (sensor.sensor_params().height + threads_per_block - 1)
                       / threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);
  AllocBlockArrayKernel << < grid_size, block_size >> > (
      candidate_entries,
          sensor.data(),
          sensor.sensor_params(), sensor.wTc(),
          main_engine.hash_table(),
          main_engine.geometry_helper(),
          main_engine.runtime_params()
  );

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  return timer.Tock();
}

double AllocVoxelArray(
    EntryArray candidate_entries,
    Sensor &sensor,
    MainEngine &main_engine
)
{
  Timer timer;
  timer.Tick();

  const dim3 grid_size(static_cast<unsigned int>(
                           std::ceil(candidate_entries.count() / static_cast<double>(CUDA_THREADS_PER_BLOCK))));
  const dim3 block_size(CUDA_THREADS_PER_BLOCK);
  if (main_engine.runtime_params().enable_directional_sdf and
      main_engine.runtime_params().update_type == UPDATE_TYPE_VOXEL_PROJECTION)
  {
    AllocateVoxelArrayKernelDirectional << < grid_size, block_size >> > (
        candidate_entries,
            candidate_entries.count(),
            sensor.data(),
            sensor.sensor_params(),
            sensor.cTw(),
            sensor.wTc(),
            main_engine.blocks(),
            main_engine.geometry_helper()
    );
  } else
  {
    AllocateVoxelArrayKernel << < grid_size, block_size >> > (
        candidate_entries,
            candidate_entries.count(),
            main_engine.blocks(),
            main_engine.runtime_params().enable_directional_sdf
    );
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return timer.Tock();
}
