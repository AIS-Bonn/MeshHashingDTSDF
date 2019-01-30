//
// Created by wei on 17-10-22.
//

#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "mapping/allocate.h"
#include "util/timer.h"

__global__
void AllocBlockArrayKernel(HashTable   hash_table,
                           SensorData  sensor_data,
                           SensorParams sensor_params,
                           float4x4     w_T_c,
                           GeometryHelper geometry_helper) {

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
  float near_depth = fminf(geometry_helper.sdf_upper_bound, depth - truncation);
  float far_depth = fminf(geometry_helper.sdf_upper_bound, depth + truncation);
  if (near_depth >= far_depth) return;

  float3 camera_pos_near = geometry_helper.ImageReprojectToCamera(x, y, near_depth,
                                                            sensor_params.fx, sensor_params.fy,
                                                            sensor_params.cx, sensor_params.cy);
  float3 camera_pos_far  = geometry_helper.ImageReprojectToCamera(x, y, far_depth,
                                                            sensor_params.fx, sensor_params.fy,
                                                            sensor_params.cx, sensor_params.cy);

  /// 2. Set range where blocks are allocated
  float3 world_pos_near  = w_T_c * camera_pos_near;
  float3 world_pos_far   = w_T_c * camera_pos_far;
  float3 world_ray_dir = normalize(world_pos_far - world_pos_near);

  int3 block_pos_near = geometry_helper.WorldToBlock(world_pos_near);
  int3 block_pos_far  = geometry_helper.WorldToBlock(world_pos_far);
  float3 block_step = make_float3(sign(world_ray_dir));

  /// 3. Init zig-zag steps
  float3 world_pos_nearest_voxel_center
      = geometry_helper.BlockToWorld(block_pos_near + make_int3(clamp(block_step, 0.0, 1.0f)))
        - 0.5f * geometry_helper.voxel_size;
  float3 t = (world_pos_nearest_voxel_center - world_pos_near) / world_ray_dir;
  float3 dt = (block_step * BLOCK_SIDE_LENGTH * geometry_helper.voxel_size) / world_ray_dir;
  int3 block_pos_bound = make_int3(make_float3(block_pos_far) + block_step);

  if (world_ray_dir.x == 0.0f) {
    t.x = PINF;
    dt.x = PINF;
  }
  if (world_ray_dir.y == 0.0f) {
    t.y = PINF;
    dt.y = PINF;
  }
  if (world_ray_dir.z == 0.0f) {
    t.z = PINF;
    dt.z = PINF;
  }

  int3 block_pos_curr = block_pos_near;
  /// 4. Go a zig-zag path to ensure all voxels are visited
  const uint kMaxIterTime = 1024;
#pragma unroll 1
  for (uint iter = 0; iter < kMaxIterTime; ++iter) {
    if (geometry_helper.IsBlockInCameraFrustum(
        w_T_c.getInverse(),
        block_pos_curr,
        sensor_params)) {
      /// Disable streaming at current
      // && !isSDFBlockStreamedOut(idCurrentVoxel, hash_table, is_streamed_mask)) {
      hash_table.AllocEntry(block_pos_curr);
    }

    // Traverse voxel grid
    if (t.x < t.y && t.x < t.z) {
      block_pos_curr.x += block_step.x;
      if (block_pos_curr.x == block_pos_bound.x) return;
      t.x += dt.x;
    } else if (t.y < t.z) {
      block_pos_curr.y += block_step.y;
      if (block_pos_curr.y == block_pos_bound.y) return;
      t.y += dt.y;
    } else {
      block_pos_curr.z += block_step.z;
      if (block_pos_curr.z == block_pos_bound.z) return;
      t.z += dt.z;
    }
  }
}

double AllocBlockArray(
    HashTable& hash_table,
    Sensor& sensor,
    GeometryHelper& geometry_helper
) {
  Timer timer;
  timer.Tick();
  hash_table.ResetMutexes();

  const uint threads_per_block = 8;
  const dim3 grid_size((sensor.sensor_params().width + threads_per_block - 1)
                       /threads_per_block,
                       (sensor.sensor_params().height + threads_per_block - 1)
                       /threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  AllocBlockArrayKernel<<<grid_size, block_size>>>(
      hash_table,
      sensor.data(),
      sensor.sensor_params(), sensor.wTc(),
      geometry_helper);
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
