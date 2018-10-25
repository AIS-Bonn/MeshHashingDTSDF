#include <device_launch_parameters.h>
#include <util/timer.h>

#include "core/block_array.h"
#include "core/directional_tsdf.h"
#include "mapping/update_simple.h"
#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/spatial_query.h"

////////////////////
/// Device code
////////////////////

/** Allocates the first Voxel Arrays for every given Block
 */
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

/** Allocates Voxel Arrays for the given Blocks with respect to the input normal map.
 */
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
    normal.w = 1;

    float4x4 wTcRotOnly = wTc;
    wTcRotOnly.m14 = 0;
    wTcRotOnly.m24 = 0;
    wTcRotOnly.m34 = 0;
    float4 normal_world = wTcRotOnly * normal;
    TSDFDirection direction = VectorToTSDFDirection(normal_world);
    allocate_directions[static_cast<size_t>(direction)] = 1;
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
void UpdateBlocksSimpleKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    const size_t voxel_array_idx,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    HashTable hash_table,
    GeometryHelper geometry_helper
)
{

  //TODO check if we should load this in shared memory (entries)
  /// 1. Select voxel
  const HashEntry &entry = candidate_entries[blockIdx.x];
  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

//  Voxel &this_voxel = blocks[entry.ptr].voxel_array[voxel_array_idx]->voxels[local_idx];
  Voxel &this_voxel = blocks.GetVoxelArray(entry.ptr, voxel_array_idx).voxels[local_idx];
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
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= geometry_helper.sdf_upper_bound)
    return;

  float sdf = depth - camera_pos.z;
  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float inv_sigma2 = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth),
                           1.0f);
  float truncation = geometry_helper.truncate_distance(depth);
  if (sdf <= -truncation)
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
    HashTable hash_table,
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
  float depth = tex2D<float>(sensor_data.depth_texture, image_pos.x, image_pos.y);
  if (depth == MINF || depth == 0.0f || depth >= geometry_helper.sdf_upper_bound)
    return;

  float sdf = depth - camera_pos.z;
  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float inv_sigma2 = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth),
                           1.0f);
  float truncation = geometry_helper.truncate_distance(depth);
  if (sdf <= -truncation)
    return;
  if (sdf >= 0.0f)
  {
    sdf = fminf(truncation, sdf);
  } else
  {
    sdf = fmaxf(-truncation, sdf);
  }

  /// 4. Find TSDF direction to fuse into
  float4 normal = tex2D<float4>(sensor_data.normal_texture, image_pos.x, image_pos.y);
  normal.w = 1;
  if (normal.x == 0 and normal.y == 0 and normal.z == 0)
  { // No normal value for this coordinate
    return;
  }

  float4x4 wTcRotOnly = wTc;
  wTcRotOnly.m14 = 0;
  wTcRotOnly.m24 = 0;
  wTcRotOnly.m34 = 0;
  float4 normal_world = wTcRotOnly * normal;
  TSDFDirection direction = VectorToTSDFDirection(normal_world);
//  printf("(%f, %f, %f)[%f, %f] -> %s\n", normal_world.x, normal_world.y, normal_world.z,
//      std::atan2(normal_world.x, normal_world.z), std::sin(normal_world.y), TSDFDirectionToString(direction));
  Voxel &this_voxel = blocks.GetVoxelArray(entry.ptr, static_cast<size_t>(direction)).voxels[local_idx];

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

double UpdateBlocksSimple(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    const size_t voxel_array_idx,
    Sensor &sensor,
    HashTable &hash_table,
    GeometryHelper &geometry_helper
)
{

  Timer timer;
  timer.Tick();

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return timer.Tock();

  // 1) Make sure VoxelArrays are allocated
  const dim3 num_blocks(static_cast<unsigned int>(
                            std::ceil(candidate_entry_count / static_cast<double>(CUDA_THREADS_PER_BLOCK))));
  const dim3 num_threads(CUDA_THREADS_PER_BLOCK);
  AllocateVoxelArrayKernel << < num_blocks, num_threads >> > (
      candidate_entries,
          candidate_entry_count,
          blocks
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);

  // 2) Update blocks
  UpdateBlocksSimpleKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          voxel_array_idx,
          sensor.data(),
          sensor.sensor_params(),
          sensor.cTw(),
          hash_table,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  return timer.Tock();
}

double UpdateBlocksSimpleDirectional(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Sensor &sensor,
    HashTable &hash_table,
    GeometryHelper &geometry_helper
)
{
  Timer timer;
  timer.Tick();

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return timer.Tock();

  // 1) Make sure VoxelArrays are allocated
  const dim3 num_blocks(static_cast<unsigned int>(
                            std::ceil(candidate_entry_count / static_cast<double>(CUDA_THREADS_PER_BLOCK))));
  const dim3 num_threads(CUDA_THREADS_PER_BLOCK);
  AllocateVoxelArrayKernelDirectional << < num_blocks, num_threads >> > (
      candidate_entries,
          candidate_entry_count,
          blocks,
          sensor.data(),
          sensor.sensor_params(),
          sensor.cTw(),
          sensor.wTc(),
          geometry_helper
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);

  // 2) Update blocks
  UpdateBlocksSimpleKernelDirectional << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          sensor.data(),
          sensor.sensor_params(),
          sensor.cTw(),
          sensor.wTc(),
          hash_table,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  return timer.Tock();
}
