#include "core/functions.h"
#include "mapping/allocate.h"
#include "mapping/update_raycasting.h"
#include "mapping/block_traversal.hpp"
#include "util/timer.h"
#include "geometry/geometry_helper.h"

////////////////////
/// Device code
////////////////////

#define RAY_DIRECTION_CAMERA 0
#define RAY_DIRECTION_NORMAL 1
#define RAY_DIRECTION_POS_CAMERA_NEG_NORMAL 2

__global__
void UpdateRaycastingKernel(
    BlockArray blocks,
    const size_t voxel_array_idx,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    float4x4 wTc,
    bool enable_point_to_plane,
    const int mode,
    HashTable hash_table,
    GeometryHelper geometry_helper
)
{
  // Pixel coordinates
  const int ux = blockIdx.x * blockDim.x + threadIdx.x;
  const int uy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ux >= sensor_params.width || uy >= sensor_params.height)
    return;

  float depth = tex2D<float>(sensor_data.depth_texture, ux, uy);
  float4 normal_camera = tex2D<float4>(sensor_data.normal_texture, ux, uy);

  if (not IsValidDepth(depth) or depth >= geometry_helper.sdf_upper_bound or not IsValidNormal(normal_camera))
    return;

  float3 point_camera_pos = GeometryHelper::ImageReprojectToCamera(ux, uy, depth,
                                                                   sensor_params.fx, sensor_params.fy,
                                                                   sensor_params.cx, sensor_params.cy);
  float3 point_world_pos = make_float3(wTc * make_float4(point_camera_pos, 1));

  float4x4 wTcRotOnly = wTc;
  wTcRotOnly.m14 = 0;
  wTcRotOnly.m24 = 0;
  wTcRotOnly.m34 = 0;
  float3 normal_world = make_float3(wTcRotOnly * normal_camera);

  float truncation_distance = geometry_helper.truncate_distance(depth);


  // Traverse voxels in normal's direction through measured surface point
  float3 origin;
  float3 direction;

  if (mode == RAY_DIRECTION_CAMERA)
  {
    float3 camera_world_pos = make_float3(wTc * make_float4(0, 0, 0, 1));
    direction = normalize(point_world_pos - camera_world_pos);
    origin = point_world_pos - truncation_distance * direction;
  } else // rmode == RAY_DIRECTION_NORMAL)
  {
    origin = point_world_pos - truncation_distance * normal_world;
    direction = normal_world;
  }


  BlockTraversal voxel_traversal(
      origin,
      direction,
      2 * truncation_distance, // 2 * truncation, because it covers both positive and negative range
      geometry_helper.voxel_size);
  while (voxel_traversal.HasNextBlock())
  {
    int3 voxel_idx = voxel_traversal.GetNextBlock();

    int3 block_idx = geometry_helper.VoxelToBlock(voxel_idx);
    uint local_idx = geometry_helper.VectorizeOffset(geometry_helper.VoxelToOffset(block_idx, voxel_idx));
    if (not blocks.HasVoxelArray(hash_table.GetEntry(block_idx).ptr, voxel_array_idx))
    {
//      printf("(%i, %i, %i) ", voxel_idx.x, voxel_idx.y, voxel_idx.z);
      continue; // TODO: throw warning. This should have been allocated beforehand.
    }
    Voxel &voxel = blocks.GetVoxelArray(hash_table.GetEntry(block_idx).ptr, voxel_array_idx).voxels[local_idx];

    float normalized_depth = geometry_helper.NormalizeDepth(
        depth,
        sensor_params.min_depth_range,
        sensor_params.max_depth_range
    );
    float3 voxel_world_pos = geometry_helper.VoxelToWorld(voxel_idx);
    float3 voxel_camera_pos = cTw * voxel_world_pos;


//    float weight = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth), 1.0f);

//    // linear voxel-observation-distance weight
//    float weight = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth) *
//                         length(point_world_pos - voxel_world_pos) / truncation_distance, 1.0f);
    // linear voxel-observation-distance weight + normal angle
    float weight = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth) *
                         length(point_world_pos - voxel_world_pos) / truncation_distance *
                         (2 - normal_camera.x + normal_camera.y),
                         1.0f);

    float sdf;
    if (enable_point_to_plane)
      sdf = dot(point_world_pos - voxel_world_pos, -normal_world);
    else
      sdf = sign(dot(point_world_pos - voxel_world_pos, -normal_world)) * length(point_world_pos - voxel_world_pos);

    atomicAdd(&voxel.a, weight * sdf);
    atomicAdd(&voxel.b, weight);
    atomicAdd(&voxel.num_updates, 1);
  }
}

__global__
void UpdateRaycastedBlocksKernel(
    EntryArray candidate_entries,
    uint num_entries,
    BlockArray blocks)
{
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= num_entries)
  {
    return;
  }
  const HashEntry &entry = candidate_entries[idx];
  VoxelArray &voxel_array = blocks.GetVoxelArray(entry.ptr, 0);
  for (size_t i = 0; i < BLOCK_SIZE; i++)
  {
    Voxel &voxel = voxel_array.voxels[i];
    if (voxel.num_updates == 0)
      continue;
    Voxel delta;
    delta.sdf = voxel.a / voxel.b;
    delta.inv_sigma2 = voxel.b; // / voxel.num_updates;

    voxel_array.voxels[i].Update(delta);

    // Reset summation values for next iteration
    voxel_array.voxels[i].a = 0;
    voxel_array.voxels[i].b = 0;
    voxel_array.voxels[i].num_updates = 0;
  }
}

double UpdateRaycasting(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    size_t voxel_array_idx,
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

  // 1) Make sure VoxelArrays are allocated
  const dim3 num_blocks_alloc(static_cast<unsigned int>(
                                  std::ceil(candidate_entry_count / static_cast<double>(CUDA_THREADS_PER_BLOCK))));
  const dim3 num_threads_alloc(CUDA_THREADS_PER_BLOCK);
  AllocateVoxelArrayKernel << < num_blocks_alloc, num_threads_alloc >> > (
      candidate_entries,
          candidate_entry_count,
          blocks
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  // 2) Fuse depth data
  const int threads_per_direction = 16;
  const dim3 grid_size_fusion((sensor.width() + threads_per_direction - 1) / threads_per_direction,
                              (sensor.height() + threads_per_direction - 1) / threads_per_direction);
  const dim3 block_size_fusion(threads_per_direction, threads_per_direction);
  UpdateRaycastingKernel << < grid_size_fusion, block_size_fusion >> > (
      blocks,
          voxel_array_idx,
          sensor.data(),
          sensor.sensor_params(),
          sensor.cTw(),
          sensor.wTc(),
          runtime_params.enable_point_to_plane,
          runtime_params.raycasting_mode,
          hash_table,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  // 3) Update SDF with fused values
  UpdateRaycastedBlocksKernel << < num_blocks_alloc, num_threads_alloc >> > (
      candidate_entries,
          candidate_entry_count,
          blocks
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return timer.Tock();
}
