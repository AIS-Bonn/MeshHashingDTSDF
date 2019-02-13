#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "mapping/allocate.h"
#include "mapping/update_raycasting.h"
#include "mapping/block_traversal.hpp"
#include "util/timer.h"
#include "geometry/geometry_helper.h"

////////////////////
/// Device code
////////////////////

__device__
inline void UpdateVoxel(
    const int3 &voxel_idx,
    const size_t voxel_array_idx,
    const float normalized_depth,
    const float truncation_distance,
    const float3 &surface_point_world,
    const float3 &normal_world,
    const float4 &normal_camera,
    const BlockArray &blocks,
    const SensorParams &sensor_params,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    const bool enable_point_to_plane
)
{
  int3 block_idx = geometry_helper.VoxelToBlock(voxel_idx);
  uint local_idx = geometry_helper.VectorizeOffset(geometry_helper.VoxelToOffset(block_idx, voxel_idx));

  if (not blocks.HasVoxelArray(hash_table.GetEntry(block_idx).ptr, voxel_array_idx))
  {
//      printf("(%i, %i, %i) ", voxel_idx.x, voxel_idx.y, voxel_idx.z);
    return; // TODO: throw warning. This should have been allocated beforehand.
  }
  Voxel &voxel = blocks.GetVoxelArray(hash_table.GetEntry(block_idx).ptr, voxel_array_idx).voxels[local_idx];

  float3 voxel_pos_world = geometry_helper.VoxelToWorld(voxel_idx);

//    float weight = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth), 1.0f);

//    // linear voxel-observation-distance weight
//    float weight = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth) *
//                         length(point_world_pos - voxel_pos_world) / truncation_distance, 1.0f);
  // linear voxel-observation-distance weight + normal angle
  float weight = fmaxf(10 * geometry_helper.weight_sample * (1.0f - normalized_depth) *
                       length(surface_point_world - voxel_pos_world) / truncation_distance *
                       (2 - normal_camera.x + normal_camera.y),
                       1.0f);

  float3 observation_ray = voxel_pos_world - surface_point_world;
  float sdf;
  if (enable_point_to_plane)
    sdf = dot(observation_ray, normal_world);
  else
    sdf = sign(dot(observation_ray, normal_world)) * length(observation_ray);

  atomicAdd(&voxel.a, weight * sdf);
  atomicAdd(&voxel.b, weight);
  atomicAdd(&voxel.num_updates, 1);
}

/**
 * For every pixel casts a ray into multiple voxels (truncation range and updates the
 * SDF and weight summation values)
 * @param blocks
 * @param sensor_data
 * @param sensor_params
 * @param runtime_params
 * @param wTc
 * @param hash_table
 * @param geometry_helper
 */
__global__
void UpdateRaycastingKernel(
    BlockArray blocks,
    SensorData sensor_data,
    SensorParams sensor_params,
    RuntimeParams runtime_params,
    float4x4 wTc,
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
  float3 ray_origin;
  float3 ray_direction;

  if (runtime_params.raycasting_mode == RAY_DIRECTION_CAMERA)
  {
    float3 camera_world_pos = make_float3(wTc * make_float4(0, 0, 0, 1));
    ray_direction = normalize(point_world_pos - camera_world_pos);
    ray_origin = point_world_pos - truncation_distance * ray_direction;
  } else // (mode == RAY_DIRECTION_NORMAL)
  {
    ray_origin = point_world_pos - truncation_distance * normal_world;
    ray_direction = normal_world;
  }

  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float directional_weights[N_DIRECTIONS];
  ComputeDirectionWeights(normal_world, directional_weights);

  BlockTraversal voxel_traversal(
      ray_origin,
      ray_direction,
      2 * truncation_distance, // 2 * truncation, because it covers both positive and negative range
      geometry_helper.voxel_size);
  while (voxel_traversal.HasNextBlock())
  {
    const size_t voxel_array_idx = 0;

    int3 voxel_idx = voxel_traversal.GetNextBlock();
    if (runtime_params.enable_directional_sdf)
    {
      for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
      {
        if (directional_weights[direction] > direction_weight_threshold)
        {
          UpdateVoxel(
              voxel_idx,
              direction,
              normalized_depth,
              truncation_distance,
              point_world_pos,
              normal_world,
              normal_camera,
              blocks,
              sensor_params,
              hash_table,
              geometry_helper,
              runtime_params.enable_point_to_plane
          );
        }
      }
    } else
    {
      UpdateVoxel(
          voxel_idx,
          voxel_array_idx,
          normalized_depth,
          truncation_distance,
          point_world_pos,
          normal_world,
          normal_camera,
          blocks,
          sensor_params,
          hash_table,
          geometry_helper,
          runtime_params.enable_point_to_plane
      );
    }

  }
}

/**
 * For every updated Voxel compute perform the SDF update (take summation values, perform single update)
 * @param candidate_entries
 * @param num_entries
 * @param blocks
 */
__global__
void UpdateRaycastedBlocksKernel(
    EntryArray candidate_entries,
    uint num_entries,
    BlockArray blocks,
    RuntimeParams runtime_params)
{
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= num_entries)
  {
    return;
  }
  const HashEntry &entry = candidate_entries[idx];

  size_t max_voxel_idx = 0;
  if (runtime_params.enable_directional_sdf)
    max_voxel_idx = 5;
  for (size_t direction = 0; direction <= max_voxel_idx; direction++)
  {
    if (not blocks.HasVoxelArray(entry.ptr, direction))
    {
      return;
    }
    VoxelArray &voxel_array = blocks.GetVoxelArray(entry.ptr, direction);
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
}

double UpdateRaycasting(
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

  /// 1) Fuse depth data
  const int threads_per_direction = 16;
  const dim3 grid_size_fusion((sensor.width() + threads_per_direction - 1) / threads_per_direction,
                              (sensor.height() + threads_per_direction - 1) / threads_per_direction);
  const dim3 block_size_fusion(threads_per_direction, threads_per_direction);
  UpdateRaycastingKernel << < grid_size_fusion, block_size_fusion >> > (
      blocks,
          sensor.data(),
          sensor.sensor_params(),
          runtime_params,
          sensor.wTc(),
          hash_table,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  /// 2) Update SDF with fused values
  const dim3 num_blocks_alloc(static_cast<unsigned int>(
                                  std::ceil(candidate_entry_count / static_cast<double>(CUDA_THREADS_PER_BLOCK))));
  const dim3 num_threads_alloc(CUDA_THREADS_PER_BLOCK);
  UpdateRaycastedBlocksKernel << < num_blocks_alloc, num_threads_alloc >> > (
      candidate_entries,
          candidate_entry_count,
          blocks,
          runtime_params
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return timer.Tock();
}
