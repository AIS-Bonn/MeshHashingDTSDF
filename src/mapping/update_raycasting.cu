#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "engine/main_engine.h"
#include "mapping/allocate.h"
#include "mapping/block_traversal.hpp"
#include "mapping/update_raycasting.h"
#include "mapping/weight_functions.h"
#include "util/timer.h"
#include "geometry/geometry_helper.h"

////////////////////
/// Device code
////////////////////

__device__
inline void CarveVoxel(
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
  Voxel &voxel = blocks.GetVoxelArray(hash_table.GetEntry(block_idx).ptr, voxel_array_idx).voxels[local_idx];

  if (voxel.inv_sigma2 <= 0)
    return;

  float sdf = 1;
//  float weight = 1;
  float weight = fmaxf(geometry_helper.weight_sample *
                       weight_depth(normalized_depth) *
                       weight_normal_angle(make_float3(normal_camera)), 1.0f);

  atomicAdd(&voxel.a, weight * sdf);
  atomicAdd(&voxel.b, weight);
  atomicAdd(&voxel.num_updates, 1);
  return;
}

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
    // FIXME: Issues from parallel allocation
//    printf("(%i, %i, %i) ", voxel_idx.x, voxel_idx.y, voxel_idx.z);
    return;
  }
  Voxel &voxel = blocks.GetVoxelArray(hash_table.GetEntry(block_idx).ptr, voxel_array_idx).voxels[local_idx];

  float3 voxel_pos_world = geometry_helper.VoxelToWorld(voxel_idx);

                     DirectionWeight(DirectionAngle(normal_world, voxel_array_idx));
  float weight = fmaxf(geometry_helper.weight_sample *
                       weight_depth(normalized_depth) *
                       //                       weight_voxel_correlation(surface_point_world, voxel_pos_world, truncation_distance) *
                       weight_normal_angle(make_float3(normal_camera)) *

//                       DirectionWeight(DirectionAngle(normal_world, voxel_array_idx)),
                       weight_direction_compliance(voxel_array_idx, normal_world),
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
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 wTc,
    BlockArray blocks,
    RuntimeParams runtime_params,
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
  float3 ray_direction_before;
  float3 ray_direction_behind;

  if (runtime_params.raycasting_mode == RAY_DIRECTION_CAMERA)
  {
    float3 camera_world_pos = make_float3(wTc * make_float4(0, 0, 0, 1));
    ray_direction_before = ray_direction_behind = normalize(point_world_pos - camera_world_pos);
    ray_origin = point_world_pos - truncation_distance * ray_direction_before;
  }
  if (runtime_params.raycasting_mode == RAY_DIRECTION_POS_CAMERA_NEG_NORMAL)
  {
    float3 camera_world_pos = make_float3(wTc * make_float4(0, 0, 0, 1));
    ray_direction_before = normalize(point_world_pos - camera_world_pos);
    ray_direction_behind = -normal_world;
    ray_origin = point_world_pos - truncation_distance * ray_direction_before;
  } else // (runtime_params.raycasting_mode == RAY_DIRECTION_NORMAL)
  {
    ray_direction_before = normal_world;
    ray_origin = point_world_pos - truncation_distance * ray_direction_before;
  }

  float normalized_depth = geometry_helper.NormalizeDepth(
      depth,
      sensor_params.min_depth_range,
      sensor_params.max_depth_range
  );
  float directional_weights[N_DIRECTIONS];
  ComputeDirectionWeights(normal_world, directional_weights);


  /////////////////////////////////// voxel carving
#if 0
//  float3 carving_origin = make_float3(wTc * make_float4(normalize(point_camera_pos) * 0.5, 1));
  float3 carving_origin = make_float3(wTc *
                                      make_float4(
                                          GeometryHelper::ImageReprojectToCamera(ux, uy, 0.5,
                                                                                 sensor_params.fx, sensor_params.fy,
                                                                                 sensor_params.cx, sensor_params.cy),
                                          1));
  float3 carving_ray_direction = normalize(point_world_pos - carving_origin);
//  float carving_distance =
//      length(point_world_pos - carving_origin) - 2 / (-normal_camera.z) * truncation_distance;
  float carving_distance =
      length(point_world_pos - carving_origin) - fabs(1.5 / dot(normal_world, carving_ray_direction)) * truncation_distance;
//  float carving_distance = length(point_world_pos - carving_origin) - truncation_distance;

  BlockTraversal voxel_traversal_carving(
      carving_origin,
      carving_ray_direction,
      carving_distance,
      geometry_helper.voxel_size);
  int3 last_block_idx = make_int3(1000000, 1000000, 1000000);
  int direction_block_exists[6];
  while (voxel_traversal_carving.HasNextBlock())
  {
    if (voxel_traversal_carving.distance > length(point_world_pos - carving_origin))
      printf("%f / %f (%f)\n", voxel_traversal_carving.distance, length(point_world_pos - carving_origin),
          carving_distance);
    int3 voxel_idx = voxel_traversal_carving.GetNextBlock();
    int3 block_idx = geometry_helper.VoxelToBlock(voxel_idx);
    if (not(block_idx == last_block_idx))
    {
      last_block_idx = block_idx;
      for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
        direction_block_exists[direction] = -1;
    }

    if (runtime_params.enable_directional_sdf)
    {
      for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
      {
        if (direction_block_exists[direction] == -1)
        {
          if (blocks.HasVoxelArray(hash_table.GetEntry(block_idx).ptr, direction))
            direction_block_exists[direction] = 1;
          else
            direction_block_exists[direction] = 0;
        }
        if (direction_block_exists[direction] == 0)
          continue;
        CarveVoxel(
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
    } else
    {
      const size_t voxel_array_idx = 0;
      CarveVoxel(
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
#endif
  //////////////////////////////////

  BlockTraversal voxel_traversal_before(
      ray_origin,
      ray_direction_before,
      truncation_distance,
      geometry_helper.voxel_size);
  BlockTraversal voxel_traversal_behind(
      point_world_pos,
      ray_direction_behind,
      truncation_distance,
      geometry_helper.voxel_size);
  if (voxel_traversal_behind.HasNextBlock()) voxel_traversal_behind.GetNextBlock(); // Skip first voxel to prevent duplicate fusion
  while (voxel_traversal_before.HasNextBlock() or voxel_traversal_behind.HasNextBlock())
  {
    const size_t voxel_array_idx = 0;

    int3 voxel_idx;
    if (voxel_traversal_before.HasNextBlock())
      voxel_idx = voxel_traversal_before.GetNextBlock();
    else
      voxel_idx = voxel_traversal_behind.GetNextBlock();

    if (runtime_params.enable_directional_sdf)
    {
      for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
      {
        if (directional_weights[direction] > 0)
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
      continue;
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

struct VoxelUpdateStatistics
{
  /** Mean number of updates per voxel (all voxels) */
  float num_updates_mean;
  /** Mean number of updates per voxels that are actually updated */
  float num_updates_hit_mean;
  /** Max number of updates per voxels that are actually updated */
  int num_updates_hit_min;
  /** Min number of updates per voxels that are actually updated */
  int num_updates_hit_max;
};

__global__
void CollectUpdateStatisticsKernel(
    EntryArray candidate_entries,
    uint num_entries,
    BlockArray blocks,
    RuntimeParams runtime_params,
    VoxelUpdateStatistics *stats
)
{
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx >= num_entries)
  {
    return;
  }
  stats[idx].num_updates_hit_max = -1;
  stats[idx].num_updates_hit_min = -1;
  stats[idx].num_updates_hit_mean = 0;
  stats[idx].num_updates_mean = 0;

  const HashEntry &entry = candidate_entries[idx];
  size_t max_voxel_idx = 0;
  if (runtime_params.enable_directional_sdf)
    max_voxel_idx = 5;
  int count = 0;
  int count_hit = 0;
  for (size_t direction = 0; direction <= max_voxel_idx; direction++)
  {
    if (not blocks.HasVoxelArray(entry.ptr, direction))
    {
      continue;
    }
    VoxelArray &voxel_array = blocks.GetVoxelArray(entry.ptr, direction);
    for (size_t i = 0; i < BLOCK_SIZE; i++)
    {
      Voxel &voxel = voxel_array.voxels[i];
      stats[idx].num_updates_mean += voxel.num_updates;
      if (voxel.num_updates)
      {
        stats[idx].num_updates_hit_mean += voxel.num_updates;
        stats[idx].num_updates_hit_max = max(stats[idx].num_updates_hit_max, (int) voxel.num_updates);
        if (stats[idx].num_updates_hit_min == -1)
          stats[idx].num_updates_hit_min = voxel.num_updates;
        else
          stats[idx].num_updates_hit_min = min(stats[idx].num_updates_hit_min, voxel.num_updates);
        count_hit++;
      }
      count++;
    }
  }
  stats[idx].num_updates_mean /= count;
  stats[idx].num_updates_hit_mean /= count_hit;
}

void CollectUpdateStatistics(
    EntryArray candidate_entries,
    MainEngine &main_engine)
{
  uint num_entries = candidate_entries.count();

  VoxelUpdateStatistics *stats, *stats_cpu;
  checkCudaErrors(cudaMalloc(&stats, sizeof(VoxelUpdateStatistics) * num_entries));
  stats_cpu = (VoxelUpdateStatistics *) malloc(sizeof(VoxelUpdateStatistics) * num_entries);

  const dim3 num_blocks_alloc(static_cast<unsigned int>(
                                  std::ceil(num_entries / static_cast<double>(CUDA_THREADS_PER_BLOCK))));
  const dim3 num_threads_alloc(CUDA_THREADS_PER_BLOCK);
  CollectUpdateStatisticsKernel << < num_blocks_alloc, num_threads_alloc >> > (
      candidate_entries,
          num_entries,
          main_engine.blocks(),
          main_engine.runtime_params(),
          stats);
  checkCudaErrors(
      cudaMemcpy(stats_cpu, stats, num_entries * sizeof(VoxelUpdateStatistics), cudaMemcpyDeviceToHost));
  int total_hit_max = 0;
  int total_hit_min = std::numeric_limits<int>::max();
  float total_hit_mean = 0;
  float total_mean = 0;
  for (size_t i = 0; i < num_entries; i++)
  {
    total_hit_max = max(total_hit_max, stats_cpu[i].num_updates_hit_max);
    if (stats_cpu[i].num_updates_hit_min > 0)
      total_hit_min = min(total_hit_min, stats_cpu[i].num_updates_hit_min);
    total_mean += stats_cpu[i].num_updates_mean / num_entries;
    total_hit_mean += stats_cpu[i].num_updates_hit_mean / num_entries;
  }
  free(stats_cpu);
  checkCudaErrors(cudaFree(stats));
//  printf("MIN: %i, MAX: %i, MEAN(hit): %f, MEAN(total): %f\n", total_hit_min, total_hit_max, total_hit_mean,
//         total_mean);
  main_engine.log_engine().WriteVoxelUpdate(total_hit_max, total_hit_mean, total_mean);
}

double UpdateRaycasting(
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

  /// 1) Fuse depth data
  const int threads_per_direction = 16;
  const dim3 grid_size_fusion((sensor.width() + threads_per_direction - 1) / threads_per_direction,
                              (sensor.height() + threads_per_direction - 1) / threads_per_direction);
  const dim3 block_size_fusion(threads_per_direction, threads_per_direction);
  UpdateRaycastingKernel << < grid_size_fusion, block_size_fusion >> > (
      sensor.data(),
          sensor.sensor_params(),
          sensor.wTc(),
          main_engine.blocks(),
          main_engine.runtime_params(),
          main_engine.hash_table(),
          main_engine.geometry_helper());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

//  CollectUpdateStatistics(candidate_entries, main_engine);

  /// 2) Update SDF with fused values
  const dim3 num_blocks_alloc(static_cast<unsigned int>(
                                  std::ceil(candidate_entry_count / static_cast<double>(CUDA_THREADS_PER_BLOCK))));
  const dim3 num_threads_alloc(CUDA_THREADS_PER_BLOCK);
  UpdateRaycastedBlocksKernel << < num_blocks_alloc, num_threads_alloc >> > (
      candidate_entries,
          candidate_entry_count,
          main_engine.blocks(),
          main_engine.runtime_params()
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return timer.Tock();
}
