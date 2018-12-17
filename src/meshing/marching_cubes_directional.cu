#include <device_launch_parameters.h>
#include "meshing/marching_cubes.h"
#include "geometry/spatial_query.h"
#include "visualization/color_util.h"
#include "core/directional_tsdf.h"

////////////////////
/// class MappingEngine - meshing
////////////////////

////////////////////
/// Device code
////////////////////

/**
 * Interpolate the surface offset between two voxel vertices given their SDF values.
 * The offset denotes the distance from v1 to the iso surface in [0, 1] (-> voxel side length)
 * @param v1 SDF value of corner 1
 * @param v2 SDF value of corner 2
 * @param isolevel Surface iso level
 * @return Vertex offset between two voxel vertices
 */
__device__
static inline float InterpolateSurfaceOffset(const float &v1, const float &v2,
                                             const float &isolevel)
{
  if (fabs(v1 - isolevel) < 0.008) return 0;
  if (fabs(v2 - isolevel) < 0.008) return 1;
  return (isolevel - v1) / (v2 - v1);
}

/**
 * Interpolate vertex position on voxel edge for the given SDF values.
 * @param p1 Voxel corner 1
 * @param p2 Voxel corner 2
 * @param v1 SDF value of corner 1
 * @param v2 SDF value of corner 2
 * @param isolevel Surface iso level
 * @return Vertex position on voxel edge
 */
__device__
static float3 VertexIntersection(const float3 &p1, const float3 p2,
                                 const float &v1, const float &v2,
                                 const float &isolevel)
{
  float mu = InterpolateSurfaceOffset(v1, v2, isolevel);

  float3 p = make_float3(p1.x + mu * (p2.x - p1.x),
                         p1.y + mu * (p2.y - p1.y),
                         p1.z + mu * (p2.z - p1.z));
  return p;
}

__device__
static inline int AllocateVertexWithMutex(
    MeshUnit &mesh_unit,
    uint vertex_idx,
    const float3 &vertex_pos,
    Mesh &mesh,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool enable_sdf_gradient
)
{
  int ptr = mesh_unit.vertex_ptrs[vertex_idx];
  if (ptr == FREE_PTR)
  {
    int lock = atomicExch(&mesh_unit.vertex_mutexes[vertex_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY)
    {
      ptr = mesh.AllocVertex();
    } /// Ensure that it is only allocated once
  }

  if (ptr >= 0)
  {
    Voxel voxel_query;
    mesh_unit.vertex_ptrs[vertex_idx] = ptr;
    mesh.vertex(ptr).pos = vertex_pos;
    mesh.vertex(ptr).radius = sqrtf(1.0f / voxel_query.inv_sigma2);

    float3 grad = make_float3(0.0f);
    if (enable_sdf_gradient)
    {
      bool valid = false;
      float l = 0;
      for (size_t voxel_array_idx = 0; voxel_array_idx < 6 and not valid and not l > 0; voxel_array_idx++)
      {
        valid |= GetSpatialSDFGradient(
            vertex_pos,
            blocks, voxel_array_idx, hash_table,
            geometry_helper,
            &grad);
        l = length(grad);
      }
      mesh.vertex(ptr).normal = l > 0 && valid ? grad / l : make_float3(0);
    }

//    mesh.vertex(ptr).color = ValToRGB(voxel_query.inv_sigma2 / 10000.0f, 0, 1.0f);
    mesh.vertex(ptr).color = make_float3(0.5f);
  }
  return ptr;
}

/**
 * Check, whether the given MC index is compatible to the direction.
 *
 * @param mc_index MC index
 * @param direction Direction
 * @param sdf SDF values of voxel corners
 * @return
 */
__device__
static bool IsMCIndexDirectionCompatible(const short mc_index, const TSDFDirection direction, const float sdf[8])
{
  // Table containing for each direction:
  // 4 opposite edge pairs, each of which is checked individually.
  const static size_t view_direction_edges_to_check[6][8] = {
      {0, 4, 1, 5, 2,  6,  3,  7},
      {4, 0, 5, 1, 6,  2,  7,  3},
      {1, 3, 5, 7, 9,  8,  10, 11},
      {3, 1, 7, 5, 8,  9,  11, 10},
      {2, 0, 6, 4, 10, 9,  11, 8},
      {0, 2, 4, 6, 8,  11, 9,  10}
  };
  if (kIndexDirectionCompatibility[mc_index][static_cast<size_t>(direction)] == 0)
    return false;
  if (kIndexDirectionCompatibility[mc_index][static_cast<size_t>(direction)] == 2)
  {
    for (int e = 0; e < 4; e++)
    {
      const size_t edge_idx = view_direction_edges_to_check[static_cast<size_t>(direction)][2 * e];
      const size_t opposite_edge_idx = view_direction_edges_to_check[static_cast<size_t>(direction)][2 * e + 1];
      int2 edge = kEdgeEndpointVertices[edge_idx];
      int2 opposite_edge = kEdgeEndpointVertices[opposite_edge_idx];

      int2 endpoint_values;
      endpoint_values.x = (mc_index & (1 << edge.x)) > 0;
      endpoint_values.y = (mc_index & (1 << edge.y)) > 0;

      // If edge has NO zero-crossing -> continue
      if (endpoint_values.x + endpoint_values.y != 1)
        continue;

      // Swap vertex indices, s.t. first endpoint is behind the surface
      if (endpoint_values.y == 1)
      {
        int tmp;
        tmp = edge.x;
        edge.x = edge.y;
        edge.y = tmp;
        tmp = opposite_edge.x;
        opposite_edge.x = opposite_edge.y;
        opposite_edge.y = tmp;
      }

      float offset = InterpolateSurfaceOffset(sdf[edge.x], sdf[edge.y], 0);
      float opposite_offset = InterpolateSurfaceOffset(sdf[opposite_edge.x], sdf[opposite_edge.y], 0);

      // If interpolated surface more than 90 degrees from view direction vector -> discard
      if (offset > opposite_offset)
      {
        return false;
      }
    }
  }
  return true;
}


__device__
static short FilterMCIndexDirection(const short mc_index, const TSDFDirection direction, const float sdf[8])
{
  if (mc_index <= 0 or mc_index == 255)
    return mc_index;

  short new_index = 0;
  for (int component = 0; component < 4 and kIndexDecomposition[mc_index][component] != -1; component++)
  {
    const short part_idx = kIndexDecomposition[mc_index][component];
    if (not IsMCIndexDirectionCompatible(part_idx, direction, sdf))
      continue;
    new_index |= part_idx;
  }

  if (new_index == 0)
  { // If 0 after filtering -> invalidate, so it doesn't affect other directions during later filtering process
    new_index = -1;
  }
  return new_index;
}

/** Checks if the specified edge of two marching cubes indices is compatible
 * (if zero-crossing -> same direction?)
 *
 * @param edge
 * @return
 */
__device__
inline bool MCEdgeCompatible(short mc_index1, short mc_index2, int2 edge)
{
  int a = (((mc_index1 & (1 << edge.x)) == 0) << 1) + ((mc_index1 & (1 << edge.y)) == 0);
  int b = (((mc_index2 & (1 << edge.x)) == 0) << 1) + ((mc_index2 & (1 << edge.y)) == 0);

  return not((a == 0b01 and b == 0b10) or (a == 0b10 and b == 0b01));
}

__device__
static bool MCIndexCompatible(short mc_index1, short mc_index2)
{
  bool compatible = false;

  short intersection = mc_index1 & mc_index2;
  if (intersection)
  { // If indices overlap at least in 1 point, check if at least one edge adjacent to any overlap is compatible
    for (int edge_idx = 0; edge_idx < 12; edge_idx++)
    {
      int2 edge = kEdgeEndpointVertices[edge_idx];
      if ((intersection & (1 << edge.x)) == 0 or (intersection & (1 << edge.y)) == 0)
      {
        if (MCEdgeCompatible(mc_index1, mc_index2, edge))
        {
          compatible = true;
        }
      }
    }
  } else
  { // Indices don't overlap -> all 0-1 transition edges must be compatible !
    compatible = true;
    for (int edge_idx = 0; edge_idx < 12; edge_idx++)
    {
      compatible &= MCEdgeCompatible(mc_index1, mc_index2, kEdgeEndpointVertices[edge_idx]);
    }
  }

  return compatible;
}

__device__
static short2 ComputeCombinedMCIndices(const short mc_indices[6])
{
  short combined[2] = {0, 0};
  for (size_t i = 0; i < 6; i++)
  {
    short mc_index = mc_indices[i];

    if (mc_index <= 0 or mc_index == 255)
      continue;

    for (int cmp_idx = 0; cmp_idx < 4 and kIndexDecomposition[mc_index][cmp_idx] != -1; cmp_idx++)
    {
      short mc_component = kIndexDecomposition[mc_index][cmp_idx];
      for (int j = 0; j < 2; j++)
      {
        if (combined[j] == 0)
        {
          combined[j] = mc_component;
          break;
        } else if (MCIndexCompatible(mc_index, combined[j]))
        {
          combined[j] &= mc_component;
          break;
        } else if (j == 1)
        {
          printf("Error: marching cubes index could not be combined: (%i, %i) !! %i\n",
                 combined[0], combined[1], mc_component);
        }
      }
    }
  }

  return {combined[0], combined[1]};
}

/**
 * Given a direction and a voxel position, fetches the SDF values of the corner points and
 * computes the MC index.
 *
 * @param entry
 * @param blocks
 * @param hash_table
 * @param geometry_helper
 * @param voxel_pos Voxel position
 * @param direction SDF direction
 * @param sdf_array Output array SDF values of 8 voxel corners
 * @param mc_index Output marching cubes index
 * @return
 */
__device__
static bool GetVoxelSDFValues(const HashEntry &entry, BlockArray &blocks,
                              HashTable &hash_table, GeometryHelper &geometry_helper,
                              int3 voxel_pos, TSDFDirection direction,
                              float *sdf_array, short &mc_index)
{
  const float kVoxelSize = geometry_helper.voxel_size;
  const float kThreshold = 4 * kVoxelSize;
  const float kIsoLevel = 0;


  mc_index = 0;
  Voxel voxel_query;
  for (int i = 0; i < 8; ++i)
  {
    if (not GetVoxelValue(entry, voxel_pos + kVtxOffset[i],
                          blocks, static_cast<size_t>(direction), hash_table,
                          geometry_helper, &voxel_query))
    {
      mc_index = -1;
      return false;
    }

    sdf_array[i] = voxel_query.sdf;

    if (fabs(sdf_array[i]) > kThreshold)
    { // too far away from surface
      mc_index = -1;
      return false;
    }

    float rho = voxel_query.a / (voxel_query.a + voxel_query.b);
    if (rho < 0.1f or voxel_query.inv_sigma2 < squaref(1.0f / kVoxelSize) / 4)
    { // too uncertain (small weight)
      mc_index = -1;
      return false;
    }

    mc_index |= (sdf_array[i] < kIsoLevel) * (1 << i);
  }

  return true;
}

__global__
static void VertexExtractionKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    bool enable_sdf_gradient
)
{
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block &block = blocks[entry.ptr];

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3 offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int3 voxel_pos = voxel_base_pos + make_int3(offset);
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
  //////////
  /// 1. Read the scalar values, see mc_tables.h
  const int kVertexCount = 8;
  const float kVoxelSize = geometry_helper.voxel_size;
  const float kIsoLevel = 0;

  float3 p[kVertexCount];

  this_mesh_unit.mc_idx[0] = 0;
  this_mesh_unit.mc_idx[1] = 0;

  for (int i = 0; i < 8; ++i)
  {
    p[i] = world_pos + kVoxelSize * make_float3(kVtxOffset[i]);
  }

  // 1) Collect SDF values and MC indices for each direction
  short mc_indices[6];
  short mc_indices_[6];
  float sdf_arrays[6][8];
  bool is_valid[6];
  for (int direction = 0; direction < 6; direction++)
  {
    is_valid[direction] = GetVoxelSDFValues(entry, blocks, hash_table, geometry_helper,
                                            voxel_pos, TSDFDirection(direction),
                                            sdf_arrays[direction], mc_indices[direction]);
    mc_indices_[direction] = mc_indices[direction];

    mc_indices[direction] = FilterMCIndexDirection(mc_indices[direction], static_cast<TSDFDirection>(direction),
                                                   sdf_arrays[direction]);
  }

  // 2) Check compatibility of each pair of mc indices to filter out wrong contours
  for (int direction = 0; direction < 6; direction++)
  {
    short &mc_index = mc_indices[direction];
    if (mc_index <= 0 or mc_index == 255)
      continue;

    int support = 0;
    for (int i = 0; i < 6; i++)
    {
      // Only use directions, which are potentially compatible for exclusion
      if (kIndexDirectionCompatibility[mc_index][i] == 0)
        continue;

      if (mc_indices[i] < 0)
        continue;
      if (mc_indices[i] == 0)
      { // If any of the compatible directions states, that voxel is in front of surface -> down-weight
        support -= 1;
        // Hard threshold:
        mc_index = 0;
        break;
      } else if (i != direction and MCIndexCompatible(mc_index, mc_indices[i]))
      {
        support += 1;
      }
    }
    if (support < 0)
    {
      mc_index = 0;
    }
  }

  // 3) For every edge: find and intersect (directions) the up to 2 surface offsets (two possible directions)
  const int kEdgeCount = 12;
  for (int i = 0; i < kEdgeCount; ++i)
  {
    float2 surface_offsets = make_float2(MINF, MINF);
    int2 edge_endpoint_vertices = kEdgeEndpointVertices[i];

    for (int direction = 0; direction < 6; direction++)
    {
      short mc_index = mc_indices[direction];
      if (mc_index < 0)
        continue;

      float *sdf = sdf_arrays[direction];

      if (not(kCubeEdges[mc_index] & (1 << i)))
        continue;

      float surface_offset = InterpolateSurfaceOffset(sdf[edge_endpoint_vertices.x],
                                                      sdf[edge_endpoint_vertices.y], kIsoLevel);

      if (mc_index & (1 << edge_endpoint_vertices.x))
      { // surface points towards first endpoint
        surface_offsets.x = fmaxf(surface_offsets.x, surface_offset);
      } else
      { // surface points towards second endpoint
        if (surface_offsets.y == MINF)
        {
          surface_offsets.y = surface_offset;
        }
        surface_offsets.y = -fmaxf(-surface_offsets.y, -surface_offset);
      }
    }

    // If edge is unaffected -> continue
    if (surface_offsets.x == MINF and surface_offsets.y == MINF)
      continue;

    // Ensure that inverse surfaces don't cut
    if (surface_offsets.x > MINF and surface_offsets.y > MINF and surface_offsets.y < surface_offsets.x)
    {
      float mean = (surface_offsets.x + surface_offsets.y) * 0.5f;
      surface_offsets.x = surface_offsets.y = mean;
    }

    // Determine MeshUnit responsible for edge
    uint4 edge_cube_owner_offset = kEdgeOwnerCubeOffset[i];
    MeshUnit &mesh_unit = GetMeshUnitRef(
        entry,
        voxel_pos + make_int3(edge_cube_owner_offset.x,
                              edge_cube_owner_offset.y,
                              edge_cube_owner_offset.z),
        blocks, hash_table,
        geometry_helper);

    if (surface_offsets.x > MINF)
    { // If surface pointing towards first endpoint exists
      float3 vertex_pos = p[edge_endpoint_vertices.x]
                          + surface_offsets.x * (p[edge_endpoint_vertices.y] - p[edge_endpoint_vertices.x]);
      // Store vertex
      AllocateVertexWithMutex(
          mesh_unit,
          edge_cube_owner_offset.w,
          vertex_pos,
          mesh,
          blocks,
          hash_table, geometry_helper,
          enable_sdf_gradient);
    }
    if (surface_offsets.y > MINF)
    { // If surface pointing towards second endpoint exists
      float3 vertex_pos = p[edge_endpoint_vertices.x]
                          + surface_offsets.y * (p[edge_endpoint_vertices.y] - p[edge_endpoint_vertices.x]);
      // Store vertex
      AllocateVertexWithMutex(
          mesh_unit,
          edge_cube_owner_offset.w +
          3,  // surface facing towards second endpoint, store in latter triplet of vertex pointers
          vertex_pos,
          mesh,
          blocks,
          hash_table, geometry_helper,
          enable_sdf_gradient);
    }
  }

  // 4) Compute combined MC indices from all directions for triangle extraction
  short2 combined_mc_indices = ComputeCombinedMCIndices(mc_indices);
  this_mesh_unit.mc_idx[0] = combined_mc_indices.x;
  this_mesh_unit.mc_idx[1] = combined_mc_indices.y;
}

__device__
static inline bool IsInner(uint3 offset)
{
  return (offset.x >= 1 && offset.y >= 1 && offset.z >= 1
          && offset.x < BLOCK_SIDE_LENGTH - 1
          && offset.y < BLOCK_SIDE_LENGTH - 1
          && offset.z < BLOCK_SIDE_LENGTH - 1);
}

__global__
static void TriangleExtractionKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    bool enable_sdf_gradient
)
{
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block &block = blocks[entry.ptr];
  if (threadIdx.x == 0)
  {
    block.boundary_surfel_count = 0;
    block.inner_surfel_count = 0;
  }
  __syncthreads();

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3 offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int3 voxel_pos = voxel_base_pos + make_int3(offset);

  MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
  bool is_inner = IsInner(offset);
  for (int i = 0; i < N_VERTEX; ++i)
  {
    if (this_mesh_unit.vertex_ptrs[i] >= 0)
    {
      if (is_inner)
      {
        atomicAdd(&block.inner_surfel_count, 1);
      } else
      {
        atomicAdd(&block.boundary_surfel_count, 1);
      }
    }
  }
  for (int idx = 0; idx < 2; idx++)
  {
    short mc_index = this_mesh_unit.mc_idx[idx];

    /// Cube type unchanged: NO need to update triangles
//  if (this_cube.curr_cube_idx == this_cube.prev_cube_idx) {
//    blocks[entry.ptr].voxels[local_idx].stats.duration += 1.0f;
//    return;
//  }
//  blocks[entry.ptr].voxels[local_idx].stats.duration = 0;


    if (mc_index == 0 or mc_index == 255)
    {
      return;
    }

    //////////
    /// 2. Determine vertices (ptr allocated via (shared) edges
    /// If the program reach here, the voxels holding edges must exist
    /// This operation is in 2-pass
    /// pass2: Assign
    const int kEdgeCount = 12;
    int vertex_ptrs[kEdgeCount];

#pragma unroll 1
    for (int i = 0; i < kEdgeCount; ++i)
    {
      if (kCubeEdges[mc_index] & (1 << i))
      {
        // Compute for which surface direction the vertex is stored on
        // (compare which endpoint lies in front/behind surface)
        int ptr_offset;
        if (mc_index & (1 << kEdgeEndpointVertices[i].x))
        {
          ptr_offset = 0;
        } else
        {
          ptr_offset = 3;
        }

        uint4 edge_owner_cube_offset = kEdgeOwnerCubeOffset[i];
        MeshUnit &mesh_unit = GetMeshUnitRef(
            entry,
            voxel_pos + make_int3(edge_owner_cube_offset.x,
                                  edge_owner_cube_offset.y,
                                  edge_owner_cube_offset.z),
            blocks,
            hash_table,
            geometry_helper);

        vertex_ptrs[i] = mesh_unit.GetVertex(edge_owner_cube_offset.w + ptr_offset);
        if (vertex_ptrs[i] < 0)
        {
          printf("Error: Missing vertex MC: %x, (%i,%i, %i, %i, %i, %i) \n", mc_index,
                 mesh_unit.GetVertex(0), mesh_unit.GetVertex(1), mesh_unit.GetVertex(2), mesh_unit.GetVertex(3),
                 mesh_unit.GetVertex(4), mesh_unit.GetVertex(5));
        }
        mesh_unit.ResetMutexes();
      }
    }

    int offset = idx * 3;

    //////////
    /// 3. Assign triangles
    int i = 0;
    for (int t = 0;
         kTriangleVertexEdge[mc_index][t] != -1;
         t += 3, ++i)
    {
      int triangle_ptr = this_mesh_unit.triangle_ptrs[i + offset];
      if (triangle_ptr == FREE_PTR)
      {
        triangle_ptr = mesh.AllocTriangle();
      } else
      {
        mesh.ReleaseTriangle(mesh.triangle(triangle_ptr));
      }
      this_mesh_unit.triangle_ptrs[i + offset] = triangle_ptr;

      mesh.AssignTriangle(
          mesh.triangle(triangle_ptr),
          make_int3(vertex_ptrs[kTriangleVertexEdge[mc_index][t + 0]],
                    vertex_ptrs[kTriangleVertexEdge[mc_index][t + 1]],
                    vertex_ptrs[kTriangleVertexEdge[mc_index][t + 2]]));
      if (!enable_sdf_gradient)
      {
        mesh.ComputeTriangleNormal(mesh.triangle(triangle_ptr));
      }
    }
  }
}

/// Garbage collection (ref count)
__global__
static void RecycleTrianglesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh)
{
  const HashEntry &entry = candidate_entries[blockIdx.x];
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[threadIdx.x];


  for (int idx = 0; idx < 2; idx++)
  {
    short mc_index = mesh_unit.mc_idx[idx];
    int i = 0;
    for (int t = 0; kTriangleVertexEdge[mc_index][t] != -1; t += 3, ++i);

    i += idx * N_TRIANGLE / 2; // offset

    for (; i < N_TRIANGLE / 2; ++i)
    {
      int triangle_ptr = mesh_unit.triangle_ptrs[i];
      if (triangle_ptr == FREE_PTR) continue;

      // Clear ref_count of its pointed vertices
      mesh.ReleaseTriangle(mesh.triangle(triangle_ptr));
      mesh.FreeTriangle(triangle_ptr);
      mesh_unit.triangle_ptrs[i] = FREE_PTR;
    }
  }
}

__global__
static void RecycleVerticesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh
)
{
  const HashEntry &entry = candidate_entries[blockIdx.x];
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[threadIdx.x];

#pragma unroll 1
  for (int i = 0; i < 3; ++i)
  {
    if (mesh_unit.vertex_ptrs[i] != FREE_PTR &&
        mesh.vertex(mesh_unit.vertex_ptrs[i]).ref_count == 0)
    {
      mesh.FreeVertex(mesh_unit.vertex_ptrs[i]);
      mesh_unit.vertex_ptrs[i] = FREE_PTR;
      mesh_unit.vertex_mutexes[i] = FREE_ENTRY;
    }
  }
}

////////////////////
/// Host code
////////////////////
float MarchingCubesDirectional(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh,
    HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool enable_sdf_gradient
)
{
  uint occupied_block_count = candidate_entries.count();
  LOG(INFO) << "Marching cubes block count: " << occupied_block_count;
  if (occupied_block_count == 0)
    return -1;

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  /// Use divide and conquer to avoid read-write conflict
  Timer timer;
  timer.Tick();
  VertexExtractionKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          mesh,
          hash_table,
          geometry_helper,
          enable_sdf_gradient);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass1_seconds = timer.Tock();
  LOG(INFO) << "Pass1 duration: " << pass1_seconds;

  timer.Tick();
  TriangleExtractionKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          mesh,
          hash_table,
          geometry_helper,
          enable_sdf_gradient);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass2_seconds = timer.Tock();
  LOG(INFO) << "Pass2 duration: " << pass2_seconds;

  RecycleTrianglesKernel << < grid_size, block_size >> > (
      candidate_entries, blocks, mesh);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleVerticesKernel << < grid_size, block_size >> > (
      candidate_entries, blocks, mesh);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return (float) (pass1_seconds + pass2_seconds);
}
