#include <device_launch_parameters.h>
#include "meshing/marching_cubes.h"
#include "geometry/spatial_query.h"
#include "visualization/color_util.h"
#include "core/directional_tsdf.h"
#include "core/functions.h"

////////////////////
/// Device code
////////////////////

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

  /// 1) Collect SDF values and MC indices for each direction
  short mc_indices[6];
  short mc_indices_[6];
  float sdf_arrays[6][8];
  float sdf_weights[8];
  bool is_valid[6];
  for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
  {
    is_valid[direction] = GetVoxelSDFValues(entry, blocks, hash_table, geometry_helper,
                                            voxel_pos, direction,
                                            sdf_arrays[direction], sdf_weights[direction],
                                            mc_indices[direction]);
    mc_indices_[direction] = mc_indices[direction];

    mc_indices[direction] = FilterMCIndexDirection(mc_indices[direction], static_cast<TSDFDirection>(direction),
                                                   sdf_arrays[direction]);
  }

  /// 2) Check compatibility of each pair of mc indices to filter out wrong contours
  for (int direction = 0; direction < 6; direction++)
  {
    short &mc_index = mc_indices[direction];
    if (mc_index <= 0 or mc_index == 255)
      continue;

    float support_weight = 1.0f;
    for (int i = 0; i < N_DIRECTIONS; i++)
    {
      if (mc_indices[i] < 0)
        continue;
      // Only use directions, which are potentially compatible for exclusion
//      if (kIndexDirectionCompatibility[mc_index][i] == 0)
//        continue;

      if (mc_indices[i] == 0)
      { // If any of the compatible directions states, that voxel is in front of surface -> down-weight
        support_weight -= sdf_weights[i] / sdf_weights[direction];
//        // Hard threshold:
//        mc_index = 0;
//        break;
      } else if (i != direction and MCIndexCompatible(mc_index, mc_indices[i]))
      {
        support_weight += sdf_weights[i] / sdf_weights[direction];
      }
    }
    if (support_weight < 0)
    {
      mc_index = 0;
    }
  }

  /// 3) For every edge: find and intersect (directions) the up to 2 surface offsets (two possible directions)
  const int kEdgeCount = 12;
  for (int i = 0; i < kEdgeCount; ++i)
  {
    float2 surface_offsets = make_float2(0, 0);
    float2 weight_sum = make_float2(0, 0);
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
        surface_offsets.x += sdf_weights[direction] * surface_offset;
        weight_sum.x += sdf_weights[direction];
      } else
      { // surface points towards second endpoint
        surface_offsets.y += sdf_weights[direction] * surface_offset;
        weight_sum.y += sdf_weights[direction];
      }
    }

    if (weight_sum.x > 0)
      surface_offsets.x /= weight_sum.x;
    else
      surface_offsets.x = MINF;
    if (weight_sum.y > 0)
      surface_offsets.y /= weight_sum.y;
    else
      surface_offsets.y = MINF;



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

  /// 4) Compute combined MC indices from all directions for triangle extraction
  short2 combined_mc_indices = ComputeCombinedMCIndices(mc_indices);
  this_mesh_unit.mc_idx[0] = combined_mc_indices.x;
  this_mesh_unit.mc_idx[1] = combined_mc_indices.y;
}


/**
 * Finds the mesh unit at the given voxel position, even across block borders.
 * @param blocks
 * @param hash_table
 * @param geometry_helper
 * @param voxel_pos
 * @return
 */
__device__
MeshUnit *GetMeshUnitRelative(
    BlockArray blocks,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    int3 voxel_pos
)
{
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  HashEntry entry = hash_table.GetEntry(block_pos);
  if (entry.ptr == FREE_ENTRY)
    return nullptr;
  Block &block = blocks[entry.ptr];
  uint3 voxel_pos_relative = make_uint3(voxel_pos - geometry_helper.BlockToVoxel(block_pos));
  return &block.mesh_units[geometry_helper.VectorizeOffset(voxel_pos_relative)];
}

/**
 * Applies equalization to a voxels marching cubes index with respect to neighboring voxels
 * @param candidate_entries
 * @param blocks
 * @param hash_table
 * @param geometry_helper
 */
__global__
static void MCIndexEqualizationKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    HashTable hash_table,
    GeometryHelper geometry_helper
)
{
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block &block = blocks[entry.ptr];

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3 offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int3 voxel_pos = voxel_base_pos + make_int3(offset);

  MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
  if (this_mesh_unit.mc_idx[0] <= 0 or this_mesh_unit.mc_idx[0] == 255)
  {
    return;
  }

  const static size_t neighbors = 26;
  const static int3 offsets[neighbors] = {
      {-1, -1, -1},
      {-1, -1, 0},
      {-1, -1, 1},
      {-1, 0,  -1},
      {-1, 0,  0},
      {-1, 0,  1},
      {-1, 1,  -1},
      {-1, 1,  0},
      {-1, 1,  1},
      {0,  -1, -1},
      {0,  -1, 0},
      {0,  -1, 1},
      {0,  0,  -1},
      {0,  0,  1},
      {0,  1,  -1},
      {0,  1,  0},
      {0,  1,  1},
      {1,  -1, -1},
      {1,  -1, 0},
      {1,  -1, 1},
      {1,  0,  -1},
      {1,  0,  0},
      {1,  0,  1},
      {1,  1,  -1},
      {1,  1,  0},
      {1,  1,  1},
//      {-1, 0,  0},
//      {1,  0,  0},
//      {0,  -1, 0},
//      {0,  1,  0},
//      {0,  0,  -1},
//      {0,  0,  1},
  };
  MeshUnit *neighbor_mesh_units[neighbors];
  for (size_t i = 0; i < neighbors; i++)
  {
    neighbor_mesh_units[i] = GetMeshUnitRelative(blocks, hash_table, geometry_helper,
                                                 voxel_pos + offsets[i]);

  }

  // for every corner: 3 pairs of (neighbor, corner), that are coincident to this corner
  const static int2 corner_to_neighbor_corners[8][7] = {
      {{4,  1}, {5,  2}, {7,  5}, {8,  6}, {13, 3}, {15, 4}, {16, 7}},
      {{13, 2}, {15, 5}, {16, 6}, {21, 0}, {22, 3}, {24, 4}, {25, 7}},
      {{12, 1}, {14, 5}, {15, 6}, {20, 0}, {21, 3}, {23, 4}, {24, 7}},
      {{3,  1}, {4,  2}, {6,  5}, {7,  6}, {12, 0}, {14, 4}, {15, 7}},
      {{1,  1}, {2,  2}, {4,  5}, {5,  6}, {10, 0}, {11, 3}, {13, 7}},
      {{10, 1}, {11, 2}, {13, 6}, {18, 0}, {19, 3}, {21, 4}, {22, 7}},
      {{9,  1}, {10, 2}, {12, 5}, {17, 0}, {18, 3}, {20, 4}, {21, 7}},
      {{0,  1}, {1,  2}, {3,  5}, {4,  6}, {9,  0}, {10, 3}, {12, 4}},
//      {{0, 1}, {3, 4}, {5, 3}},
//      {{1, 0}, {3, 5}, {5, 2}},
//      {{1, 3}, {3, 6}, {4, 1}},
//      {{0, 2}, {3, 7}, {4, 0}},
//      {{0, 5}, {2, 0}, {5, 7}},
//      {{1, 4}, {2, 1}, {5, 6}},
//      {{1, 7}, {2, 2}, {4, 5}},
//      {{0, 6}, {2, 3}, {4, 4}}
  };

  short before = this_mesh_unit.mc_idx[0];

//  for (size_t c = 0; c < 8; c++)
//  {
//    int this_corner_value = (this_mesh_unit.mc_idx[0] & (1 << c)) > 0;
//    for (size_t neighbor = 0; neighbor < 7; neighbor++)
//    {
//      int2 p = corner_to_neighbor_corners[c][neighbor];
//      int neighbor_idx = p.x;
//      int neighbor_corner_idx = p.y;
//      if (not neighbor_mesh_units[neighbor_idx])
//        continue;
//      if (neighbor_mesh_units[neighbor_idx]->mc_idx[0] <= 0)
//        continue;
//      int corner_value = (neighbor_mesh_units[neighbor_idx]->mc_idx[0] & (1 << neighbor_corner_idx)) > 0;
//      this_mesh_unit.mc_idx[0] &= (0xff ^ (1 << c)) | (corner_value << c);
//    }
//  }


  for (size_t c = 0; c < 8; c++)
  {
    int this_corner_value = (this_mesh_unit.mc_idx[0] & (1 << c)) > 0;
    int votes = 1;
    for (size_t neighbor = 0; neighbor < 7; neighbor++)
    {
      int2 p = corner_to_neighbor_corners[c][neighbor];
      int neighbor_idx = p.x;
      int neighbor_corner_idx = p.y;
      if (not neighbor_mesh_units[neighbor_idx])
        continue;
      if (neighbor_mesh_units[neighbor_idx]->mc_idx[0] <= 0)
        continue;
      int corner_value = (neighbor_mesh_units[neighbor_idx]->mc_idx[0] & (1 << neighbor_corner_idx)) > 0;
      this_corner_value += corner_value;
      votes++;
    }
    // TODO: check compatibility before combining
    if (this_corner_value > votes / 2)
    {
      this_mesh_unit.mc_idx[0] |= (1 << c);
    } else
    {
      this_mesh_unit.mc_idx[0] &= 0xff ^ (1 << c);
    }
  }
//  if (before != this_mesh_unit.mc_idx[0])
//  {
//    printf("[%i, %i, %i] %i -> %i\n", voxel_pos.x, voxel_pos.y, voxel_pos.z, before, this_mesh_unit.mc_idx[0]);
//  }
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
        mesh_unit.ResetMutexes();
      }
    }

    int ptr_offset = idx * 3;

    //////////
    /// 3. Assign triangles
    int i = 0;
    for (int t = 0;
         kTriangleVertexEdge[mc_index][t] != -1;
         t += 3, ++i)
    {
      int triangle_ptr = this_mesh_unit.triangle_ptrs[i + ptr_offset];
      if (triangle_ptr == FREE_PTR)
      {
        triangle_ptr = mesh.AllocTriangle();
      } else
      {
        mesh.ReleaseTriangleVertexReferences(mesh.triangle(triangle_ptr));
      }
      this_mesh_unit.triangle_ptrs[i + ptr_offset] = triangle_ptr;


      if (vertex_ptrs[kTriangleVertexEdge[mc_index][t + 0]] < 0 or
          vertex_ptrs[kTriangleVertexEdge[mc_index][t + 1]] < 0 or
          vertex_ptrs[kTriangleVertexEdge[mc_index][t + 2]] < 0)
      {
        // If one of the vertex pointers does not exist, don't assign triangle
        // (This is expected behavior from combining neighboring marching cubes indices)
        continue;
      }
      mesh.AssignTriangleVertexReferences(
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
    int i = idx * N_TRIANGLE / 2; // offset

    // Remove triangles with at least one invalid vertex pointer
    for (int t = 0; kTriangleVertexEdge[mc_index][t] != -1; t += 3, ++i)
    {
      int triangle_ptr = mesh_unit.triangle_ptrs[i];
      if (triangle_ptr == FREE_PTR) continue;
      Triangle &triangle = mesh.triangle(mesh_unit.triangle_ptrs[i]);
      if (triangle.vertex_ptrs.x < 0 or triangle.vertex_ptrs.y < 0 or triangle.vertex_ptrs.z < 0)
      {
        mesh.FreeTriangle(triangle_ptr);
        mesh_unit.triangle_ptrs[i] = FREE_PTR;
      }
    }

    // Remove triangles, that cannot exist according to MC index
    for (; i < N_TRIANGLE / (2 - idx); ++i)
    {
      int triangle_ptr = mesh_unit.triangle_ptrs[i];
      if (triangle_ptr == FREE_PTR) continue;

      // Clear ref_count of its pointed vertices
      mesh.ReleaseTriangleVertexReferences(mesh.triangle(triangle_ptr));
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
  MCIndexEqualizationKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          hash_table,
          geometry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass11_seconds = timer.Tock();
  LOG(INFO) << "Pass1.1 duration: " << pass11_seconds;

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
