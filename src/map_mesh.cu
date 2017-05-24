#include <glog/logging.h>
#include <unordered_map>

#include "mc_tables.h"
#include "map.h"

//#define REDUCTION

////////////////////
/// class Map - meshing
////////////////////

////////////////////
/// Device code
////////////////////

/// Marching Cubes
__device__
float3 VertexIntersection(const float3& p1, const float3 p2,
                          const float& v1,  const float& v2,
                          const float& isolevel) {
  if (fabs(v1 - isolevel) < 0.001) return p1;
  if (fabs(v2 - isolevel) < 0.001) return p2;
  float mu = (isolevel - v1) / (v2 - v1);
  float3 p = make_float3(p1.x + mu * (p2.x - p1.x),
                         p1.y + mu * (p2.y - p1.y),
                         p1.z + mu * (p2.z - p1.z));
  return p;
}

__device__
inline Voxel GetVoxel(HashTableGPU&    hash_table,
                      VoxelBlocksGPU&  blocks,
                      const HashEntry& curr_entry,
                      uint3 voxel_local_pos,
                      const uint3 local_offset) {
  Voxel v; v.Clear();

  uint3 voxel_local_pos_offset = voxel_local_pos + local_offset;
  int3 block_offset = make_int3(voxel_local_pos_offset.x / BLOCK_SIDE_LENGTH,
                                voxel_local_pos_offset.y / BLOCK_SIDE_LENGTH,
                                voxel_local_pos_offset.z / BLOCK_SIDE_LENGTH);

  /// Inside the block -- no need to look up in the table
  if (block_offset.x == 0 && block_offset.y == 0 && block_offset.z == 0) {
    uint i = VoxelLocalPosToIdx(voxel_local_pos_offset);
    v = blocks[curr_entry.ptr].voxels[i];
  } else { // Outside the block -- look for it
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) return v;
    uint i = VoxelLocalPosToIdx(make_uint3(
            voxel_local_pos_offset.x % BLOCK_SIDE_LENGTH,
            voxel_local_pos_offset.y % BLOCK_SIDE_LENGTH,
            voxel_local_pos_offset.z % BLOCK_SIDE_LENGTH));
    v = blocks[entry.ptr].voxels[i];
  }

  return v;
}

__device__
inline MeshCube& GetMeshCube(HashTableGPU&  hash_table,
                             VoxelBlocksGPU blocks,
                             const HashEntry& curr_entry,
                             uint3 voxel_local_pos,
                             const uint3 local_offset) {

  uint3 voxel_local_pos_offset = voxel_local_pos + local_offset;
  int3 block_offset = make_int3(voxel_local_pos_offset.x / BLOCK_SIDE_LENGTH,
                                voxel_local_pos_offset.y / BLOCK_SIDE_LENGTH,
                                voxel_local_pos_offset.z / BLOCK_SIDE_LENGTH);

  if (block_offset.x == 0 && block_offset.y == 0 && block_offset.z == 0) {
    uint i = VoxelLocalPosToIdx(voxel_local_pos_offset);
    return blocks[curr_entry.ptr].cubes[i];
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetMeshCube: should never reach here! %d %d %d\n",
             voxel_local_pos.x,
             voxel_local_pos.y,
             voxel_local_pos.z);
    }
    uint i = VoxelLocalPosToIdx(make_uint3(
            voxel_local_pos_offset.x % BLOCK_SIDE_LENGTH,
            voxel_local_pos_offset.y % BLOCK_SIDE_LENGTH,
            voxel_local_pos_offset.z % BLOCK_SIDE_LENGTH));
    return blocks[entry.ptr].cubes[i];
  }
}

__device__
inline int AllocateVertex(MeshGPU& mesh,
                          int& vertex_ptr,
                          const float3& vertex_pos) {
  int ptr = vertex_ptr;
  if (ptr == -1) ptr = mesh.AllocVertex();
  mesh.vertices[ptr].pos = vertex_pos;
  vertex_ptr = ptr;
  return ptr;
}

__device__
inline bool check_mask(uint3 pos, uchar3 mask) {
  return ((pos.x & 1) == mask.x)
         && ((pos.y & 1) == mask.y)
         && ((pos.z & 1) == mask.z);
}

// TODO(wei): add locks
__global__
void MarchingCubesKernel(HashTableGPU        hash_table,
                         CompactHashTableGPU compact_hash_table,
                         VoxelBlocksGPU      blocks,
                         uchar3 mask1, uchar3 mask2,// use this to avoid conflict
                         MeshGPU mesh_data) {
  const float isolevel = 0;

  const HashEntry &map_entry = compact_hash_table.compacted_entries[blockIdx.x];

  int3  voxel_base_pos = BlockToVoxel(map_entry.pos);

  const uint local_idx = threadIdx.x;
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
#ifdef REDUCTION
  if (! check_mask(voxel_local_pos, mask1)
    && ! check_mask(voxel_local_pos, mask2)) {
    return;
  }
#endif

  MeshCube &this_cube = blocks[map_entry.ptr].cubes[local_idx];
  this_cube.cube_index = 0;

  int3 voxel_pos = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos = VoxelToWorld(voxel_pos);

  //////////
  /// 1. Read the scalar values
  /// Refer to paulbourke.net/geometry/polygonise
  /// Our coordinate system:
  ///       ^
  ///      /
  ///    z
  ///   /
  /// o -- x -->
  /// |
  /// y
  /// |
  /// v
  // 0 -> 011
  // 1 -> 111
  // 2 -> 110
  // 3 -> 010
  // 4 -> 001
  // 5 -> 101
  // 6 -> 100
  // 7 -> 000
  Voxel v;
  float d[8];
  float3 p[8];

  float voxel_size = kSDFParams.voxel_size;
  v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos, make_uint3(0, 1, 1));
  if (v.weight == 0) return;
  p[0] = world_pos + voxel_size * make_float3(0, 1, 1);
  d[0] = v.sdf;

  v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos, make_uint3(1, 1, 1));
  if (v.weight == 0) return;
  p[1] = world_pos + voxel_size * make_float3(1, 1, 1);
  d[1] = v.sdf;

  v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos, make_uint3(1, 1, 0));
  if (v.weight == 0) return;
  p[2] = world_pos + voxel_size * make_float3(1, 1, 0);
  d[2] = v.sdf;

  v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos, make_uint3(0, 1, 0));
  if (v.weight == 0) return;
  p[3] = world_pos + voxel_size * make_float3(0, 1, 0);
  d[3] = v.sdf;

  v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos, make_uint3(0, 0, 1));
  if (v.weight == 0) return;
  p[4] = world_pos + voxel_size * make_float3(0, 0, 1);
  d[4] = v.sdf;

  v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos, make_uint3(1, 0, 1));
  if (v.weight == 0) return;
  p[5] = world_pos + voxel_size * make_float3(1, 0, 1);
  d[5] = v.sdf;

  v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos, make_uint3(1, 0, 0));
  if (v.weight == 0) return;
  p[6] = world_pos + voxel_size * make_float3(1, 0, 0);
  d[6] = v.sdf;

  v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos, make_uint3(0, 0, 0));
  if (v.weight == 0) return;
  p[7] = world_pos + voxel_size * make_float3(0, 0, 0);
  d[7] = v.sdf;

  //////////
  /// 2. Determine cube type
  int cube_index = 0;
  if (d[0] < isolevel) cube_index |= 1;
  if (d[1] < isolevel) cube_index |= 2;
  if (d[2] < isolevel) cube_index |= 4;
  if (d[3] < isolevel) cube_index |= 8;
  if (d[4] < isolevel) cube_index |= 16;
  if (d[5] < isolevel) cube_index |= 32;
  if (d[6] < isolevel) cube_index |= 64;
  if (d[7] < isolevel) cube_index |= 128;

  const float kThreshold = 0.2f;
  if (fabs(d[0]) > kThreshold) return;
  if (fabs(d[1]) > kThreshold) return;
  if (fabs(d[2]) > kThreshold) return;
  if (fabs(d[3]) > kThreshold) return;
  if (fabs(d[4]) > kThreshold) return;
  if (fabs(d[5]) > kThreshold) return;
  if (fabs(d[6]) > kThreshold) return;
  if (fabs(d[7]) > kThreshold) return;
  for (uint k = 0; k < 8; k++) {
    for (uint l = 0; l < 8; l++) {
      if (d[k] * d[l] < 0.0f) {
        if (fabs(d[k]) + fabs(d[l]) > kThreshold) return;
      } else {
        if (fabs(d[k] - d[l]) > kThreshold) return;
      }
    }
  }

  if (kEdgeTable[cube_index] == 0 || kEdgeTable[cube_index] == 255)
    return;

  //////////
  /// 3. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  // 0 -> 011.x, (0, 1)
  // 1 -> 110.z, (1, 2)
  // 2 -> 010.x, (2, 3)
  // 3 -> 010.z, (3, 0)
  // 4 -> 001.x, (4, 5)
  // 5 -> 100.z, (5, 6)
  // 6 -> 000.x, (6, 7)
  // 7 -> 000.z, (7, 4)
  // 8 -> 001.y, (4, 0)
  // 9 -> 101.y, (5, 1)
  //10 -> 100.y, (6, 2)
  //11 -> 000.y, (7, 3)
  int vertex_ptr[12];
  float3 vertex_pos;

  /// plane y = 1
  if (kEdgeTable[cube_index] & 1) {
    vertex_pos = VertexIntersection(p[0], p[1], d[0], d[1], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(0, 1, 1));
    vertex_ptr[0] = AllocateVertex(mesh_data, cube.vertex_ptrs.x, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 2) {
    vertex_pos = VertexIntersection(p[1], p[2], d[1], d[2], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(1, 1, 0));
    vertex_ptr[1] = AllocateVertex(mesh_data, cube.vertex_ptrs.z, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 4) {
    vertex_pos = VertexIntersection(p[2], p[3], d[2], d[3], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(0, 1, 0));
    vertex_ptr[2] = AllocateVertex(mesh_data, cube.vertex_ptrs.x, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 8) {
    vertex_pos = VertexIntersection(p[3], p[0], d[3], d[0], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(0, 1, 0));
    vertex_ptr[3] = AllocateVertex(mesh_data, cube.vertex_ptrs.z, vertex_pos);
  }

  /// plane y = 0
  if (kEdgeTable[cube_index] & 16) {
    vertex_pos = VertexIntersection(p[4], p[5], d[4], d[5], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(0, 0, 1));
    vertex_ptr[4] = AllocateVertex(mesh_data, cube.vertex_ptrs.x, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 32) {
    vertex_pos = VertexIntersection(p[5], p[6], d[5], d[6], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(1, 0, 0));
    vertex_ptr[5] = AllocateVertex(mesh_data, cube.vertex_ptrs.z, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 64) {
    vertex_pos = VertexIntersection(p[6], p[7], d[6], d[7], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(0, 0, 0));
    vertex_ptr[6] = AllocateVertex(mesh_data, cube.vertex_ptrs.x, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 128) {
    vertex_pos = VertexIntersection(p[7], p[4], d[7], d[4], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(0, 0, 0));
    vertex_ptr[7] = AllocateVertex(mesh_data, cube.vertex_ptrs.z, vertex_pos);
  }

  /// vertical
  if (kEdgeTable[cube_index] & 256) {
    vertex_pos = VertexIntersection(p[4], p[0], d[4], d[0], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(0, 0, 1));
    vertex_ptr[8] = AllocateVertex(mesh_data, cube.vertex_ptrs.y, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 512) {
    vertex_pos = VertexIntersection(p[5], p[1], d[5], d[1], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(1, 0, 1));
    vertex_ptr[9] = AllocateVertex(mesh_data, cube.vertex_ptrs.y, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 1024) {
    vertex_pos = VertexIntersection(p[6], p[2], d[6], d[2], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(1, 0, 0));
    vertex_ptr[10] = AllocateVertex(mesh_data, cube.vertex_ptrs.y,
                                    vertex_pos);
  }
  if (kEdgeTable[cube_index] & 2048) {
    vertex_pos = VertexIntersection(p[7], p[3], d[7], d[3], isolevel);

    MeshCube &cube = GetMeshCube(hash_table, blocks, map_entry,
                                 voxel_local_pos, make_uint3(0, 0, 0));
    vertex_ptr[11] = AllocateVertex(mesh_data, cube.vertex_ptrs.y,
                                    vertex_pos);
  }

  int i = 0;
  for (int t = 0; kTriangleTable[cube_index][t] != -1; t += 3, ++i) {
    int triangle_ptr = this_cube.triangle_ptr[i];

    /// If the cube type is not changed, do not modify triangles,
    /// as they are what they are
    if (kTriangleTable[cube_index][t]
        != kTriangleTable[this_cube.cube_index][t]) {
      if (triangle_ptr == -1) {
        triangle_ptr = mesh_data.AllocTriangle();
      } else { // recycle the rubbish (TODO: more sophisticated operations)
        int3 vertex_ptrs = mesh_data.triangles[triangle_ptr].vertex_ptrs;
        atomicSub(&mesh_data.vertices[vertex_ptrs.x].ref_count, 1);
        atomicSub(&mesh_data.vertices[vertex_ptrs.y].ref_count, 1);
        atomicSub(&mesh_data.vertices[vertex_ptrs.z].ref_count, 1);
      }
    }

    this_cube.triangle_ptr[i] = triangle_ptr;

    Triangle triangle;
    triangle.Clear();
    triangle.vertex_ptrs.x = vertex_ptr[kTriangleTable[cube_index][t + 0]];
    triangle.vertex_ptrs.y = vertex_ptr[kTriangleTable[cube_index][t + 1]];
    triangle.vertex_ptrs.z = vertex_ptr[kTriangleTable[cube_index][t + 2]];

    float3 p0 = mesh_data.vertices[triangle.vertex_ptrs.x].pos;
    float3 p1 = mesh_data.vertices[triangle.vertex_ptrs.y].pos;
    float3 p2 = mesh_data.vertices[triangle.vertex_ptrs.z].pos;

    // TODO: make it more reasonable
    float3 n = normalize(cross((p1 - p0), (p2 - p0)));
    mesh_data.vertices[triangle.vertex_ptrs.x].normal = n;
    mesh_data.vertices[triangle.vertex_ptrs.y].normal = n;
    mesh_data.vertices[triangle.vertex_ptrs.z].normal = n;

    atomicAdd(&mesh_data.vertices[triangle.vertex_ptrs.y].ref_count, 1);
    atomicAdd(&mesh_data.vertices[triangle.vertex_ptrs.x].ref_count, 1);
    atomicAdd(&mesh_data.vertices[triangle.vertex_ptrs.z].ref_count, 1);

    mesh_data.triangles[triangle_ptr] = triangle;
  }
  this_cube.cube_index = cube_index;
}

/// Garbage collection (ref count)
__global__
void RecycleTrianglesKernel(CompactHashTableGPU compact_hash_table,
                            VoxelBlocksGPU      blocks,
                            MeshGPU             mesh_data) {
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];

  const uint local_idx = threadIdx.x;  //inside an SDF block
  MeshCube &cube = blocks[entry.ptr].cubes[local_idx];

  int i = 0;
  for (int t = 0; kTriangleTable[cube.cube_index][t] != -1; t += 3, ++i);

  for (; i < MeshCube::kTrianglePerCube; ++i) {
    int triangle_ptr = cube.triangle_ptr[i];
    if (triangle_ptr == -1) continue;

    int3 vertex_ptrs = mesh_data.triangles[triangle_ptr].vertex_ptrs;
    atomicSub(&mesh_data.vertices[vertex_ptrs.x].ref_count, 1);
    atomicSub(&mesh_data.vertices[vertex_ptrs.y].ref_count, 1);
    atomicSub(&mesh_data.vertices[vertex_ptrs.z].ref_count, 1);

    cube.triangle_ptr[i] = -1;
    mesh_data.triangles[triangle_ptr].Clear();
    mesh_data.FreeTriangle(triangle_ptr);
  }
}

__global__
void RecycleVerticesKernel(CompactHashTableGPU compact_hash_table,
                           VoxelBlocksGPU      blocks,
                           MeshGPU             mesh_data) {
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  MeshCube &cube = blocks[entry.ptr].cubes[local_idx];

  if (cube.vertex_ptrs.x != -1 &&
      mesh_data.vertices[cube.vertex_ptrs.x].ref_count <= 0) {
    mesh_data.vertices[cube.vertex_ptrs.x].Clear();
    mesh_data.FreeVertex(cube.vertex_ptrs.x);
    cube.vertex_ptrs.x = -1;
  }
  if (cube.vertex_ptrs.y != -1 &&
      mesh_data.vertices[cube.vertex_ptrs.y].ref_count <= 0) {
    mesh_data.vertices[cube.vertex_ptrs.y].Clear();
    mesh_data.FreeVertex(cube.vertex_ptrs.y);
    cube.vertex_ptrs.y = -1;
  }
  if (cube.vertex_ptrs.z != -1 &&
      mesh_data.vertices[cube.vertex_ptrs.z].ref_count <= 0) {
    mesh_data.vertices[cube.vertex_ptrs.z].Clear();
    mesh_data.FreeVertex(cube.vertex_ptrs.z);
    cube.vertex_ptrs.z = -1;
  }
}

/// Compress discrete vertices and triangles
__global__
void CollectVerticesAndTrianglesKernel(CompactHashTableGPU compact_hash_table,
                                       VoxelBlocksGPU      blocks,
                                       MeshGPU             mesh,
                                       CompactMeshGPU      compact_mesh) {
  const HashEntry &map_entry = compact_hash_table.compacted_entries[blockIdx.x];
  MeshCube &cube = blocks[map_entry.ptr].cubes[threadIdx.x];

  for (int i = 0; i < MeshCube::kTrianglePerCube; ++i) {
    int triangle_ptr = cube.triangle_ptr[i];
    if (triangle_ptr != -1) {
      int3& triangle = mesh.triangles[triangle_ptr].vertex_ptrs;
      atomicAdd(&compact_mesh.triangles_ref_count[triangle_ptr], 1);
      atomicAdd(&compact_mesh.vertices_ref_count[triangle.x], 1);
      atomicAdd(&compact_mesh.vertices_ref_count[triangle.y], 1);
      atomicAdd(&compact_mesh.vertices_ref_count[triangle.z], 1);
    }
  }
}

__global__
void AssignVertexRemapperKernel(MeshGPU        mesh,
                                CompactMeshGPU compact_mesh) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < kMaxVertexCount && compact_mesh.vertices_ref_count[idx] > 0) {
    int addr = atomicAdd(compact_mesh.vertex_counter, 1);
    compact_mesh.vertex_index_remapper[idx] = addr;
    compact_mesh.vertices[addr] = mesh.vertices[idx].pos;
    compact_mesh.normals[addr]  = mesh.vertices[idx].normal;
  }
}

__global__
void AssignTrianglesKernel(MeshGPU        mesh,
                           CompactMeshGPU compact_mesh) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < kMaxVertexCount && compact_mesh.triangles_ref_count[idx] > 0) {
    int addr = atomicAdd(compact_mesh.triangle_counter, 1);
    compact_mesh.triangles[addr].x
            = compact_mesh.vertex_index_remapper[
            mesh.triangles[idx].vertex_ptrs.x];
    compact_mesh.triangles[addr].y
            = compact_mesh.vertex_index_remapper[
            mesh.triangles[idx].vertex_ptrs.y];
    compact_mesh.triangles[addr].z
            = compact_mesh.vertex_index_remapper[
            mesh.triangles[idx].vertex_ptrs.z];
  }
}

////////////////////
/// Host code
////////////////////
void Map::MarchingCubes() {
  uint occupied_block_count = compact_hash_table_.entry_count();
  LOG(INFO) << "Marching cubes block count: " << occupied_block_count;
  if (occupied_block_count <= 0)
    return;

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  /// Use divide and conquer to avoid read-write conflict
  MarchingCubesKernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
          compact_hash_table_.gpu_data(),
          blocks_.gpu_data(),
          make_uchar3(0, 0, 0), make_uchar3(1, 1, 1),
          mesh_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

#ifdef REDUCTION
  MarchingCubesKernel<<<grid_size, block_size>>>(map->gpu_data(),
          make_uchar3(0, 0, 1), make_uchar3(1, 1, 0),
          mesh_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  MarchingCubesKernel<<<grid_size, block_size>>>(map->gpu_data(),
          make_uchar3(0, 1, 0), make_uchar3(1, 0, 1),
          mesh_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  MarchingCubesKernel<<<grid_size, block_size>>>(map->gpu_data(),
          make_uchar3(1, 0, 0), make_uchar3(0, 1, 1),
          mesh_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
#endif

  RecycleTrianglesKernel<<<grid_size, block_size>>>(
          compact_hash_table_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleVerticesKernel<<<grid_size, block_size>>>(
          compact_hash_table_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

/// Assume this operation is following
/// CollectInFrustumBlocks or
/// CollectAllBlocks
void Map::CompressMesh() {
  compact_mesh_.Reset();

  int occupied_block_count = compact_hash_table_.entry_count();
  if (occupied_block_count <= 0) return;

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size(occupied_block_count, 1);
    const dim3 block_size(threads_per_block, 1);

    CollectVerticesAndTrianglesKernel <<< grid_size, block_size >>> (
            compact_hash_table_.gpu_data(),
            blocks_.gpu_data(),
            mesh_.gpu_data(),
            compact_mesh_.gpu_data());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const uint threads_per_block = 256;
    const dim3 grid_size((kMaxVertexCount + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    AssignVertexRemapperKernel <<< grid_size, block_size >>> (
            mesh_.gpu_data(),
            compact_mesh_.gpu_data());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const uint threads_per_block = 256;
    const dim3 grid_size((kMaxVertexCount + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    AssignTrianglesKernel <<< grid_size, block_size >>> (
            mesh_.gpu_data(),
            compact_mesh_.gpu_data());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  LOG(INFO) << "Vertices: " << compact_mesh_.vertex_count();
  LOG(INFO) << "Triangles: " << compact_mesh_.triangle_count();
}

void Map::SaveMesh(std::string path) {
  LOG(INFO) << "Copying data from GPU";

  CollectAllBlocks();
  CompressMesh();

  uint compact_vertex_count = compact_mesh_.vertex_count();
  uint compact_triangle_count = compact_mesh_.triangle_count();
  LOG(INFO) << "Vertices: " << compact_vertex_count;
  LOG(INFO) << "Triangles: " << compact_triangle_count;

  float3* vertices = new float3[compact_vertex_count];
  float3* normals  = new float3[compact_vertex_count];
  int3* triangles  = new int3  [compact_triangle_count];
  checkCudaErrors(cudaMemcpy(vertices, compact_mesh_.gpu_data().vertices,
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(normals, compact_mesh_.gpu_data().normals,
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(triangles, compact_mesh_.gpu_data().triangles,
                             sizeof(int3) * compact_triangle_count,
                             cudaMemcpyDeviceToHost));

  std::ofstream out(path);
  std::stringstream ss;
  LOG(INFO) << "Writing vertices";
  for (uint i = 0; i < compact_vertex_count; ++i) {
    ss.str("");
    ss <<  "v " << vertices[i].x << " "
       << vertices[i].y << " "
       << vertices[i].z << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Writing normals";
  for (uint i = 0; i < compact_vertex_count; ++i) {
    ss.str("");
    ss <<  "vn " << normals[i].x << " "
       << normals[i].y << " "
       << normals[i].z << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Writing faces";
  for (uint i = 0; i < compact_triangle_count; ++i) {
    ss.str("");
    int3 idx = triangles[i] + make_int3(1);
    ss << "f "
       << idx.x << "//" << idx.x << " "
       << idx.y << "//" << idx.y << " "
       << idx.z << "//" << idx.z << "\n";
    out << ss.str();
  }
  out.close();

  LOG(INFO) << "Finishing vertices";
  delete[] vertices;
  LOG(INFO) << "Finishing normals";
  delete[] normals;
  LOG(INFO) << "Finishing triangles";
  delete[] triangles;
}