//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_MARCHING_CUBES_H
#define MESH_HASHING_MARCHING_CUBES_H

#include <glog/logging.h>
#include <unordered_map>
#include <chrono>

#include <ctime>
#include "mc_tables.h"
#include "util/timer.h"
#include "engine/main_engine.h"
#include "core/collect_block_array.h"

/**
 * Given a voxel array index and a voxel position, fetches the SDF values of the corner points and
 * computes the MC index.
 *
 * @param entry
 * @param blocks
 * @param hash_table
 * @param geometry_helper
 * @param voxel_pos
 * @param voxel_array_idx
 * @param sdf_array Output array SDF values of 8 voxel corners
 * @param sdf_weight
 * @param mc_index Output marching cubes index
 * @return
 */
__device__
bool GetVoxelSDFValues(const HashEntry &entry,
                       BlockArray &blocks,
                       HashTable &hash_table,
                       GeometryHelper &geometry_helper,
                       int3 voxel_pos,
                       size_t voxel_array_idx,
                       float *sdf_array,
                       float &sdf_weight,
                       short &mc_index);

float MarchingCubes(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    size_t voxel_array_idx,
    Mesh &mesh,
    HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool enable_sdf_gradient,
    bool enable_mc_direction_filtering);

void ClearMesh(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh
);

#endif //MESH_HASHING_MARCHING_CUBES_H
