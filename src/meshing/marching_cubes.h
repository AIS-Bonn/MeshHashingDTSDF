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

float MarchingCubes(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    size_t voxel_array_idx,
    Mesh& mesh,
    HashTable& hash_table,
    GeometryHelper& geometry_helper,
    bool enable_sdf_gradient,
    bool enable_mc_direction_filtering);
#endif //MESH_HASHING_MARCHING_CUBES_H
