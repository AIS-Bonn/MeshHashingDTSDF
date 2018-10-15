//
// Created by wei on 17-10-22.
//

#pragma once

#include <glog/logging.h>
#include <unordered_map>
#include <chrono>

#include <ctime>
#include "mc_tables.h"
#include "util/timer.h"
#include "engine/main_engine.h"
#include "core/collect_block_array.h"

float MarchingCubesDirectional(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    Mesh& mesh,
    HashTable& hash_table,
    GeometryHelper& geometry_helper,
    bool enable_sdf_gradient);
