//
// Created by wei on 17-10-22.
//
#pragma once

#include "core/entry_array.h"
#include "core/block_array.h"
#include "core/hash_table.h"
#include "geometry/geometry_helper.h"
#include "sensor/rgbd_sensor.h"

class MainEngine;

// @function
// See what entries of @param hash_table
// was affected by @param sensor
// with the help of @param geometry_helper
/**
 * Check which blocks are affected by the current sensor readings and allocate new blocks, if necessary.
 * Also sets flags in the candidate_entries array for collecting blocks which are affected by the update.
 *
 * @param hash_table
 * @param sensor
 * @param runtime_params
 * @param geometry_helper
 * @param candidate_entries
 * @return Runtime in s
 */
double AllocBlockArray(
    EntryArray candidate_entries,
    Sensor &sensor,
    MainEngine &main_engine
);

/**
 * Allocates the Voxel arrays for the candidate entries, if necessary
 * @param candidate_entries
 * @param blocks
 * @param runtime_params
 * @return Runtime in s
 */
double AllocVoxelArray(
    EntryArray candidate_entries,
    Sensor &sensor,
    MainEngine &main_engine
);
