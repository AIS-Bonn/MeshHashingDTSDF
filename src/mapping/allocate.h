//
// Created by wei on 17-10-22.
//
#pragma once

#include "core/entry_array.h"
#include "core/block_array.h"
#include "core/hash_table.h"
#include "geometry/geometry_helper.h"
#include "sensor/rgbd_sensor.h"

// @function
// See what entries of @param hash_table
// was affected by @param sensor
// with the help of @param geometry_helper
double AllocBlockArray(
    HashTable& hash_table,
    Sensor& sensor,
    GeometryHelper& geometry_helper
);

/** Allocates the first Voxel Arrays for every given Block
 */
__global__
void AllocateVoxelArrayKernel(
    EntryArray candidate_entries,
    uint num_entries,
    BlockArray blocks
);

/** Allocates Voxel Arrays for the given Blocks with respect to the input normal map.
 */
__global__
void AllocateVoxelArrayKernelDirectional(
    EntryArray candidate_entries,
    uint num_entries,
    BlockArray blocks,
    SensorData sensor_data,
    SensorParams sensor_params,
    float4x4 cTw,
    float4x4 wTc,
    GeometryHelper geometry_helper
);
