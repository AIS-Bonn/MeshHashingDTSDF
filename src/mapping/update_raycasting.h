#pragma once

#include "core/hash_table.h"
#include "core/block_array.h"
#include "core/entry_array.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"

double UpdateRaycasting(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    Sensor& sensor,
    const RuntimeParams &runtime_params,
    HashTable& hash_table,
    GeometryHelper& geometry_helper
);
