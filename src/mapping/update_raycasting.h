#pragma once

#include "core/hash_table.h"
#include "core/block_array.h"
#include "core/entry_array.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"

class MainEngine;

double UpdateRaycasting(
    EntryArray &candidate_entries,
    Sensor &sensor,
    MainEngine &main_engine
);
