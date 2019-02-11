//
// Created by wei on 17-10-22.
//

#ifndef CORE_COLLECT_H
#define CORE_COLLECT_H

#include "core/entry_array.h"
#include "core/hash_table.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"

/**
 * Read the entries in hash_table
 * Write to the candidate_entries (for parallel computation)
 */
void CollectAllBlocks(
    HashTable &hash_table,
    EntryArray &candidate_entries
);

/**
 * Read the entries in hash_table
 * Filter the positions with sensor info (pose and params),
 *                       and geometry helper
 * Write to the candidate_entries (for parallel computation)
 */
double CollectBlocksInFrustum(
    HashTable &hash_table,
    Sensor &sensor,
    GeometryHelper &geometry_helper,
    EntryArray &candidate_entries
);

/**
 * Collect all blocks flagged for update during allocation in candidate_entries and
 * write them in sequential order for fast processing
 */
double CollectFlaggedBlocks(
    HashTable &hash_table,
    EntryArray &candidate_entries
);

#endif //CORE_COLLECT_H
