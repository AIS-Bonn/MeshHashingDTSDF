//
// Created by wei on 17-4-28.
//

#ifndef CORE_HASH_TABLE_H
#define CORE_HASH_TABLE_H

#include "helper_cuda.h"
#include "helper_math.h"

#include "core/common.h"
#include "core/params.h"
#include "core/hash_entry.h"
#include "geometry/geometry_helper.h"

class HashTable {
public:
  /// Parameters
  uint      bucket_count;
  uint      bucket_size;
  uint      entry_count;
  uint      value_capacity;
  uint      linked_list_size;

  __host__ HashTable() = default;
  __host__ explicit HashTable(const HashParams &params);
  // ~HashTable();
  __host__ void Alloc(const HashParams &params);
  __host__ void Free();

__host__ void Resize(const HashParams &params);
  __host__ void Reset();
  __host__ void ResetMutexes();

  __host__ __device__ HashEntry& entry(uint i) {
    return entries_[i];
  }
  //__host__ void Debug();

  /////////////////
  // Device part //

private:
  bool  is_allocated_on_gpu_ = false;
  // @param array
  uint      *heap_;             /// index to free values
  // @param read-write element
  uint      *heap_counter_;     /// single element; used as an atomic counter (points to the next free block)

  // @param array
  HashEntry *entries_;          /// hash entries that stores pointers to sdf values
  // @param array
  int       *bucket_mutexes_;   /// binary flag per hash bucket; used for allocation to atomically lock a bucket

  __device__ uint HashBucketForBlockPos(const int3& pos) const;

  __device__ bool IsPosAllocated(const int3& pos, const HashEntry& hash_entry) const;

  __device__ uint Alloc();

  __device__ void Free(uint ptr);

public:
  __device__ HashEntry GetEntry(const int3& pos) const;

  __device__ void AllocEntry(const int3& pos);

  __device__ bool FreeEntry(const int3& pos);
};

#endif //VH_HASH_TABLE_H
