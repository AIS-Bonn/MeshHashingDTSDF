//
// Created by wei on 17-10-21.
//

#ifndef CORE_ENTRY_ARRAY_H
#define CORE_ENTRY_ARRAY_H

#include "hash_entry.h"

class EntryArray {
public:
  __host__ EntryArray() = default;
  __host__ explicit EntryArray(uint entry_count);
  // __host__ ~EntryArray();

  __host__ void Alloc(uint entry_count);
  __host__ void Resize(uint entry_count);
  __host__ void Free();

  __host__ uint count();
  __host__ void reset_count();

  __host__ void Reset();

  __host__ __device__
  HashEntry& operator [] (int i) {
    return entries_[i];
  }
  __host__ __device__ uchar& flag(int i) {
    return flags_[i];
  }
  __host__ __device__ int& counter() {
    return counter_[0];
  }

  __host__ HashEntry* GetGPUPtr() const{
    return entries_;
  }
private:
  bool      is_allocated_on_gpu_ = false;
  // @param const element
  uint       entry_count_;
  // @param array
  HashEntry *entries_;
  // @param read-write element
  int       *counter_;       /// atomic counter

  /** Flag dor every entry with dual purpose:
   * 1) Allocation phase: the entry is ORed with the directions, that are applicable (|= (1 << direction))
   * 2) Garbage collection: 1 means remove entry, 0 means do nothing.
   *
   * After each phase the flags are cleared (set to 0)
   *    1) collect_block_array.cu
   *    2) recycle.cu
   */
  uchar     *flags_;
};

#endif //MESH_HASHING_ENTRY_ARRAY_H
