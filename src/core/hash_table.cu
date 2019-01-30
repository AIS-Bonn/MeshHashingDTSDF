#include <unordered_set>
#include <device_launch_parameters.h>

#include "core/hash_table.h"

////////////////////
/// Device code
////////////////////
__global__
void HashTableResetBucketMutexesKernel(
    int *bucket_mutexes,
    uint bucket_count
) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < bucket_count) {
    bucket_mutexes[idx] = FREE_ENTRY;
  }
}

__global__
void HashTableResetHeapKernel(
    uint *heap,
    uint value_capacity
) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < value_capacity) {
    heap[idx] = value_capacity - idx - 1;
  }
}

__global__
void HashTableResetEntriesKernel(
    HashEntry *entries,
    uint entry_count
) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < entry_count) {
    entries[idx].Clear();
  }
}

__device__
HashEntry HashTable::GetEntry(const int3& pos) const {
  uint bucket_idx             = HashBucketForBlockPos(pos);
  uint bucket_first_entry_idx = bucket_idx * bucket_size;

  HashEntry entry;
  entry.pos    = pos;
  entry.offset = 0;
  entry.ptr    = FREE_ENTRY;

  for (uint i = 0; i < bucket_size; ++i) {
    HashEntry curr_entry = entries_[i + bucket_first_entry_idx];
    if (IsPosAllocated(pos, curr_entry)) {
      return curr_entry;
    }
  }

  /// The last entry is visted twice, but it's OK
#ifdef HANDLE_COLLISIONS
  const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
  int i = bucket_last_entry_idx;

#pragma unroll 1
  for (uint iter = 0; iter < linked_list_size; ++iter) {
    HashEntry curr_entry = entries_[i];

    if (IsPosAllocated(pos, curr_entry)) {
      return curr_entry;
    }
    if (curr_entry.offset == 0) {
      break;
    }
    i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
  }
#endif
  return entry;
}

//pos in SDF block coordinates
__device__
void HashTable::AllocEntry(const int3& pos) {
  uint bucket_idx             = HashBucketForBlockPos(pos);		//hash bucket
  uint bucket_first_entry_idx = bucket_idx * bucket_size;	//hash position

  /// 1. Try GetEntry, meanwhile collect an empty entry potentially suitable
  int empty_entry_idx = -1;
  for (uint j = 0; j < bucket_size; j++) {
    uint i = j + bucket_first_entry_idx;
    const HashEntry& curr_entry = entries_[i];
    if (IsPosAllocated(pos, curr_entry)) {
      return;
    }

    /// wei: should not break and alloc before a thorough searching is over:
    if (empty_entry_idx == -1 && curr_entry.ptr == FREE_ENTRY) {
      empty_entry_idx = i;
    }
  }

#ifdef HANDLE_COLLISIONS
  const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
  uint i = bucket_last_entry_idx;
  for (uint iter = 0; iter < linked_list_size; ++iter) {
    HashEntry curr_entry = entries_[i];

    if (IsPosAllocated(pos, curr_entry)) {
      return;
    }
    if (curr_entry.offset == 0) {
      break;
    }
    i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
  }
#endif

  /// 2. NOT FOUND, Allocate
  if (empty_entry_idx != -1) {
    int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY) {
      HashEntry& entry = entries_[empty_entry_idx];
      entry.pos    = pos;
      entry.ptr    = Alloc();
      entry.offset = NO_OFFSET;
    }
    return;
  }

#ifdef HANDLE_COLLISIONS
  i = bucket_last_entry_idx;
  int offset = 0;

#pragma  unroll 1
  for (uint iter = 0; iter < linked_list_size; ++iter) {
    offset ++;
    if ((offset % bucket_size) == 0) continue;

    i = (bucket_last_entry_idx + offset) % (entry_count);

    HashEntry& curr_entry = entries_[i];

    if (curr_entry.ptr == FREE_ENTRY) {
      int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
      if (lock != LOCK_ENTRY) {
        HashEntry& bucket_last_entry = entries_[bucket_last_entry_idx];
        uint alloc_bucket_idx = i / bucket_size;

        lock = atomicExch(&bucket_mutexes_[alloc_bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          HashEntry& entry = entries_[i];
          entry.pos    = pos;
          entry.offset = bucket_last_entry.offset; // pointer assignment in linked list
          entry.ptr    = Alloc();	//memory alloc

          // Not sure if it is ok to directly assign to reference
          bucket_last_entry.offset = offset;
          entries_[bucket_last_entry_idx] = bucket_last_entry;
        }
      }
      return;	//bucket was already locked
    }
  }
#endif
}

//! deletes a hash entry position for a given pos index
// returns true uppon successful deletion; otherwise returns false
__device__
bool HashTable::FreeEntry(const int3& pos) {
  uint bucket_idx = HashBucketForBlockPos(pos);	//hash bucket
  uint bucket_first_entry_idx = bucket_idx * bucket_size;		//hash position

  for (uint j = 0; j < bucket_size; j++) {
    uint i = j + bucket_first_entry_idx;
    const HashEntry& curr = entries_[i];
    if (IsPosAllocated(pos, curr)) {

#ifndef HANDLE_COLLISIONS
      Free(curr.ptr);
        entries_[i].Clear();
        return true;
#else
      // Deal with linked list: curr = curr->next
      if (curr.offset != 0) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          Free(curr.ptr);
          int next_idx = (i + curr.offset) % (entry_count);
          entries_[i] = entries_[next_idx];
          entries_[next_idx].Clear();
          return true;
        } else {
          return false;
        }
      } else {
        Free(curr.ptr);
        entries_[i].Clear();
        return true;
      }
#endif
    }
  }

#ifdef HANDLE_COLLISIONS
  // Init with linked list traverse
  const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
  int i = bucket_last_entry_idx;
  HashEntry& curr = entries_[i];

  int prev_idx = i;
  i = (bucket_last_entry_idx + curr.offset) % (entry_count);

#pragma unroll 1
  for (uint iter = 0; iter < linked_list_size; ++iter) {
    curr = entries_[i];

    if (IsPosAllocated(pos, curr)) {
      int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
      if (lock != LOCK_ENTRY) {
        Free(curr.ptr);
        entries_[i].Clear();
        HashEntry prev = entries_[prev_idx];
        prev.offset = curr.offset;
        entries_[prev_idx] = prev;
        return true;
      } else {
        return false;
      }
    }

    if (curr.offset == 0) {	//we have found the end of the list
      return false;	//should actually never happen because we need to find that guy before
    }

    prev_idx = i;
    i = (bucket_last_entry_idx + curr.offset) % (entry_count);
  }
#endif	// HANDLE_COLLSISION
  return false;
}

//! see Teschner et al. (but with correct prime values)
__device__ uint HashTable::HashBucketForBlockPos(const int3& pos) const {
  const int p0 = 73856093;
  const int p1 = 19349669;
  const int p2 = 83492791;

  int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2))
            % bucket_count;
  if (res < 0) res += bucket_count;
  return (uint) res;
}

__device__
bool HashTable::IsPosAllocated(const int3& pos, const HashEntry& hash_entry) const {
  return pos.x == hash_entry.pos.x
         && pos.y == hash_entry.pos.y
         && pos.z == hash_entry.pos.z
         && hash_entry.ptr != FREE_ENTRY;
}

__device__ uint HashTable::Alloc() {
  uint addr = atomicSub(&heap_counter_[0], 1);
  if (addr < MEMORY_LIMIT) {
    printf("Memory nearly exhausted! %d -> %d\n", addr, heap_[addr]);
  }
  return heap_[addr];
}

__device__ void HashTable::Free(uint ptr) {
  uint addr = atomicAdd(&heap_counter_[0], 1);
  heap_[addr + 1] = ptr;
}

////////////////////
/// Host code
////////////////////
HashTable::HashTable(const HashParams &params) {
  Alloc(params);
  Reset();
}

//HashTable::~HashTable() {
//  Free();
//}

void HashTable::Alloc(const HashParams &params) {
  if (!is_allocated_on_gpu_) {
    /// Parameters
    bucket_count = params.bucket_count;
    bucket_size = params.bucket_size;
    entry_count = params.entry_count;
    value_capacity = params.max_block_count;
    linked_list_size = params.linked_list_size;

    /// Values
    checkCudaErrors(cudaMalloc(&heap_,
                               sizeof(uint) * params.max_block_count));
    checkCudaErrors(cudaMalloc(&heap_counter_,
                               sizeof(uint)));

    /// Entries
    checkCudaErrors(cudaMalloc(&entries_,
                               sizeof(HashEntry) * params.entry_count));

    /// Mutexes
    checkCudaErrors(cudaMalloc(&bucket_mutexes_,
                               sizeof(int) * params.bucket_count));
    is_allocated_on_gpu_ = true;
  }
}

void HashTable::Free() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(heap_));
    checkCudaErrors(cudaFree(heap_counter_));

    checkCudaErrors(cudaFree(entries_));
    checkCudaErrors(cudaFree(bucket_mutexes_));

    is_allocated_on_gpu_ = false;
  }
}

void HashTable::Resize(const HashParams &params) {
  Alloc(params);
  Reset();
}
/// Reset
void HashTable::Reset() {
  /// Reset mutexes
  ResetMutexes();

  {
    /// Reset entries
    const int threads_per_block = 64;
    const dim3 grid_size((entry_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    HashTableResetEntriesKernel <<<grid_size, block_size>>>(entries_, entry_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    /// Reset allocated memory
    uint heap_counter_init = value_capacity - 1;
    checkCudaErrors(cudaMemcpy(heap_counter_, &heap_counter_init,
                               sizeof(uint),
                               cudaMemcpyHostToDevice));

    const int threads_per_block = 64;
    const dim3 grid_size((value_capacity + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    HashTableResetHeapKernel <<<grid_size, block_size>>>(heap_, value_capacity);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

void HashTable::ResetMutexes() {
  const int threads_per_block = 64;
  const dim3 grid_size((bucket_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  HashTableResetBucketMutexesKernel <<<grid_size, block_size>>>(bucket_mutexes_, bucket_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

/// Member function: Others
//void HashTable::Debug() {
//  HashEntry *entries = new HashEntry[hash_params_.bucket_size * hash_params_.bucket_count];
//  uint *heap_ = new uint[hash_params_.max_block_count];
//  uint  heap_counter_;
//
//  checkCudaErrors(cudaMemcpy(&heap_counter_, heap_counter_, sizeof(uint), cudaMemcpyDeviceToHost));
//  heap_counter_++; //points to the first free entry: number of blocks is one more
//
//  checkCudaErrors(cudaMemcpy(heap_, heap_,
//                             sizeof(uint) * hash_params_.max_block_count,
//                             cudaMemcpyDeviceToHost));
//  checkCudaErrors(cudaMemcpy(entries, entries,
//                             sizeof(HashEntry) * hash_params_.bucket_size * hash_params_.bucket_count,
//                             cudaMemcpyDeviceToHost));
////  checkCudaErrors(cudaMemcpy(values, values,
////                             sizeof(T) * hash_params_.value_capacity,
////                             cudaMemcpyDeviceToHost));
//
//  LOG(INFO) << "GPU -> CPU data transfer finished";
//
//  //Check for duplicates
//  class Entry {
//  public:
//    Entry() {}
//    Entry(int x_, int y_, int z_, int i_, int offset_, int ptr_) :
//            x(x_), y(y_), z(z_), i(i_), offset(offset_), ptr(ptr_) {}
//    ~Entry() {}
//
//    bool operator< (const Entry &other) const {
//      if (x == other.x) {
//        if (y == other.y) {
//          return z < other.z;
//        } return y < other.y;
//      } return x < other.x;
//    }
//
//    bool operator== (const Entry &other) const {
//      return x == other.x && y == other.y && z == other.z;
//    }
//
//    int x, y, z, i;
//    int offset;
//    int ptr;
//  };
//
//  /// Iterate over free heap_
//  std::unordered_set<uint> free_heap_index;
//  std::vector<int> free_value_index(hash_params_.max_block_count, 0);
//  for (uint i = 0; i < heap_counter_; i++) {
//    free_heap_index.insert(heap_[i]);
//    free_value_index[heap_[i]] = FREE_ENTRY;
//  }
//  if (free_heap_index.size() != heap_counter_) {
//    LOG(ERROR) << "heap_ check invalid";
//  }
//
//  uint not_free_entry_count = 0;
//  uint not_locked_entry_count = 0;
//
//  /// Iterate over entries
//  std::list<Entry> l;
//  uint entry_count = hash_params_.count;
//  for (uint i = 0; i < count; i++) {
//    if (entries[i].ptr != LOCK_ENTRY) {
//      not_locked_entry_count++;
//    }
//
//    if (entries[i].ptr != FREE_ENTRY) {
//      not_free_entry_count++;
//      Entry occupied_entry(entries[i].pos.x, entries[i].pos.y, entries[i].pos.z,
//                           i, entries[i].offset, entries[i].ptr);
//      l.push_back(occupied_entry);
//
//      if (free_heap_index.find(occupied_entry.ptr) != free_heap_index.end()) {
//        LOG(ERROR) << "ERROR: ptr is on free heap_, but also marked as an allocated entry";
//      }
//      free_value_index[entries[i].ptr] = LOCK_ENTRY;
//    }
//  }
//
//  /// Iterate over values
//  uint free_value_count = 0;
//  uint not_free_value_count = 0;
//  for (uint i = 0; i < hash_params_.max_block_count; i++) {
//    if (free_value_index[i] == FREE_ENTRY) {
//      free_value_count++;
//    } else if (free_value_index[i] == LOCK_ENTRY) {
//      not_free_value_count++;
//    } else {
//      LOG(ERROR) << "memory leak detected: neither free nor allocated";
//      return;
//    }
//  }
//
//  if (free_value_count + not_free_value_count == hash_params_.max_block_count)
//    LOG(INFO) << "heap_ OK!";
//  else {
//    LOG(ERROR) << "heap_ CORRUPTED";
//    return;
//  }
//
//  l.sort();
//  size_t size_before = l.size();
//  l.unique();
//  size_t size_after = l.size();
//
//
//  LOG(INFO) << "Duplicated entry count: " << size_before - size_after;
//  LOG(INFO) << "Not locked entry count: " << not_locked_entry_count;
//  LOG(INFO) << "Not free value count: " << not_free_value_count
//            << "; free value count: " << free_value_count;
//  LOG(INFO) << "not_free + free entry count: "
//            << not_free_value_count + free_value_count;
//
//  delete [] entries;
//  //delete [] values;
//  delete [] heap_;
//}