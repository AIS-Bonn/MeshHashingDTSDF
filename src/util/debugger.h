//
// Created by wei on 17-6-4.
//

/// !!!! NOW a test only version !!!
/// Rewrite it entirely !!!
#ifndef VOXEL_HASHING_DEBUGGER_H
#define VOXEL_HASHING_DEBUGGER_H

#include <vector>
#include <unordered_map>
#include "../core/hash_table.h"
#include "../core/block.h"
#include "../core/mesh.h"


struct Hash3D {
  const static int bucket_count = 1000000;

  std::size_t operator()(const int3 &pos) const {
    const int p0 = 73856093;
    const int p1 = 19349669;
    const int p2 = 83492791;

    int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2))
              % (bucket_count);
    if (res < 0) res += (bucket_count);
    return (size_t) res;
  }
};

class Debugger {
private:
  std::unordered_map<int3, Block, Hash3D> block_map_;

  HashEntry *entries_;
  uint *heap_;
  uint *heap_counter__;
  ///           |
  ///           v
  Block *blocks_;
  ///           |
  ///           v
  uint *vertex_heap__;
  uint *vertex_heap_counter__;
  Vertex *vertices_;

  uint *triangle_heap__;
  uint *triangle_heap_counter__;
  Triangle *triangles_;

  int entry_count_;
  int block_count_;
  int vertex_count_;
  int triangle_count_;

  float voxel_size_;

public:
  Debugger(int entry_count, int block_count,
           int vertex_count, int triangle_count,
           float voxel_size);
  ~Debugger();

  void CoreDump(EntryArrayGPU &hash_table);

  void CoreDump(BlockGPUMemory &blocks);

  void CoreDump(MeshGPU &mesh);

  void DebugAll();
};


#endif //VOXEL_HASHING_DEBUGGER_H
