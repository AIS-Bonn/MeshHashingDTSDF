#pragma once

#include "core/common.h"
#include <helper_cuda.h>

/**
 * Template implementation of a memory heap in cuda
 *
 * Allows to allocate and free elements of the given template type.
 * IMPORTANT: The type T muss have a function Clear() that wipes the content.
 * @tparam T
 */
template<typename T>
class CudaMemoryHeap
{
public:
  __host__ void Alloc(const size_t number_elements);

  __host__ void Free();

  __host__ void Reset();

  __device__
  uint AllocElement();

  __device__
  void FreeElement(const uint ptr);

  __device__
  T &GetElement(const uint ptr) const;

private:
  uint *heap_;
  uint *heap_counter_;
  T *elements_;
  size_t elements_count_;
};


// Explicit instantiation of required templates (Cuda issue)
#include "core/block.h"

template
class CudaMemoryHeap<VoxelArray>;
