#include "core/cuda_memory_heap.h"
#include "cuda_memory_heap.h"


#include <typeinfo>
#include <glog/logging.h>

////////////////////
/// Device code
////////////////////
template<typename T>
__global__
void ResetHeapKernel(uint *heap, T *elements, int max_count)
{
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < max_count)
  {
    heap[idx] = max_count - idx - 1;
    elements[idx].Clear();
  }
}

template<typename T>
__device__
uint CudaMemoryHeap<T>::AllocElement()
{
  uint addr = atomicSub(heap_counter_, 1);
  if (addr < MEMORY_LIMIT)
  {
    printf("WARNING %s heap out of memory: %d/%d left\n", __PRETTY_FUNCTION__, addr, heap_[addr]);
  }
  return heap_[addr];
}

template<typename T>
__device__
void CudaMemoryHeap<T>::FreeElement(const uint ptr)
{
  uint addr = atomicAdd(heap_counter_, 1) + 1;
  heap_[addr] = ptr;
}

template<typename T>
__device__
T &CudaMemoryHeap<T>::GetElement(const uint ptr) const
{
  return elements_[ptr];
}

template<typename T>
void CudaMemoryHeap<T>::CopyPtrsToHost(uint *array, uint &counter)
{
  checkCudaErrors(cudaMemcpy(&counter, heap_counter_, sizeof(uint), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(array, heap_, elements_count_ * sizeof(uint), cudaMemcpyDeviceToHost));
}

////////////////////
/// Host code
////////////////////

template<typename T>
__host__
void CudaMemoryHeap<T>::Alloc(const size_t number_elements)
{
  elements_count_ = number_elements;
  LOG(INFO) << "Allocating " << number_elements << " elements of size "
            << sizeof(T) << " Bytes each => " << sizeof(T) * number_elements << " Bytes in total";
  checkCudaErrors(cudaMalloc(&elements_, sizeof(T) * number_elements));
  checkCudaErrors(cudaMalloc(&heap_, sizeof(uint) * number_elements));
  checkCudaErrors(cudaMalloc(&heap_counter_, sizeof(uint)));

  Reset();
}

template<typename T>
__host__
void CudaMemoryHeap<T>::Reset()
{
  uint val = elements_count_ - 1;
  checkCudaErrors(cudaMemcpy(heap_counter_,
                             &val,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));

  {
    const int threads_per_block = 64;
    const dim3 grid_size((elements_count_ + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 cuda_block_size(threads_per_block, 1);

    ResetHeapKernel << < grid_size, cuda_block_size >> > (heap_, elements_,
        elements_count_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

template<typename T>
__host__
void CudaMemoryHeap<T>::Free()
{

}
