#include "core/block_array.h"
#include "helper_cuda.h"
#include "block_array.h"


#include <glog/logging.h>
#include <device_launch_parameters.h>

////////////////////
/// Device code
////////////////////
__global__
void BlockArrayResetKernel(
    BlockArray blocks,
    CudaMemoryHeap<VoxelArray> &voxel_array_heap,
    int block_count
)
{
  const uint block_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (block_idx < block_count)
  {
    blocks[block_idx].Clear();
  }
}

__device__
VoxelArray &BlockArray::GetVoxelArray(uint block_idx, size_t voxel_array_idx) const
{
  Block &block = blocks_[block_idx];
  int ptr = block.voxel_array_ptrs[voxel_array_idx];
  if (ptr < 0)
  {
    printf("ERROR: Trying to acces VoxelArray %u of block %u which does not exist.", voxel_array_idx, block_idx);
  }
  return voxel_array_heap_.GetElement(ptr);
}

__device__
bool BlockArray::HasVoxelArray(uint block_idx, size_t voxel_array_idx) const
{
  Block &block = blocks_[block_idx];
  return block.voxel_array_ptrs[voxel_array_idx] != FREE_PTR;
}

__device__
int BlockArray::AllocateVoxelArrayWithMutex(
    const uint &block_idx,
    const size_t &voxel_array_idx)
{
  Block &block = blocks_[block_idx];
  int ptr = block.voxel_array_ptrs[voxel_array_idx];
  if (ptr == FREE_PTR)
  {
    int lock = atomicExch(&block.voxel_array_mutexes[voxel_array_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY)
    {
      ptr = voxel_array_heap_.AllocElement();
      if (ptr >= 0)
      {
        block.voxel_array_ptrs[voxel_array_idx] = ptr;
        block.voxel_arrays[voxel_array_idx] = &voxel_array_heap_.GetElement(ptr);
      }
    } // Ensure that it is only allocated once
  }

  return ptr;
}

////////////////////
/// Host code
//////////////////////
__host__
BlockArray::BlockArray(uint block_count)
{
  Resize(block_count);
}

//BlockArray::~BlockArray() {
//  Free();
//}

__host__
void BlockArray::Alloc(uint block_count)
{
  if (!is_allocated_on_gpu_)
  {
    block_count_ = block_count;
    LOG(INFO) << "Allocating " << block_count << " blocks of size " << sizeof(Block) << " Bytes => "
              << sizeof(Block) * block_count << " Bytes in total";
    checkCudaErrors(cudaMalloc(&blocks_, sizeof(Block) * block_count));
    is_allocated_on_gpu_ = true;
  }
}

__host__
void BlockArray::Free()
{
  if (is_allocated_on_gpu_)
  {
    checkCudaErrors(cudaFree(blocks_));
    block_count_ = 0;
    blocks_ = NULL;
    is_allocated_on_gpu_ = false;
  }
}

__host__
void BlockArray::Resize(uint block_count)
{
  if (is_allocated_on_gpu_)
  {
    Free();
    voxel_array_heap_.Free();
  }
  Alloc(block_count);
  voxel_array_heap_.Alloc(static_cast<size_t>(block_count));
  Reset();
}

__host__
void BlockArray::Reset()
{
  if (block_count_ == 0) return;

  voxel_array_heap_.Reset();

  // NOTE: this block is the parallel unit in CUDA, not the data structure Block
  const uint cuda_blocks = (block_count_ + CUDA_THREADS_PER_BLOCK - 1) / CUDA_THREADS_PER_BLOCK;

  const dim3 grid_size(cuda_blocks, 1);
  const dim3 block_size(CUDA_THREADS_PER_BLOCK, 1);

  BlockArrayResetKernel << < grid_size, block_size >> > (*this, voxel_array_heap_, block_count_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}
