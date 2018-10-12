//
// Created by splietke on 12.10.18.
//

#ifndef MESH_HASHING_UTILS_H
#define MESH_HASHING_UTILS_H

inline void CheckCUDADevices()
{
  int N;
  cudaGetDeviceCount(&N);

  if (N == 0)
  {
    LOG(ERROR) << "No CUDA capable devices found";
    exit(1);
  }

  for (int i = 0; i < N; i++)
  {
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, i);
    printf("CUDA Device (%i)\n", i);
    printf("\tDevice name: %s\n", properties.name);
    printf("\tCompute capability: %i.%i\n", properties.major, properties.minor);
    printf("\tMemory total: %lu MB\n", properties.totalGlobalMem / (1024 * 1024));
    printf("\tMultiprocessor count: %i\n", properties.multiProcessorCount);
    printf("\tMax Threads per Block: %i, Max Threads per Dimension: %i\n", properties.maxThreadsPerBlock, properties.maxThreadsDim[0]);

  }
}


#endif //MESH_HASHING_UTILS_H
