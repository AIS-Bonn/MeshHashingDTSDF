#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include <extern/cuda/matrix.h>
#include <geometry/geometry_helper.h>
#include <device_launch_parameters.h>
#include <extern/cuda/helper_cuda.h>
#include "core/params.h"
#include "preprocess.h"

__global__
void ResetInlierRatioKernel(
    float *inlier_ratio,
    int width,
    int height
)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  const int idx = y * width + x;
  /// Convert mm -> m
  inlier_ratio[idx] = 0.1f;
}

__global__
void ConvertDepthFormatKernel(
    float *dst, short *src,
    uint width, uint height,
    float range_factor,
    float min_depth_range,
    float max_depth_range
)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  const int idx = y * width + x;
  /// Convert mm -> m
  const float depth = range_factor * src[idx];
  bool is_valid = (depth >= min_depth_range && depth <= max_depth_range);
  dst[idx] = is_valid ? depth : MINF;
}

__global__
void ConvertColorFormatKernel(float4 *dst, uchar4 *src,
                              uint width, uint height)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  const int idx = y * width + x;

  uchar4 c = src[idx];
  bool is_valid = (c.x != 0 && c.y != 0 && c.z != 0);
  dst[idx] = is_valid ? make_float4(c.z / 255.0f, c.y / 255.0f,
                                    c.x / 255.0f, c.w / 255.0f)
                      : make_float4(MINF, MINF, MINF, MINF);
}

__device__
size_t GetArrayIndex(int x, int y, int width)
{
  return static_cast<size_t>(y * width + x);
}

__global__
void NormalizeNormalsKernel(float4 *normal, uint width)
{
  const int ux = blockIdx.x * blockDim.x + threadIdx.x;
  const int uy = blockIdx.y * blockDim.y + threadIdx.y;

  const size_t idx = GetArrayIndex(ux, uy, width);

  normal[idx] = make_float4(normalize(make_float3(normal[idx])), 1.0f);
}

/**
 * Implementation from BundleFusion
 * Copyright (c) 2017 by Angela Dai and Matthias Niessner
 */
inline __device__ float gaussD(float sigma, int x, int y)
{
  return exp(-((x * x + y * y) / (2.0f * sigma * sigma)));
}

/**
 * Implementation from BundleFusion
 * Copyright (c) 2017 by Angela Dai and Matthias Niessner
 */
inline __device__ float gaussR(float sigma, float dist)
{
  return exp(-(dist * dist) / (2.0 * sigma * sigma));
}

/**
 * Implementation from BundleFusion
 * Copyright (c) 2017 by Angela Dai and Matthias Niessner
 *
 * @param input
 * @param output
 * @param sigma_d
 * @param sigma_r
 */
__global__
void BilateralFilterKernel(float4 *input, float4 *output, float sigma_d, float sigma_r, uint width, uint height)
{
  const int ux = blockIdx.x * blockDim.x + threadIdx.x;
  const int uy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ux >= width or uy >= height)
    return;

  const uint idx = uy * width + ux;

  output[idx] = make_float4(MINF);

  const float4 center = input[idx];
  if (center.x == MINF or center.y == MINF or center.z == MINF or center.w == MINF)
    return;

  float4 sum = make_float4(0.0f);
  float sum_weight = 0.0f;

  const uint radius = (uint) ceil(2.0 * sigma_d);
  for (int i = ux - radius; i <= ux + radius; i++)
  {
    for (int j = uy - radius; j <= uy + radius; j++)
    {
      if (i < 0 or j < 0 or i >= width or j >= height)
        continue;

      const float4 value = input[j * width + i];

      if (value.x == MINF or value.y == MINF or value.z == MINF or value.w == MINF)
        continue;

      const float weight = gaussD(sigma_d, i - ux, j - uy) * gaussR(sigma_r, length(value - center));

      sum += weight * value;
      sum_weight += weight;
    }
  }

  if (sum_weight >= 0.0f)
  {
    output[idx] = sum / sum_weight;
  }
}

/**
 * Implementation from BundleFusion
 * Copyright (c) 2017 by Angela Dai and Matthias Niessner
 *
 * @param input
 * @param output
 * @param sigma_d
 * @param sigma_r
 */
__global__
void BilateralFilterKernelFloat(float *input, float *output, float sigma_d, float sigma_r, uint width, uint height)
{
  const int ux = blockIdx.x * blockDim.x + threadIdx.x;
  const int uy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ux >= width or uy >= height)
    return;

  const uint idx = uy * width + ux;

  output[idx] = MINF;

  const float center = input[idx];
  if (center == MINF)
    return;

  float sum = 0.0f;
  float sum_weight = 0.0f;

  const uint radius = (uint) ceil(2.0 * sigma_d);
  for (int i = ux - radius; i <= ux + radius; i++)
  {
    for (int j = uy - radius; j <= uy + radius; j++)
    {
      if (i < 0 or j < 0 or i >= width or j >= height)
        continue;

      const float value = input[j * width + i];

      if (value == MINF)
        continue;

      const float weight = gaussD(sigma_d, i - ux, j - uy) * gaussR(sigma_r, abs(value - center));

      sum += weight * value;
      sum_weight += weight;
    }
  }

  if (sum_weight >= 0.0f)
  {
    output[idx] = sum / sum_weight;
  }
}

__global__
void ComputeNormalMapKernel(float4 *normal, float *depth,
                            uint width, uint height,
                            float fx, float fy, float cx, float cy)
{
  const int ux = blockIdx.x * blockDim.x + threadIdx.x;
  const int uy = blockIdx.y * blockDim.y + threadIdx.y;

  const size_t idx = GetArrayIndex(ux, uy, width);

  if (ux < 1 or uy < 1 or ux + 1 >= width or uy + 1 >= height)
  {
    if (ux == 1 or uy == 1 or ux + 1 == width or uy + 1 == height)
      normal[idx] = make_float4(0);
    return;
  }

  float3 normal_ = make_float3(0);
  float3 center = GeometryHelper::ImageReprojectToCamera(ux, uy, depth[idx], fx, fy, cx, cy);
  float3 neighbors[8];
  size_t count = 0;
  static const int2 coords[8] = {{-1, -1},
                                 {0,  -1},
                                 {1,  -1},
                                 {1,  0},
                                 {1,  1},
                                 {0,  1},
                                 {-1, 1},
                                 {-1, 0}};
  for (int i = 0; i < 8; i++)
  {
    int u = ux + coords[i].x;
    int v = uy + coords[i].y;
    float depth_value = depth[GetArrayIndex(u, v, width)];
    if (depth_value == 0.0f or depth_value == MINF)
    {
      normal[idx] = make_float4(0);
      return;
    }
    neighbors[count] = GeometryHelper::ImageReprojectToCamera(u, v, depth_value, fx, fy, cx, cy);
    count++;
  }

  for (int i = 0; i < count; i++)
  {
    float3 n = normalize(cross(neighbors[i] - center, neighbors[(i + 1) % 4] - center));
    if (n.z > 0) // This is an outlier case caused by faulty depth data!
      continue;

    normal_ += n;
  }

  normal[idx] = make_float4(normalize(make_float3(normal_.x, normal_.y, normal_.z)), 0.0f);
}

//////////
/// Member function: (CPU calling GPU kernels)
__host__
void ResetInlierRatio(
    float *inlier_ratio,
    SensorParams &params
)
{
  uint width = params.width;
  uint height = params.height;

  const uint threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1) / threads_per_block,
                       (height + threads_per_block - 1) / threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);
  ResetInlierRatioKernel << < grid_size, block_size >> > (
      inlier_ratio, width, height);
}

__host__
void ConvertDepthFormat(
    cv::Mat &depth_img,
    short *depth_buffer,
    float *depth_data,
    SensorParams &params
)
{
  /// First copy cpu data in to cuda short
  uint width = params.width;
  uint height = params.height;
  uint image_size = width * height;

  checkCudaErrors(cudaMemcpy(depth_buffer, (short *) depth_img.data,
                             sizeof(short) * image_size,
                             cudaMemcpyHostToDevice));

  const uint threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1) / threads_per_block,
                       (height + threads_per_block - 1) / threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  ConvertDepthFormatKernel << < grid_size, block_size >> > (
      depth_data,
          depth_buffer,
          width, height,
          params.range_factor,
          params.min_depth_range,
          params.max_depth_range);

//  float *depth_tmp;
//  checkCudaErrors(cudaMalloc(&depth_tmp, sizeof(float) * width * height));
//  ConvertDepthFormatKernel << < grid_size, block_size >> > (
//      depth_tmp,
//          depth_buffer,
//          width, height,
//          params.range_factor,
//          params.min_depth_range,
//          params.max_depth_range);
//  BilateralFilterKernelFloat << < grid_size, block_size >> > (
//      depth_tmp,
//          depth_data,
//          5,
//          5,
//          width,
//          height
//  );
//  checkCudaErrors(cudaFree(depth_tmp));
}

__host__
void ConvertColorFormat(
    cv::Mat &color_img,
    uchar4 *color_buffer,
    float4 *color_data,
    SensorParams &params
)
{

  uint width = params.width;
  uint height = params.height;
  uint image_size = width * height;

  checkCudaErrors(cudaMemcpy(color_buffer, color_img.data,
                             sizeof(uchar4) * image_size,
                             cudaMemcpyHostToDevice));

  const int threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1) / threads_per_block,
                       (height + threads_per_block - 1) / threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  ConvertColorFormatKernel << < grid_size, block_size >> > (
      color_data,
          color_buffer,
          width,
          height);
}

__host__
void ComputeNormalMap(
    float *depth_data,
    float4 *normal_data,
    SensorParams &params
)
{
  uint width = params.width;
  uint height = params.height;

  const int threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1) / threads_per_block,
                       (height + threads_per_block - 1) / threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  // Filter depth image BEFORE normal estimation
//  cv::cuda::GpuMat depth_img(height, width, CV_32FC1, depth_data);
//  cv::cuda::GpuMat depth_img_filtered;
//  cv::cuda::bilateralFilter(depth_img, depth_img_filtered, -1, 5, 5, cv::BORDER_DEFAULT);

  float4 *normals_tmp;
  checkCudaErrors(cudaMalloc(&normals_tmp, sizeof(float4) * width * height));
  ComputeNormalMapKernel << < grid_size, block_size >> > (
      normals_tmp,
//          reinterpret_cast<float *>(depth_img_filtered.data),
          depth_data,
          width,
          height,
          params.fx, params.fy, params.cx, params.cy
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  BilateralFilterKernel << < grid_size, block_size >> > (
      normals_tmp,
          normal_data,
          2,
          2,
          width,
          height
  );
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaFree(normals_tmp));

  // Filter normal data AFTER normal estimation
//  cv::cuda::GpuMat normal_map(height, width, CV_32FC4, normal_data);
//  cv::cuda::GpuMat normal_map_filtered;
//  cv::cuda::bilateralFilter(normal_map, normal_map_filtered, -1, 5, 5, cv::BORDER_DEFAULT);
//  checkCudaErrors(cudaMemcpy(normal_data, normal_map_filtered.data,
//                             sizeof(float4) * height * width,
//                             cudaMemcpyDeviceToDevice));

  NormalizeNormalsKernel << < grid_size, block_size >> > (normal_data, width);
}
