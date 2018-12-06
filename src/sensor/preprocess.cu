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

__global__
void ComputeNormalMapKernel(float4 *normal, float *depth,
                            uint width, uint height,
                            float fx, float fy, float cx, float cy)
{
  const int ux = blockIdx.x * blockDim.x + threadIdx.x;
  const int uy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ux < 1 or uy < 1 or ux + 1 >= width or uy + 1 >= height)
    return;

  const size_t idx = GetArrayIndex(ux, uy, width);

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

  ComputeNormalMapKernel << < grid_size, block_size >> > (
      normal_data,
//          reinterpret_cast<float *>(depth_img_filtered.data),
          depth_data,
          width,
          height,
          params.fx, params.fy, params.cx, params.cy
  );

  // Filter normal data AFTER normal estimation
  cv::cuda::GpuMat normal_map(height, width, CV_32FC4, normal_data);
  cv::cuda::GpuMat normal_map_filtered;
  cv::cuda::bilateralFilter(normal_map, normal_map_filtered, -1, 5, 5, cv::BORDER_DEFAULT);
  checkCudaErrors(cudaMemcpy(normal_data, normal_map_filtered.data,
                             sizeof(float4) * height * width,
                             cudaMemcpyDeviceToDevice));

  NormalizeNormalsKernel << < grid_size, block_size >> > (normal_data, width);
}
