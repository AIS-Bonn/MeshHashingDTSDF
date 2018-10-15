#include <opencv2/opencv.hpp>
#include <helper_cuda.h>
#include <geometry/geometry_helper.h>
#include <device_launch_parameters.h>
#include <extern/cuda/helper_cuda.h>
#include "core/params.h"
#include "preprocess.h"

#define MINF __int_as_float(0xff800000)

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
void ComputeNormalMapKernel(float3 *normal, float *depth,
                            uint width, uint height,
                            float fx, float fy, float cx, float cy)
{
  const int ux = blockIdx.x * blockDim.x + threadIdx.x;
  const int uy = blockIdx.y * blockDim.y + threadIdx.y;

  if (ux < 1 or uy < 1 or ux + 1 >= width or uy + 1 >= height)
    return;

  const size_t idx = GetArrayIndex(ux, uy, width);

  float3 points[5];
  static const int coords[2][5] = {{0, -1, 0,  1, 0},
                                 {0, 0,  -1, 0, +1}};
#pragma unroll 5
  for (int i = 0; i < 5; i++)
  {
    int u = ux + coords[0][i];
    int v = uy + coords[1][i];
    float depth_value = depth[GetArrayIndex(u, v, width)];
    if (depth_value == 0.0f or depth_value == MINF)
    {
      normal[idx] = make_float3(0);
      return;
    }
    points[i] = GeometryHelper::ImageReprojectToCamera(u, v, depth_value, fx, fy, cx, cy);
  }

#pragma unroll 4
  for (int i = 0; i < 4; i++)
  {
    normal[idx] += normalize(cross(points[i + 1] - points[0],
                                 points[((i + 1) % 4) + 1] - points[0]));
  }

  normal[idx] = normalize(normal[idx]);
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
    float3 *normal_data,
    SensorParams &params
)
{
  uint width = params.width;
  uint height = params.height;

  const int threads_per_block = 16;
  const dim3 grid_size((width + threads_per_block - 1) / threads_per_block,
                       (height + threads_per_block - 1) / threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  ComputeNormalMapKernel << < grid_size, block_size >> > (
      normal_data,
          depth_data,
          width,
          height,
          params.fx, params.fy, params.cx, params.cy
  );
}
