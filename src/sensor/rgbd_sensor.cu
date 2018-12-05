/// 16 threads per block

#include "rgbd_sensor.h"
#include <extern/cuda/helper_cuda.h>
#include <extern/cuda/helper_cuda.h>
#include <extern/cuda/helper_math.h>
#include <geometry/geometry_helper.h>
#include <sensor/preprocess.h>
#include <util/debugging.hpp>
#include <visualization/color_util.h>

#include <driver_types.h>
#include <glog/logging.h>


/// Member functions: (CPU code)
Sensor::Sensor(SensorParams &sensor_params) {
  const uint image_size = sensor_params.height * sensor_params.width;

  params_ = sensor_params; // Is it copy constructing?
  checkCudaErrors(cudaMalloc(&data_.depth_buffer, sizeof(short) * image_size));
  checkCudaErrors(cudaMalloc(&data_.color_buffer, sizeof(uchar4) * image_size));

  checkCudaErrors(cudaMalloc(&data_.depth_data, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc(&data_.inlier_ratio, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc(&data_.filtered_depth_data, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc(&data_.color_data, sizeof(float4) * image_size));
  checkCudaErrors(cudaMalloc(&data_.normal_data, sizeof(float4) * image_size));

  data_.depth_channel_desc = cudaCreateChannelDesc<float>();
  checkCudaErrors(cudaMallocArray(&data_.depth_array,
                                  &data_.depth_channel_desc,
                                  params_.width, params_.height));

  data_.color_channel_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaMallocArray(&data_.color_array,
                                  &data_.color_channel_desc,
                                  params_.width, params_.height));

  data_.normal_channel_desc = cudaCreateChannelDesc<float4>();
  checkCudaErrors(cudaMallocArray(&data_.normal_array,
                                  &data_.normal_channel_desc,
                                  params_.width, params_.height));

  data_.depth_texture = 0;
  data_.color_texture = 0;
  data_.normal_texture = 0;

  BindCUDATexture();
  is_allocated_on_gpu_ = true;
}

Sensor::~Sensor() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(data_.depth_buffer));
    checkCudaErrors(cudaFree(data_.color_buffer));

    checkCudaErrors(cudaFree(data_.depth_data));
    checkCudaErrors(cudaFree(data_.inlier_ratio));
    checkCudaErrors(cudaFree(data_.filtered_depth_data));
    checkCudaErrors(cudaFree(data_.color_data));
    checkCudaErrors(cudaFree(data_.normal_data));

    checkCudaErrors(cudaFreeArray(data_.depth_array));
    checkCudaErrors(cudaFreeArray(data_.color_array));
    checkCudaErrors(cudaFreeArray(data_.normal_array));
  }
}

void Sensor::BindCUDATexture() {
  cudaResourceDesc depth_resource;
  memset(&depth_resource, 0, sizeof(depth_resource));
  depth_resource.resType = cudaResourceTypeArray;
  depth_resource.res.array.array = data_.depth_array;

  cudaTextureDesc depth_tex_desc;
  memset(&depth_tex_desc, 0, sizeof(depth_tex_desc));
  depth_tex_desc.readMode = cudaReadModeElementType;

  if (data_.depth_texture != 0)
    checkCudaErrors(cudaDestroyTextureObject(data_.depth_texture));
  checkCudaErrors(cudaCreateTextureObject(&data_.depth_texture,
                                          &depth_resource,
                                          &depth_tex_desc,
                                          NULL));

  cudaResourceDesc color_resource;
  memset(&color_resource, 0, sizeof(color_resource));
  color_resource.resType = cudaResourceTypeArray;
  color_resource.res.array.array = data_.color_array;

  cudaTextureDesc color_tex_desc;
  memset(&color_tex_desc, 0, sizeof(color_tex_desc));
  color_tex_desc.readMode = cudaReadModeElementType;

  if (data_.color_texture != 0)
    checkCudaErrors(cudaDestroyTextureObject(data_.color_texture));
  checkCudaErrors(cudaCreateTextureObject(&data_.color_texture,
                                          &color_resource,
                                          &color_tex_desc,
                                          NULL));

  cudaResourceDesc normal_resource;
  memset(&normal_resource, 0, sizeof(normal_resource));
  normal_resource.resType = cudaResourceTypeArray;
  normal_resource.res.array.array = data_.normal_array;

  cudaTextureDesc normal_tex_desc;
  memset(&normal_tex_desc, 0, sizeof(normal_tex_desc));
  normal_tex_desc.readMode = cudaReadModeElementType;

  if (data_.normal_texture != 0)
    checkCudaErrors(cudaDestroyTextureObject(data_.normal_texture));
  checkCudaErrors(cudaCreateTextureObject(&data_.normal_texture,
                                          &normal_resource,
                                          &normal_tex_desc,
                                          NULL));
}

int Sensor::Process(cv::Mat &depth, cv::Mat &color) {
  // TODO(wei): deal with distortion
  /// Disable all filters at current
  ConvertDepthFormat(depth, data_.depth_buffer, data_.depth_data, params_);
  ConvertColorFormat(color, data_.color_buffer, data_.color_data, params_);
  ResetInlierRatio(data_.inlier_ratio, params_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  ComputeNormalMap(data_.depth_data, data_.normal_data, params_);

  /// Array used as texture in mapper
  checkCudaErrors(cudaMemcpyToArray(data_.depth_array, 0, 0,
                                    data_.depth_data,
                                    sizeof(float)*params_.height*params_.width,
                                    cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpyToArray(data_.color_array, 0, 0,
                                    data_.color_data,
                                    sizeof(float4)*params_.height*params_.width,
                                    cudaMemcpyDeviceToDevice));

  checkCudaErrors(cudaMemcpyToArray(data_.normal_array, 0, 0,
                                    data_.normal_data,
                                    sizeof(float4)*params_.height*params_.width,
                                    cudaMemcpyDeviceToDevice));

  // Save debug normal image
//  static uint counter = 0;
//  std::stringstream ss;
//  ss << "/tmp/normals/normals" << std::setfill('0') << std::setw(4) << counter << ".png";
//  SaveNormalImage(ss.str(), data_, params_);
//  counter +=1;

  BindCUDATexture();
  return 0;
}
