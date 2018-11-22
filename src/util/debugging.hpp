#pragma once

#include <string>
#include <sensor/rgbd_sensor.h>

inline void SaveNormalImage(const std::string &path, const SensorData &sensor_data, const SensorParams &params)
{
  cv::Mat normal_map(params.height, params.width, CV_32FC4);
  checkCudaErrors(cudaMemcpy(normal_map.data, sensor_data.normal_data,
                             sizeof(float4) * params.height * params.width,
                             cudaMemcpyDeviceToHost));

  cv::cvtColor(normal_map, normal_map, CV_RGBA2BGR);
  normal_map = cv::abs(normal_map * 255);

//  cv::Mat bilat;
//  cv::bilateralFilter(normal_map, bilat, 15, 20, 80, cv::BORDER_DEFAULT);
//  normal_map = cv::abs(bilat * 255);

//  cv::cuda::GpuMat input(normal_map);
//  cv::cuda::GpuMat output;
//  cv::cuda::bilateralFilter(input, output, 15, 100, 10, cv::BORDER_DEFAULT);
//  output.download(normal_map);
//  normal_map = cv::abs(normal_map * 255);

  cv::imwrite(path, normal_map);
}