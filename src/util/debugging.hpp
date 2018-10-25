#pragma once

#include <string>
#include <sensor/rgbd_sensor.h>

inline void SaveNormalImage(const std::string &path, const SensorData &sensor_data, const SensorParams &params)
{
  cv::Mat normal_map(params.height, params.width, CV_32FC3);
  checkCudaErrors(cudaMemcpy(normal_map.data, sensor_data.normal_data,
                             sizeof(float3) * params.height * params.width,
                             cudaMemcpyDeviceToHost));
  cv::cvtColor(normal_map, normal_map, CV_RGB2BGR);
  normal_map = cv::abs(normal_map * 255);
  cv::imwrite(path, normal_map);
}