#pragma once

inline void SaveNormalImage(const std::string &path, const SensorData &sensor_data, const SensorParams &params)
{
  cv::Mat normal_map(params.height, params.width, CV_32FC3);
  checkCudaErrors(cudaMemcpy(normal_map.data, sensor_data.normal_data,
                             sizeof(float3) * params.height * params.width,
                             cudaMemcpyDeviceToHost));
  normal_map = normal_map * 255;
  cv::imwrite(path, normal_map);
}