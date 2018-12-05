#pragma once

#include <string>
#include <extern/cuda/helper_cuda.h>
#include "core/directional_tsdf.h"
#include "core/functions.h"
#include "sensor/rgbd_sensor.h"

inline void SaveNormalImage(const std::string &path, const SensorData &sensor_data, const SensorParams &params)
{
  cv::Mat normal_map(params.height, params.width, CV_32FC4);
  checkCudaErrors(cudaMemcpy(normal_map.data, sensor_data.normal_data,
                             sizeof(float4) * params.height * params.width,
                             cudaMemcpyDeviceToHost));

  cv::cvtColor(normal_map, normal_map, CV_RGBA2BGR);
  normal_map = cv::abs(normal_map * 255);
  cv::imwrite(path, normal_map);
}

/**
 * Computes the directional decision for each pixel and stores a color-coded image
 * @param path
 * @param sensor
 */
__host__
inline void SaveDirectionDecisionImage(const std::string &path, const Sensor &sensor)
{
  cv::Mat normal_map(sensor.height(), sensor.width(), CV_32FC4);
  checkCudaErrors(cudaMemcpy(normal_map.data, sensor.data().normal_data,
                             sizeof(float4) * sensor.height() * sensor.width(),
                             cudaMemcpyDeviceToHost));
  cv::Mat img(sensor.height(), sensor.width(), CV_32FC3, cv::Scalar(0, 0, 0));
  for (int i = 0; i < sensor.height() * sensor.width(); i++)
  {
    float4 normal = normal_map.at<float4>(i);
    if (not IsValidNormal(normal))
    { // No normal value for this coordinate
      continue;
    }
    float4x4 wTcRotOnly = sensor.wTc();
    wTcRotOnly.m14 = 0;
    wTcRotOnly.m24 = 0;
    wTcRotOnly.m34 = 0;
    float4 normal_world = wTcRotOnly * normal;
    TSDFDirection direction = VectorToTSDFDirection(normal_world);

    cv::Vec3f color;
    switch(direction)
    {
      case TSDFDirection::UP:
        color = cv::Vec3f(0, 0, 255);
        break;
      case TSDFDirection::DOWN:
        color = cv::Vec3f(0, 255, 0);
        break;
      case TSDFDirection::LEFT:
        color = cv::Vec3f(0, 255, 255);
        break;
      case TSDFDirection::RIGHT:
        color = cv::Vec3f(255, 0, 0);
        break;
      case TSDFDirection::FORWARD:
        color = cv::Vec3f(255, 0, 255);
        break;
      case TSDFDirection::BACKWARD:
        color = cv::Vec3f(255, 255, 0);
        break;
      default:
        color = cv::Vec3f(0, 0, 0);
    }
    img.at<cv::Vec3f>(i) = color;
  }
  cv::cvtColor(img, img, CV_BGR2RGB);
  cv::imwrite(path, img);
}