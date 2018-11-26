//
// Created by wei on 17-10-21.
//

#ifndef MESH_HASHING_RGBD_LOCAL_SEQUENCE_H
#define MESH_HASHING_RGBD_LOCAL_SEQUENCE_H

#include "io/config_manager.h"

struct RGBDDataProvider {
  /// Read from Disk
  size_t frame_id = 0;
  std::vector<std::string> depth_image_list;
  std::vector<std::string> color_image_list;
  std::vector<float4x4>    wTcs;

  void LoadDataset(DatasetType dataset_type);

  void LoadDataset(std::string dataset_path,
                   DatasetType dataset_type);
  void LoadDataset(Dataset     dataset);

  /// If read from disk, then provide mat at frame_id
  /// If read from network/USB, then wait until a mat comes;
  ///                           a while loop might be inside
  bool ProvideData(cv::Mat &depth, cv::Mat &color);

  /**
   * Reads the next dataset entry (depth frame + rgb frame + pose).
   * @param depth
   * @param color
   * @param wTc
   * @param start_width_zero_pose Indicates whether the wTc pose should be converted, s.t. it starts at the origin.
   * @return
   */
  bool ProvideData(cv::Mat &depth, cv::Mat &color, float4x4 &wTc, bool start_width_zero_pose=true);
};

#endif //MESH_HASHING_RGBD_LOCAL_SEQUENCE_H
