//
// Created by wei on 17-5-31.
//

#include "config_manager.h"

#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

void LoadRuntimeParams(std::string path, RuntimeParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.dataset_type  = (int)fs["dataset_type"];
  params.update_type  = (int)fs["update_type"];
  params.raycasting_mode  = (int)fs["raycasting_mode"];

  params.enable_point_to_plane = (int)fs["enable_point_to_plane"];
  params.enable_directional_sdf = (int)fs["enable_directional_sdf"];
  params.enable_navigation   = (int)fs["enable_navigation"];
  params.enable_polygon_mode = (int)fs["enable_polygon_mode"];
  params.enable_global_mesh = (int)fs["enable_global_mesh"];
  params.enable_sdf_gradient = (int)fs["enable_sdf_gradient"];
  params.enable_color   = (int)fs["enable_color"];

  params.enable_bounding_box  = (int)fs["enable_bounding_box"];
  params.enable_trajectory  = (int)fs["enable_trajectory"];
  params.enable_ray_casting   = (int)fs["enable_ray_casting"];

  params.enable_visualization  = (int)fs["enable_visualization"];
  params.enable_video_recording  = (int)fs["enable_video_recording"];
  params.enable_block_saving     = (int)fs["enable_block_saving"];
  params.enable_ply_saving     = (int)fs["enable_ply_saving"];
  params.filename_prefix = (std::string)fs["filename_prefix"];
  params.time_profile    = (std::string)fs["time_profile"];
  params.memo_profile    = (std::string)fs["memo_profile"];

  params.run_frames    = (int)fs["run_frames"];
}

void LoadHashParams(std::string path, HashParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.bucket_count     = (int)fs["bucket_count"];
  params.bucket_size      = (int)fs["bucket_size"];
  params.entry_count      = (int)fs["count"];
  params.linked_list_size = (int)fs["linked_list_size"];
  params.max_block_count   = (int)fs["max_block_count"];
}

void LoadMeshParams(std::string path, MeshParams &params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.max_vertex_count   = (int)fs["max_vertex_count"];
  params.max_triangle_count = (int)fs["max_triangle_count"];
}

void LoadVolumeParams(std::string path, VolumeParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.voxel_size                = (float)fs["voxel_size"];
  params.sdf_upper_bound           = (float)fs["sdf_upper_bound"];
  params.truncation_distance_scale = (float)fs["truncation_distance_scale"];
  params.truncation_distance       = (float)fs["truncation_distance"];
  params.weight_sample             = (int)fs["weight_sample"];
  params.weight_upper_bound        = (int)fs["weight_upper_bound"];
}

void LoadSensorParams(std::string path, SensorParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.fx = (float)fs["fx"];
  params.fy = (float)fs["fy"];
  params.cx = (float)fs["cx"];
  params.cy = (float)fs["cy"];
  params.min_depth_range = (float)fs["min_depth_range"];
  params.max_depth_range = (float)fs["max_depth_range"];
  params.range_factor    = (float)fs["range_factor"];
  params.width  = (int)fs["width"];
  params.height = (int)fs["height"];
}

void LoadRayCasterParams(std::string path, RayCasterParams& params) {
  cv::FileStorage fs(path, cv::FileStorage::READ);
  params.min_raycast_depth    = (float)fs["min_raycast_depth"];
  params.max_raycast_depth    = (float)fs["max_raycast_depth"];
  params.raycast_step         = (float)fs["raycast_step"];
  params.sample_sdf_threshold = (float)fs["sample_sdf_threshold"];
  params.sdf_threshold        = (float)fs["sdf_threshold"];
  params.enable_gradients     = (int)fs["enable_gradient"];
}


/// 1-1-1 correspondences
void LoadICL(std::string               dataset_path,
             std::vector<std::string> &depth_image_list,
             std::vector<std::string> &color_image_list,
             std::vector<float4x4>& wTcs) {
  std::ifstream img_stream(dataset_path + "associations.txt");
  std::string time_stamp, depth_image_name, color_image_name;
  /// !!! ICL problem: pose of the 1st frame is not provided
  img_stream >> time_stamp >> depth_image_name
             >> time_stamp >> color_image_name;
  while (img_stream >> time_stamp >> depth_image_name
                    >> time_stamp >> color_image_name) {
    depth_image_list.push_back(dataset_path + "/" + depth_image_name);
    color_image_list.push_back(dataset_path + "/" + color_image_name);
  }

  std::ifstream traj_stream(dataset_path + "trajectory.txt");
  std::string ts_img, img_path, ts_gt;
  float tx, ty, tz, qx, qy, qz, qw;
  float4x4 rTl;
  rTl.setIdentity();
  rTl.entries2[1][1] = -1;

  while (traj_stream >> ts_img
                     >> tx >> ty >> tz
                     >> qx >> qy >> qz >> qw) {
    float4x4 wTc;
    wTc.setIdentity();

    wTc.m11 = 1 - 2 * qy * qy - 2 * qz * qz;
    wTc.m12 = 2 * qx * qy - 2 * qz * qw;
    wTc.m13 = 2 * qx * qz + 2 * qy * qw;
    wTc.m14 = tx;
    wTc.m21 = 2 * qx * qy + 2 * qz * qw;
    wTc.m22 = 1 - 2 * qx * qx - 2 * qz * qz;
    wTc.m23 = 2 * qy * qz - 2 * qx * qw;
    wTc.m24 = ty;
    wTc.m31 = 2 * qx * qz - 2 * qy * qw;
    wTc.m32 = 2 * qy * qz + 2 * qx * qw;
    wTc.m33 = 1 - 2 * qx * qx - 2 * qy * qy;
    wTc.m34 = tz;
    wTc.m44 = 1;

    wTc = rTl * wTc * rTl.getInverse();
    wTcs.push_back(wTc);
  }
}

void LoadSUN3D(std::string dataset_path,
               std::vector<std::string> &depth_img_list,
               std::vector<std::string> &color_img_list,
               std::vector<float4x4> &wTcs) {
  std::ifstream color_stream(dataset_path + "color.txt");
  LOG(INFO) << dataset_path + "color.txt";
  std::string img_name;
  while (color_stream >> img_name) {
    color_img_list.push_back(dataset_path + "color/" + img_name);
  }

  std::ifstream depth_stream(dataset_path + "depth.txt");
  while (depth_stream >> img_name) {
    depth_img_list.push_back(dataset_path + "depth/" + img_name);
  }

  std::ifstream traj_stream(dataset_path + "trajectory.log");
  int dummy;
  float4x4 wTc;
  while (traj_stream >> dummy >> dummy >> dummy
                     >> wTc.m11 >> wTc.m12 >> wTc.m13 >> wTc.m14
                     >> wTc.m21 >> wTc.m22 >> wTc.m23 >> wTc.m24
                     >> wTc.m31 >> wTc.m32 >> wTc.m33 >> wTc.m34
                     >> wTc.m41 >> wTc.m42 >> wTc.m43 >> wTc.m44) {
    wTcs.push_back(wTc);
  }
}

void LoadSUN3DOriginal(std::string dataset_path,
                       std::vector<std::string> &depth_img_list,
                       std::vector<std::string> &color_img_list,
                       std::vector<float4x4> &wTcs) {
  std::ifstream stream(dataset_path + "image_depth_association.txt");
  std::string ts, color_img_name, depth_img_name;
  while (stream >> ts >> color_img_name >> ts >> depth_img_name) {
    color_img_list.push_back(dataset_path + "image/" + color_img_name);
    depth_img_list.push_back(dataset_path + "depth/" + depth_img_name);
  }

  std::ifstream traj_stream(dataset_path + "trajectory.txt");
  double cTw[12];
  float4x4 wTc;
  while (traj_stream >> cTw[0] >> cTw[1] >> cTw[2] >> cTw[3]
                     >> cTw[4] >> cTw[5] >> cTw[6] >> cTw[7]
                     >> cTw[8] >> cTw[9] >> cTw[10] >> cTw[11]) {

    wTc.setIdentity();
    wTc.m11 = (float)cTw[0];
    wTc.m12 = (float)cTw[1];
    wTc.m13 = (float)cTw[2];
    wTc.m14 = (float)cTw[3];
    wTc.m21 = (float)cTw[4];
    wTc.m22 = (float)cTw[5];
    wTc.m23 = (float)cTw[6];
    wTc.m24 = (float)cTw[7];
    wTc.m31 = (float)cTw[8];
    wTc.m32 = (float)cTw[9];
    wTc.m33 = (float)cTw[10];
    wTc.m34 = (float)cTw[11];

    wTc.getInverse();
    wTcs.push_back(wTc);
  }
}

/// no 1-1-1 correspondences
void LoadTUM(std::string dataset_path,
             std::vector<std::string> &depth_image_list,
             std::vector<std::string> &color_image_list,
             std::vector<float4x4>& wTcs) {
  std::ifstream img_stream(dataset_path + "/depth_rgb_associations.txt");
  if (not img_stream.is_open())
  {
    LOG(ERROR) << "failed to open \"depth_rgb_associations.txt\". Is the dataset directory "
               << dataset_path << " correct?";
    return;
  }
  std::unordered_map<std::string, std::string> depth_color_correspondence;
  std::string depth_image_name, color_image_name, ts;
  while (img_stream >> ts >> depth_image_name >> ts >> color_image_name) {
    depth_color_correspondence.emplace(depth_image_name, color_image_name);
  }

  std::ifstream traj_stream(dataset_path + "/depth_gt_associations.txt");
  float tx, ty, tz, qx, qy, qz, qw;
  while (traj_stream >> ts >> depth_image_name
                     >> ts >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
    if (depth_color_correspondence.find(depth_image_name)
        != depth_color_correspondence.end()) {
      float4x4 wTc;
      wTc.setIdentity();

      wTc.m11 = 1 - 2 * qy * qy - 2 * qz * qz;
      wTc.m12 = 2 * qx * qy - 2 * qz * qw;
      wTc.m13 = 2 * qx * qz + 2 * qy * qw;
      wTc.m14 = tx;
      wTc.m21 = 2 * qx * qy + 2 * qz * qw;
      wTc.m22 = 1 - 2 * qx * qx - 2 * qz * qz;
      wTc.m23 = 2 * qy * qz - 2 * qx * qw;
      wTc.m24 = ty;
      wTc.m31 = 2 * qx * qz - 2 * qy * qw;
      wTc.m32 = 2 * qy * qz + 2 * qx * qw;
      wTc.m33 = 1 - 2 * qx * qx - 2 * qy * qy;
      wTc.m34 = tz;
      wTc.m44 = 1;

      depth_image_list.push_back(dataset_path + "/" + depth_image_name);
      color_image_list.push_back(dataset_path + "/"
                                 + depth_color_correspondence[depth_image_name]);
      wTcs.push_back(wTc);
//      LOG(INFO) << depth_image_name << " "
//                << depth_color_correspondence[depth_image_name];
    }
  }
}

void Load3DVCR(std::string dataset_path,
               std::vector<std::string> &depth_image_list,
               std::vector<std::string> &color_image_list,
               std::vector<float4x4>& wTcs) {

  std::ifstream traj_stream(dataset_path + "trajectory.txt");
  std::string ts_img, img_path, ts_gt;
  float ts, tx, ty, tz, qx, qy, qz, qw;

  std::unordered_set<int> tracked_ts;
  while (traj_stream >> ts
                     >> tx >> ty >> tz
                     >> qx >> qy >> qz >> qw) {
    tracked_ts.emplace((int)ts);
    LOG(INFO) << (int)ts;

    float4x4 wTc;
    wTc.setIdentity();

    wTc.m11 = 1 - 2 * qy * qy - 2 * qz * qz;
    wTc.m12 = 2 * qx * qy - 2 * qz * qw;
    wTc.m13 = 2 * qx * qz + 2 * qy * qw;
    wTc.m14 = tx;
    wTc.m21 = 2 * qx * qy + 2 * qz * qw;
    wTc.m22 = 1 - 2 * qx * qx - 2 * qz * qz;
    wTc.m23 = 2 * qy * qz - 2 * qx * qw;
    wTc.m24 = ty;
    wTc.m31 = 2 * qx * qz - 2 * qy * qw;
    wTc.m32 = 2 * qy * qz + 2 * qx * qw;
    wTc.m33 = 1 - 2 * qx * qx - 2 * qy * qy;
    wTc.m34 = tz;
    wTc.m44 = 1;
    wTcs.push_back(wTc);
  }

  std::ifstream color_stream(dataset_path + "rgb.txt");
  std::string img_name;

  int count = 0;
  while (color_stream >> img_name) {
    if (tracked_ts.find(count) != tracked_ts.end()) {
      LOG(INFO) << dataset_path + "rgb/" + img_name;
      color_image_list.push_back(dataset_path + "rgb/" + img_name);
    }
    ++count;
  }

  count = 0;
  std::ifstream depth_stream(dataset_path + "depth.txt");
  while (depth_stream >> img_name) {
    if (tracked_ts.find(count) != tracked_ts.end()) {
      LOG(INFO) << dataset_path + "depth/" + img_name;
      depth_image_list.push_back(dataset_path + "depth/" + img_name);
    }
    ++count;
  }
}

const std::string kConfigPaths[] = {
    "../config/ICL.yml",
    "../config/TUM1.yml",
    "../config/TUM2.yml",
    "../config/TUM3.yml",
    "../config/SUN3D.yml",
    "../config/SUN3D_ORIGINAL.yml",
    "../config/PKU.yml"
};

void ConfigManager::LoadConfig(std::string config_path) {
  LoadHashParams(config_path, hash_params);
  LoadMeshParams(config_path, mesh_params);
  LoadVolumeParams(config_path, sdf_params);
  LoadSensorParams(config_path, sensor_params);
  LoadRayCasterParams(config_path, ray_caster_params);

  ray_caster_params.width = sensor_params.width;
  ray_caster_params.height = sensor_params.height;
  ray_caster_params.fx = sensor_params.fx;
  ray_caster_params.fy = sensor_params.fy;
  ray_caster_params.cx = sensor_params.cx;
  ray_caster_params.cy = sensor_params.cy;
}

void ConfigManager::LoadConfig(DatasetType dataset_type) {
  std::string config_path = kConfigPaths[dataset_type];
  LoadConfig(config_path);
}