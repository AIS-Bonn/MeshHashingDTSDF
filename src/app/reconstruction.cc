//
// Created by wei on 17-3-26.
//
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <helper_cuda.h>
#include <chrono>

#include <string>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "util/timer.h"
#include <queue>
#include <engine/visualizing_engine.h>
#include <io/mesh_writer.h>
#include <meshing/marching_cubes.h>
#include <visualization/compress_mesh.h>

#include "app/utils.h"
#include "sensor/rgbd_data_provider.h"
#include "sensor/rgbd_sensor.h"
#include "visualization/ray_caster.h"

#include "io/config_manager.h"
#include "core/collect_block_array.h"
#include "glwrapper.h"

#include <glog/logging.h>


#define DEBUG_

int main(int argc, char **argv)
{
  CheckCUDADevices();

  /// Use this to substitute tedious argv parsing
  RuntimeParams args;
  LoadRuntimeParams("../config/args.yml", args);

  ConfigManager config;
  RGBDDataProvider rgbd_local_sequence;

  DatasetType dataset_type = DatasetType(args.dataset_type);
  config.LoadConfig(dataset_type);
  rgbd_local_sequence.LoadDataset(dataset_type);
  Sensor sensor(config.sensor_params);

  MainEngine main_engine(
      args,
      config.hash_params,
      config.sdf_params,
      config.mesh_params,
      config.sensor_params,
      config.ray_caster_params
  );

  gl::Light light;
  if (args.enable_visualization)
  {
    light.Load("../config/lights.yaml");
  }
  main_engine.ConfigVisualizingEngine(
      light,
      args.enable_visualization,
      args.enable_navigation,
      args.enable_global_mesh,
      args.enable_bounding_box,
      args.enable_trajectory,
      args.enable_polygon_mode,
      args.enable_ray_casting,
      args.enable_color
  );
  main_engine.ConfigLoggingEngine(
      ".",
      args.enable_video_recording,
      args.enable_ply_saving
  );
  main_engine.enable_sdf_gradient() = args.enable_sdf_gradient;

  cv::Mat color, depth;
  float4x4 wTc, cTw;
  int frame_count = 0;
  while (rgbd_local_sequence.ProvideData(depth, color, wTc, false))
  {
    frame_count++;
//    if (frame_count % 10 != 0)
//      continue;
//    if (frame_count < 300)// or frame_count > 600)
//      continue;
    if (args.run_frames > 0 && frame_count > args.run_frames)
      break;

    // Preprocess data
    wTc.m24 -= 0.00;
    sensor.Process(depth, color, main_engine.log_engine());
    sensor.set_transform(wTc);
    cTw = wTc.getInverse();

	  main_engine.Mapping(sensor);
    if (args.enable_visualization)
    {
	    main_engine.Meshing();
      if (main_engine.Visualize(cTw))
        break;
    }

    std::stringstream ss;
    ss << "block" << std::setw(4) << std::setfill('0') << frame_count;
//    main_engine.StoreBlocks(ss.str());
//    main_engine.WriteMesh("meshes/" + ss.str());

    main_engine.Log();
    main_engine.Recycle();
  }

  main_engine.FinalLog();
  if (args.enable_block_saving)
  {
    main_engine.StoreBlocks("block");
  }

  if (args.enable_visualization)
  {
    while (args.enable_navigation and not main_engine.Visualize(cTw));
  }

  return 0;
}