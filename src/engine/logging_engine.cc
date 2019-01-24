//
// Created by wei on 17-10-24.
//

#include <experimental/filesystem>
#include <yaml-cpp/yaml.h>
#include <iomanip>
#include <io/mesh_writer.h>
#include <glog/logging.h>
#include <core/block.h>
#include <core/hash_entry.h>
#include <extern/cuda/helper_cuda.h>
#include "logging_engine.h"

namespace fs = std::experimental::filesystem;

void LoggingEngine::Init(std::string path)
{
  base_path_ = path;

  time_stamp_file_.open(base_path_ + "/time_mapping.txt");
  if (!time_stamp_file_.is_open())
  {
    LOG(ERROR) << "Can't open mappoing stamp file";
    return;
  }

  time_stamp_file_.flags(std::ios::right);
  time_stamp_file_.setf(std::ios::fixed);

  meshing_time_file_.open(base_path_ + "/time_meshing.txt");

  mesh_stats_file_.open(base_path_ + "/stats_mesh.txt");

  localization_err_file_.open(base_path_ + "/localization_error.txt");
}

LoggingEngine::~LoggingEngine()
{
  if (video_writer_.isOpened())
    video_writer_.release();
  if (time_stamp_file_.is_open())
    time_stamp_file_.close();
  time_stamp_file_ << std::setprecision(4);
}

void LoggingEngine::ConfigVideoWriter(int width, int height)
{
  enable_video_ = true;
  video_writer_.open(base_path_ + "/video.avi",
                     CV_FOURCC('X', 'V', 'I', 'D'),
                     30, cv::Size(width, height));
}

void LoggingEngine::WriteVideo(cv::Mat &mat)
{
  video_writer_ << mat;
}

void LoggingEngine::ConfigPlyWriter()
{
  enable_ply_ = true;
}

void LoggingEngine::WritePly(CompactMesh &mesh, const std::string& filename)
{
  SavePly(mesh, base_path_ + "/" + filename);
}

void LoggingEngine::WriteLocalizationError(float error)
{
  localization_err_file_ << error << "\n";
}

void LoggingEngine::WriteMappingTimeStamp(double alloc_time,
                                          double collect_time,
                                          double update_time,
                                          int frame_idx)
{

  time_stamp_file_ << frame_idx << " "
                   //<< "alloc time : "
                   << alloc_time * 1000 << " "//<< "ms "
                   //<< "collect time : "
                   << collect_time * 1000 << " "// << "ms "
                   //<< "update time : "
                   << update_time * 1000 << "\n";
}

void LoggingEngine::WriteMappingTimeStamp(float alloc_time,
                                          float collect_time,
                                          float predict_time,
                                          float update_time,
                                          int frame_idx)
{

  time_stamp_file_ << frame_idx << " "
                   //<< "alloc time : "
                   << alloc_time * 1000 << " "//<< "ms "
                   //<< "collect time : "
                   << collect_time * 1000 << " "// << "ms "
                   << predict_time * 1000 << " "
                   //<< "update time : "
                   << update_time * 1000 << "\n";
}

void LoggingEngine::WriteMeshingTimeStamp(float time, int frame_idx)
{
  meshing_time_file_ << frame_idx << " " << time << "\n";
}

void LoggingEngine::WriteMeshStats(int vtx_count, int tri_count)
{
  mesh_stats_file_ << "Vertices: " << vtx_count << "\n"
                   << "Triangles: " << tri_count;
}


void LoggingEngine::WriteRawBlocks(const BlockMap &blocks, std::string filename)
{
  fs::create_directories(base_path_ + "/Blocks/");
  std::ofstream file(base_path_ + "/Blocks/" + filename + ".block",
                     std::ios::binary);
  if (!file.is_open())
  {
    LOG(WARNING) << "can't open block file.";
    return;
  }

  int N = sizeof(std::pair<int3, Block>);
  int num = blocks.size();
  file.write((char *) &num, sizeof(int));
  for (auto &&block:blocks)
  {
    file.write((char *) &block, N);
  }
  file.close();
}

BlockMap LoggingEngine::ReadRawBlocks(std::string filename)
{
  std::ifstream file(base_path_ + "/Blocks/" + filename + ".block");
  BlockMap blocks;
  if (!file.is_open())
  {
    LOG(WARNING) << " can't open block file.";
    return blocks;
  }

  int num;
  file.read((char *) &num, sizeof(int));
  if (file.bad())
  {
    LOG(WARNING) << " can't open block file.";
    return blocks;
  }

  std::pair<int3, Block> block;
  int N = sizeof(block);
  for (int i = 0; i < num; ++i)
  {
    file.read((char *) &block, N);
    if (file.bad())
    {
      LOG(WARNING) << " did not read the whole block file.";
      return std::move(blocks);
    }
    blocks.insert(block);
  }
  file.close();
  return std::move(blocks);
}

void
LoggingEngine::WriteFormattedBlocks(const BlockMap &blocks, std::string filename)
{
  YAML::Node root;
  root["number_blocks"] = blocks.size();

  for (auto &&block:blocks)
  {
    YAML::Node block_node;
    block_node["pos"]["x"] = block.first.x;
    block_node["pos"]["y"] = block.first.y;
    block_node["pos"]["z"] = block.first.z;
    block_node["ptr"] = block.second.voxel_array_ptrs[0];

    for (auto voxel_array : block.second.voxel_arrays)
    {
      YAML::Node direction_node;
      if (voxel_array)
      {
        for (auto &voxel : voxel_array->voxels)
        {
          direction_node["sdf"].push_back(voxel.sdf);
          direction_node["inv_sigma2"].push_back(voxel.inv_sigma2);
        }
      }
      block_node["directions"].push_back(direction_node);
    }
    root["blocks"].push_back(block_node);
  }


  fs::create_directories(base_path_ + "/FormatBlocks/");
  std::ofstream file(base_path_ + "/FormatBlocks/" + filename + ".formatblock");
  if (!file.is_open())
  {
    LOG(ERROR) << " can't open format block file.";
    return;
  }

  YAML::Emitter emitter;
  emitter.SetIndent(4);
  emitter.SetSeqFormat(YAML::Flow);
  emitter << root;
  file << emitter.c_str();

  file.close();
}

BlockMap LoggingEngine::ReadFormattedBlocks(std::string filename)
{
  // FIXME: This is not up to date after the data structure change concerning the voxel arrays
  std::ifstream file(base_path_ + "/FormatBlocks/" + filename + ".formatblock");
  BlockMap blocks;
  if (!file.is_open())
  {
    LOG(ERROR) << " can't open format block file.";
    return blocks;
  }

  int num;
  Block block;
  file >> num;
  for (int i = 0; i < num; ++i)
  {
    int3 pos;
    file >> pos.x >> pos.y >> pos.z;
    int size = BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH * BLOCK_SIDE_LENGTH;
    block.Clear();
    for (int i = 0; i < size; ++i)
      file >> block.voxel_arrays[0]->voxels[i].sdf;
    for (int i = 0; i < size; ++i)
      file >> block.voxel_arrays[0]->voxels[i].inv_sigma2;
    if (file.bad())
    {
      LOG(ERROR) << " can't read the whole format block file.";
      return blocks;
    }
    blocks.emplace(pos, block);
  }
  file.close();
  return blocks;
}

BlockMap LoggingEngine::RecordBlockToMemory(
    const Block *block_gpu, uint block_num,
    const HashEntry *candidate_entry_gpu, uint entry_num
)
{

  BlockMap block_map;
  auto *block_cpu = new Block[block_num];
  auto *candidate_entry_cpu = new HashEntry[entry_num];
  cudaMemcpy(block_cpu, block_gpu,
             sizeof(Block) * block_num,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(candidate_entry_cpu, candidate_entry_gpu,
             sizeof(HashEntry) * entry_num,
             cudaMemcpyDeviceToHost);

  for (uint i = 0; i < entry_num; ++i)
  {
    int3 &pos = candidate_entry_cpu[i].pos;
    Block &block = block_cpu[candidate_entry_cpu[i].ptr];
    for (size_t direction = 0; direction < N_DIRECTIONS; direction++)
    {
      if (block.voxel_arrays[direction])
      {
        VoxelArray *addr = block.voxel_arrays[direction];
        block.voxel_arrays[direction] = new VoxelArray;
        checkCudaErrors(cudaMemcpy(block.voxel_arrays[direction], addr,
                                   sizeof(VoxelArray),
                                   cudaMemcpyDeviceToHost));

      }
    }
    block_map.emplace(pos, block);
  }

  delete[] block_cpu;
  delete[] candidate_entry_cpu;
  return block_map;
}
