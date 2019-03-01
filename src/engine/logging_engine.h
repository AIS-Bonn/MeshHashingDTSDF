//
// Created by wei on 17-10-24.
//

#ifndef ENGINE_LOGGING_ENGINE_H
#define ENGINE_LOGGING_ENGINE_H

#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

class Block;
class CompactMesh;
class HashEntry;

class Int3Sort {
public:
  bool operator()(int3 const &a, int3 const &b) const {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
  }
};
typedef std::map<int3, Block, Int3Sort> BlockMap;

class LoggingEngine {
public:
  LoggingEngine() = default;
  explicit LoggingEngine(std::string path)
      : base_path_(path) {};
  void Init(std::string path);
  ~LoggingEngine();

  void ConfigVideoWriter(int width, int height);
  void ConfigPlyWriter();
  void WriteVideo(cv::Mat& mat);
  void WritePly(CompactMesh& mesh, const std::string& filename="mesh.ply");
  void WriteLocalizationError(float error);
  void WriteMappingTimeStamp(double alloc_time, double collect_time, double update_time,
                               int frame_idx);
  void WriteMappingTimeStamp(float alloc_time, float collect_time, float predict_time, float update_time,
                             int frame_idx);
  void WritePreprocessTimeStamp(double copy_time, double normal_estimation_time, double bilateral_filter_time);
  void WriteRecycleTimeStamp(double recycle_time, int frame_idx);
  void WriteMeshingTimeStamp(float time, int frame_idx);
  void WriteMeshStats(int vtx_count, int tri_count);
  void WriteBlockStats(const BlockMap &blocks, std::string filename);
  void WriteVoxelUpdate(int max, float mean_hit, float mean);

  BlockMap RecordBlockToMemory(
      const Block *block_gpu, uint block_num,
      const HashEntry *candidate_entry_gpu, uint entry_num
  );
  void WriteFormattedBlocks(const BlockMap &blocks, std::string filename);
  BlockMap ReadFormattedBlocks(std::string filename);
  void WriteRawBlocks(const BlockMap &blocks, std::string filename);
  BlockMap ReadRawBlocks(std::string filename);

  bool enable_video() {
    return enable_video_;
  }
  bool enable_ply() {
    return enable_ply_;
  }
private:
  bool enable_video_ = false;
  bool enable_ply_ = false;

  std::string base_path_;
  std::string prefix_;
  cv::VideoWriter video_writer_;
  std::ofstream time_stamp_file_;
  std::ofstream preprocess_time_file_;
  std::ofstream recycle_time_file_;
  std::ofstream meshing_time_file_;
  std::ofstream mesh_stats_file_;
  std::ofstream localization_err_file_;
  std::ofstream voxel_update_file_;
};


#endif //MESH_HASHING_LOGGING_ENGINE_H
