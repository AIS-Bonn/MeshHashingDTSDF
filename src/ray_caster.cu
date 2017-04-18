#include <matrix.h>

#include "sensor_data.h"
#include "hash_table.h"
#include "ray_caster.h"
#include "ray_caster_data.h"

#define T_PER_BLOCK 8

__device__
inline float frac(float val) {
  return (val - floorf(val));
}
__device__
inline float3 frac(const float3& val)  {
  return make_float3(frac(val.x), frac(val.y), frac(val.z));
}
__device__
bool TrilinearInterpolation(const HashTable& hash, const float3& pos,
                            float& sdf, uchar3& color) {
  const float offset = kHashParams.voxel_size;
  const float3 pos_corner = pos - 0.5f * offset;
  float3 ratio = frac(WorldToVoxelf(pos));

  float w;
  Voxel v;
  
  sdf = 0.0f;
  float3 colorf = make_float3(0.0f, 0.0f, 0.0f);
  float3 v_color;
  
  /// 000
  v = hash.GetVoxel(pos_corner+make_float3(0.0f, 0.0f, 0.0f));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = (1.0f-ratio.x)*(1.0f-ratio.y)*(1.0f-ratio.z);
  sdf    += w * v.sdf;
  colorf += w * v_color;

  /// 001
  v = hash.GetVoxel(pos_corner+make_float3(0.0f, 0.0f, offset));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = (1.0f-ratio.x)*(1.0f-ratio.y)*ratio.z;
  sdf    += w * v.sdf;
  colorf += w * v_color;

  /// 010
  v = hash.GetVoxel(pos_corner+make_float3(0.0f, offset, 0.0f));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = (1.0f-ratio.x)*ratio.y *(1.0f-ratio.z);
  sdf    += w * v.sdf;
  colorf += w * v_color;

  /// 011
  v = hash.GetVoxel(pos_corner+make_float3(0.0f, offset, offset));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = (1.0f-ratio.x)*ratio.y*ratio.z;
  sdf    += w * v.sdf;
  colorf += w * v_color;

  /// 100
  v = hash.GetVoxel(pos_corner+make_float3(offset, 0.0f, 0.0f));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = ratio.x*(1.0f-ratio.y)*(1.0f-ratio.z);
  sdf    +=	w * v.sdf;
  colorf += w * v_color;

  /// 101
  v = hash.GetVoxel(pos_corner+make_float3(offset, 0.0f, offset));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = ratio.x*(1.0f-ratio.y)*ratio.z;
  sdf    +=	w * v.sdf;
  colorf += w	* v_color;

  /// 110
  v = hash.GetVoxel(pos_corner+make_float3(offset, offset, 0.0f));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = ratio.x*ratio.y*(1.0f-ratio.z);
  sdf    +=	w * v.sdf;
  colorf += w * v_color;

  /// 111
  v = hash.GetVoxel(pos_corner+make_float3(offset, offset, offset));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = ratio.x*ratio.y*ratio.z;
  sdf    += w * v.sdf;
  colorf += w * v_color;

  color = make_uchar3(colorf.x, colorf.y, colorf.z);
  return true;
}

__device__
/// sdf_near: -, sdf_far: +
float LinearIntersection(float t_near, float t_far, float sdf_near, float sdf_far) {
  return t_near + (sdf_near / (sdf_near - sdf_far)) * (t_far - t_near);
}

// d0 near, d1 far
__device__
/// Iteratively
bool BisectionIntersection(const HashTable& hash_table,
                           const float3& world_pos_camera_origin, const float3& world_dir,
                           float sdf_near, float t_near,
                           float sdf_far,  float t_far,
                           float& t, uchar3& color) {
  float l = t_near, r = t_far, m = (l + r) * 0.5f;
  float l_sdf = sdf_near, r_sdf = sdf_far, m_sdf;

  const uint kIterations = 3;
#pragma unroll 1
  for(uint i = 0; i < kIterations; i++) {
    m = LinearIntersection(l, r, l_sdf, r_sdf);
    if(!TrilinearInterpolation(hash_table, world_pos_camera_origin + m * world_dir,
                               m_sdf, color))
      return false;

    if (l_sdf * m_sdf > 0.0) {
      l = m; l_sdf = m_sdf;
    } else {
      r = m; r_sdf = m_sdf;
    }
  }
  t = m;
  return true;
}

__device__
float3 GradientAtPoint(const HashTable& hash_table, const float3& pos) {
  const float voxelSize = kHashParams.voxel_size;
  float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

  /// negative
  float distn00; uchar3 colorn00;
  TrilinearInterpolation(hash_table, pos-make_float3(0.5f*offset.x, 0.0f, 0.0f), distn00, colorn00);
  float dist0n0; uchar3 color0n0;
  TrilinearInterpolation(hash_table, pos-make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0n0, color0n0);
  float dist00n; uchar3 color00n;
  TrilinearInterpolation(hash_table, pos-make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00n, color00n);

  /// positive
  float distp00; uchar3 colorp00;
  TrilinearInterpolation(hash_table, pos+make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00, colorp00);
  float dist0p0; uchar3 color0p0;
  TrilinearInterpolation(hash_table, pos+make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0, color0p0);
  float dist00p; uchar3 color00p;
  TrilinearInterpolation(hash_table, pos+make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p, color00p);

  float3 grad = make_float3((distp00-distn00)/offset.x,
                            (dist0p0-dist0n0)/offset.y,
                            (dist00p-dist00n)/offset.z);

  float l = length(grad);
  if(l == 0.0f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  return grad/l;
}

__global__
void CastKernel(const HashTable hash_table, RayCasterData rayCasterData,
                const float4x4 c_T_w, const float4x4 w_T_c) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  const RayCasterParams &ray_caster_params = kRayCasterParams;


  if (x >= ray_caster_params.width || y >= ray_caster_params.height)
    return;

  int pixel_idx = y * ray_caster_params.width + x;
  rayCasterData.depth_image_ [pixel_idx] = MINF;
  rayCasterData.vertex_image_[pixel_idx] = make_float4(MINF, MINF, MINF, MINF);
  rayCasterData.normal_image_[pixel_idx] = make_float4(MINF, MINF, MINF, MINF);
  rayCasterData.color_image_ [pixel_idx] = make_float4(MINF, MINF, MINF, MINF);

  /// Fix this! this uses the sensor's parameter instead of the viewer's
  /// 1. Determine ray direction
  float3 camera_dir = normalize(ImageReprojectToCamera(x, y, 1.0f));
  float ray_length_per_depth_unit = 1.0f / camera_dir.z;
  float t_min = ray_caster_params.min_raycast_depth / camera_dir.z;
  float t_max = ray_caster_params.max_raycast_depth / camera_dir.z;

  float3 world_pos_camera_origin = w_T_c * make_float3(0.0f, 0.0f, 0.0f);
  float4 world_dir_homo = w_T_c * make_float4(camera_dir, 0.0f);
  float3 world_dir = normalize(make_float3(world_dir_homo.x,
                                           world_dir_homo.y,
                                           world_dir_homo.z));

  RayCasterSample prev_sample;
  prev_sample.sdf = 0.0f;
  prev_sample.t = 0.0f;
  prev_sample.weight = 0;
  
  /// NOT zig-zag; just evenly sampling along the ray
#pragma unroll 1
  for (float t = t_min; t < t_max; t += ray_caster_params.raycast_step) {
    float3 world_pos_sample = world_pos_camera_origin + t * world_dir;
    float sdf;
    uchar3 color;

    if (TrilinearInterpolation(hash_table, world_pos_sample, sdf, color)) {
      /// Zero crossing
      if (prev_sample.weight > 0 && prev_sample.sdf > 0.0f && sdf < 0.0f) {

        float interpolated_t;
        uchar3 interpolated_color;
        /// Find isosurface
        bool is_isosurface_found = BisectionIntersection(hash_table,
                                                         world_pos_camera_origin, world_dir,
                                                         prev_sample.sdf, prev_sample.t, sdf, t,
                                                         interpolated_t, interpolated_color);

        float3 world_pos_isosurface = world_pos_camera_origin + interpolated_t * world_dir;
        /// Good enough sample
        if (is_isosurface_found && abs(prev_sample.sdf - sdf) < ray_caster_params.sample_sdf_threshold) {
          /// Trick from the original author of voxel-hashing
          if (abs(sdf) < ray_caster_params.sdf_threshold) {
            float depth = interpolated_t / ray_length_per_depth_unit;

            rayCasterData.depth_image_ [pixel_idx] = depth;
            rayCasterData.vertex_image_[pixel_idx] = make_float4(ImageReprojectToCamera(x, y, depth), 1.0f);
            rayCasterData.color_image_ [pixel_idx] = make_float4(interpolated_color.x / 255.f,
                                                                 interpolated_color.y / 255.f,
                                                                 interpolated_color.z / 255.f, 1.0f);
            if (ray_caster_params.enable_gradients) {
              float3 normal = GradientAtPoint(hash_table, world_pos_isosurface);
              normal = -normal;
              float4 n = c_T_w * make_float4(normal, 0.0f);
              rayCasterData.normal_image_[pixel_idx] = make_float4(n.x, n.y, n.z, 1.0f);
            }

            return;
          }
        }
      }

      /// No zero crossing || not good
      prev_sample.sdf = sdf;
      prev_sample.t = t;
      prev_sample.weight = 1;
    }
  }
}


__host__
void CastCudaHost(const HashTable        &hash_table,   const RayCasterData   &rayCastData,
              const RayCasterParams &ray_caster_params, const float4x4 &c_T_w, const float4x4 &w_T_c) {

  const dim3 gridSize((ray_caster_params.width + T_PER_BLOCK - 1)/T_PER_BLOCK, (ray_caster_params.height + T_PER_BLOCK - 1)/T_PER_BLOCK);
  const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

  CastKernel<<<gridSize, blockSize>>>(hash_table, rayCastData, c_T_w, w_T_c);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}