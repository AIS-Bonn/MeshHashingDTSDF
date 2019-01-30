#include <catch.hpp>

#include "mapping/voxel_traversal.hpp"

// Maximum distance along ray
const float truncation_distance = 0.4;

GeometryHelper get_geometry_helper()
{
  VolumeParams params;
  params.truncation_distance = 0.2;
  params.truncation_distance_scale = 0.0;
  params.voxel_size = 0.05;
  GeometryHelper geometry_helper(params);

  return geometry_helper;
}

TEST_CASE("test basic functionality", "[voxel_traversal]")
{
  float3 origin = make_float3(0, 0, 0);
  float3 direction = make_float3(2, 0, 0); // check, whether handles un-normalized directions

  GeometryHelper geometry_helper = get_geometry_helper();
  VoxelTraversal voxel_traversal(
      origin,
      direction,
      truncation_distance,
      geometry_helper
  );
  int count = 0;

  REQUIRE(voxel_traversal.HasNextVoxel());

  // Check if first voxel is correct
  int3 voxel = voxel_traversal.GetNextVoxel();
  count += 1;
  int3 origin_voxel = make_int3(0, 0, 0);
  REQUIRE(voxel == origin_voxel);

  while (voxel_traversal.HasNextVoxel())
  {
    voxel_traversal.GetNextVoxel();
    count += 1;
  }
  REQUIRE(count == truncation_distance / geometry_helper.voxel_size);
}

TEST_CASE("test non-axis-aligned direction", "[voxel_traversal]")
{
  float3 origin = make_float3(1.01, 1.12, 1.23);
  float3 direction = make_float3(-1, -1, -1); // 45 degrees to all axes

  GeometryHelper geometry_helper = get_geometry_helper();
  VoxelTraversal voxel_traversal(
      origin,
      direction,
      truncation_distance,
      geometry_helper
  );

  // Check if internal parameters are correctly initialized
  REQUIRE(voxel_traversal.tMax.x * voxel_traversal.direction.x == Approx(-0.01f));
  REQUIRE(voxel_traversal.tMax.y * voxel_traversal.direction.y == Approx(-0.02f));
  REQUIRE(voxel_traversal.tMax.z * voxel_traversal.direction.z == Approx(-0.03f));

  // Check outcoming voxels
  int3 voxel = make_int3(20, 22, 25); // 25 because 0.03 round up to next voxel!
  int count = 0;
  while (voxel_traversal.HasNextVoxel())
  {
    REQUIRE(voxel_traversal.GetNextVoxel() == voxel);

    // Update expected outcome for next turn (straight line, repeatedly crossing x,y,z in that order
    if (count % 3 == 0)
    {
      voxel.x -= 1;
    }
    else if (count % 3 == 1)
    {
      voxel.y -= 1;
    }
    else if (count % 3 == 2)
    {
      voxel.z -= 1;
    }
    count++;
  }
}

TEST_CASE("test truncation range", "[voxel_traversal]")
{
  float3 origin = make_float3(-0.303607, -0.516774, 0.001197);
  float3 direction = make_float3(-0.532084, -0.846631, -0.010123);

  GeometryHelper geometry_helper = get_geometry_helper();
  VoxelTraversal voxel_traversal(
      origin,
      direction,
      truncation_distance,
      geometry_helper
  );

  int count = 0;
  int3 voxel;
  while (voxel_traversal.HasNextVoxel())
  {
    voxel = voxel_traversal.GetNextVoxel();
    count += 1;
  }
  float dist = length(geometry_helper.VoxelToWorld(voxel) - origin);
  REQUIRE(dist >= truncation_distance - geometry_helper.voxel_size);
}