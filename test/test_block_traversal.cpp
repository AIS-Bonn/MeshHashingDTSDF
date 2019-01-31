#include <catch.hpp>

#include "mapping/block_traversal.hpp"

// Maximum distance along ray
const float truncation_distance = 0.4;
const float block_size = 0.05;

TEST_CASE("test basic functionality", "[block_traversal]")
{
  float3 origin = make_float3(0, 0, 0);
  float3 direction = make_float3(2, 0, 0); // check, whether handles un-normalized directions

  BlockTraversal block_traversal(
      origin,
      direction,
      truncation_distance,
      block_size
  );
  int count = 0;

  REQUIRE(block_traversal.HasNextBlock());

  // Check if first block is correct
  int3 block = block_traversal.GetNextBlock();
  count += 1;
  int3 origin_block = make_int3(0, 0, 0);
  REQUIRE(block == origin_block);

  while (block_traversal.HasNextBlock())
  {
    block_traversal.GetNextBlock();
    count += 1;
  }
  REQUIRE(count == truncation_distance / block_size);
}

TEST_CASE("test non-axis-aligned direction", "[block_traversal]")
{
  float3 origin = make_float3(1.01, 1.12, 1.23);
  float3 direction = make_float3(-1, -1, -1); // 45 degrees to all axes

  BlockTraversal block_traversal(
      origin,
      direction,
      truncation_distance,
      block_size
  );

  // Check if internal parameters are correctly initialized
  REQUIRE(block_traversal.tMax.x * block_traversal.direction.x == Approx(-0.01f));
  REQUIRE(block_traversal.tMax.y * block_traversal.direction.y == Approx(-0.02f));
  REQUIRE(block_traversal.tMax.z * block_traversal.direction.z == Approx(-0.03f));

  // Check outcoming blocks
  int3 block = make_int3(20, 22, 25); // 25 because 0.03 round up to next block!
  int count = 0;
  while (block_traversal.HasNextBlock())
  {
    REQUIRE(block_traversal.GetNextBlock() == block);

    // Update expected outcome for next turn (straight line, repeatedly crossing x,y,z in that order
    if (count % 3 == 0)
    {
      block.x -= 1;
    } else if (count % 3 == 1)
    {
      block.y -= 1;
    } else if (count % 3 == 2)
    {
      block.z -= 1;
    }
    count++;
  }
}

TEST_CASE("test truncation range", "[block_traversal]")
{
//  float3 origin = make_float3(-0.303607, -0.516774, 0.001197);
//  float3 direction = make_float3(-0.532084, -0.846631, -0.010123);
  float3 origin = make_float3(-0.013223, 0.149254, 0.003236);
  float3 direction = make_float3(0.111472, -0.993766, 0.001749);

  BlockTraversal block_traversal(
      origin,
      direction,
      truncation_distance,
      block_size
  );

  int count = 0;
  int3 block;
  while (block_traversal.HasNextBlock())
  {
    block = block_traversal.GetNextBlock();
//    printf("(%i, %i, %i), ", block.x, block.y, block.z);
    count += 1;
  }
  float dist = length(make_float3(block) * block_size - origin);
  REQUIRE(dist >= truncation_distance - block_size);
}