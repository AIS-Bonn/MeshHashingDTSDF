@0xa417735ba73742d0;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("block_serialization");

struct Map(Key, Value) {
  entries @0 :List(Entry);
  struct Entry {
    key @0 :Key;
    value @1 :Value;
  }
}

struct Coordinate {
x @0 :Int32;
y @1 :Int32;
z @2 :Int32;
}

struct VoxelArray {
sdf @0 :List(Float32);
weight @1 :List(Float32);
}

struct Block {
voxelArrays @0 :List(Int32);
}

struct BlockMap {
  blocks @0 :Map(Coordinate, Block);
  voxelArrays @1 :List(VoxelArray);
}
