//
// Created by wei on 17-10-21.
//

#ifndef CORE_HASH_ENTRY_H
#define CORE_HASH_ENTRY_H

#include "core/common.h"
#include "helper_math.h"

struct __ALIGN__(8) HashEntry
{
  /** block position (lower left corner of SDFBlock)) */
  int3 pos;
  /** pointer into heap to SDFBlock */
  int ptr;
  /** offset for linked lists */
  uint offset;
  /** Flags for direction allocation */
  uchar direction_flags;

  __device__
  void operator=(const struct HashEntry &e)
  {
    ((long long *) this)[0] = ((const long long *) &e)[0];
    ((long long *) this)[1] = ((const long long *) &e)[1];
    ((int *) this)[4] = ((const int *) &e)[4];
  }

  __device__
  void Clear()
  {
    pos = make_int3(0);
    ptr = FREE_ENTRY;
    offset = 0;
    direction_flags = 0;
  }
};

#endif //MESH_HASHING_HASH_ENTRY_H
