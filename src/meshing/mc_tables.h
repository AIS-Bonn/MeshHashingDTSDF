//
// Created by wei on 17-5-1.
//
// Tables for marching cubes

#ifndef VH_MC_TABLES_H
#define VH_MC_TABLES_H

#include <vector_types.h>

// Polygonising a scalar field
// Also known as: "3D Contouring", "Marching Cubes", "Surface Reconstruction" 
// Written by Paul Bourke
// May 1994 
// http://paulbourke.net/geometry/polygonise/
__device__
const static int kCubeEdges[256] = {
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};


__device__
const static int kTriangleVertexEdge[256][16] = {
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  1,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  8,  3,  9,  8,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  3,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9,  2,  10, 0,  2,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2,  8,  3,  2,  10, 8,  10, 9,  8,  -1, -1, -1, -1, -1, -1, -1},
    {3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  11, 2,  8,  11, 0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  9,  0,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  11, 2,  1,  9,  11, 9,  8,  11, -1, -1, -1, -1, -1, -1, -1},
    {3,  10, 1,  11, 10, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  10, 1,  0,  8,  10, 8,  11, 10, -1, -1, -1, -1, -1, -1, -1},
    {3,  9,  0,  3,  11, 9,  11, 10, 9,  -1, -1, -1, -1, -1, -1, -1},
    {9,  8,  10, 10, 8,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  3,  0,  7,  3,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  1,  9,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  1,  9,  4,  7,  1,  7,  3,  1,  -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  10, 8,  4,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3,  4,  7,  3,  0,  4,  1,  2,  10, -1, -1, -1, -1, -1, -1, -1},
    {9,  2,  10, 9,  0,  2,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1},
    {2,  10, 9,  2,  9,  7,  2,  7,  3,  7,  9,  4,  -1, -1, -1, -1},
    {8,  4,  7,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 4,  7,  11, 2,  4,  2,  0,  4,  -1, -1, -1, -1, -1, -1, -1},
    {9,  0,  1,  8,  4,  7,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1},
    {4,  7,  11, 9,  4,  11, 9,  11, 2,  9,  2,  1,  -1, -1, -1, -1},
    {3,  10, 1,  3,  11, 10, 7,  8,  4,  -1, -1, -1, -1, -1, -1, -1},
    {1,  11, 10, 1,  4,  11, 1,  0,  4,  7,  11, 4,  -1, -1, -1, -1},
    {4,  7,  8,  9,  0,  11, 9,  11, 10, 11, 0,  3,  -1, -1, -1, -1},
    {4,  7,  11, 4,  11, 9,  9,  11, 10, -1, -1, -1, -1, -1, -1, -1},
    {9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9,  5,  4,  0,  8,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  5,  4,  1,  5,  0,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8,  5,  4,  8,  3,  5,  3,  1,  5,  -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  10, 9,  5,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3,  0,  8,  1,  2,  10, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1},
    {5,  2,  10, 5,  4,  2,  4,  0,  2,  -1, -1, -1, -1, -1, -1, -1},
    {2,  10, 5,  3,  2,  5,  3,  5,  4,  3,  4,  8,  -1, -1, -1, -1},
    {9,  5,  4,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  11, 2,  0,  8,  11, 4,  9,  5,  -1, -1, -1, -1, -1, -1, -1},
    {0,  5,  4,  0,  1,  5,  2,  3,  11, -1, -1, -1, -1, -1, -1, -1},
    {2,  1,  5,  2,  5,  8,  2,  8,  11, 4,  8,  5,  -1, -1, -1, -1},
    {10, 3,  11, 10, 1,  3,  9,  5,  4,  -1, -1, -1, -1, -1, -1, -1},
    {4,  9,  5,  0,  8,  1,  8,  10, 1,  8,  11, 10, -1, -1, -1, -1},
    {5,  4,  0,  5,  0,  11, 5,  11, 10, 11, 0,  3,  -1, -1, -1, -1},
    {5,  4,  8,  5,  8,  10, 10, 8,  11, -1, -1, -1, -1, -1, -1, -1},
    {9,  7,  8,  5,  7,  9,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9,  3,  0,  9,  5,  3,  5,  7,  3,  -1, -1, -1, -1, -1, -1, -1},
    {0,  7,  8,  0,  1,  7,  1,  5,  7,  -1, -1, -1, -1, -1, -1, -1},
    {1,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9,  7,  8,  9,  5,  7,  10, 1,  2,  -1, -1, -1, -1, -1, -1, -1},
    {10, 1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3,  -1, -1, -1, -1},
    {8,  0,  2,  8,  2,  5,  8,  5,  7,  10, 5,  2,  -1, -1, -1, -1},
    {2,  10, 5,  2,  5,  3,  3,  5,  7,  -1, -1, -1, -1, -1, -1, -1},
    {7,  9,  5,  7,  8,  9,  3,  11, 2,  -1, -1, -1, -1, -1, -1, -1},
    {9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7,  11, -1, -1, -1, -1},
    {2,  3,  11, 0,  1,  8,  1,  7,  8,  1,  5,  7,  -1, -1, -1, -1},
    {11, 2,  1,  11, 1,  7,  7,  1,  5,  -1, -1, -1, -1, -1, -1, -1},
    {9,  5,  8,  8,  5,  7,  10, 1,  3,  10, 3,  11, -1, -1, -1, -1},
    {5,  7,  0,  5,  0,  9,  7,  11, 0,  1,  0,  10, 11, 10, 0,  -1},
    {11, 10, 0,  11, 0,  3,  10, 5,  0,  8,  0,  7,  5,  7,  0,  -1},
    {11, 10, 5,  7,  11, 5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  3,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9,  0,  1,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  8,  3,  1,  9,  8,  5,  10, 6,  -1, -1, -1, -1, -1, -1, -1},
    {1,  6,  5,  2,  6,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  6,  5,  1,  2,  6,  3,  0,  8,  -1, -1, -1, -1, -1, -1, -1},
    {9,  6,  5,  9,  0,  6,  0,  2,  6,  -1, -1, -1, -1, -1, -1, -1},
    {5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8,  -1, -1, -1, -1},
    {2,  3,  11, 10, 6,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 0,  8,  11, 2,  0,  10, 6,  5,  -1, -1, -1, -1, -1, -1, -1},
    {0,  1,  9,  2,  3,  11, 5,  10, 6,  -1, -1, -1, -1, -1, -1, -1},
    {5,  10, 6,  1,  9,  2,  9,  11, 2,  9,  8,  11, -1, -1, -1, -1},
    {6,  3,  11, 6,  5,  3,  5,  1,  3,  -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  11, 0,  11, 5,  0,  5,  1,  5,  11, 6,  -1, -1, -1, -1},
    {3,  11, 6,  0,  3,  6,  0,  6,  5,  0,  5,  9,  -1, -1, -1, -1},
    {6,  5,  9,  6,  9,  11, 11, 9,  8,  -1, -1, -1, -1, -1, -1, -1},
    {5,  10, 6,  4,  7,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  3,  0,  4,  7,  3,  6,  5,  10, -1, -1, -1, -1, -1, -1, -1},
    {1,  9,  0,  5,  10, 6,  8,  4,  7,  -1, -1, -1, -1, -1, -1, -1},
    {10, 6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4,  -1, -1, -1, -1},
    {6,  1,  2,  6,  5,  1,  4,  7,  8,  -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7,  -1, -1, -1, -1},
    {8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6,  -1, -1, -1, -1},
    {7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9,  -1},
    {3,  11, 2,  7,  8,  4,  10, 6,  5,  -1, -1, -1, -1, -1, -1, -1},
    {5,  10, 6,  4,  7,  2,  4,  2,  0,  2,  7,  11, -1, -1, -1, -1},
    {0,  1,  9,  4,  7,  8,  2,  3,  11, 5,  10, 6,  -1, -1, -1, -1},
    {9,  2,  1,  9,  11, 2,  9,  4,  11, 7,  11, 4,  5,  10, 6,  -1},
    {8,  4,  7,  3,  11, 5,  3,  5,  1,  5,  11, 6,  -1, -1, -1, -1},
    {5,  1,  11, 5,  11, 6,  1,  0,  11, 7,  11, 4,  0,  4,  11, -1},
    {0,  5,  9,  0,  6,  5,  0,  3,  6,  11, 6,  3,  8,  4,  7,  -1},
    {6,  5,  9,  6,  9,  11, 4,  7,  9,  7,  11, 9,  -1, -1, -1, -1},
    {10, 4,  9,  6,  4,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  10, 6,  4,  9,  10, 0,  8,  3,  -1, -1, -1, -1, -1, -1, -1},
    {10, 0,  1,  10, 6,  0,  6,  4,  0,  -1, -1, -1, -1, -1, -1, -1},
    {8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1,  10, -1, -1, -1, -1},
    {1,  4,  9,  1,  2,  4,  2,  6,  4,  -1, -1, -1, -1, -1, -1, -1},
    {3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4,  -1, -1, -1, -1},
    {0,  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8,  3,  2,  8,  2,  4,  4,  2,  6,  -1, -1, -1, -1, -1, -1, -1},
    {10, 4,  9,  10, 6,  4,  11, 2,  3,  -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  2,  2,  8,  11, 4,  9,  10, 4,  10, 6,  -1, -1, -1, -1},
    {3,  11, 2,  0,  1,  6,  0,  6,  4,  6,  1,  10, -1, -1, -1, -1},
    {6,  4,  1,  6,  1,  10, 4,  8,  1,  2,  1,  11, 8,  11, 1,  -1},
    {9,  6,  4,  9,  3,  6,  9,  1,  3,  11, 6,  3,  -1, -1, -1, -1},
    {8,  11, 1,  8,  1,  0,  11, 6,  1,  9,  1,  4,  6,  4,  1,  -1},
    {3,  11, 6,  3,  6,  0,  0,  6,  4,  -1, -1, -1, -1, -1, -1, -1},
    {6,  4,  8,  11, 6,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7,  10, 6,  7,  8,  10, 8,  9,  10, -1, -1, -1, -1, -1, -1, -1},
    {0,  7,  3,  0,  10, 7,  0,  9,  10, 6,  7,  10, -1, -1, -1, -1},
    {10, 6,  7,  1,  10, 7,  1,  7,  8,  1,  8,  0,  -1, -1, -1, -1},
    {10, 6,  7,  10, 7,  1,  1,  7,  3,  -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7,  -1, -1, -1, -1},
    {2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9,  -1},
    {7,  8,  0,  7,  0,  6,  6,  0,  2,  -1, -1, -1, -1, -1, -1, -1},
    {7,  3,  2,  6,  7,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2,  3,  11, 10, 6,  8,  10, 8,  9,  8,  6,  7,  -1, -1, -1, -1},
    {2,  0,  7,  2,  7,  11, 0,  9,  7,  6,  7,  10, 9,  10, 7,  -1},
    {1,  8,  0,  1,  7,  8,  1,  10, 7,  6,  7,  10, 2,  3,  11, -1},
    {11, 2,  1,  11, 1,  7,  10, 6,  1,  6,  7,  1,  -1, -1, -1, -1},
    {8,  9,  6,  8,  6,  7,  9,  1,  6,  11, 6,  3,  1,  3,  6,  -1},
    {0,  9,  1,  11, 6,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7,  8,  0,  7,  0,  6,  3,  11, 0,  11, 6,  0,  -1, -1, -1, -1},
    {7,  11, 6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7,  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3,  0,  8,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  1,  9,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8,  1,  9,  8,  3,  1,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1},
    {10, 1,  2,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  10, 3,  0,  8,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1},
    {2,  9,  0,  2,  10, 9,  6,  11, 7,  -1, -1, -1, -1, -1, -1, -1},
    {6,  11, 7,  2,  10, 3,  10, 8,  3,  10, 9,  8,  -1, -1, -1, -1},
    {7,  2,  3,  6,  2,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {7,  0,  8,  7,  6,  0,  6,  2,  0,  -1, -1, -1, -1, -1, -1, -1},
    {2,  7,  6,  2,  3,  7,  0,  1,  9,  -1, -1, -1, -1, -1, -1, -1},
    {1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6,  -1, -1, -1, -1},
    {10, 7,  6,  10, 1,  7,  1,  3,  7,  -1, -1, -1, -1, -1, -1, -1},
    {10, 7,  6,  1,  7,  10, 1,  8,  7,  1,  0,  8,  -1, -1, -1, -1},
    {0,  3,  7,  0,  7,  10, 0,  10, 9,  6,  10, 7,  -1, -1, -1, -1},
    {7,  6,  10, 7,  10, 8,  8,  10, 9,  -1, -1, -1, -1, -1, -1, -1},
    {6,  8,  4,  11, 8,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3,  6,  11, 3,  0,  6,  0,  4,  6,  -1, -1, -1, -1, -1, -1, -1},
    {8,  6,  11, 8,  4,  6,  9,  0,  1,  -1, -1, -1, -1, -1, -1, -1},
    {9,  4,  6,  9,  6,  3,  9,  3,  1,  11, 3,  6,  -1, -1, -1, -1},
    {6,  8,  4,  6,  11, 8,  2,  10, 1,  -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  10, 3,  0,  11, 0,  6,  11, 0,  4,  6,  -1, -1, -1, -1},
    {4,  11, 8,  4,  6,  11, 0,  2,  9,  2,  10, 9,  -1, -1, -1, -1},
    {10, 9,  3,  10, 3,  2,  9,  4,  3,  11, 3,  6,  4,  6,  3,  -1},
    {8,  2,  3,  8,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1},
    {0,  4,  2,  4,  6,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8,  -1, -1, -1, -1},
    {1,  9,  4,  1,  4,  2,  2,  4,  6,  -1, -1, -1, -1, -1, -1, -1},
    {8,  1,  3,  8,  6,  1,  8,  4,  6,  6,  10, 1,  -1, -1, -1, -1},
    {10, 1,  0,  10, 0,  6,  6,  0,  4,  -1, -1, -1, -1, -1, -1, -1},
    {4,  6,  3,  4,  3,  8,  6,  10, 3,  0,  3,  9,  10, 9,  3,  -1},
    {10, 9,  4,  6,  10, 4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  9,  5,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  3,  4,  9,  5,  11, 7,  6,  -1, -1, -1, -1, -1, -1, -1},
    {5,  0,  1,  5,  4,  0,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1},
    {11, 7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5,  -1, -1, -1, -1},
    {9,  5,  4,  10, 1,  2,  7,  6,  11, -1, -1, -1, -1, -1, -1, -1},
    {6,  11, 7,  1,  2,  10, 0,  8,  3,  4,  9,  5,  -1, -1, -1, -1},
    {7,  6,  11, 5,  4,  10, 4,  2,  10, 4,  0,  2,  -1, -1, -1, -1},
    {3,  4,  8,  3,  5,  4,  3,  2,  5,  10, 5,  2,  11, 7,  6,  -1},
    {7,  2,  3,  7,  6,  2,  5,  4,  9,  -1, -1, -1, -1, -1, -1, -1},
    {9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7,  -1, -1, -1, -1},
    {3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0,  -1, -1, -1, -1},
    {6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8,  -1},
    {9,  5,  4,  10, 1,  6,  1,  7,  6,  1,  3,  7,  -1, -1, -1, -1},
    {1,  6,  10, 1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4,  -1},
    {4,  0,  10, 4,  10, 5,  0,  3,  10, 6,  10, 7,  3,  7,  10, -1},
    {7,  6,  10, 7,  10, 8,  5,  4,  10, 4,  8,  10, -1, -1, -1, -1},
    {6,  9,  5,  6,  11, 9,  11, 8,  9,  -1, -1, -1, -1, -1, -1, -1},
    {3,  6,  11, 0,  6,  3,  0,  5,  6,  0,  9,  5,  -1, -1, -1, -1},
    {0,  11, 8,  0,  5,  11, 0,  1,  5,  5,  6,  11, -1, -1, -1, -1},
    {6,  11, 3,  6,  3,  5,  5,  3,  1,  -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  10, 9,  5,  11, 9,  11, 8,  11, 5,  6,  -1, -1, -1, -1},
    {0,  11, 3,  0,  6,  11, 0,  9,  6,  5,  6,  9,  1,  2,  10, -1},
    {11, 8,  5,  11, 5,  6,  8,  0,  5,  10, 5,  2,  0,  2,  5,  -1},
    {6,  11, 3,  6,  3,  5,  2,  10, 3,  10, 5,  3,  -1, -1, -1, -1},
    {5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2,  -1, -1, -1, -1},
    {9,  5,  6,  9,  6,  0,  0,  6,  2,  -1, -1, -1, -1, -1, -1, -1},
    {1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8,  -1},
    {1,  5,  6,  2,  1,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  3,  6,  1,  6,  10, 3,  8,  6,  5,  6,  9,  8,  9,  6,  -1},
    {10, 1,  0,  10, 0,  6,  9,  5,  0,  5,  6,  0,  -1, -1, -1, -1},
    {0,  3,  8,  5,  6,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {10, 5,  6,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5,  10, 7,  5,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {11, 5,  10, 11, 7,  5,  8,  3,  0,  -1, -1, -1, -1, -1, -1, -1},
    {5,  11, 7,  5,  10, 11, 1,  9,  0,  -1, -1, -1, -1, -1, -1, -1},
    {10, 7,  5,  10, 11, 7,  9,  8,  1,  8,  3,  1,  -1, -1, -1, -1},
    {11, 1,  2,  11, 7,  1,  7,  5,  1,  -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2,  11, -1, -1, -1, -1},
    {9,  7,  5,  9,  2,  7,  9,  0,  2,  2,  11, 7,  -1, -1, -1, -1},
    {7,  5,  2,  7,  2,  11, 5,  9,  2,  3,  2,  8,  9,  8,  2,  -1},
    {2,  5,  10, 2,  3,  5,  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1},
    {8,  2,  0,  8,  5,  2,  8,  7,  5,  10, 2,  5,  -1, -1, -1, -1},
    {9,  0,  1,  5,  10, 3,  5,  3,  7,  3,  10, 2,  -1, -1, -1, -1},
    {9,  8,  2,  9,  2,  1,  8,  7,  2,  10, 2,  5,  7,  5,  2,  -1},
    {1,  3,  5,  3,  7,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  7,  0,  7,  1,  1,  7,  5,  -1, -1, -1, -1, -1, -1, -1},
    {9,  0,  3,  9,  3,  5,  5,  3,  7,  -1, -1, -1, -1, -1, -1, -1},
    {9,  8,  7,  5,  9,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {5,  8,  4,  5,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1},
    {5,  0,  4,  5,  11, 0,  5,  10, 11, 11, 3,  0,  -1, -1, -1, -1},
    {0,  1,  9,  8,  4,  10, 8,  10, 11, 10, 4,  5,  -1, -1, -1, -1},
    {10, 11, 4,  10, 4,  5,  11, 3,  4,  9,  4,  1,  3,  1,  4,  -1},
    {2,  5,  1,  2,  8,  5,  2,  11, 8,  4,  5,  8,  -1, -1, -1, -1},
    {0,  4,  11, 0,  11, 3,  4,  5,  11, 2,  11, 1,  5,  1,  11, -1},
    {0,  2,  5,  0,  5,  9,  2,  11, 5,  4,  5,  8,  11, 8,  5,  -1},
    {9,  4,  5,  2,  11, 3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2,  5,  10, 3,  5,  2,  3,  4,  5,  3,  8,  4,  -1, -1, -1, -1},
    {5,  10, 2,  5,  2,  4,  4,  2,  0,  -1, -1, -1, -1, -1, -1, -1},
    {3,  10, 2,  3,  5,  10, 3,  8,  5,  4,  5,  8,  0,  1,  9,  -1},
    {5,  10, 2,  5,  2,  4,  1,  9,  2,  9,  4,  2,  -1, -1, -1, -1},
    {8,  4,  5,  8,  5,  3,  3,  5,  1,  -1, -1, -1, -1, -1, -1, -1},
    {0,  4,  5,  1,  0,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5,  -1, -1, -1, -1},
    {9,  4,  5,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  11, 7,  4,  9,  11, 9,  10, 11, -1, -1, -1, -1, -1, -1, -1},
    {0,  8,  3,  4,  9,  7,  9,  11, 7,  9,  10, 11, -1, -1, -1, -1},
    {1,  10, 11, 1,  11, 4,  1,  4,  0,  7,  4,  11, -1, -1, -1, -1},
    {3,  1,  4,  3,  4,  8,  1,  10, 4,  7,  4,  11, 10, 11, 4,  -1},
    {4,  11, 7,  9,  11, 4,  9,  2,  11, 9,  1,  2,  -1, -1, -1, -1},
    {9,  7,  4,  9,  11, 7,  9,  1,  11, 2,  11, 1,  0,  8,  3,  -1},
    {11, 7,  4,  11, 4,  2,  2,  4,  0,  -1, -1, -1, -1, -1, -1, -1},
    {11, 7,  4,  11, 4,  2,  8,  3,  4,  3,  2,  4,  -1, -1, -1, -1},
    {2,  9,  10, 2,  7,  9,  2,  3,  7,  7,  4,  9,  -1, -1, -1, -1},
    {9,  10, 7,  9,  7,  4,  10, 2,  7,  8,  7,  0,  2,  0,  7,  -1},
    {3,  7,  10, 3,  10, 2,  7,  4,  10, 1,  10, 0,  4,  0,  10, -1},
    {1,  10, 2,  8,  7,  4,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  9,  1,  4,  1,  7,  7,  1,  3,  -1, -1, -1, -1, -1, -1, -1},
    {4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1,  -1, -1, -1, -1},
    {4,  0,  3,  7,  4,  3,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {4,  8,  7,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {9,  10, 8,  10, 11, 8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3,  0,  9,  3,  9,  11, 11, 9,  10, -1, -1, -1, -1, -1, -1, -1},
    {0,  1,  10, 0,  10, 8,  8,  10, 11, -1, -1, -1, -1, -1, -1, -1},
    {3,  1,  10, 11, 3,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  2,  11, 1,  11, 9,  9,  11, 8,  -1, -1, -1, -1, -1, -1, -1},
    {3,  0,  9,  3,  9,  11, 1,  2,  9,  2,  11, 9,  -1, -1, -1, -1},
    {0,  2,  11, 8,  0,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {3,  2,  11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2,  3,  8,  2,  8,  10, 10, 8,  9,  -1, -1, -1, -1, -1, -1, -1},
    {9,  10, 2,  0,  9,  2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {2,  3,  8,  2,  8,  10, 0,  1,  8,  1,  10, 8,  -1, -1, -1, -1},
    {1,  10, 2,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {1,  3,  8,  9,  1,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  9,  1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {0,  3,  8,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

// Vertex mapping
// {0 ... 7} -> ({0,1}, {0,1}, {0,1})
/// Refer to http://paulbourke.net/geometry/polygonise
/// Our coordinate system:
///       ^
///      /
///    z
///   /
/// o -- x -->
/// |
/// y
/// |
/// v
// 0 -> 011
// 1 -> 111
// 2 -> 110
// 3 -> 010
// 4 -> 001
// 5 -> 101
// 6 -> 100
// 7 -> 000
__device__
const static int3 kVtxOffset[8] = {
    {0, 1, 1},
    {1, 1, 1},
    {1, 1, 0},
    {0, 1, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 0, 0},
    {0, 0, 0}
};

/**
 * Table to map from each voxel edge to an offset for determining the responsible MeshUnit.
 *
 * FORMAT:
 *      offset + vertex_index, ({0,1}^3, {0,1,2})
 *
 * Offset denotes how many steps the MeshUnit is away in (x,y,z) direction.
 * vertex_index determines which of the 3 vertices inside the MeshUnit correspond to that edge.
 *
 *  0 -> 011.x, (0, 1)
 *  1 -> 110.z, (1, 2)
 *  2 -> 010.x, (2, 3)
 *  3 -> 010.z, (3, 0)
 *  4 -> 001.x, (4, 5)
 *  5 -> 100.z, (5, 6)
 *  6 -> 000.x, (6, 7)
 *  7 -> 000.z, (7, 4)
 *  8 -> 001.y, (4, 0)
 *  9 -> 101.y, (5, 1)
 * 10 -> 100.y, (6, 2)
 * 11 -> 000.y, (7, 3)
 */
__device__
const static uint4 kEdgeOwnerCubeOffset[12] = {
    {0, 1, 1, 0},
    {1, 1, 0, 2},
    {0, 1, 0, 0},
    {0, 1, 0, 2},
    {0, 0, 1, 0},
    {1, 0, 0, 2},
    {0, 0, 0, 0},
    {0, 0, 0, 2},
    {0, 0, 1, 1},
    {1, 0, 1, 1},
    {1, 0, 0, 1},
    {0, 0, 0, 1}
};

/**
 * Map from edge index to both endpoint voxel corner indices
 */
__device__
const static int2 kEdgeEndpointVertices[12] = {
    // (Changed order of vertices, s.t. all edges point from left-to-right, top-to-bottom, front-to-back)
    {0, 1},
    {2, 1},
    {3, 2},
    {3, 0},
    {4, 5},
    {6, 5},
    {7, 6},
    {7, 4},
    {4, 0},
    {5, 1},
    {6, 2},
    {7, 3}
};

/**
 * Decomposes a marching cubes index into its up to 4 separate surface components (again mc indices)
 */
__device__
const static short kIndexDecomposition[256][4] = {
    {0,   -1,  -1,  -1},
    {1,   -1,  -1,  -1},
    {2,   -1,  -1,  -1},
    {3,   -1,  -1,  -1},
    {4,   -1,  -1,  -1},
    {1,   4,   -1,  -1},
    {6,   -1,  -1,  -1},
    {7,   -1,  -1,  -1},
    {8,   -1,  -1,  -1},
    {9,   -1,  -1,  -1},
    {2,   8,   -1,  -1},
    {11,  -1,  -1,  -1},
    {12,  -1,  -1,  -1},
    {13,  -1,  -1,  -1},
    {14,  -1,  -1,  -1},
    {15,  -1,  -1,  -1},
    {16,  -1,  -1,  -1},
    {17,  -1,  -1,  -1},
    {2,   16,  -1,  -1},
    {19,  -1,  -1,  -1},
    {4,   16,  -1,  -1},
    {17,  4,   -1,  -1},
    {6,   16,  -1,  -1},
    {23,  -1,  -1,  -1},
    {8,   16,  -1,  -1},
    {25,  -1,  -1,  -1},
    {2,   8,   16,  -1},
    {27,  -1,  -1,  -1},
    {12,  16,  -1,  -1},
    {29,  -1,  -1,  -1},
    {14,  16,  -1,  -1},
    {31,  -1,  -1,  -1},
    {32,  -1,  -1,  -1},
    {1,   32,  -1,  -1},
    {34,  -1,  -1,  -1},
    {35,  -1,  -1,  -1},
    {4,   32,  -1,  -1},
    {1,   4,   32,  -1},
    {38,  -1,  -1,  -1},
    {39,  -1,  -1,  -1},
    {8,   32,  -1,  -1},
    {9,   32,  -1,  -1},
    {34,  8,   -1,  -1},
    {43,  -1,  -1,  -1},
    {12,  32,  -1,  -1},
    {13,  32,  -1,  -1},
    {46,  -1,  -1,  -1},
    {47,  -1,  -1,  -1},
    {48,  -1,  -1,  -1},
    {49,  -1,  -1,  -1},
    {50,  -1,  -1,  -1},
    {51,  -1,  -1,  -1},
    {4,   48,  -1,  -1},
    {49,  4,   -1,  -1},
    {54,  -1,  -1,  -1},
    {55,  -1,  -1,  -1},
    {8,   48,  -1,  -1},
    {57,  -1,  -1,  -1},
    {50,  8,   -1,  -1},
    {59,  -1,  -1,  -1},
    {12,  48,  -1,  -1},
    {61,  -1,  -1,  -1},
    {62,  -1,  -1,  -1},
    {63,  -1,  -1,  -1},
    {64,  -1,  -1,  -1},
    {1,   64,  -1,  -1},
    {2,   64,  -1,  -1},
    {3,   64,  -1,  -1},
    {68,  -1,  -1,  -1},
    {1,   68,  -1,  -1},
    {70,  -1,  -1,  -1},
    {71,  -1,  -1,  -1},
    {8,   64,  -1,  -1},
    {9,   64,  -1,  -1},
    {2,   8,   64,  -1},
    {11,  64,  -1,  -1},
    {76,  -1,  -1,  -1},
    {77,  -1,  -1,  -1},
    {78,  -1,  -1,  -1},
    {79,  -1,  -1,  -1},
    {16,  64,  -1,  -1},
    {17,  64,  -1,  -1},
    {2,   16,  64,  -1},
    {19,  64,  -1,  -1},
    {68,  16,  -1,  -1},
    {17,  68,  -1,  -1},
    {70,  16,  -1,  -1},
    {87,  -1,  -1,  -1},
    {8,   16,  64,  -1},
    {25,  64,  -1,  -1},
    {2,   8,   16,  64},
    {27,  64,  -1,  -1},
    {76,  16,  -1,  -1},
    {93,  -1,  -1,  -1},
    {78,  16,  -1,  -1},
    {95,  -1,  -1,  -1},
    {96,  -1,  -1,  -1},
    {1,   96,  -1,  -1},
    {98,  -1,  -1,  -1},
    {99,  -1,  -1,  -1},
    {100, -1,  -1,  -1},
    {1,   100, -1,  -1},
    {102, -1,  -1,  -1},
    {103, -1,  -1,  -1},
    {8,   96,  -1,  -1},
    {9,   96,  -1,  -1},
    {98,  8,   -1,  -1},
    {107, -1,  -1,  -1},
    {108, -1,  -1,  -1},
    {109, -1,  -1,  -1},
    {110, -1,  -1,  -1},
    {111, -1,  -1,  -1},
    {112, -1,  -1,  -1},
    {113, -1,  -1,  -1},
    {114, -1,  -1,  -1},
    {115, -1,  -1,  -1},
    {116, -1,  -1,  -1},
    {117, -1,  -1,  -1},
    {118, -1,  -1,  -1},
    {119, -1,  -1,  -1},
    {8,   112, -1,  -1},
    {121, -1,  -1,  -1},
    {114, 8,   -1,  -1},
    {123, -1,  -1,  -1},
    {124, -1,  -1,  -1},
    {125, -1,  -1,  -1},
    {126, -1,  -1,  -1},
    {127, -1,  -1,  -1},
    {128, -1,  -1,  -1},
    {1,   128, -1,  -1},
    {2,   128, -1,  -1},
    {3,   128, -1,  -1},
    {4,   128, -1,  -1},
    {1,   4,   128, -1},
    {6,   128, -1,  -1},
    {7,   128, -1,  -1},
    {136, -1,  -1,  -1},
    {137, -1,  -1,  -1},
    {2,   136, -1,  -1},
    {139, -1,  -1,  -1},
    {140, -1,  -1,  -1},
    {141, -1,  -1,  -1},
    {142, -1,  -1,  -1},
    {143, -1,  -1,  -1},
    {144, -1,  -1,  -1},
    {145, -1,  -1,  -1},
    {2,   144, -1,  -1},
    {147, -1,  -1,  -1},
    {4,   144, -1,  -1},
    {145, 4,   -1,  -1},
    {6,   144, -1,  -1},
    {151, -1,  -1,  -1},
    {152, -1,  -1,  -1},
    {153, -1,  -1,  -1},
    {2,   152, -1,  -1},
    {155, -1,  -1,  -1},
    {156, -1,  -1,  -1},
    {157, -1,  -1,  -1},
    {158, -1,  -1,  -1},
    {159, -1,  -1,  -1},
    {32,  128, -1,  -1},
    {1,   32,  128, -1},
    {34,  128, -1,  -1},
    {35,  128, -1,  -1},
    {4,   32,  128, -1},
    {1,   4,   32,  128},
    {38,  128, -1,  -1},
    {39,  128, -1,  -1},
    {136, 32,  -1,  -1},
    {137, 32,  -1,  -1},
    {34,  136, -1,  -1},
    {171, -1,  -1,  -1},
    {140, 32,  -1,  -1},
    {141, 32,  -1,  -1},
    {174, -1,  -1,  -1},
    {175, -1,  -1,  -1},
    {176, -1,  -1,  -1},
    {177, -1,  -1,  -1},
    {178, -1,  -1,  -1},
    {179, -1,  -1,  -1},
    {4,   176, -1,  -1},
    {177, 4,   -1,  -1},
    {182, -1,  -1,  -1},
    {183, -1,  -1,  -1},
    {184, -1,  -1,  -1},
    {185, -1,  -1,  -1},
    {186, -1,  -1,  -1},
    {187, -1,  -1,  -1},
    {188, -1,  -1,  -1},
    {189, -1,  -1,  -1},
    {190, -1,  -1,  -1},
    {191, -1,  -1,  -1},
    {192, -1,  -1,  -1},
    {1,   192, -1,  -1},
    {2,   192, -1,  -1},
    {3,   192, -1,  -1},
    {196, -1,  -1,  -1},
    {1,   196, -1,  -1},
    {198, -1,  -1,  -1},
    {199, -1,  -1,  -1},
    {200, -1,  -1,  -1},
    {201, -1,  -1,  -1},
    {2,   200, -1,  -1},
    {203, -1,  -1,  -1},
    {204, -1,  -1,  -1},
    {205, -1,  -1,  -1},
    {206, -1,  -1,  -1},
    {207, -1,  -1,  -1},
    {208, -1,  -1,  -1},
    {209, -1,  -1,  -1},
    {2,   208, -1,  -1},
    {211, -1,  -1,  -1},
    {212, -1,  -1,  -1},
    {213, -1,  -1,  -1},
    {214, -1,  -1,  -1},
    {215, -1,  -1,  -1},
    {216, -1,  -1,  -1},
    {217, -1,  -1,  -1},
    {2,   216, -1,  -1},
    {219, -1,  -1,  -1},
    {220, -1,  -1,  -1},
    {221, -1,  -1,  -1},
    {222, -1,  -1,  -1},
    {223, -1,  -1,  -1},
    {224, -1,  -1,  -1},
    {1,   224, -1,  -1},
    {226, -1,  -1,  -1},
    {227, -1,  -1,  -1},
    {228, -1,  -1,  -1},
    {1,   228, -1,  -1},
    {230, -1,  -1,  -1},
    {231, -1,  -1,  -1},
    {232, -1,  -1,  -1},
    {233, -1,  -1,  -1},
    {234, -1,  -1,  -1},
    {235, -1,  -1,  -1},
    {236, -1,  -1,  -1},
    {237, -1,  -1,  -1},
    {238, -1,  -1,  -1},
    {239, -1,  -1,  -1},
    {240, -1,  -1,  -1},
    {241, -1,  -1,  -1},
    {242, -1,  -1,  -1},
    {243, -1,  -1,  -1},
    {244, -1,  -1,  -1},
    {245, -1,  -1,  -1},
    {246, -1,  -1,  -1},
    {247, -1,  -1,  -1},
    {248, -1,  -1,  -1},
    {249, -1,  -1,  -1},
    {250, -1,  -1,  -1},
    {251, -1,  -1,  -1},
    {252, -1,  -1,  -1},
    {253, -1,  -1,  -1},
    {254, -1,  -1,  -1},
    {255, -1,  -1,  -1}
};


/**
 * Compatibility between marching cubes indices and directions.
 */
__device__
const static unsigned char kIndexDirectionCompatibility[256][6] = {
    {1, 1, 1, 1, 1, 1},
    {0, 1, 1, 0, 1, 0},
    {0, 1, 0, 1, 1, 0},
    {0, 1, 2, 2, 1, 0},
    {0, 1, 0, 1, 0, 1},
    {0, 1, 0, 0, 0, 0},
    {0, 1, 0, 1, 2, 2},
    {0, 1, 0, 1, 1, 0},
    {0, 1, 1, 0, 0, 1},
    {0, 1, 1, 0, 2, 2},
    {0, 1, 0, 0, 0, 0},
    {0, 1, 1, 0, 1, 0},
    {0, 1, 2, 2, 0, 1},
    {0, 1, 1, 0, 0, 1},
    {0, 1, 0, 1, 0, 1},
    {0, 1, 2, 2, 2, 2},
    {1, 0, 1, 0, 1, 0},
    {2, 2, 1, 0, 1, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 1, 1, 0, 1, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 1, 0, 0, 1, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 1, 1, 0, 1, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 1, 0, 1, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 1, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 1, 0, 1, 0},
    {1, 0, 0, 1, 1, 0},
    {0, 0, 0, 0, 1, 0},
    {2, 2, 0, 1, 1, 0},
    {0, 1, 0, 1, 1, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 0, 1, 1, 0},
    {0, 1, 0, 1, 1, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 1, 0, 0, 0, 0},
    {0, 1, 0, 0, 1, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 0, 1, 0, 0},
    {0, 1, 0, 1, 1, 0},
    {1, 0, 2, 2, 1, 0},
    {1, 0, 1, 0, 1, 0},
    {1, 0, 0, 1, 1, 0},
    {2, 2, 2, 2, 1, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 0},
    {0, 1, 0, 1, 1, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 1, 0, 1, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 1, 0, 1, 0},
    {0, 0, 2, 2, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 1, 2, 2, 1, 0},
    {1, 0, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 0, 1, 0, 0},
    {2, 2, 0, 1, 0, 1},
    {0, 1, 0, 0, 0, 0},
    {0, 1, 0, 1, 0, 1},
    {0, 1, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 0, 1, 0, 1},
    {0, 1, 0, 0, 0, 1},
    {0, 1, 0, 1, 0, 1},
    {0, 1, 0, 1, 0, 1},
    {1, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0},
    {2, 2, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {1, 0, 0, 1, 2, 2},
    {0, 0, 0, 0, 1, 0},
    {1, 0, 0, 1, 1, 0},
    {0, 0, 0, 1, 1, 0},
    {1, 0, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {2, 2, 0, 1, 2, 2},
    {0, 1, 0, 1, 1, 0},
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 1},
    {0, 1, 0, 1, 0, 1},
    {0, 1, 0, 1, 2, 2},
    {1, 0, 0, 1, 1, 0},
    {1, 0, 0, 0, 1, 0},
    {1, 0, 0, 1, 1, 0},
    {1, 0, 0, 1, 1, 0},
    {1, 0, 0, 1, 0, 0},
    {1, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 1, 0},
    {2, 2, 0, 1, 1, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 1, 0, 1, 1, 0},
    {1, 0, 1, 0, 0, 1},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {2, 2, 1, 0, 0, 1},
    {0, 1, 1, 0, 0, 1},
    {0, 1, 0, 0, 0, 0},
    {0, 1, 1, 0, 0, 0},
    {0, 1, 1, 0, 0, 1},
    {0, 1, 1, 0, 0, 1},
    {0, 1, 0, 0, 0, 1},
    {0, 1, 1, 0, 0, 1},
    {1, 0, 1, 0, 2, 2},
    {1, 0, 1, 0, 1, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 1, 0, 1, 0},
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 2, 2},
    {0, 0, 0, 0, 1, 0},
    {1, 0, 1, 0, 0, 1},
    {2, 2, 1, 0, 2, 2},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 1, 0, 1, 0},
    {0, 0, 1, 0, 0, 1},
    {0, 1, 1, 0, 0, 1},
    {0, 0, 0, 0, 0, 1},
    {0, 1, 1, 0, 2, 2},
    {1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {2, 2, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {0, 1, 0, 0, 0, 0},
    {1, 0, 1, 0, 1, 0},
    {1, 0, 1, 0, 1, 0},
    {1, 0, 0, 0, 1, 0},
    {1, 0, 1, 0, 1, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 1, 0},
    {1, 0, 1, 0, 0, 0},
    {1, 0, 1, 0, 1, 0},
    {1, 0, 0, 0, 0, 0},
    {2, 2, 1, 0, 1, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 1, 1, 0, 1, 0},
    {1, 0, 2, 2, 0, 1},
    {0, 0, 1, 0, 0, 0},
    {0, 0, 0, 1, 0, 0},
    {0, 0, 2, 2, 0, 0},
    {1, 0, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 0, 1, 0, 1},
    {0, 0, 0, 1, 0, 0},
    {1, 0, 1, 0, 0, 1},
    {0, 0, 1, 0, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {2, 2, 2, 2, 0, 1},
    {0, 1, 1, 0, 0, 1},
    {0, 1, 0, 1, 0, 1},
    {0, 1, 2, 2, 0, 1},
    {1, 0, 1, 0, 0, 1},
    {1, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {1, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {1, 0, 1, 0, 0, 1},
    {1, 0, 1, 0, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {0, 0, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 1},
    {2, 2, 1, 0, 0, 1},
    {0, 0, 0, 0, 0, 1},
    {0, 1, 1, 0, 0, 1},
    {1, 0, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 0, 0},
    {0, 0, 0, 1, 0, 0},
    {1, 0, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 0, 1},
    {0, 0, 0, 1, 0, 0},
    {1, 0, 0, 0, 0, 1},
    {0, 0, 0, 0, 0, 1},
    {1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 0, 1},
    {0, 0, 0, 0, 0, 1},
    {2, 2, 0, 1, 0, 1},
    {0, 1, 0, 1, 0, 1},
    {1, 0, 2, 2, 2, 2},
    {1, 0, 1, 0, 1, 0},
    {1, 0, 0, 1, 1, 0},
    {1, 0, 2, 2, 1, 0},
    {1, 0, 0, 1, 0, 1},
    {1, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 2, 2},
    {1, 0, 0, 1, 1, 0},
    {1, 0, 1, 0, 0, 1},
    {1, 0, 1, 0, 2, 2},
    {1, 0, 0, 0, 0, 0},
    {1, 0, 1, 0, 1, 0},
    {1, 0, 2, 2, 0, 1},
    {1, 0, 1, 0, 0, 1},
    {1, 0, 0, 1, 0, 1},
    {1, 1, 1, 1, 1, 1}
};

#endif //VH_MC_TABLES_H
