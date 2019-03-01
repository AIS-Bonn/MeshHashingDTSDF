#!/usr/bin/env python3
import itertools
import numpy as np


# Corner 7 is origin (cf. paper)
corner_positions = [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)]
corner_positions = [np.array(x) for x in corner_positions]

offsets = [np.array(x) for x in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]) if x != (0, 0, 0)]

def get_corner_adjacent_voxels(corner):
    """
    For the given corner index return the indices of the adjacent voxels (inside the "offsets" list)
    """
    corner_pos = corner_positions[corner]
    results = []
    for idx in range(len(offsets)):
        voxel_offset = offsets[idx]
        for c in range(8):
            c_pos = voxel_offset + corner_positions[c]
            if np.array_equal(c_pos, corner_pos):
                results.append((idx, c))
    return results


def main():
    for o in offsets:
        print("{{{}, {}, {}}},".format(*o))
    for corner in range(8):
        res = get_corner_adjacent_voxels(corner)
        print("{", end="")
        for pair in res[:-1]:
            print("{{{}, {}}}, ".format(pair[0], pair[1]), end="")
        print("{{{}, {}}}".format(res[-1][0], res[-1][1]), end="")
        print("},")

if __name__ == "__main__":
    main()
