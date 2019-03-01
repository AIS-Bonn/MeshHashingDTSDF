#!/usr/bin/env python3
import networkx as nx
import enum


EdgeVertexTable = [[0, 1],
                   [2, 1],
                   [3, 2],
                   [3, 0],
                   [4, 5],
                   [6, 5],
                   [7, 6],
                   [7, 4],
                   [4, 0],
                   [5, 1],
                   [6, 2],
                   [7, 3]]

def decompose(mc_idx):
    """
    Decomposes the given mc index into its up to 4 unconnected surfaces
    """
    G = nx.Graph()
    # 1) find connected positive corners of mc index
    for e in range(0, 12):
        v1 = EdgeVertexTable[e][0]
        v2 = EdgeVertexTable[e][1]
        G.add_edge(v1, v2)

    for i in range(0, 8):
        if (mc_idx & (1 << i)) <= 0:
            G.remove_node(i)

    # 2) create new mc indices
    mc_indices = [-1, -1, -1, -1]
    count = 0
    for comp in nx.connected_components(G):
        idx = 0
        for i in comp:
            idx |= (1 << i)
        mc_indices[count] = idx
        count += 1
    return mc_indices


class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    FORWARD = 4
    BACKWARD = 5
directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT, Direction.FORWARD, Direction.BACKWARD]

# For every direction all 4 edges parallel to the direction vector (vertex order in view directinon as well!)
ViewDirectionParallelEdges = [
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        [(4, 0), (5, 1), (6, 2), (7, 3)],
        [(1, 0), (2, 3), (5, 4), (6, 7)],
        [(0, 1), (3, 2), (4, 5), (7, 6)],
        [(2, 1), (3, 0), (6, 5), (7, 4)],
        [(1, 2), (0, 3), (5, 6), (4, 7)]
        ]

def compatibility(mc_idx):
    """
    Computes compatibility between directions and the given mc index
    """
    # compat = [1, 1, 1, 1, 1, 1]
    # for direction in directions:
    #     for (v1, v2) in ViewDirectionParallelEdges[direction.value]:
    #         if (mc_idx & (1 << v1) > 0) and (mc_idx & (1 << v2) == 0):
    #             compat[direction.value] = 0
    compat = [1, 1, 1, 1, 1, 1]
    for direction in directions:
        parallel_count = 0
        for (v1, v2) in ViewDirectionParallelEdges[direction.value]: 
            if (mc_idx & (1 << v1) > 0) and (mc_idx & (1 << v2) == 0):
                compat[direction.value] = 0
            parallel_count += ((mc_idx & (1 << v1) > 0) and (mc_idx & (1 << v2) > 0)) or ((mc_idx & (1 << v1) == 0) and (mc_idx & (1 << v2) == 0))
        if parallel_count == 4 and mc_idx not in [0, 255]:
            compat[direction.value] = 2

    return compat

    

def main():
    # # Generate MC index decomposition table:
    # #     For every MC index the separate MC components are: a, b, c, d  (-1 means no component)
    # for mc_idx in range(0, 256):
    #     mc_dec = decompose(mc_idx)
    #     print("{{{}, {}, {}, {}}}".format(*mc_dec), end="")
    #     if mc_idx < 255:
    #         print(",")
    #     else:
    #         print()

    # Generate MC index compatibility table:
    #   For every MC index a 6-vector is printed (one value per direction). 0 = incompatible, 1 = compatible, 2 = angle check required
    for mc_idx in range(0, 256):
        comp = compatibility(mc_idx)
        # print("{{{}, {}, {}, {}, {}, {}}}".format(*comp), end="")
        print("{}{{{}, {}, {}, {}, {}, {}}}".format(*comp), end="")
        if mc_idx < 255:
            print(",")
        else:
            print()

if __name__ == "__main__":
    main()
