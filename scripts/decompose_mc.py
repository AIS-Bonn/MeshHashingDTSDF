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
    """ Decomposes the mc index into its up to 4 unconnected surfaces
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



  # UP = 0,
  # DOWN,
  # LEFT,
  # RIGHT,
  # FORWARD,
  # BACKWARD

class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    FORWARD = 4
    BACKWARD = 5

directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT, Direction.FORWARD, Direction.BACKWARD]


ViewDirectionParallelEdges = [
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        [(4, 0), (5, 1), (6, 2), (7, 3)],
        [(1, 0), (2, 3), (5, 4), (6, 7)],
        [(0, 1), (3, 2), (4, 5), (7, 6)],
        [(2, 1), (3, 0), (6, 5), (7, 4)],
        [(1, 2), (0, 3), (5, 6), (4, 7)]
        ]

def compatibility(mc_idx):
    """ Computes compatibility between direction and mc index
    """
    # compat = [0, 0, 0, 0, 0, 0]
    # for direction in directions:
    #     for (v1, v2) in ViewDirectionParallelEdges[direction.value]:
    #         if (mc_idx & (1 << v1) == 0) and (mc_idx & (1 << v2) > 0):
    #             compat[direction.value] = 1
    #             break
    compat = [1, 1, 1, 1, 1, 1]
    for direction in directions:
        for (v1, v2) in ViewDirectionParallelEdges[direction.value]: 
            if (mc_idx & (1 << v1) > 0) and (mc_idx & (1 << v2) == 0):
                compat[direction.value] = 0

    return compat

    

def main():
    # for mc_idx in range(0, 256):
    #     mc_dec = decompose(mc_idx)
    #     print("{{{}, {}, {}, {}}}".format(*mc_dec), end="")
    #     if mc_idx < 255:
    #         print(",")
    #     else:
    #         print()

    for mc_idx in range(0, 256):
        comp = compatibility(mc_idx)
        print("{{{}, {}, {}, {}, {}, {}}}".format(*comp), end="")
        if mc_idx < 255:
            print(",")
        else:
            print()

if __name__ == "__main__":
    main()
