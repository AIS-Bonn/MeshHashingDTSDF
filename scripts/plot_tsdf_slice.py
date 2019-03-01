import enum
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import yaml

methods = [None, 'bilinear', 'bicubic']

side_length = 4


class Direction(enum.Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    FORWARD = 4
    BACKWARD = 5


directions = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT, Direction.FORWARD, Direction.BACKWARD]


def get_array_offset(coordinates_xyz):
    return coordinates_xyz[2] * side_length * side_length + coordinates_xyz[1] * side_length + coordinates_xyz[0]


def main():
    with open("/home/splietke/code/MeshHashing/bin/FormatBlocks/block.formatblock", "r") as stream:
        data = yaml.load(stream, Loader=yaml.CLoader)

    print("finished reading block file")

    interpolation_method = methods[0]

    dimensions = [0, 0, 0]
    min_z = 0
    max_z = 0
    for block in data["blocks"]:
        (x, y, z) = (int(block["pos"]["x"]), int(block["pos"]["y"]), int(block["pos"]["z"]))
        dimensions[0] = max(dimensions[0], abs(x))
        dimensions[1] = max(dimensions[1], abs(y))
        dimensions[2] = max(dimensions[2], abs(z))
        min_z = min(min_z, z)
        max_z = max(max_z, z)
    num_blocks_z = max_z - min_z + 1
    print(dimensions, min_z, max_z)

    size = math.ceil((max(dimensions) + 1) * side_length / 2) * 4
    grids = [np.full((size, size), np.inf) for i in range(len(directions) * side_length * num_blocks_z)]

    for block_z, direction, sl in itertools.product(range(min_z, max_z + 1), directions, range(side_length)):
        grid = grids[side_length * len(directions) * (block_z - min_z) + len(directions) * sl + direction.value]
        for block in data["blocks"]:
            (x, y, z) = (int(block["pos"]["x"]), int(block["pos"]["y"]), int(block["pos"]["z"]))
            if z != block_z:
                continue
            for (row, col) in itertools.product(range(side_length), range(side_length)):
                if not block["directions"][direction.value]:
                    continue
                # change value, so zero crossing becomes jump from -1 to 1 (better visibility)
                val = float(block["directions"][direction.value]["sdf"][get_array_offset([col, row, sl])])
                # val = math.copysign(1, val) * (1 - abs(val)) # invert color-fade for sharper +/- contrast
                grid[side_length * y + row + size // 2, side_length * x + col + size // 2] = val

    fig_size = (8 * max(dimensions[0], dimensions[1]), 8 * max(dimensions[0], dimensions[1]) * dimensions[2])
    fig, axs = plt.subplots(nrows=side_length * num_blocks_z, ncols=len(directions), figsize=fig_size,
                            subplot_kw={'xticks': np.arange(-0.5, size, step=side_length),
                                        'xticklabels': [i for i in range(-size // 2, size // 2, side_length)],
                                        'yticks': np.arange(-0.5, size, step=side_length),
                                        'yticklabels': [i for i in range(-size // 2, size // 2, side_length)]})

    fig.subplots_adjust(left=0.10, right=0.97, hspace=0.3, wspace=0.05)

    # https://matplotlib.org/examples/color/colormaps_reference.html
    for ax, (block_z, sl, direction) in zip(axs.flat, itertools.product(range(max_z, min_z - 1, -1),
                                                                        range(side_length - 1, -1, -1), directions)):
        ax.imshow(grids[side_length * len(directions) * (block_z - min_z) + len(directions) * sl + direction.value],
                  interpolation=interpolation_method, cmap='RdYlGn')
        ax.grid(color='k', linestyle='-', linewidth=1)
        ax.set_title(direction.name + " " + str(block_z * side_length + sl))
    plt.savefig("/tmp/fig.png")
    # plt.show()


if __name__ == "__main__":
    main()
