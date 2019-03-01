#!/usr/bin/env python3
import sys

def mc_to_ascii(mc_index):

    cube = """  {}-------{}
 /|      /|
{}-------{} |
| |     | |      {}
| {}-----|-{}
|/      |/
{}-------{}"""
    symbols = []
    for i in range(8):
        if mc_index & (1 << i):
            symbols.append("X")
        else:
            symbols.append(".")

    print(cube.format(symbols[4], symbols[5], symbols[7], symbols[6], mc_index, symbols[0], symbols[1], symbols[3], symbols[2]))

if __name__ == "__main__":
    for i in range(len(sys.argv) - 1):
        mc_to_ascii(int(sys.argv[i + 1]))
        if i < len(sys.argv) - 2:
            print()
