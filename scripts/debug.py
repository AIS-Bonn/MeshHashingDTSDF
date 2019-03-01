#!/usr/bin/env python3

import yaml

with open("../bin/FormatBlocks/block.formatblock", "r") as f:
    ys = yaml.load(f)

ptrs = [x["ptr"] for x in ys["blocks"]]
dups = []
for i in range(0, len(ptrs)):
    for j in range(i+1, len(ptrs)):
        if ptrs[i] == ptrs[j]:
            dups.append((i, j))
for m in dups:
    print("({}, {}, {})". format(m[0], m[1], ptrs[m[0]]))
