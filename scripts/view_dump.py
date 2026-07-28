#!/usr/bin/env python3
"""Visualise un dump .bqd du test headless (scatter 3D anime).

Usage : python view_dump.py frames.bqd [--stride 2]
Format .bqd : int32 n, int32 frames, puis frames * n * 3 float32.
Si un sidecar frames.mat existe (n uint8), colore par materiau :
eau en bleu, elastique en orange.
"""
import argparse
import pathlib
import struct

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MAT_COLORS = ["#ff9944", "#3388ff"]  # 0 = elastique, 1 = eau


def load(path):
    with open(path, "rb") as f:
        n, frames = struct.unpack("ii", f.read(8))
        data = np.fromfile(f, dtype=np.float32, count=frames * n * 3)
    return data.reshape(frames, n, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--stride", type=int, default=1,
                    help="n'affiche qu'une particule sur N (perf matplotlib)")
    args = ap.parse_args()

    frames = load(args.path)
    matpath = pathlib.Path(args.path).with_suffix(".mat")
    colors = None
    if matpath.exists():
        mats = np.fromfile(matpath, dtype=np.uint8)[::args.stride]
        colors = np.array(MAT_COLORS)[np.clip(mats, 0, 1)]

    frames = frames[:, ::args.stride, :]
    nf, n, _ = frames.shape
    print(f"{nf} frames, {n} particules affichees")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    lo = frames.min(axis=(0, 1)); hi = frames.max(axis=(0, 1))
    if colors is not None:
        sc = ax.scatter(*frames[0].T, s=2, c=colors)
    else:
        sc = ax.scatter(*frames[0].T, s=2, c=frames[0][:, 1], cmap="Blues_r")

    def update(i):
        sc._offsets3d = tuple(frames[i].T)
        if colors is None:
            sc.set_array(frames[i][:, 1])
        ax.set_title(f"frame {i}/{nf}")
        return (sc,)

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    anim = FuncAnimation(fig, update, frames=nf, interval=1000 / 24)
    plt.show()


if __name__ == "__main__":
    main()
