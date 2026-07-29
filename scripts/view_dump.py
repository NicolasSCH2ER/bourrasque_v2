#!/usr/bin/env python3
"""Visualise un dump .bqd du solveur (scatter 3D anime).

Usage : python view_dump.py frames.bqd [--stride 2]

Outil de validation en ligne de commande, INDEPENDANT de l'extension
Blender : aucune dependance a `extension/cache.py` n'est prise ici, meme si
ce module contient deja un lecteur des deux formats. La raison : ce script
doit rester utilisable sur une machine ou seul le core CUDA a ete compile,
sans que `extension/` (ni ses eventuelles dependances a l'ecosystem
Blender/bpy) ne soit necessairement present. Le format est donc relu ici de
facon courte et independante ; la specification de reference reste le
docstring de tete de `extension/cache.py`, a consulter en cas de doute ou
d'evolution du format.

Formats supportes (detection automatique par les 4 premiers octets) :

    v1 : int32 n, int32 frames, puis frames * n * 3 float32.
         Nombre de particules CONSTANT sur toute la simulation.

    v2 : magie b"BQD2", int32 version=2, int32 frames, int32 n_max,
         int64 index_off, puis par frame : int32 count, count*3 float32,
         puis a l'offset index_off : frames * int64 (offset absolu de
         chaque frame). Nombre de particules VARIABLE par frame (emission
         continue) ; la table d'index n'est pas necessaire ici car ce
         script lit le fichier sequentiellement une seule fois.

Si un sidecar frames.mat existe (n_max uint8, id materiau de la DERNIERE
frame), colore par materiau : eau en bleu, elastique en orange. Les
particules n'etant jamais retirees, la frame i (compte count_i) utilise les
count_i premieres entrees du sidecar.
"""
import argparse
import pathlib
import struct

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MAT_COLORS = ["#ff9944", "#3388ff"]  # 0 = elastique, 1 = eau

_MAGIC = b"BQD2"
_V1_HEADER = struct.Struct("<ii")
_V2_HEADER = struct.Struct("<4siiiq")
_V2_FRAME_COUNT = struct.Struct("<i")


def _load_v1(f):
    n, nframes = _V1_HEADER.unpack(f.read(_V1_HEADER.size))
    data = np.fromfile(f, dtype="<f4", count=nframes * n * 3)
    data = data.reshape(nframes, n, 3)
    return [data[i] for i in range(nframes)]


def _load_v2(f):
    header = f.read(_V2_HEADER.size)
    _magic, version, nframes, _n_max, _index_off = _V2_HEADER.unpack(header)
    if version != 2:
        raise ValueError(f"version BQD v2 inconnue : {version}")
    frames = []
    for _ in range(nframes):
        (count,) = _V2_FRAME_COUNT.unpack(f.read(_V2_FRAME_COUNT.size))
        arr = np.fromfile(f, dtype="<f4", count=count * 3).reshape(count, 3)
        frames.append(arr)
    return frames


def load(path):
    """Renvoie `(version, frames)` : `version` in (1, 2), `frames` une
    liste de `nframes` ndarrays `(count_i, 3)` float32 (count_i variable
    en v2, constant en v1)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        f.seek(0)
        if magic == _MAGIC:
            return 2, _load_v2(f)
        return 1, _load_v1(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--stride", type=int, default=1,
                    help="n'affiche qu'une particule sur N (perf matplotlib)")
    args = ap.parse_args()

    version, frames = load(args.path)
    nf = len(frames)
    counts = [a.shape[0] for a in frames]

    if counts and min(counts) != max(counts):
        n_desc = f"{min(counts)}..{max(counts)} particules (variable)"
    else:
        n_desc = f"{counts[0] if counts else 0} particules"
    print(f"format v{version}, {nf} frames, {n_desc}")

    matpath = pathlib.Path(args.path).with_suffix(".mat")
    mats_full = None
    if matpath.exists():
        mats_full = np.fromfile(matpath, dtype=np.uint8)

    stride = args.stride

    # Bornes des axes calculees sur l'ensemble des frames (avant
    # decimation par --stride, qui n'est qu'une aide au rendu) pour
    # rester stables d'une frame a l'autre malgre le compte variable.
    if nf and any(a.size for a in frames):
        all_pts = np.concatenate([a for a in frames if a.size], axis=0)
        lo = all_pts.min(axis=0)
        hi = all_pts.max(axis=0)
    else:
        lo = np.zeros(3, dtype=np.float32)
        hi = np.ones(3, dtype=np.float32)

    def frame_render_data(i):
        pos = frames[i][::stride]
        if mats_full is not None:
            mats_i = mats_full[:counts[i]][::stride]
            colors = np.array(MAT_COLORS)[np.clip(mats_i, 0, 1)]
            return pos, colors, None
        # pas de sidecar materiau : colore par hauteur (Y)
        return pos, None, pos[:, 1] if pos.size else np.empty(0)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])

    # Le nombre de particules variant d'une frame a l'autre (v2), un
    # scatter matplotlib cree pour N points ne peut pas etre agrandi a
    # N' > N via `_offsets3d` seul (les tableaux de couleur/array internes
    # restent a la taille initiale et desynchronisent l'affichage, voire
    # levent une exception au dessin). On recree donc l'artiste scatter a
    # chaque frame plutot que de tronquer ou de tenter une mutation en
    # place : le cout (un scatter par frame) est negligeable ici, ce
    # script n'etant pas le chemin chaud du bake.
    state = {"sc": None}

    def draw_frame(i):
        if state["sc"] is not None:
            state["sc"].remove()
        pos, colors, cvals = frame_render_data(i)
        if colors is not None:
            state["sc"] = ax.scatter(*pos.T, s=2, c=colors)
        else:
            state["sc"] = ax.scatter(*pos.T, s=2, c=cvals, cmap="Blues_r")
        ax.set_title(f"frame {i}/{nf} ({pos.shape[0]} particules)")
        return (state["sc"],)

    def update(i):
        return draw_frame(i)

    draw_frame(0)
    anim = FuncAnimation(fig, update, frames=nf, interval=1000 / 24)
    plt.show()


if __name__ == "__main__":
    main()
