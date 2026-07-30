"""Lecteur GLB minimal : renvoie, par noeud de la scene, ses triangles en
coordonnees monde. Aucune dependance hors numpy.

Convention glTF : Y vers le haut, Z vers l'avant, main droite. Blender exporte
en convertissant depuis sa propre convention Z-up. Le solveur Bourrasque est
Y-up lui aussi, avec la meme correspondance que le frontend : solveur X <- monde
X, solveur Y <- monde Z_blender, solveur Z <- -monde Y_blender. Comme le GLB a
DEJA subi la conversion Blender -> glTF, ses coordonnees sont directement dans
un repere Y-up : on les utilise telles quelles.
"""
import json
import struct

import numpy as np

_COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
         5125: ("I", 4), 5126: ("f", 4)}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67:
        raise ValueError("pas un GLB")
    off = 12
    js, bin_chunk = None, b""
    while off < len(data):
        clen, ctype = struct.unpack_from("<II", data, off)
        chunk = data[off + 8:off + 8 + clen]
        if ctype == 0x4E4F534A:
            js = json.loads(chunk.decode("utf-8"))
        elif ctype == 0x004E4942:
            bin_chunk = chunk
        off += 8 + clen + ((4 - clen % 4) % 4 if clen % 4 else 0)
    return js, bin_chunk


def accessor(js, blob, idx):
    acc = js["accessors"][idx]
    fmt, size = _COMP[acc["componentType"]]
    ncomp = _NCOMP[acc["type"]]
    view = js["bufferViews"][acc["bufferView"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride") or size * ncomp
    out = np.empty((acc["count"], ncomp), dtype=np.dtype(fmt))
    for i in range(acc["count"]):
        out[i] = struct.unpack_from("<" + fmt * ncomp, blob, base + i * stride)
    return out


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    m = np.eye(4)
    if "scale" in node:
        m = np.diag(list(node["scale"]) + [1.0]) @ m
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
            [0, 0, 0, 1]])
        m = r @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


def objects(path):
    """[(nom, triangles (n,3,3) monde), ...]"""
    js, blob = load(path)
    out = []

    def walk(ni, parent):
        node = js["nodes"][ni]
        world = parent @ node_matrix(node)
        if "mesh" in node:
            tris = []
            for prim in js["meshes"][node["mesh"]].get("primitives", []):
                if prim.get("mode", 4) != 4:
                    continue
                pos = accessor(js, blob, prim["attributes"]["POSITION"]).astype(np.float64)
                if "indices" in prim:
                    idx = accessor(js, blob, prim["indices"]).reshape(-1).astype(np.int64)
                else:
                    idx = np.arange(len(pos), dtype=np.int64)
                h = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
                wp = (world @ h.T).T[:, :3]
                tris.append(wp[idx].reshape(-1, 3, 3))
            if tris:
                out.append((node.get("name", f"node{ni}"), np.concatenate(tris)))
        for c in node.get("children", []):
            walk(c, world)

    scene = js.get("scene", 0)
    for ni in js["scenes"][scene].get("nodes", []):
        walk(ni, np.eye(4))
    return out


if __name__ == "__main__":
    import sys
    for name, tri in objects(sys.argv[1]):
        lo, hi = tri.reshape(-1, 3).min(0), tri.reshape(-1, 3).max(0)
        ext = hi - lo
        print(f"{name!r}: {len(tri)} triangles")
        print(f"    bbox min {np.round(lo, 4)}  max {np.round(hi, 4)}")
        print(f"    extent   {np.round(ext, 4)}  (cm: {np.round(ext * 100, 2)})")
