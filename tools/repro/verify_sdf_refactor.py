"""Repro deterministe pour la tache A0 du jalon M17 (decouplage des noyaux
SDF de la grille du solveur, core/src/mlsmpm.cu).

Construit une Sim de configuration FIXE (grille non cubique 48x64x40, pour
attraper toute confusion d'axe), pose des colliders FIXES (une boite fermee
+ un objet en biais non aligne sur la grille), appelle read_sdf()/read_cnrm()
et imprime un hachage SHA-256 de chaque champ, plus quelques statistiques.

La construction du champ SDF est deterministe (contrairement au reste du
solveur, qui utilise des atomicAdd flottants sur les particules) : les deux
hachages -- avant et apres le refactor A0 -- doivent etre rigoureusement
identiques.

Usage :
    Python312/python.exe tools/repro/verify_sdf_refactor.py
"""

import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "extension"))
import lib  # noqa: E402


def sha256_of_array(arr: np.ndarray) -> str:
    # tobytes() sur un tableau C-contigu float32 : hachage bit a bit exact.
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.tobytes()).hexdigest()


def make_box_triangles(lo, hi):
    """Boite fermee (6 faces, 2 triangles chacune, normales sortantes),
    coordonnees monde."""
    lo = np.array(lo, dtype=np.float64)
    hi = np.array(hi, dtype=np.float64)
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # bas z0
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # haut z1
    ])
    # (a,b,c,d) par face, ordre CCW vu depuis l'exterieur
    faces = [
        (0, 3, 2, 1),  # z0, normale -z
        (4, 5, 6, 7),  # z1, normale +z
        (0, 1, 5, 4),  # y0, normale -y
        (2, 3, 7, 6),  # y1, normale +y
        (1, 2, 6, 5),  # x1, normale +x
        (3, 0, 4, 7),  # x0, normale -x
    ]
    tris = []
    for a, b, c, d in faces:
        tris.append([v[a], v[b], v[c]])
        tris.append([v[a], v[c], v[d]])
    return np.array(tris, dtype=np.float64)


def make_skew_tetra(center, scale, rot_deg=(17.0, 31.0, 53.0)):
    """Tetraedre place en biais (rotation composee sur les 3 axes, angles non
    triviaux), non aligne sur la grille -- pour attraper toute confusion
    d'axe/permutation dans le refactor."""
    verts = np.array([
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ]) * scale

    def rot_x(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    rx, ry, rz = (np.deg2rad(a) for a in rot_deg)
    R = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    verts = verts @ R.T + np.array(center)

    faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
    tris = []
    for a, b, c in faces:
        tris.append([verts[a], verts[b], verts[c]])
    return np.array(tris, dtype=np.float64)


def main():
    grid_res = (48, 64, 40)  # non cubique expres
    dx = 0.02

    cfg = lib.default_config()
    cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2] = grid_res
    cfg.cell_size = dx

    domain_hi = (grid_res[0] * dx, grid_res[1] * dx, grid_res[2] * dx)
    print(f"domaine solveur : {domain_hi[0]:.4f} x {domain_hi[1]:.4f} x {domain_hi[2]:.4f}")

    # Boite fermee, centree dans le domaine, pas alignee sur des multiples
    # exacts de dx (evite la degenerescence "noeud pile sur une face").
    box = make_box_triangles(
        lo=(0.31 * domain_hi[0] + 0.0037, 0.28 * domain_hi[1] + 0.0021, 0.35 * domain_hi[2] + 0.0013),
        hi=(0.62 * domain_hi[0] + 0.0037, 0.71 * domain_hi[1] + 0.0021, 0.68 * domain_hi[2] + 0.0013),
    )

    # Objet en biais, place ailleurs dans le domaine, sans intersecter la boite.
    skew = make_skew_tetra(
        center=(0.15 * domain_hi[0], 0.15 * domain_hi[1], 0.15 * domain_hi[2]),
        scale=0.08 * min(domain_hi),
    )

    tris = np.concatenate([box, skew], axis=0).astype(np.float32)
    n_tri = tris.shape[0]
    vel = np.zeros_like(tris, dtype=np.float32)
    fric = np.full((n_tri,), 0.3, dtype=np.float32)

    print(f"n_tri = {n_tri} (boite: {box.shape[0]}, biais: {skew.shape[0]})")

    with lib.Sim(cfg) as sim:
        sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4e4, gamma=3.0)
        sim.set_colliders(tris, vel, fric)
        sdf = sim.read_sdf()
        cnrm = sim.read_cnrm()

    sdf_hash = sha256_of_array(sdf)
    cnrm_hash = sha256_of_array(cnrm)

    n_neg = int(np.sum(sdf < 0.0))
    print()
    print(f"sdf   shape={sdf.shape} dtype={sdf.dtype}")
    print(f"sdf   min={sdf.min():.6f} max={sdf.max():.6f} n_negatif={n_neg}")
    print(f"sdf   sha256={sdf_hash}")
    print()
    print(f"cnrm  shape={cnrm.shape} dtype={cnrm.dtype}")
    print(f"cnrm  min={cnrm.min():.6f} max={cnrm.max():.6f}")
    print(f"cnrm  sha256={cnrm_hash}")


if __name__ == "__main__":
    main()
