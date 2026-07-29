"""Transformation monde Blender <-> espace solveur.

Aucune dependance a bpy : ce module est pur, importable et testable en
dehors de Blender (voir extension/tests/test_transform.py).

Blender est Z-up droitier, le solveur Y-up droitier. Un simple echange
d'axes inverserait la chiralite et mirroirait la simulation. Le mapping
retenu (voir docs/plan-milestone-3.md, decision D2) est :

    world_to_solver, avec origin = (mx, my, mz) et size = (sx_size, sy_size, sz_size)
    (etendues du pave SOLVEUR sur ses trois axes, voir docs/plan-milestone-5.md,
    decision D3 : size[0] correspond a l'etendue monde X, size[1] a l'etendue
    monde Z, size[2] a l'etendue monde Y) :
        sx = bx - mx
        sy = bz - mz
        sz = (my + size[2]) - by

    solver_to_world :
        bx = sx + mx
        by = my + size[2] - sz
        bz = sy + mz

Depuis M5 (domaine en pave, dx uniforme mais nombre de cellules different
par axe), `size` est un TRIPLET et non plus un scalaire : seul le terme
`size[2]` (etendue solveur Z, issue de l'etendue monde Y inversee)
intervient dans la formule, car c'est le seul axe ou le sens est inverse
(`sz` decroit quand `by` croit). Les axes sx/sy restent de simples
translations, sans terme d'echelle.
"""

import numpy as np


def world_to_solver(p, origin, size):
    """Convertit un point monde Blender `p` en coordonnees solveur.

    `origin` est le coin min de la bounding box monde du domaine, `size`
    est le triplet `(size_x, size_y, size_z)` des etendues du pave SOLVEUR
    (voir docstring du module pour le mapping des axes).
    """
    bx, by, bz = p
    mx, my, mz = origin
    size_z = size[2]
    sx = bx - mx
    sy = bz - mz
    sz = (my + size_z) - by
    return (sx, sy, sz)


def solver_to_world(p, origin, size):
    """Inverse de `world_to_solver`."""
    sx, sy, sz = p
    mx, my, mz = origin
    size_z = size[2]
    bx = sx + mx
    by = my + size_z - sz
    bz = sy + mz
    return (bx, by, bz)


def world_to_solver_dir(v):
    """Convertit un vecteur directionnel (vitesse) monde -> solveur.

    Meme transformation lineaire que `world_to_solver`, sans le terme de
    translation.
    """
    vx, vy, vz = v
    return (vx, vz, -vy)


def solver_to_world_array(positions, origin, size, out=None):
    """Conversion vectorisee numpy, equivalente a `solver_to_world` appliquee
    point par point, mais sur un tableau (n, 3) entier d'un coup.

    Utilisee par `display.py` (chemin chaud du scrub de timeline, jusqu'a
    200 000 particules a 24 images/s : une boucle scalaire Python couterait
    de l'ordre de la seconde par frame) et par
    `tests/test_display_math.py`, qui verifie qu'elle coincide avec
    `solver_to_world` plutot que de la dupliquer (source de verite unique).
    """
    mx, my, mz = origin
    size_z = size[2]
    if out is None or out.shape != positions.shape:
        out = np.empty_like(positions)
    out[:, 0] = positions[:, 0] + mx
    out[:, 1] = my + size_z - positions[:, 2]
    out[:, 2] = positions[:, 1] + mz
    return out
