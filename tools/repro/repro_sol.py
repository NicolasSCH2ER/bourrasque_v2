"""Collider pose au sol dont la face inferieure SORT du domaine.

Hypothese : la partie basse du solide n'a plus aucune surface a l'interieur
de la grille, donc aucune graine ne l'amorce comme interieure ; les cellules
concernees restent UNKNOWN et la regle de securite les declare EXTERIEURES.
Il se creuse alors un trou sous le collider, par lequel le fluide passe.
"""
import ctypes

import numpy as np

DLL = r"C:\Users\nicol\Code\bourrasque_v2\extension\bin\bourrasque.dll"
d = ctypes.CDLL(DLL)
F = ctypes.POINTER(ctypes.c_float)


class Cfg(ctypes.Structure):
    _fields_ = [("grid_res", ctypes.c_int * 3), ("cell_size", ctypes.c_float),
                ("gravity_y", ctypes.c_float), ("cfl", ctypes.c_float),
                ("ppc_axis", ctypes.c_int), ("max_particles", ctypes.c_int)]


class Mat(ctypes.Structure):
    _fields_ = [("model", ctypes.c_int), ("rho", ctypes.c_float),
                ("E", ctypes.c_float), ("nu", ctypes.c_float),
                ("bulk", ctypes.c_float), ("gamma", ctypes.c_float)]


d.bq_default_config.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.restype = ctypes.c_void_p
d.bq_destroy.argtypes = [ctypes.c_void_p]
d.bq_add_material.argtypes = [ctypes.c_void_p, ctypes.POINTER(Mat)]
d.bq_add_material.restype = ctypes.c_int
d.bq_set_colliders.argtypes = [ctypes.c_void_p, F, F, F, ctypes.c_int]
d.bq_set_colliders.restype = ctypes.c_int
d.bq_read_sdf.argtypes = [ctypes.c_void_p, F]
d.bq_read_sdf.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p


def box_tris(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
             (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float32)


cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
d.bq_add_material(sim, ctypes.byref(mat))

R = cfg.grid_res[0]
dx = cfg.cell_size
L = R * dx
sdf = np.empty(R * R * R, dtype=np.float32)
# Y est l'axe vertical du solveur.
XC = (0.3 * L, 0.7 * L)

for label, y0 in (("cube POSE sur le sol (face basse a y=0.25L)", 0.25 * L),
                  ("cube DEBORDANT sous le domaine (face basse a y=-0.1L)", -0.1 * L)):
    tri = box_tris((XC[0], y0, XC[0]), (XC[1], 0.5 * L, XC[1]))
    n = tri.shape[0]
    vel = np.zeros_like(tri)
    fric = np.full(n, 0.2, dtype=np.float32)
    if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                          fric.ctypes.data_as(F), n) != 0:
        raise SystemExit(d.bq_last_error().decode())
    d.bq_read_sdf(sim, sdf.ctypes.data_as(F))
    g = sdf.reshape(R, R, R)          # (i=x, j=y, k=z)

    # Colonne centrale du cube : que voit-on en descendant ?
    i = k = R // 2
    col = g[i, :, k]
    solid = col < 0
    # Cellules qui devraient etre solides : celles sous le haut du cube et
    # au-dessus de la face basse (clampee au domaine).
    # Le champ est echantillonne AUX NOEUDS (j*dx) : le noeud j est dans le
    # solide si j*dx appartient a [y0, 0.5L], d'ou ceil/floor et non int().
    j_hi = int(np.floor(0.5 * L / dx))
    j_lo = max(0, int(np.ceil(y0 / dx)))
    attendu = np.zeros(R, dtype=bool)
    attendu[j_lo:j_hi] = True
    manquantes = int((attendu & ~solid).sum())
    print(f"{label}")
    print(f"   cellules solides sur la colonne : {int(solid.sum())}"
          f"   attendues : {int(attendu.sum())}")
    print(f"   TROU sous le collider : {manquantes} cellules solides manquantes")
    if manquantes:
        js = np.nonzero(attendu & ~solid)[0]
        print(f"   indices j concernes : {js.min()}..{js.max()} (bas du domaine)")

d.bq_destroy(sim)
