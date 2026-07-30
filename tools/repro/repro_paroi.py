"""Une paroi mince est-elle vue par le champ de distance ?

Hypothese : le SDF est echantillonne AUX CELLULES. Une paroi d'epaisseur t < dx
peut ne contenir aucun centre de cellule : phi >= 0 partout, aucune condition
aux limites, le fluide traverse. Un cube plein est epais de plusieurs dx, d'ou
la difference observee a echelle egale.

On construit une boite creuse etanche (deux boites imbriquees, l'interieure a
normales inversees) et on fait varier t / dx.
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

_FACES = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
          (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
          (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]


def box_tris(lo, hi, flip=False):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(a, c, b) for a, b, c in _FACES] if flip else _FACES
    return np.array([[v[a], v[b], v[c]] for a, b, c in f], dtype=np.float32)


cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
d.bq_add_material(sim, ctypes.byref(mat))

R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx
sdf = np.empty(R * R * R, dtype=np.float32)
print(f"grille {R}^3, dx = {dx * 1000:.2f} mm, domaine {L:.3f} m\n")

OLO = np.array([0.25 * L] * 3)
OHI = np.array([0.75 * L] * 3)

for tm in (0.25, 0.5, 1.0, 2.0, 4.0):
    t = tm * dx
    tri = np.concatenate([box_tris(OLO, OHI),
                          box_tris(OLO + t, OHI - t, flip=True)])
    n = tri.shape[0]
    vel = np.zeros_like(tri)
    fric = np.full(n, 0.2, dtype=np.float32)
    if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                          fric.ctypes.data_as(F), n) != 0:
        raise SystemExit(d.bq_last_error().decode())
    d.bq_read_sdf(sim, sdf.ctypes.data_as(F))
    g = sdf.reshape(R, R, R)
    solid = g < 0

    vol_paroi = np.prod(OHI - OLO) - np.prod(OHI - OLO - 2 * t)
    attendu = vol_paroi / dx ** 3

    # Etancheite reelle : un rayon horizontal traversant la paroi -X
    # rencontre-t-il au moins une cellule solide ?
    j0, j1 = int((OLO[1] + t) / dx) + 1, int((OHI[1] - t) / dx)
    k0, k1 = int((OLO[2] + t) / dx) + 1, int((OHI[2] - t) / dx)
    i0, i1 = int(OLO[0] / dx), int((OLO[0] + t) / dx) + 1
    mur = solid[i0:i1 + 1, j0:j1, k0:k1]
    n_ray = mur.shape[1] * mur.shape[2]
    trous = int((~mur.any(axis=0)).sum()) if n_ray else 0

    print(f"t = {tm:>4} dx ({t * 1000:5.2f} mm)")
    print(f"   cellules solides : {int(solid.sum()):>6}   attendu ~{attendu:>7.0f}")
    print(f"   paroi -X : {trous} / {n_ray} rayons TRAVERSENT sans obstacle"
          f"  ({trous / max(n_ray, 1) * 100:5.1f} %)")

d.bq_destroy(sim)
