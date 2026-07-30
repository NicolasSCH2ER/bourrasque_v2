"""Le fluide passe-t-il SOUS un collider pose au sol du domaine ?

Le champ de distance est correct (verifie par repro_sol.py) : on teste donc
le COMPORTEMENT. De l'eau est lachee a cote d'un cube pose au sol ; on compte
les particules qui finissent sous lui, c'est-a-dire dans son ombre verticale
et sous sa face inferieure.
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


for name, argt, rest in (
    ("bq_default_config", [ctypes.POINTER(Cfg)], None),
    ("bq_create", [ctypes.POINTER(Cfg)], ctypes.c_void_p),
    ("bq_destroy", [ctypes.c_void_p], None),
    ("bq_add_material", [ctypes.c_void_p, ctypes.POINTER(Mat)], ctypes.c_int),
    ("bq_set_colliders", [ctypes.c_void_p, F, F, F, ctypes.c_int], ctypes.c_int),
    ("bq_emit_box", [ctypes.c_void_p, ctypes.c_int, F, F, F], ctypes.c_int),
    ("bq_step", [ctypes.c_void_p, ctypes.c_float], ctypes.c_int),
    ("bq_particle_count", [ctypes.c_void_p], ctypes.c_int),
    ("bq_read_positions", [ctypes.c_void_p, F], ctypes.c_int),
):
    getattr(d, name).argtypes = argt
    if rest:
        getattr(d, name).restype = rest
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
cfg.gravity_y = -9.8
R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx
B = 3 * dx                                  # couche de bord du domaine

# Cube pose au sol : face basse SOUS le domaine (deborde), sommet a 0.45L.
CLO = (0.30 * L, -0.10 * L, 0.30 * L)
CHI = (0.70 * L, 0.45 * L, 0.70 * L)

for label, cy0 in (("face basse DANS le domaine (y=0.10L)", 0.10 * L),
                   ("face basse SOUS le domaine  (y=-0.10L)", -0.10 * L)):
    sim = d.bq_create(ctypes.byref(cfg))
    mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
    m = d.bq_add_material(sim, ctypes.byref(mat))
    tri = box_tris((CLO[0], cy0, CLO[2]), CHI)
    n_t = tri.shape[0]
    vel = np.zeros_like(tri)
    fric = np.full(n_t, 0.2, dtype=np.float32)
    d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                       fric.ctypes.data_as(F), n_t)

    # Colonne d'eau posee a cote, qui va s'ecrouler contre le cube.
    lo = np.array([B, 0.55 * L, B], dtype=np.float32)
    hi = np.array([0.28 * L, 0.9 * L, L - B], dtype=np.float32)
    vz = np.zeros(3, dtype=np.float32)
    n = d.bq_emit_box(sim, m, lo.ctypes.data_as(F), hi.ctypes.data_as(F),
                      vz.ctypes.data_as(F))
    for _ in range(120):
        d.bq_step(sim, 1.0 / 24.0)
    pos = np.empty((d.bq_particle_count(sim), 3), dtype=np.float32)
    d.bq_read_positions(sim, pos.ctypes.data_as(F))
    d.bq_destroy(sim)

    # Sous le cube : dans son ombre en x/z, et sous sa face inferieure.
    ombre = ((pos[:, 0] > CLO[0]) & (pos[:, 0] < CHI[0])
             & (pos[:, 2] > CLO[2]) & (pos[:, 2] < CHI[2]))
    dessous = ombre & (pos[:, 1] < max(cy0, 0.0))
    dedans = ombre & (pos[:, 1] >= max(cy0, 0.0)) & (pos[:, 1] < CHI[1])
    print(f"{label} : {n} particules")
    print(f"   SOUS le cube  : {int(dessous.sum())}")
    print(f"   DANS le cube  : {int(dedans.sum())}")
