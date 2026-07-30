"""Un plan de sol tres etendu (50x la taille du domaine) doit etre accepte
sans allocation delirante ni depassement d'entier."""
import ctypes
import time

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
d.bq_last_error.restype = ctypes.c_char_p

cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx

sim = d.bq_create(ctypes.byref(cfg))
mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
m = d.bq_add_material(sim, ctypes.byref(mat))


def box_tris(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
             (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float32)


# Plan de sol : 50x l'etendue du domaine en x/z, fin en y, centre sur le domaine.
S = 50.0 * L
tri = box_tris((-S / 2, -0.05 * L, -S / 2), (S / 2, 0.0, S / 2))
n_t = tri.shape[0]
vel = np.zeros_like(tri)
fric = np.full(n_t, 0.2, dtype=np.float32)

t0 = time.perf_counter()
rc = d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                        fric.ctypes.data_as(F), n_t)
t1 = time.perf_counter()
if rc != 0:
    raise SystemExit(f"echec : {d.bq_last_error().decode()}")

print(f"plan de sol {S:.1f} m ({S/L:.0f}x L) : accepte")
print(f"bq_set_colliders : {(t1 - t0) * 1e3:.1f} ms (1er appel, inclut rodage cuda)")

REP = 5
t0 = time.perf_counter()
for _ in range(REP):
    d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                       fric.ctypes.data_as(F), n_t)
ms = (time.perf_counter() - t0) / REP * 1e3
print(f"bq_set_colliders : {ms:.1f} ms/appel (moyenne sur {REP})")

d.bq_destroy(sim)
