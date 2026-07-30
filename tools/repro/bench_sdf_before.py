"""Cout du kernel de champ de distance, apres tuilage en memoire partagee.

Passe par ctypes directement : `lib.py` n'expose pas encore `set_colliders`
(lot frontend en cours). Mesure `bq_set_colliders`, qui lance le kernel puis
synchronise, donc le temps mesure est bien celui du kernel.
"""
import ctypes
import time

import numpy as np

DLL = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\bourrasque_before.dll"
d = ctypes.CDLL(DLL)

F = ctypes.POINTER(ctypes.c_float)


class Cfg(ctypes.Structure):
    _fields_ = [("grid_res", ctypes.c_int * 3), ("cell_size", ctypes.c_float),
                ("gravity_y", ctypes.c_float), ("cfl", ctypes.c_float),
                ("ppc_axis", ctypes.c_int), ("max_particles", ctypes.c_int)]


d.bq_abi_version.restype = ctypes.c_int
d.bq_default_config.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.restype = ctypes.c_void_p
d.bq_destroy.argtypes = [ctypes.c_void_p]
d.bq_set_colliders.argtypes = [ctypes.c_void_p, F, F, F, ctypes.c_int]
d.bq_set_colliders.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p

print("ABI version :", d.bq_abi_version())


def sphere(n_sub, centre=(0.5, 0.5, 0.5), r=0.2):
    """Sphere UV triangulee, ~2*n_sub^2 triangles."""
    u = np.linspace(0, np.pi, n_sub + 1)
    v = np.linspace(0, 2 * np.pi, n_sub + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    P = np.stack([centre[0] + r * np.sin(uu) * np.cos(vv),
                  centre[1] + r * np.sin(uu) * np.sin(vv),
                  centre[2] + r * np.cos(uu)], -1)
    tris = []
    for i in range(n_sub):
        for j in range(n_sub):
            a, b, c, e = P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]
            tris.append([a, b, c])
            tris.append([a, c, e])
    return np.array(tris, dtype=np.float32)


class Mat(ctypes.Structure):
    _fields_ = [("model", ctypes.c_int), ("rho", ctypes.c_float),
                ("E", ctypes.c_float), ("nu", ctypes.c_float),
                ("bulk", ctypes.c_float), ("gamma", ctypes.c_float)]


d.bq_add_material.argtypes = [ctypes.c_void_p, ctypes.POINTER(Mat)]
d.bq_add_material.restype = ctypes.c_int

cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
if not sim:
    raise SystemExit("bq_create a echoue")

# Indispensable : `prm` (donc `res` et `dx`) n'est renseigne que par
# `upload_params`, declenche par le premier ajout de materiau. Sans lui,
# `bq_set_colliders` lit res=(0,0,0) et lance une grille de 0 bloc.
mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
if d.bq_add_material(sim, ctypes.byref(mat)) < 0:
    raise SystemExit(d.bq_last_error().decode())

print(f"domaine {tuple(cfg.grid_res)}  dx={cfg.cell_size}\n")
print(f"{'triangles':>10} {'ms/appel':>10} {'us/1000 tri':>12}")
for n_sub in (50, 100, 158):
    tri = sphere(n_sub)
    n_tri = tri.shape[0]
    vel = np.zeros_like(tri)
    fric = np.full(n_tri, 0.2, dtype=np.float32)
    args = (sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
            fric.ctypes.data_as(F), n_tri)
    if d.bq_set_colliders(*args) != 0:          # rodage
        raise SystemExit(d.bq_last_error().decode())
    t0 = time.perf_counter()
    REP = 5
    for _ in range(REP):
        d.bq_set_colliders(*args)
    ms = (time.perf_counter() - t0) / REP * 1e3
    print(f"{n_tri:>10} {ms:>10.2f} {ms * 1000 / n_tri * 1000:>12.1f}")

d.bq_destroy(sim)
