"""Defaut 2 : un triangle degenere isole (aire nulle) ne doit pas creer de
coquille de cellules solides parasites collee a la surface.

Compare le nombre de cellules solides d'une sphere avec / sans un triangle
degenere ajoute au maillage (trois sommets confondus, et un cas colineaire).
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


def sphere(n_sub, centre=(0.5, 0.5, 0.5), r=0.2):
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


def run(tri):
    n_tri = tri.shape[0]
    cfg = Cfg()
    d.bq_default_config(ctypes.byref(cfg))
    sim = d.bq_create(ctypes.byref(cfg))
    mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
    if d.bq_add_material(sim, ctypes.byref(mat)) < 0:
        raise SystemExit(d.bq_last_error().decode())
    vel = np.zeros_like(tri)
    fric = np.full(n_tri, 0.2, dtype=np.float32)
    rc = d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                            fric.ctypes.data_as(F), n_tri)
    if rc != 0:
        raise SystemExit(d.bq_last_error().decode())
    rx, ry, rz = cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2]
    ncell = rx * ry * rz
    buf = (ctypes.c_float * ncell)()
    d.bq_read_sdf(sim, buf)
    sdf = np.frombuffer(buf, dtype=np.float32, count=ncell)
    n_solid = int((sdf < 0).sum())
    d.bq_destroy(sim)
    return n_solid, ncell


CENTRE = (0.5, 0.5, 0.5)
RADIUS = 0.2
base = sphere(50, centre=CENTRE, r=RADIUS)

# triangle degenere 1 : trois sommets confondus
degen_point = np.array([[[0.5, 0.5, 0.7], [0.5, 0.5, 0.7], [0.5, 0.5, 0.7]]], dtype=np.float32)
# triangle degenere 2 : trois sommets colineaires
degen_line = np.array([[[0.5, 0.5, 0.7], [0.51, 0.5, 0.7], [0.52, 0.5, 0.7]]], dtype=np.float32)

tri_with_degen = np.concatenate([base, degen_point, degen_line], axis=0)

n_solid_base, ncell = run(base)
n_solid_degen, _ = run(tri_with_degen)

print(f"sans triangle degenere : {n_solid_base} / {ncell} cellules solides")
print(f"avec triangle degenere : {n_solid_degen} / {ncell} cellules solides")
print(f"difference : {n_solid_degen - n_solid_base}")
