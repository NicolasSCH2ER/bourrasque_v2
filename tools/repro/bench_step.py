"""Cout d'un `step` de solveur, pour situer celui du champ de distance.

Le SDF coute ~112 ms/frame a 5000 triangles. Est-ce grand ou petit devant
le solveur lui-meme ? C'est le ratio qui decide s'il faut optimiser.
"""
import ctypes
import time

import numpy as np

DLL = r"C:\Users\nicol\Code\bourrasque_v2\build\Release\bourrasque.dll"
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
d.bq_emit_box.argtypes = [ctypes.c_void_p, ctypes.c_int, F, F, F]
d.bq_emit_box.restype = ctypes.c_int
d.bq_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
d.bq_step.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p

cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
m = d.bq_add_material(sim, ctypes.byref(mat))

lo = np.array([0.2, 0.2, 0.2], dtype=np.float32)
hi = np.array([0.8, 0.6, 0.8], dtype=np.float32)
vz = np.zeros(3, dtype=np.float32)
n = d.bq_emit_box(sim, m, lo.ctypes.data_as(F), hi.ctypes.data_as(F),
                  vz.ctypes.data_as(F))
if n < 0:
    raise SystemExit(d.bq_last_error().decode())

FRAME_DT = 1.0 / 24.0
sub = d.bq_step(sim, FRAME_DT)                      # rodage
t0 = time.perf_counter()
REP = 10
for _ in range(REP):
    d.bq_step(sim, FRAME_DT)
ms = (time.perf_counter() - t0) / REP * 1e3

print(f"particules      : {n}")
print(f"substeps/frame  : {sub}")
print(f"step            : {ms:.1f} ms/frame")
print(f"SDF (5000 tri)  : 111.8 ms/frame")
print(f"-> le SDF represente {111.8 / ms * 100:.0f} % du cout d'un step")
d.bq_destroy(sim)
