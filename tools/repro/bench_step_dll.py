"""bench_step parametrable par chemin de DLL, pour comparaison interleavee."""
import ctypes
import sys
import time

import numpy as np

DLL = sys.argv[1]
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
for _ in range(3):
    d.bq_step(sim, FRAME_DT)                      # rodage
times = []
REP = 15
for _ in range(REP):
    t0 = time.perf_counter()
    d.bq_step(sim, FRAME_DT)
    times.append((time.perf_counter() - t0) * 1e3)
times = np.array(times)
print(f"n={n} median={np.median(times):.1f}ms mean={times.mean():.1f}ms "
      f"min={times.min():.1f}ms max={times.max():.1f}ms")
d.bq_destroy(sim)
