"""M18/S2 -- verifie l'absence de NaN sur une scene de sable apres plusieurs
centaines de frames (colonne de sable versee sur un sol plat, colliders
actifs -- scenario proche de l'usage reel, contrairement a repro_sand_apex.py
qui teste la chute libre isolee)."""
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
                ("bulk", ctypes.c_float), ("gamma", ctypes.c_float),
                ("friction_angle", ctypes.c_float), ("cohesion", ctypes.c_float)]


d.bq_default_config.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.restype = ctypes.c_void_p
d.bq_destroy.argtypes = [ctypes.c_void_p]
d.bq_add_material.argtypes = [ctypes.c_void_p, ctypes.POINTER(Mat)]
d.bq_add_material.restype = ctypes.c_int
d.bq_set_colliders.argtypes = [ctypes.c_void_p, F, F, F,
                               ctypes.POINTER(ctypes.c_int), ctypes.c_int]
d.bq_set_colliders.restype = ctypes.c_int
d.bq_emit_box.argtypes = [ctypes.c_void_p, ctypes.c_int, F, F, F]
d.bq_emit_box.restype = ctypes.c_int
d.bq_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
d.bq_step.restype = ctypes.c_int
d.bq_read_positions.argtypes = [ctypes.c_void_p, F]
d.bq_read_positions.restype = ctypes.c_int
d.bq_particle_count.argtypes = [ctypes.c_void_p]
d.bq_particle_count.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p

cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
if not sim:
    raise SystemExit(d.bq_last_error().decode())

mat = Mat(model=2, rho=1600.0, E=3.5e5, nu=0.3, bulk=0.0, gamma=0.0,
         friction_angle=35.0, cohesion=0.0)
mid = d.bq_add_material(sim, ctypes.byref(mat))
if mid < 0:
    raise SystemExit(d.bq_last_error().decode())

R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx

# sol plat : un seul quad (2 triangles), couvrant tout le domaine, a y=0.10*L
Y_FLOOR = 0.10 * L
pad = 2.0 * L
tri = np.array([
    [(-pad, Y_FLOOR, -pad), (L + pad, Y_FLOOR, -pad), (L + pad, Y_FLOOR, L + pad)],
    [(-pad, Y_FLOOR, -pad), (L + pad, Y_FLOOR, L + pad), (-pad, Y_FLOOR, L + pad)],
], dtype=np.float32)
vel_tri = np.zeros_like(tri)
fric = np.full(2, 0.5, dtype=np.float32)
if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel_tri.ctypes.data_as(F),
                      fric.ctypes.data_as(F), None, 2) != 0:
    raise SystemExit(d.bq_last_error().decode())

# colonne de sable au-dessus du sol
lo = (ctypes.c_float * 3)(0.40 * L, 0.45 * L, 0.40 * L)
hi = (ctypes.c_float * 3)(0.60 * L, 0.85 * L, 0.60 * L)
vel = (ctypes.c_float * 3)(0.0, 0.0, 0.0)
n_emit = d.bq_emit_box(sim, mid, lo, hi, vel)
if n_emit < 0:
    raise SystemExit(d.bq_last_error().decode())
print(f"emis {n_emit} particules SAND au-dessus d'un sol a y={Y_FLOOR:.4f}")

FRAME_DT = 1.0 / 30.0
N_FRAMES = 300
CHECK_EVERY = 30
for f in range(N_FRAMES):
    substeps = d.bq_step(sim, FRAME_DT)
    if substeps < 0:
        raise SystemExit(f"bq_step a echoue frame {f}: {d.bq_last_error().decode()}")
    if f % CHECK_EVERY == 0 or f == N_FRAMES - 1:
        n = d.bq_particle_count(sim)
        xb = np.empty(n * 3, dtype=np.float32)
        d.bq_read_positions(sim, xb.ctypes.data_as(F))
        nan_count = int(np.isnan(xb).sum())
        inf_count = int(np.isinf(xb).sum())
        print(f"  frame {f:3d}: substeps={substeps:5d} n={n:6d} "
              f"nan={nan_count} inf={inf_count}")
        if nan_count or inf_count:
            raise SystemExit(f"FAIL: NaN/Inf detecte frame {f}")

print(f"\nOK : {N_FRAMES} frames, aucun NaN/Inf detecte")
d.bq_destroy(sim)
