"""Non-regression etancheite (niveau core, sans extension Blender) :
colonne d'eau tombant sur une sphere statique, compte de particules finissant
a l'interieur. Reference (avant la refonte bucket/propagation) : 2 sur 46208,
penetration ~9% d'une cellule.
"""
import ctypes

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
d.bq_set_colliders.argtypes = [ctypes.c_void_p, F, F, F, ctypes.c_int]
d.bq_set_colliders.restype = ctypes.c_int
d.bq_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
d.bq_step.restype = ctypes.c_int
d.bq_particle_count.argtypes = [ctypes.c_void_p]
d.bq_particle_count.restype = ctypes.c_int
d.bq_read_positions.argtypes = [ctypes.c_void_p, F]
d.bq_read_positions.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p


def sphere(n_sub, centre, r):
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


cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
if not sim:
    raise SystemExit("bq_create a echoue")

water = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
m_water = d.bq_add_material(sim, ctypes.byref(water))
if m_water < 0:
    raise SystemExit(d.bq_last_error().decode())

CENTRE = (0.5, 0.30, 0.5)
RADIUS = 0.15
tri = sphere(48, centre=CENTRE, r=RADIUS)
n_tri = tri.shape[0]
vel = np.zeros_like(tri)
fric = np.full(n_tri, 0.2, dtype=np.float32)

lo = np.array([0.35, 0.55, 0.35], dtype=np.float32)
hi = np.array([0.65, 0.85, 0.65], dtype=np.float32)
v0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
n_emit = d.bq_emit_box(sim, m_water, lo.ctypes.data_as(F), hi.ctypes.data_as(F), v0.ctypes.data_as(F))
if n_emit < 0:
    raise SystemExit(d.bq_last_error().decode())
print(f"particules emises : {n_emit}")

rc = d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F), fric.ctypes.data_as(F), n_tri)
if rc != 0:
    raise SystemExit(d.bq_last_error().decode())

N_FRAMES = 90
for fr in range(N_FRAMES):
    sub = d.bq_step(sim, 1.0 / 24.0)
    if sub < 0:
        raise SystemExit(d.bq_last_error().decode())

n = d.bq_particle_count(sim)
pos = (ctypes.c_float * (3 * n))()
d.bq_read_positions(sim, pos)
pos = np.frombuffer(pos, dtype=np.float32, count=3 * n).reshape(n, 3).astype(np.float64)

dist = np.linalg.norm(pos - np.array(CENTRE), axis=1)
n_inside = int(np.sum(dist < RADIUS))
penetration = float(RADIUS - dist.min()) if n > 0 else 0.0
dx = cfg.cell_size

print(f"n={n}  n_inside={n_inside}  penetration_max={penetration:.5f} m  dx={dx:.5f} m "
      f"({penetration/dx*100:.1f}% de dx)")

d.bq_destroy(sim)
