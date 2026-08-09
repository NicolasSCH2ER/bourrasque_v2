"""Mesure honnete AVANT/APRES du tri spatial (perf) : rejoue un scenario avec
un nombre de substeps eleve (materiau mou bulk=20 + vitesse initiale forte,
meme motif de plancher dt que repro_ccd_dt_floor.py, sur une grille plus fine
que le defaut pour amplifier le cout par substep), laisse le fluide s'ecrouler
et se desordonner en memoire pendant plusieurs frames (chute + rebond +
reseeding), PUIS mesure le temps de bq_step sur les frames suivantes -- pas
sur les toutes premieres frames ou l'ordre d'emission est deja spatialement
coherent (biais qui masquerait tout gain du tri).

Usage: python bench_sort_perf.py <dll_path>
"""
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
d.bq_emit_points_vel.argtypes = [ctypes.c_void_p, ctypes.c_int, F, F, ctypes.c_int]
d.bq_emit_points_vel.restype = ctypes.c_int
d.bq_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
d.bq_step.restype = ctypes.c_int
d.bq_particle_count.argtypes = [ctypes.c_void_p]
d.bq_particle_count.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p

cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 96
cfg.cell_size = 1.0 / 96.0
cfg.ppc_axis = 2
cfg.max_particles = 2_000_000
sim = d.bq_create(ctypes.byref(cfg))
if not sim:
    raise SystemExit(d.bq_last_error().decode())

# materiau raide (bulk eleve -> vitesse du son elevee -> dt CFL petit) combine
# a une grille fine : produit beaucoup de sous-pas par frame, le regime qui
# amplifie le cout relatif de P2G/G2P (les deux kernels les plus chauds,
# executes une fois par sous-pas) par rapport au tri spatial (une fois par
# frame). Note : contrairement a l'intuition, un materiau MOU (bulk faible,
# cf. repro_ccd_dt_floor.py) donne ICI PEU de sous-pas -- dt = cfl*dx/c_son,
# et c_son = sqrt(bulk/rho) est plus FAIBLE pour un materiau mou, donc dt plus
# GRAND. Le materiau mou de repro_ccd_dt_floor.py ne produit beaucoup de
# sous-pas que via le plancher de VITESSE REELLE (v=10 m/s), pas via le bulk
# lui-meme -- pour ce bench on prend le chemin direct (bulk eleve).
mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e5, gamma=3.0)
mid = d.bq_add_material(sim, ctypes.byref(mat))
if mid < 0:
    raise SystemExit(d.bq_last_error().decode())

R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx
rng = np.random.default_rng(0)
n_pts = 250_000
pos = np.empty((n_pts, 3), dtype=np.float32)
pos[:, 0] = rng.uniform(0.15 * L, 0.85 * L, n_pts)
pos[:, 1] = rng.uniform(0.55 * L, 0.85 * L, n_pts)
pos[:, 2] = rng.uniform(0.15 * L, 0.85 * L, n_pts)
vel = np.zeros((n_pts, 3), dtype=np.float32)
vel[:, 1] = -8.0  # chute rapide -> plancher dt actif, chute puis eclaboussure

n_emit = d.bq_emit_points_vel(sim, mid, pos.ctypes.data_as(F),
                              vel.ctypes.data_as(F), n_pts)
if n_emit < 0:
    raise SystemExit(d.bq_last_error().decode())

FRAME_DT = 1.0 / 24.0

# rodage : laisser le fluide chuter, eclabousser, se reordonner par le
# reseeding/l'advection -- N frames desordonnent completement le tableau par
# rapport a l'ordre d'emission (spatialement coherent au depart).
WARMUP = 20
substeps_seen = []
for _ in range(WARMUP):
    ss = d.bq_step(sim, FRAME_DT)
    if ss < 0:
        raise SystemExit(d.bq_last_error().decode())
    substeps_seen.append(ss)

n_now = d.bq_particle_count(sim)

# mesure : plusieurs frames, apres rodage
times = []
REP = 20
for _ in range(REP):
    t0 = time.perf_counter()
    ss = d.bq_step(sim, FRAME_DT)
    times.append((time.perf_counter() - t0) * 1e3)
times = np.array(times)
print(f"n~{n_now} substeps(warmup last)={substeps_seen[-1]} substeps(measured last)={ss} "
      f"median={np.median(times):.1f}ms mean={times.mean():.1f}ms "
      f"min={times.min():.1f}ms max={times.max():.1f}ms")
d.bq_destroy(sim)
