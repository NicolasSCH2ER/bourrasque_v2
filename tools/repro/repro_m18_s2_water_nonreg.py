"""M18/S2 -- non-regression EAU (bruit contre bruit).

Le dispatch constitutif (k_p2g), material_sound_speed et le reseeding ont ete
touches par l'ajout du modele SAND (D2/D4/D5). Ce script rejoue une scene d'eau
identique (colonne en chute libre dans un bac ouvert, avec reseeding actif) sur
DEUX binaires -- la DLL courante (post-S2) et une DLL de reference reconstruite
en revertant EXACTEMENT les edits de S2 sur une copie de mlsmpm.cu (pre-S2,
c-a-d juste apres S1/SVD3) -- et compare des agregats macro (bounding box,
vitesse moyenne, nombre de particules vivantes) sur plusieurs frames.

Comparaison : bruit run-a-run (3 executions par cote, meme binaire, seed RNG
d'emission fixe mais le solveur n'est pas deterministe bit-a-bit d'une
execution a l'autre -- atomicAdd flottants) contre l'ecart entre binaires. Si
l'ecart inter-binaires est du meme ordre que le bruit intra-binaire, pas de
regression detectable.

Usage: python repro_m18_s2_water_nonreg.py <dll_path> <label>
Ecrit un resume JSON-like sur stdout, une ligne par frame echantillonnee.
"""
import sys
import ctypes

import numpy as np

DLL = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else DLL

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
d.bq_emit_box.argtypes = [ctypes.c_void_p, ctypes.c_int, F, F, F]
d.bq_emit_box.restype = ctypes.c_int
d.bq_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
d.bq_step.restype = ctypes.c_int
d.bq_read_positions.argtypes = [ctypes.c_void_p, F]
d.bq_read_positions.restype = ctypes.c_int
d.bq_read_velocities.argtypes = [ctypes.c_void_p, F]
d.bq_read_velocities.restype = ctypes.c_int
d.bq_particle_count.argtypes = [ctypes.c_void_p]
d.bq_particle_count.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p

cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
if not sim:
    raise SystemExit(d.bq_last_error().decode())

mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=5e4, gamma=7.0,
         friction_angle=0.0, cohesion=0.0)
mid = d.bq_add_material(sim, ctypes.byref(mat))
if mid < 0:
    raise SystemExit(d.bq_last_error().decode())

R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx

lo = (ctypes.c_float * 3)(0.15 * L, 0.15 * L, 0.15 * L)
hi = (ctypes.c_float * 3)(0.45 * L, 0.55 * L, 0.85 * L)
vel = (ctypes.c_float * 3)(0.0, 0.0, 0.0)
n_emit = d.bq_emit_box(sim, mid, lo, hi, vel)
if n_emit < 0:
    raise SystemExit(d.bq_last_error().decode())

FRAME_DT = 1.0 / 30.0
N_FRAMES = 40
SAMPLE = {5, 15, 25, 39}

print(f"# label={LABEL} n_emit={n_emit}")
for f in range(N_FRAMES):
    substeps = d.bq_step(sim, FRAME_DT)
    if substeps < 0:
        raise SystemExit(d.bq_last_error().decode())
    if f in SAMPLE:
        n = d.bq_particle_count(sim)
        xb = np.empty(n * 3, dtype=np.float32)
        vb = np.empty(n * 3, dtype=np.float32)
        d.bq_read_positions(sim, xb.ctypes.data_as(F))
        d.bq_read_velocities(sim, vb.ctypes.data_as(F))
        x = xb.reshape(n, 3)
        v = vb.reshape(n, 3)
        speed = np.linalg.norm(v, axis=1)
        bbox_lo = x.min(axis=0)
        bbox_hi = x.max(axis=0)
        print(f"RESULT label={LABEL} frame={f:3d} n={n:6d} "
              f"bbox_lo={bbox_lo[0]:.5f},{bbox_lo[1]:.5f},{bbox_lo[2]:.5f} "
              f"bbox_hi={bbox_hi[0]:.5f},{bbox_hi[1]:.5f},{bbox_hi[2]:.5f} "
              f"mean_speed={speed.mean():.6f} max_speed={speed.max():.6f} "
              f"centroid={x.mean(axis=0)[0]:.5f},{x.mean(axis=0)[1]:.5f},{x.mean(axis=0)[2]:.5f}")

d.bq_destroy(sim)
