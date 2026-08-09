"""Verification du garde-fou dt (plancher de vitesse reelle, cf. spec T?) :

Avant correctif, dt n'etait derive que de material_sound_speed(bulk, rho), jamais
de la vitesse effective des particules. A bulk=20 (eau tres molle) et vitesse
reelle 10 m/s, le deplacement par sous-pas atteignait ~21 dx -- bien au-dela du
rayon de recherche de la CCD (~6 dx) -- et une fraction significative des
particules traversait completement un mur fin sans jamais etre rattrapee par la
CCD, restant bloquee de l'autre cote en permanence.

Ce script emet un bloc de particules bulk=20 avec une vitesse initiale +10 m/s
en x, en face d'un mur fin FERME (boite creuse etanche, ~1.3 dx d'epaisseur --
pas un simple quad ouvert, qui casse le flood-fill de signe du SDF), et mesure
la fraction de particules ayant fini de l'autre cote du mur (x > 0.70) apres
plusieurs frames.
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
d.bq_emit_points_vel.argtypes = [ctypes.c_void_p, ctypes.c_int, F, F, ctypes.c_int]
d.bq_emit_points_vel.restype = ctypes.c_int
d.bq_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
d.bq_step.restype = ctypes.c_int
d.bq_read_positions.argtypes = [ctypes.c_void_p, F]
d.bq_read_positions.restype = ctypes.c_int
d.bq_read_velocities.argtypes = [ctypes.c_void_p, F]
d.bq_read_velocities.restype = ctypes.c_int
d.bq_particle_count.argtypes = [ctypes.c_void_p]
d.bq_particle_count.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p

_FACES = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
          (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
          (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]


def box_tris(lo, hi, flip=False):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(a, c, b) for a, b, c in _FACES] if flip else _FACES
    return np.array([[v[a], v[b], v[c]] for a, b, c in f], dtype=np.float32)


cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
if not sim:
    raise SystemExit(d.bq_last_error().decode())

mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=20.0, gamma=3.0)
mid = d.bq_add_material(sim, ctypes.byref(mat))
if mid < 0:
    raise SystemExit(d.bq_last_error().decode())

R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx
print(f"grille {R}^3, dx = {dx * 1000:.3f} mm, domaine {L:.3f} m")

# mur fin ferme (boite creuse etanche), ~1.3 dx d'epaisseur, a x = 0.59..0.61.
# Le mur deborde largement du domaine en y/z (au-dela des parois du domaine
# lui-meme) pour former un veritable septum qui coupe le domaine en deux
# chambres -- sinon les particules peuvent simplement contourner un mur fini
# par les cotes, ce qui n'a rien a voir avec une traversee (fuite CCD).
t = 1.3 * dx
WALL_LO = np.array([0.60 * L - t / 2, -0.5 * L, -0.5 * L])
WALL_HI = np.array([0.60 * L + t / 2, 1.5 * L, 1.5 * L])
tri = np.concatenate([box_tris(WALL_LO, WALL_HI),
                      box_tris(WALL_LO + t, WALL_HI - t, flip=True)])
n_tri = tri.shape[0]
vel_tri = np.zeros_like(tri)
fric = np.full(n_tri, 0.2, dtype=np.float32)
if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel_tri.ctypes.data_as(F),
                      fric.ctypes.data_as(F), n_tri) != 0:
    raise SystemExit(d.bq_last_error().decode())
print(f"mur : x = [{WALL_LO[0] / L:.3f}, {WALL_HI[0] / L:.3f}] * L "
      f"(epaisseur {t / dx:.2f} dx)")

# bloc de particules a gauche du mur, vitesse initiale +10 m/s en x
rng = np.random.default_rng(0)
n_pts = 20000
pos = np.empty((n_pts, 3), dtype=np.float32)
pos[:, 0] = rng.uniform(0.30 * L, 0.50 * L, n_pts)
pos[:, 1] = rng.uniform(0.20 * L, 0.80 * L, n_pts)
pos[:, 2] = rng.uniform(0.20 * L, 0.80 * L, n_pts)
vel = np.zeros((n_pts, 3), dtype=np.float32)
vel[:, 0] = 10.0

n_emit = d.bq_emit_points_vel(sim, mid, pos.ctypes.data_as(F),
                              vel.ctypes.data_as(F), n_pts)
if n_emit < 0:
    raise SystemExit(d.bq_last_error().decode())
print(f"emis : {n_emit} particules, vitesse initiale 10 m/s en x")

N_FRAMES = 15
FRAME_DT = 1.0 / 30.0
for f in range(N_FRAMES):
    substeps = d.bq_step(sim, FRAME_DT)
    if substeps < 0:
        raise SystemExit(d.bq_last_error().decode())
    n_now = d.bq_particle_count(sim)
    vbuf = np.empty(n_now * 3, dtype=np.float32)
    d.bq_read_velocities(sim, vbuf.ctypes.data_as(F))
    speed = np.linalg.norm(vbuf.reshape(n_now, 3), axis=1)
    print(f"  frame {f:2d}: substeps={substeps:6d}  n={n_now:6d}  "
          f"max|v|={speed.max():10.3f}  mean|v|={speed.mean():8.3f}")

n = d.bq_particle_count(sim)
buf = np.empty(n * 3, dtype=np.float32)
d.bq_read_positions(sim, buf.ctypes.data_as(F))
xs = buf.reshape(n, 3)[:, 0]

crossed = int((xs > 0.70 * L).sum())
rate = crossed / n if n else 0.0
print(f"\napres {N_FRAMES} frames : {n} particules vivantes")
print(f"particules avec x > 0.70*L (traversees) : {crossed} / {n} "
      f"({rate * 100:.1f} %)")

d.bq_destroy(sim)
