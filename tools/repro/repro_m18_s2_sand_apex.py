"""M18/S2 -- verifie le cas "sommet du cone" (D2) : un bloc de sable lache en
chute libre, SANS collider, doit se DISPERSER (le sable ne tire pas -- tr(eps)
> 0 efface toute trace elastique, S -> 1). Le meme bloc en ELASTIC doit rester
COHERENT (le corotationnel fixe ne connait pas cette regle).

Mesure : ecart-type des positions (proxy de "coherence du bloc") a la meme
frame, sans aucun collider pour ne pas confondre avec un effet de contact.
Le bloc SABLE doit avoir un ecart-type nettement plus grand que le bloc
ELASTIC a la meme frame (les deux partent du meme volume, meme vitesse
initiale nulle, seule la gravite agit).
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
d.bq_particle_count.argtypes = [ctypes.c_void_p]
d.bq_particle_count.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p


def run(model, extra):
    cfg = Cfg()
    d.bq_default_config(ctypes.byref(cfg))
    sim = d.bq_create(ctypes.byref(cfg))
    if not sim:
        raise SystemExit(d.bq_last_error().decode())

    mat = Mat(model=model, rho=1600.0, E=3.5e5, nu=0.3, bulk=0.0, gamma=0.0,
             friction_angle=extra.get("friction_angle", 0.0), cohesion=0.0)
    mid = d.bq_add_material(sim, ctypes.byref(mat))
    if mid < 0:
        raise SystemExit(d.bq_last_error().decode())

    R, dx = cfg.grid_res[0], cfg.cell_size
    L = R * dx
    # bloc suspendu en milieu de domaine, loin de tout bord (pas de collider,
    # pas de rebond sur les bords du domaine pendant la fenetre mesuree)
    lo = (ctypes.c_float * 3)(0.40 * L, 0.55 * L, 0.40 * L)
    hi = (ctypes.c_float * 3)(0.60 * L, 0.70 * L, 0.60 * L)
    vel = (ctypes.c_float * 3)(0.0, 0.0, 0.0)
    n_emit = d.bq_emit_box(sim, mid, lo, hi, vel)
    if n_emit < 0:
        raise SystemExit(d.bq_last_error().decode())

    FRAME_DT = 1.0 / 30.0
    N_FRAMES = 20
    for f in range(N_FRAMES):
        substeps = d.bq_step(sim, FRAME_DT)
        if substeps < 0:
            raise SystemExit(d.bq_last_error().decode())

    n = d.bq_particle_count(sim)
    xb = np.empty(n * 3, dtype=np.float32)
    d.bq_read_positions(sim, xb.ctypes.data_as(F))
    x = xb.reshape(n, 3)
    nan_count = int(np.isnan(x).sum())
    d.bq_destroy(sim)
    return n_emit, n, x, nan_count


n_emit_e, n_e, x_e, nan_e = run(0, {})   # ELASTIC
n_emit_s, n_s, x_s, nan_s = run(2, {"friction_angle": 35.0})  # SAND

std_e = x_e.std(axis=0)
std_s = x_s.std(axis=0)
bbox_e = x_e.max(axis=0) - x_e.min(axis=0)
bbox_s = x_s.max(axis=0) - x_s.min(axis=0)

print(f"ELASTIC : n_emit={n_emit_e} n_final={n_e} nan={nan_e} "
      f"std=({std_e[0]:.5f},{std_e[1]:.5f},{std_e[2]:.5f}) "
      f"bbox=({bbox_e[0]:.5f},{bbox_e[1]:.5f},{bbox_e[2]:.5f})")
print(f"SAND    : n_emit={n_emit_s} n_final={n_s} nan={nan_s} "
      f"std=({std_s[0]:.5f},{std_s[1]:.5f},{std_s[2]:.5f}) "
      f"bbox=({bbox_s[0]:.5f},{bbox_s[1]:.5f},{bbox_s[2]:.5f})")

vol_std_e = float(np.prod(std_e))
vol_std_s = float(np.prod(std_s))
print(f"\nprod(std) ELASTIC = {vol_std_e:.8e}")
print(f"prod(std) SAND    = {vol_std_s:.8e}")
print(f"ratio SAND/ELASTIC = {vol_std_s / vol_std_e:.3f}x")

assert nan_e == 0 and nan_s == 0, "NaN detecte"
assert vol_std_s > vol_std_e, "le sable ne s'est pas disperse plus que l'elastique"
print("\nOK : le bloc SAND se disperse nettement plus que le bloc ELASTIC "
      "(sommet du cone actif)")
