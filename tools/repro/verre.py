"""Etancheite COMPORTEMENTALE d'un collider a paroi fine (le cas « verre »).

Le fluide est emis directement dans la cavite d'une boite creuse etanche, dont on
fait varier l'epaisseur de paroi. Sous la gravite, il doit rester dedans. Le SDF
signe est aveugle sous 1 dx (mesure par repro_paroi.py : 0 cellule solide) : seule
la couche de contact sur distance non signee peut retenir le fluide.

Reference du bas : la meme emission SANS aucun collider. Le fluide tombe alors au
sol du domaine, tres au-dessous de la cavite -- c'est la mesure de « rien ne
retient », donc ce que donnait necessairement l'ancien code sur une paroi fine.
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


for name, argt, rest in (
    ("bq_default_config", [ctypes.POINTER(Cfg)], None),
    ("bq_create", [ctypes.POINTER(Cfg)], ctypes.c_void_p),
    ("bq_destroy", [ctypes.c_void_p], None),
    ("bq_add_material", [ctypes.c_void_p, ctypes.POINTER(Mat)], ctypes.c_int),
    ("bq_set_colliders", [ctypes.c_void_p, F, F, F, ctypes.c_int], ctypes.c_int),
    ("bq_emit_box", [ctypes.c_void_p, ctypes.c_int, F, F, F], ctypes.c_int),
    ("bq_step", [ctypes.c_void_p, ctypes.c_float], ctypes.c_int),
    ("bq_particle_count", [ctypes.c_void_p], ctypes.c_int),
    ("bq_read_positions", [ctypes.c_void_p, F], ctypes.c_int),
):
    getattr(d, name).argtypes = argt
    if rest:
        getattr(d, name).restype = rest
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
cfg.gravity_y = -9.8
R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx
N_FRAME = 120

OLO = np.array([0.25 * L] * 3, dtype=np.float64)
OHI = np.array([0.75 * L] * 3, dtype=np.float64)
print(f"grille {R}^3, dx = {dx * 1000:.2f} mm, boite {OLO[0]:.3f}..{OHI[0]:.3f} m")
print(f"{N_FRAME} frames a 1/24 s\n")


def run(tri):
    sim = d.bq_create(ctypes.byref(cfg))
    mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
    m = d.bq_add_material(sim, ctypes.byref(mat))
    if tri is not None:
        n = tri.shape[0]
        vel = np.zeros_like(tri)
        fric = np.full(n, 0.2, dtype=np.float32)
        if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                              fric.ctypes.data_as(F), n) != 0:
            raise SystemExit(d.bq_last_error().decode())
    # Emission dans la cavite, avec 2 dx de marge pour ne naitre dans aucune paroi.
    lo = (OLO + 4 * dx).astype(np.float32)
    hi = np.array([OHI[0] - 4 * dx, OLO[1] + 0.15 * L, OHI[2] - 4 * dx],
                  dtype=np.float32)
    vz = np.zeros(3, dtype=np.float32)
    n_p = d.bq_emit_box(sim, m, lo.ctypes.data_as(F), hi.ctypes.data_as(F),
                        vz.ctypes.data_as(F))
    for _ in range(N_FRAME):
        d.bq_step(sim, 1.0 / 24.0)
    pos = np.empty((d.bq_particle_count(sim), 3), dtype=np.float32)
    d.bq_read_positions(sim, pos.ctypes.data_as(F))
    d.bq_destroy(sim)
    return n_p, pos.astype(np.float64)


def rapport(label, n_p, pos, off=0.0):
    # Fuite incontestable : la particule est sortie de la boite EXTERIEURE.
    lo, hi = OLO + off, OHI + off
    dehors = ((pos < lo).any(axis=1) | (pos > hi).any(axis=1))
    sous = pos[:, 1] < lo[1]
    print(f"{label}")
    print(f"   {n_p} particules emises")
    print(f"   SORTIES de la boite : {int(dehors.sum()):>6}"
          f"  ({dehors.sum() / n_p * 100:5.1f} %)   dont sous le fond : {int(sous.sum())}")
    print(f"   y min = {pos[:, 1].min():.4f} m   (fond de la cavite ~ "
          f"{OLO[1]:.4f} m, sol du domaine = {3 * dx:.4f} m)")


n_p, pos = run(None)
rapport("SANS collider (reference « rien ne retient »)", n_p, pos)
print()

# OLO vaut 0.25*L = 16 dx : la boite est PARFAITEMENT alignee sur la grille, donc
# ses faces tombent pile sur des noeuds -- distance nulle, direction du point le
# plus proche indeterminee. On teste donc aussi un decalage d'un demi-dx, qui est
# le cas general (une geometrie d'artiste n'est jamais alignee).
for off_name, off in (("aligne sur la grille", 0.0), ("decale de dx/2", 0.5 * dx)):
    print(f"=== boite {off_name} ===")
    for tm in (0.25, 0.5, 1.0, 2.0):
        t = tm * dx
        tri = np.concatenate([box_tris(OLO + off, OHI + off),
                              box_tris(OLO + off + t, OHI + off - t, flip=True)]
                             ).astype(np.float32)
        n_p, pos = run(tri)
        rapport(f"paroi t = {tm} dx ({t * 1000:.2f} mm)", n_p, pos, off)
    print()
