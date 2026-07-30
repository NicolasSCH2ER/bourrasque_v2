"""Reproduction de la scene reelle : domaine 40 cm, verre cylindrique ouvert,
paroi de 2 mm, rempli par un jet.

Trois differences avec le test sur boite fermee (verre.py), qui donnait 0.1 % de
fuite a epaisseur comparable :
  - le verre est OUVERT en haut : le maillage n'est pas etanche, donc aucune
    cellule ne peut etre classee INTERIOR ;
  - c'est un CYLINDRE : surfaces courbes, jamais alignees sur la grille ;
  - il est rempli par un JET : le fluide arrive avec de la vitesse, la ou le
    test precedent partait d'un fluide au repos.
"""
import ctypes
import sys

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
    ("bq_emit_points_vel", [ctypes.c_void_p, ctypes.c_int, F, F, ctypes.c_int], ctypes.c_int),
    ("bq_step", [ctypes.c_void_p, ctypes.c_float], ctypes.c_int),
    ("bq_particle_count", [ctypes.c_void_p], ctypes.c_int),
    ("bq_read_positions", [ctypes.c_void_p, F], ctypes.c_int),
    ("bq_read_sdf", [ctypes.c_void_p, F], ctypes.c_int),
):
    getattr(d, name).argtypes = argt
    if rest:
        getattr(d, name).restype = rest
d.bq_last_error.restype = ctypes.c_char_p

DOM = 0.40          # domaine cubique de 40 cm
R_EXT = 0.10        # verre : cylindre de 20 cm de diametre
H = 0.20            # hauteur 20 cm
T = 0.002           # paroi 2 mm
N_SEG = 48


def verre_tris(cx, cz, y0):
    """Cylindre creux ouvert en haut, normales sortantes. Maillage etanche
    partout SAUF l'ouverture superieure (c'est justement le cas reel)."""
    ri = R_EXT - T
    y1 = y0 + H
    yi = y0 + T
    a = np.linspace(0.0, 2.0 * np.pi, N_SEG + 1)[:-1]
    a2 = np.roll(a, -1)
    tris = []

    def quad(p0, p1, p2, p3):
        tris.append([p0, p1, p2])
        tris.append([p0, p2, p3])

    for t0, t1 in zip(a, a2):
        eo0 = (cx + R_EXT * np.cos(t0), cz + R_EXT * np.sin(t0))
        eo1 = (cx + R_EXT * np.cos(t1), cz + R_EXT * np.sin(t1))
        ei0 = (cx + ri * np.cos(t0), cz + ri * np.sin(t0))
        ei1 = (cx + ri * np.cos(t1), cz + ri * np.sin(t1))
        # paroi exterieure (normale vers l'exterieur)
        quad((eo0[0], y0, eo0[1]), (eo1[0], y0, eo1[1]),
             (eo1[0], y1, eo1[1]), (eo0[0], y1, eo0[1]))
        # paroi interieure (normale vers l'axe)
        quad((ei0[0], yi, ei0[1]), (ei0[0], y1, ei0[1]),
             (ei1[0], y1, ei1[1]), (ei1[0], yi, ei1[1]))
        # dessous exterieur (normale vers le bas)
        tris.append([(cx, y0, cz), (eo1[0], y0, eo1[1]), (eo0[0], y0, eo0[1])])
        # fond interieur (normale vers le haut)
        tris.append([(cx, yi, cz), (ei0[0], yi, ei0[1]), (ei1[0], yi, ei1[1])])
        # couronne du bord superieur (normale vers le haut)
        quad((ei0[0], y1, ei0[1]), (eo0[0], y1, eo0[1]),
             (eo1[0], y1, eo1[1]), (ei1[0], y1, ei1[1]))
    return np.array(tris, dtype=np.float32)


def run(R, avec_collider, n_frame=96):
    cfg = Cfg()
    d.bq_default_config(ctypes.byref(cfg))
    for i in range(3):
        cfg.grid_res[i] = R
    cfg.cell_size = DOM / R
    cfg.gravity_y = -9.8
    dx = cfg.cell_size
    sim = d.bq_create(ctypes.byref(cfg))
    mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
    m = d.bq_add_material(sim, ctypes.byref(mat))

    c = DOM / 2.0
    y0 = 3 * dx                       # verre pose sur le sol du domaine
    tri = verre_tris(c, c, y0)
    if avec_collider:
        n_t = tri.shape[0]
        vel = np.zeros_like(tri)
        fric = np.full(n_t, 0.2, dtype=np.float32)
        if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                              fric.ctypes.data_as(F), n_t) != 0:
            raise SystemExit(d.bq_last_error().decode())

    # Jet vertical au-dessus du centre du verre, emis a chaque frame.
    jet_r = 0.02
    jet_y = y0 + H + 0.05
    jet_v = -1.0
    spacing = dx / 2.0
    g = np.arange(-jet_r, jet_r + 1e-9, spacing)
    gx, gz = np.meshgrid(g, g, indexing="ij")
    keep = gx ** 2 + gz ** 2 <= jet_r ** 2
    site = np.stack([c + gx[keep], np.full(keep.sum(), jet_y), c + gz[keep]], -1)
    site = np.ascontiguousarray(site, dtype=np.float32)
    jvel = np.tile(np.array([0.0, jet_v, 0.0], dtype=np.float32), (len(site), 1))
    jvel = np.ascontiguousarray(jvel)

    for _ in range(n_frame):
        d.bq_emit_points_vel(sim, m, site.ctypes.data_as(F),
                             jvel.ctypes.data_as(F), len(site))
        d.bq_step(sim, 1.0 / 24.0)

    n = d.bq_particle_count(sim)
    pos = np.empty((n, 3), dtype=np.float32)
    d.bq_read_positions(sim, pos.ctypes.data_as(F))

    ncell = R * R * R
    sdf = np.empty(ncell, dtype=np.float32)
    d.bq_read_sdf(sim, sdf.ctypes.data_as(F))
    d.bq_destroy(sim)

    pos = pos.astype(np.float64)
    rr = np.hypot(pos[:, 0] - c, pos[:, 2] - c)
    # Fuite : sous le fond du verre, ou traversant la paroi laterale sous le bord.
    sous = (rr < R_EXT) & (pos[:, 1] < y0)
    lateral = (rr > R_EXT) & (pos[:, 1] < y0 + H) & (pos[:, 1] > y0)
    dedans = (rr < R_EXT - T) & (pos[:, 1] >= y0 + T) & (pos[:, 1] <= y0 + H)
    print(f"  R={R:>3}  dx={dx * 1000:5.2f} mm  paroi={T / dx:4.2f} dx  "
          f"cellules solides={int((sdf < 0).sum()):>6}")
    print(f"     {n:>6} particules : dedans {int(dedans.sum()):>6}   "
          f"SOUS le fond {int(sous.sum()):>6}   HORS paroi laterale {int(lateral.sum()):>6}")
    return int(sous.sum()) + int(lateral.sum()), n


print(f"domaine {DOM * 100:.0f} cm, verre cylindrique D={2 * R_EXT * 100:.0f} cm "
      f"h={H * 100:.0f} cm, paroi {T * 1000:.0f} mm, jet 1 m/s, 96 frames\n")
print("SANS collider (reference) :")
run(64, False)
print("\nAVEC le verre :")
for R in (48, 64, 96):
    run(R, True)
