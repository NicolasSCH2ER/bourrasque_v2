"""Simulation de la scene reelle, avec la geometrie exacte du GLB.

Disposition reprise du frontend (_domain_layout) : dx = max(extent monde) / R,
res = ceil(extent / dx) + 2*3 cellules de bord, et la boite de l'artiste est
placee a 3*dx du bord de la grille. Le GLB etant deja Y-up, la correspondance
d'axes vers le solveur est l'identite.
"""
import ctypes
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\nicol\AppData\Local\Temp\claude"
                   r"\C--Users-nicol-Code-bourrasque-v2"
                   r"\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad")
import glb

GLB = r"C:\Users\nicol\Code\bourrasque_v2\Bourrasque_test_verre.glb"
DLL = r"C:\Users\nicol\Code\bourrasque_v2\extension\bin\bourrasque.dll"
BOUND = 3

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

objs = dict(glb.objects(GLB))
dom = objs["domaine"].reshape(-1, 3)
DLO, DHI = dom.min(0), dom.max(0)
verre = objs["verre"]
emit = objs["emitter"].reshape(-1, 3)
ELO, EHI = emit.min(0), emit.max(0)

# Geometrie du verre, en coordonnees monde : cavite et fond.
VW = verre.reshape(-1, 3)
CX, CZ = VW[:, 0].mean(), VW[:, 2].mean()
R_EXT = np.hypot(VW[:, 0] - CX, VW[:, 2] - CZ).max()
Y_BAS, Y_HAUT = VW[:, 1].min(), VW[:, 1].max()
print(f"verre : R_ext={R_EXT * 100:.2f} cm, y {Y_BAS * 100:.2f}..{Y_HAUT * 100:.2f} cm, "
      f"{len(verre)} triangles")
print(f"domaine monde : {np.round(DLO, 4)} .. {np.round(DHI, 4)}")
print(f"emetteur      : {np.round(ELO, 4)} .. {np.round(EHI, 4)}\n")


def run(R_user, n_frame=96, jet_v=0.0, poser_au_sol=True, trace=False):
    ext = DHI - DLO
    dx = float(ext.max()) / R_user
    res = [int(math.ceil(ext[i] / dx)) + 2 * BOUND for i in range(3)]
    cfg = Cfg()
    d.bq_default_config(ctypes.byref(cfg))
    for i in range(3):
        cfg.grid_res[i] = res[i]
    cfg.cell_size = dx
    cfg.gravity_y = -9.8
    sim = d.bq_create(ctypes.byref(cfg))
    if not sim:
        raise RuntimeError(f"echec coeur : {d.bq_last_error().decode()!r}")
    mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
    m = d.bq_add_material(sim, ctypes.byref(mat))

    def to_solver(p):
        return (p - DLO) + BOUND * dx

    # Poser le verre sur le fond du domaine : sans cet espace de 1.3 cm, une
    # particule sous le fond du verre est forcement passee AU TRAVERS, alors
    # qu'autrement elle peut y avoir coule apres avoir debordé.
    v_geo = verre.copy()
    dy = (DLO[1] - Y_BAS) if poser_au_sol else 0.0
    v_geo[:, :, 1] += dy
    tri = np.ascontiguousarray(to_solver(v_geo), dtype=np.float32)
    n_t = tri.shape[0]
    vel = np.zeros_like(tri)
    fric = np.full(n_t, 0.2, dtype=np.float32)
    if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                          fric.ctypes.data_as(F), n_t) != 0:
        raise RuntimeError(f"echec coeur : {d.bq_last_error().decode()!r}")

    # Emission continue depuis le volume de l'emetteur, pas de dx/2.
    sp = dx / 2.0
    axes = [np.arange(ELO[i] + sp / 2, EHI[i], sp) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    site = to_solver(np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1))
    site = np.ascontiguousarray(site, dtype=np.float32)
    jv = np.tile(np.array([0.0, jet_v, 0.0], dtype=np.float32), (len(site), 1))
    jv = np.ascontiguousarray(jv)

    y_bas = Y_BAS + dy
    y_haut = Y_HAUT + dy
    buf = np.empty((cfg.max_particles, 3), dtype=np.float32)

    def etat():
        n = d.bq_particle_count(sim)
        d.bq_read_positions(sim, buf.ctypes.data_as(F))
        w = buf[:n].astype(np.float64) - BOUND * dx + DLO
        rr = np.hypot(w[:, 0] - CX, w[:, 2] - CZ)
        cav = (rr < R_EXT) & (w[:, 1] > y_bas) & (w[:, 1] < y_haut)
        sous = (rr < R_EXT) & (w[:, 1] <= y_bas)
        niveau = np.percentile(w[cav, 1], 95) if cav.sum() > 20 else y_bas
        return n, int(cav.sum()), int(sous.sum()), niveau

    for fr in range(n_frame):
        d.bq_emit_points_vel(sim, m, site.ctypes.data_as(F),
                             jv.ctypes.data_as(F), len(site))
        if d.bq_step(sim, 1.0 / 24.0) < 0:   # renvoie le nombre de substeps
            raise RuntimeError(f"echec coeur : {d.bq_last_error().decode()!r}")
        if trace and (fr + 1) % 16 == 0:
            n, cav, sous, niv = etat()
            print(f"     frame {fr + 1:>3} : {n:>7} particules   cavite {cav:>7}"
                  f"   SOUS le fond {sous:>7}   niveau {(niv - y_bas) * 100:6.2f} cm")

    n = d.bq_particle_count(sim)
    pos = np.empty((n, 3), dtype=np.float32)
    d.bq_read_positions(sim, pos.ctypes.data_as(F))
    ncell = res[0] * res[1] * res[2]
    sdf = np.empty(ncell, dtype=np.float32)
    d.bq_read_sdf(sim, sdf.ctypes.data_as(F))
    d.bq_destroy(sim)

    # Particules situees dans une cellule SOLIDE : elles n'ont rien a y faire.
    # Le champ est echantillonne AUX NOEUDS (i*dx) : noeud le plus proche.
    ci = np.clip(np.floor(pos[:, 0] / dx + 0.5).astype(int), 0, res[0] - 1)
    cj = np.clip(np.floor(pos[:, 1] / dx + 0.5).astype(int), 0, res[1] - 1)
    ck = np.clip(np.floor(pos[:, 2] / dx + 0.5).astype(int), 0, res[2] - 1)
    dans_solide = int((sdf[(ci * res[1] + cj) * res[2] + ck] < 0).sum())

    w = pos.astype(np.float64) - BOUND * dx + DLO
    rr = np.hypot(w[:, 0] - CX, w[:, 2] - CZ)
    cav = (rr < R_EXT) & (w[:, 1] > y_bas) & (w[:, 1] < y_haut)
    sous = (rr < R_EXT) & (w[:, 1] <= y_bas)
    niveau = np.percentile(w[cav, 1], 95) if cav.sum() > 20 else y_bas
    print(f"  R={R_user:>3} grille {res} dx={dx * 1000:5.2f} mm  "
          f"paroi~{2.07 / (dx * 1000):.2f} dx  cellules solides={int((sdf < 0).sum()):>6}")
    print(f"     {n:>7} particules : cavite {int(cav.sum()):>7}"
          f"   SOUS le fond {int(sous.sum()):>7}"
          f"   dans le solide {dans_solide:>6}"
          f"   niveau {(niveau - y_bas) * 100:6.2f} cm")
    return int(sous.sum())


print("verre POSE sur le fond du domaine : toute particule sous son fond a")
print("necessairement traverse la paroi.\n")
for R in (48, 64, 96):
    run(R, trace=True)
    print()
