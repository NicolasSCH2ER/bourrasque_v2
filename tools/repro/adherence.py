"""Le contact retient-il le fluide qui S'ELOIGNE d'un collider ?

Reserve P2 de la revue : la corroboration ne serait vraie que pour les noeuds a
moins de ~dx de la surface, donc les noeuds situes entre dx et 1.5 dx seraient
traites en contact BIDIRECTIONNEL, y compris autour d'un collider epais. Le
fluide y serait retenu en s'eloignant -- une membrane collee, artefact de forme
que le maillage rendrait visible.

Protocole : un mur collider epais, une nappe de particules placee a une distance
donnee de sa surface, une vitesse initiale qui S'EN ELOIGNE, gravite nulle. En
contact purement separant, la nappe part librement. Si elle est retenue, le
deplacement s'effondre. On balaie la distance initiale de 0.25 a 3 dx.
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
    ("bq_emit_points_vel", [ctypes.c_void_p, ctypes.c_int, F, F, ctypes.c_int], ctypes.c_int),
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


def box_tris(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in _FACES], dtype=np.float32)


cfg = Cfg()
d.bq_default_config(cfg := Cfg()) if False else d.bq_default_config(ctypes.byref(cfg))
cfg.gravity_y = 0.0                      # on isole le contact
R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx
N_FRAME = 12
DT = 1.0 / 24.0
# Mur epais occupant x < X_MUR (5 cellules d'epaisseur), decale de dx/2 pour ne
# pas faire coincider sa face avec un noeud.
X_MUR = 12.5 * dx


def run(dist, avec_mur, V0=1.0):
    sim = d.bq_create(ctypes.byref(cfg))
    mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
    m = d.bq_add_material(sim, ctypes.byref(mat))
    if avec_mur:
        tri = box_tris((X_MUR - 5 * dx, 0.1 * L, 0.1 * L),
                       (X_MUR, 0.9 * L, 0.9 * L))
        n_t = tri.shape[0]
        vel = np.zeros_like(tri)
        fric = np.full(n_t, 0.2, dtype=np.float32)
        if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                              fric.ctypes.data_as(F), n_t) != 0:
            raise RuntimeError(d.bq_last_error().decode())

    # Nappe mince (un seul plan de particules) a `dist` de la face du mur.
    x0 = X_MUR + dist
    g = np.arange(0.35 * L, 0.65 * L, dx / 2.0)
    yy, zz = np.meshgrid(g, g, indexing="ij")
    site = np.stack([np.full(yy.size, x0), yy.ravel(), zz.ravel()], -1)
    site = np.ascontiguousarray(site, dtype=np.float32)
    vel = np.tile(np.array([V0, 0.0, 0.0], dtype=np.float32), (len(site), 1))
    n = d.bq_emit_points_vel(sim, m, site.ctypes.data_as(F),
                             np.ascontiguousarray(vel).ctypes.data_as(F), len(site))
    for _ in range(N_FRAME):
        if d.bq_step(sim, DT) < 0:
            raise RuntimeError(d.bq_last_error().decode())
    pos = np.empty((d.bq_particle_count(sim), 3), dtype=np.float32)
    d.bq_read_positions(sim, pos.ctypes.data_as(F))
    d.bq_destroy(sim)
    return float(pos[:, 0].mean()) - x0, n


print(f"dx = {dx * 1000:.2f} mm, gravite nulle, bande de contact = 1.5 dx = "
      f"{1.5 * dx * 1000:.1f} mm")
print("Une vitesse FAIBLE est le cas le plus defavorable : la particule reste")
print("plus longtemps dans la bande, donc subit la contrainte sur plus de substeps.\n")
print("=== CONTROLE POSITIF : meme protocole, mais vitesse VERS le mur.")
print("    Le contact doit bloquer. Sans cela, le test ci-dessous ne prouve rien.")
print(f"{'dist au mur':>12} {'sans mur':>12} {'avec mur':>12} {'blocage':>9}")
for k in (0.25, 0.75, 1.25, 2.0):
    dist = k * dx
    d_sans, _ = run(dist, False, -1.0)
    d_avec, _ = run(dist, True, -1.0)
    frein = 100.0 * (1.0 - d_avec / d_sans) if abs(d_sans) > 1e-9 else float("nan")
    print(f"{k:>9.2f} dx {d_sans * 1000:>10.2f} mm {d_avec * 1000:>10.2f} mm "
          f"{frein:>8.1f} %")
print()

for V0 in (0.05, 0.2, 1.0):
    print(f"--- v0 = {V0} m/s   (deplacement libre = {V0 * N_FRAME * DT * 1000:.1f} mm)")
    print(f"{'dist au mur':>12} {'sans mur':>12} {'avec mur':>12} {'retenue':>9}")
    for k in (0.25, 0.75, 1.0, 1.25, 1.5, 2.0):
        dist = k * dx
        d_sans, _ = run(dist, False, V0)
        d_avec, n = run(dist, True, V0)
        frein = 100.0 * (1.0 - d_avec / d_sans) if abs(d_sans) > 1e-9 else float("nan")
        print(f"{k:>9.2f} dx {d_sans * 1000:>10.2f} mm {d_avec * 1000:>10.2f} mm "
              f"{frein:>8.1f} %")
    print()
