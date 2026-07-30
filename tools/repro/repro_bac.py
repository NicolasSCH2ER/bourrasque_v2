"""Ecart 1 de la revue : un collider dont l'AABB couvre tout le domaine.

Si aucune cellule n'est amorcee comme exterieure, la propagation ne demarre
pas et TOUTE la grille est signee negative -> tout le domaine est declare
solide et le fluide gele. Cas d'usage vise : un bac / conteneur dimensionne
sur le domaine.
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
d.bq_read_sdf.argtypes = [ctypes.c_void_p, F]
d.bq_read_sdf.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p


def box_tris(lo, hi):
    """Les 12 triangles d'une boite axis-aligned, normales sortantes."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
             (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float32)


cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
d.bq_add_material(sim, ctypes.byref(mat))

R = cfg.grid_res[0]
ncell = R * R * R
L = R * cfg.cell_size
sdf = np.empty(ncell, dtype=np.float32)

def sphere_tris(centre, r, n_sub=40):
    """Sphere UV triangulee, normales sortantes."""
    u = np.linspace(0, np.pi, n_sub + 1)
    v = np.linspace(0, 2 * np.pi, n_sub + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    P = np.stack([centre[0] + r * np.sin(uu) * np.cos(vv),
                  centre[1] + r * np.sin(uu) * np.sin(vv),
                  centre[2] + r * np.cos(uu)], -1)
    t = []
    for i in range(n_sub):
        for j in range(n_sub):
            a, b, c, e = P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]
            t.append([a, b, c])
            t.append([a, c, e])
    return np.array(t, dtype=np.float32)


# Le cas discriminant : une SPHERE dont l'AABB couvre tout le domaine.
# Son volume ne remplit pas le domaine — les 8 coins sont dehors. Le rapport
# exact volume/domaine vaut pi/6 = 52.4 %, donc ~47.6 % des cellules doivent
# rester fluides. Si la propagation ne s'amorce pas, on lira 100 %.
CENTRE = (L / 2, L / 2, L / 2)
for label, tri in (
    ("sphere r=L/4, AABB au centre du domaine", sphere_tris(CENTRE, L / 4)),
    ("SPHERE r=L/2 : AABB = domaine entier", sphere_tris(CENTRE, L / 2)),
):
    n = tri.shape[0]
    vel = np.zeros_like(tri)
    fric = np.full(n, 0.2, dtype=np.float32)
    if d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                          fric.ctypes.data_as(F), n) != 0:
        raise SystemExit(d.bq_last_error().decode())
    d.bq_read_sdf(sim, sdf.ctypes.data_as(F))
    neg = int((sdf < 0).sum())
    print(f"{label}")
    print(f"   cellules signees 'solide' : {neg:>7} / {ncell}  ({neg/ncell*100:5.1f} %)")

print("\n-> un bac creux devrait laisser son INTERIEUR non solide.")
print("   100 % de cellules solides = la propagation n'a jamais demarre.")
d.bq_destroy(sim)
