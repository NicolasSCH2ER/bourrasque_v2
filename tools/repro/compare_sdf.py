"""Verification 1 : comparaison cellule par cellule du champ de distance
signee, avant (winding number exhaustif) / apres (buckets + propagation),
sur la meme configuration de collider (sphere UV, plusieurs densites)."""
import ctypes
import numpy as np

F = ctypes.POINTER(ctypes.c_float)


class Cfg(ctypes.Structure):
    _fields_ = [("grid_res", ctypes.c_int * 3), ("cell_size", ctypes.c_float),
                ("gravity_y", ctypes.c_float), ("cfl", ctypes.c_float),
                ("ppc_axis", ctypes.c_int), ("max_particles", ctypes.c_int)]


class Mat(ctypes.Structure):
    _fields_ = [("model", ctypes.c_int), ("rho", ctypes.c_float),
                ("E", ctypes.c_float), ("nu", ctypes.c_float),
                ("bulk", ctypes.c_float), ("gamma", ctypes.c_float)]


def load(path):
    d = ctypes.CDLL(path)
    d.bq_abi_version.restype = ctypes.c_int
    d.bq_default_config.argtypes = [ctypes.POINTER(Cfg)]
    d.bq_create.argtypes = [ctypes.POINTER(Cfg)]
    d.bq_create.restype = ctypes.c_void_p
    d.bq_destroy.argtypes = [ctypes.c_void_p]
    d.bq_add_material.argtypes = [ctypes.c_void_p, ctypes.POINTER(Mat)]
    d.bq_add_material.restype = ctypes.c_int
    d.bq_set_colliders.argtypes = [ctypes.c_void_p, F, F, F, ctypes.c_int]
    d.bq_set_colliders.restype = ctypes.c_int
    d.bq_debug_read_sdf.argtypes = [ctypes.c_void_p, F]
    d.bq_debug_read_sdf.restype = ctypes.c_int
    d.bq_last_error.restype = ctypes.c_char_p
    return d


def sphere(n_sub, centre=(0.5, 0.5, 0.5), r=0.2):
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


def run(d, n_sub):
    cfg = Cfg()
    d.bq_default_config(ctypes.byref(cfg))
    sim = d.bq_create(ctypes.byref(cfg))
    if not sim:
        raise SystemExit("bq_create a echoue")
    mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
    if d.bq_add_material(sim, ctypes.byref(mat)) < 0:
        raise SystemExit(d.bq_last_error().decode())

    tri = sphere(n_sub)
    n_tri = tri.shape[0]
    vel = np.zeros_like(tri)
    fric = np.full(n_tri, 0.2, dtype=np.float32)
    args = (sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
            fric.ctypes.data_as(F), n_tri)
    if d.bq_set_colliders(*args) != 0:
        raise SystemExit(d.bq_last_error().decode())

    ncell = cfg.grid_res[0] * cfg.grid_res[1] * cfg.grid_res[2]
    buf = np.zeros(ncell, dtype=np.float32)
    got = d.bq_debug_read_sdf(sim, buf.ctypes.data_as(F))
    assert got == ncell
    d.bq_destroy(sim)
    return buf.copy(), tuple(cfg.grid_res), n_tri


BEFORE = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\bourrasque_before.dll"
AFTER = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\bourrasque_after.dll"

d_before = load(BEFORE)
d_after = load(AFTER)

print("ABI avant:", d_before.bq_abi_version(), " apres:", d_after.bq_abi_version())

for n_sub in (50, 100, 158):
    sdf_b, res, n_tri = run(d_before, n_sub)
    sdf_a, res2, n_tri2 = run(d_after, n_sub)
    assert res == res2 and n_tri == n_tri2

    diff = np.abs(sdf_a - sdf_b)
    # cellules "grande distance" (hors bande) : 1e6 des deux cotes, ignorees
    # de la comparaison de magnitude (juste verifier egalite exacte)
    far_a = sdf_a >= 1e6
    far_b = sdf_b >= 1e6
    assert np.array_equal(far_a, far_b), "desaccord sur les cellules hors bande"

    near = ~far_a
    dist_diff = diff[near]
    max_diff = dist_diff.max() if dist_diff.size else 0.0
    mean_diff = dist_diff.mean() if dist_diff.size else 0.0

    sign_b = np.sign(sdf_b[near])
    sign_a = np.sign(sdf_a[near])
    sign_mismatch = np.sum(sign_a != sign_b)

    print(f"\n--- n_tri={n_tri} (n_sub={n_sub}) grid={res} ---")
    print(f"cellules dans la bande : {near.sum()} / {sdf_a.size}")
    print(f"ecart max |d_apres - d_avant| : {max_diff:.6g}")
    print(f"ecart moyen : {mean_diff:.6g}")
    print(f"cellules avec signe different : {sign_mismatch} / {near.sum()}")
    if sign_mismatch > 0:
        idx = np.where((sign_a != sign_b) & near)[0]
        print(f"  exemples (jusqu'a 10) : sdf_avant={sdf_b[idx[:10]]}  sdf_apres={sdf_a[idx[:10]]}")
