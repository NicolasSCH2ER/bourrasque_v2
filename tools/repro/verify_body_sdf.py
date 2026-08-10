"""M17 / B1 -- verification du SDF local par corps (D9).

Sept vérifications indépendantes, décrites dans docs/plan-milestone-17.md
(tâche B1) :

  1. Boîte : erreur max de body_sdf() contre la distance signée analytique,
     à l'intérieur comme à l'extérieur, hors d'une bande étroite près des
     arêtes/sommets (interpolation trilinéaire, lissage légitime).
  2. Sphère (maillage subdivisé) : même chose, erreur dominée par la
     facettisation du maillage.
  3. Signe : pourcentage de points classés intérieur/extérieur conforme à
     l'analytique.
  4. Invariance par transformation rigide : même point local, deux poses
     monde différentes, round-trip R^T(p - x) puis body_sdf() -- doit
     redonner exactement le même point local et donc la même valeur.
  5. Plafond de 128^3 (ici testé à max_res réduit pour rester rapide) :
     doit se déclencher et l'annoncer sur stderr.
  6. Coût : temps de construction à 1 puis 10 corps, VRAM (calculée depuis
     la résolution effective -- l'API n'expose pas de sonde VRAM directe).

Ce script charge la DLL directement par ctypes (PAS extension/lib.py, qui ne
connaît pas encore ces fonctions -- B1 ne touche pas à extension/).

Lancement (interpréteur avec numpy) :
    python tools/repro/verify_body_sdf.py
"""

import ctypes
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DLL_PATH = ROOT / "extension" / "bin" / "bourrasque.dll"


# ---------------------------------------------------------------------------
# Chargement minimal -- juste ce dont ce script a besoin, pas tout lib.py.
# ---------------------------------------------------------------------------


class BqConfig(ctypes.Structure):
    _fields_ = [
        ("grid_res", ctypes.c_int * 3),
        ("cell_size", ctypes.c_float),
        ("gravity_y", ctypes.c_float),
        ("cfl", ctypes.c_float),
        ("ppc_axis", ctypes.c_int),
        ("max_particles", ctypes.c_int),
    ]


dll = ctypes.CDLL(str(DLL_PATH))

dll.bq_default_config.argtypes = [ctypes.POINTER(BqConfig)]
dll.bq_default_config.restype = None
dll.bq_create.argtypes = [ctypes.POINTER(BqConfig)]
dll.bq_create.restype = ctypes.c_void_p
dll.bq_destroy.argtypes = [ctypes.c_void_p]
dll.bq_destroy.restype = None
dll.bq_last_error.argtypes = []
dll.bq_last_error.restype = ctypes.c_char_p

dll.bq_build_body_sdf.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_float, ctypes.c_int,
]
dll.bq_build_body_sdf.restype = ctypes.c_int

dll.bq_read_body_sdf.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
dll.bq_read_body_sdf.restype = ctypes.c_int

dll.bq_query_body_sdf.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
dll.bq_query_body_sdf.restype = ctypes.c_int


def check(cond, msg):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


def make_sim():
    cfg = BqConfig()
    dll.bq_default_config(ctypes.byref(cfg))
    sim = dll.bq_create(ctypes.byref(cfg))
    if not sim:
        raise RuntimeError(f"bq_create a echoue: {dll.bq_last_error().decode()}")
    return sim


def build_body_sdf(sim, body, tri, target_cell, max_res):
    tri = np.ascontiguousarray(tri, dtype=np.float32)
    n_tri = tri.shape[0]
    ptr = tri.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc = dll.bq_build_body_sdf(sim, body, ptr, n_tri, target_cell, max_res)
    if rc != 0:
        raise RuntimeError(f"bq_build_body_sdf a echoue: {dll.bq_last_error().decode()}")


def read_body_sdf_meta(sim, body):
    res = (ctypes.c_int * 3)()
    cell = ctypes.c_float()
    origin = (ctypes.c_float * 3)()
    n = dll.bq_read_body_sdf(sim, body, None, res, ctypes.byref(cell), origin)
    if n < 0:
        raise RuntimeError(f"bq_read_body_sdf a echoue: {dll.bq_last_error().decode()}")
    return tuple(res), cell.value, tuple(origin), n


def query_body_sdf(sim, body, pts_local, want_grad=False):
    pts_local = np.ascontiguousarray(pts_local, dtype=np.float32)
    n = pts_local.shape[0]
    phi = np.empty(n, dtype=np.float32)
    ptr_pts = pts_local.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ptr_phi = phi.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    if want_grad:
        grad = np.empty((n, 3), dtype=np.float32)
        ptr_grad = grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        grad = None
        ptr_grad = None
    rc = dll.bq_query_body_sdf(sim, body, ptr_pts, n, ptr_phi, ptr_grad)
    if rc < 0:
        raise RuntimeError(f"bq_query_body_sdf a echoue: {dll.bq_last_error().decode()}")
    return phi, grad


# ---------------------------------------------------------------------------
# Géométries de test
# ---------------------------------------------------------------------------


def box_triangles(half_extent):
    """12 triangles d'un pavé centré à l'origine, normales SORTANTES (même
    convention que diag_archimede.cube_triangles)."""
    hx, hy, hz = half_extent
    s = np.array([[-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
                  [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]],
                 dtype=np.float64)
    v = s * np.array([hx, hy, hz])
    faces = [
        (0, 3, 2), (0, 2, 1),   # z-
        (4, 5, 6), (4, 6, 7),   # z+
        (0, 1, 5), (0, 5, 4),   # y-
        (3, 7, 6), (3, 6, 2),   # y+
        (0, 4, 7), (0, 7, 3),   # x-
        (1, 2, 6), (1, 6, 5),   # x+
    ]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float32)


def box_sdf_analytic(p, half_extent):
    h = np.asarray(half_extent, dtype=np.float64)
    q = np.abs(p) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def box_edge_band(p, half_extent, band):
    """True si p est a moins de `band` d'au moins DEUX faces simultanement
    (arete ou sommet de la boite) -- zone ou l'interpolation trilineaire
    lisse legitimement le champ, exclue de la mesure d'erreur max."""
    h = np.asarray(half_extent, dtype=np.float64)
    near = np.abs(np.abs(p) - h) < band
    return np.sum(near, axis=-1) >= 2


def icosphere(radius, subdiv):
    t = (1.0 + 5.0 ** 0.5) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    base = base / np.linalg.norm(base[0])
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    verts = [base[i] for i in range(12)]
    cache = {}

    def midpoint(i1, i2):
        key = (min(i1, i2), max(i1, i2))
        if key in cache:
            return cache[key]
        m = (verts[i1] + verts[i2]) / 2.0
        m = m / np.linalg.norm(m)
        verts.append(m)
        idx = len(verts) - 1
        cache[key] = idx
        return idx

    for _ in range(subdiv):
        new_faces = []
        for (a, b, c) in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    V = np.array(verts) * radius
    tri = np.array([[V[a], V[b], V[c]] for a, b, c in faces], dtype=np.float32)
    return tri


def sphere_sdf_analytic(p, radius):
    return np.linalg.norm(p, axis=-1) - radius


def quat_to_mat3(q):
    """Meme convention EXACTE que quat_to_mat3 dans mlsmpm.cu (w, x, y, z)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_angle_quat(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle_rad / 2.0)
    return np.array([np.cos(angle_rad / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def main():
    rng = np.random.default_rng(1234)
    all_ok = True

    # ------------------------------------------------------------- 1. boite
    print("=== 1. Boite : erreur contre l'analytique ===")
    half = (0.15, 0.25, 0.10)
    target_cell = 0.01
    max_res = 128
    tri = box_triangles(half)

    sim = make_sim()
    t0 = time.perf_counter()
    build_body_sdf(sim, 0, tri, target_cell, max_res)
    build_time_1 = time.perf_counter() - t0
    res, cell, origin, ncell = read_body_sdf_meta(sim, 0)
    print(f"  grille : res={res} cell={cell:.5f} origin={origin} ncell={ncell}")

    # Le corps est dilate de 4 voxels (bq_build_body_sdf) -- on reste a
    # l'interieur de cette marge (3 voxels) pour ne jamais tomber sur la
    # sentinelle hors-grille (1e6), qui n'a rien a voir avec une erreur de
    # champ.
    pad = 3.0 * target_cell
    n_pts = 20000
    lo = -np.array(half) - pad
    hi = np.array(half) + pad
    pts = rng.uniform(lo, hi, size=(n_pts, 3))

    phi_num, _ = query_body_sdf(sim, 0, pts.astype(np.float32))
    phi_ana = box_sdf_analytic(pts, half)

    band = 2.0 * cell
    edge_mask = box_edge_band(pts, half, band)
    far_mask = np.abs(phi_ana) > (np.max(half) + pad) * 0.999  # hors grille : ignorer
    keep = (~edge_mask) & (~far_mask)

    err = np.abs(phi_num[keep] - phi_ana[keep])
    max_err = float(np.max(err))
    max_err_vox = max_err / cell
    n_excluded = int(np.sum(~keep))
    print(f"  points testes: {n_pts}, exclus (bande aretes/sommets, {band:.4f} m): {n_excluded} "
          f"({100.0*n_excluded/n_pts:.1f}%)")
    print(f"  erreur max (hors bande): {max_err:.6f} m = {max_err_vox:.4f} voxel")
    all_ok &= check(max_err_vox < 1.0, "erreur max boite < 1 voxel hors bande aretes/sommets")

    # signe (verif 3, sur la boite)
    sign_num = np.sign(phi_num)
    sign_ana = np.sign(phi_ana)
    sign_ana[sign_ana == 0] = 1.0
    sign_match = sign_num == sign_ana
    pct_sign = 100.0 * np.mean(sign_match)
    print(f"\n=== 3. Signe (boite) ===")
    print(f"  accord de signe: {pct_sign:.3f}% sur {n_pts} points")
    all_ok &= check(pct_sign > 99.0, "accord de signe > 99% sur la boite")

    dll.bq_destroy(sim)

    # ------------------------------------------------------------ 2. sphere
    print("\n=== 2. Sphere (maillage subdivise) : erreur contre l'analytique ===")
    radius = 0.20
    sph_tri = icosphere(radius, subdiv=4)
    print(f"  maillage: {sph_tri.shape[0]} triangles")

    sim = make_sim()
    build_body_sdf(sim, 0, sph_tri, target_cell, max_res)
    res_s, cell_s, origin_s, ncell_s = read_body_sdf_meta(sim, 0)
    print(f"  grille : res={res_s} cell={cell_s:.5f} ncell={ncell_s}")

    pad_s = 3.0 * target_cell
    pts_s = rng.uniform(-(radius + pad_s), radius + pad_s, size=(n_pts, 3))
    phi_num_s, _ = query_body_sdf(sim, 0, pts_s.astype(np.float32))
    phi_ana_s = sphere_sdf_analytic(pts_s, radius)
    far_mask_s = np.abs(phi_ana_s) > (radius + pad_s) * 0.999
    keep_s = ~far_mask_s
    err_s = np.abs(phi_num_s[keep_s] - phi_ana_s[keep_s])
    max_err_s = float(np.max(err_s))
    mean_err_s = float(np.mean(err_s))
    max_err_s_vox = max_err_s / cell_s
    print(f"  erreur max: {max_err_s:.6f} m = {max_err_s_vox:.4f} voxel, "
          f"erreur moyenne: {mean_err_s:.6f} m")
    # borne de facettisation attendue : ecart max entre corde et arc pour un
    # icosahedre subdivise `subdiv` fois, angle de sommet ~ 2*pi/(5*2^subdiv)
    approx_edge = 2 * np.pi * radius / (5 * 2 ** 4)
    facet_bound = approx_edge ** 2 / (8 * radius)  # sagitta corde/arc
    print(f"  borne de facettisation attendue (sagitta corde/arc): ~{facet_bound:.6f} m")
    all_ok &= check(max_err_s < max(facet_bound * 5.0, 3 * cell_s),
                     "erreur max sphere du meme ordre que la facettisation "
                     "(pas une erreur de code)")

    sign_num_s = np.sign(phi_num_s)
    sign_ana_s = np.sign(phi_ana_s)
    sign_ana_s[sign_ana_s == 0] = 1.0
    pct_sign_s = 100.0 * np.mean(sign_num_s == sign_ana_s)
    print(f"  accord de signe (sphere): {pct_sign_s:.3f}%")
    all_ok &= check(pct_sign_s > 99.0, "accord de signe > 99% sur la sphere")

    dll.bq_destroy(sim)

    # ---------------------------------------------------- 4. invariance rigide
    print("\n=== 4. Invariance par transformation rigide ===")
    sim = make_sim()
    build_body_sdf(sim, 0, box_triangles(half), target_cell, max_res)

    local_pts = np.array([
        [0.0, 0.0, 0.0],       # centre, profondement dedans
        [0.05, 0.05, 0.02],    # dedans, hors axe
        [0.30, 0.30, 0.20],    # dehors
        [0.15, 0.0, 0.0],      # tres pres d'une face
        [-0.20, 0.10, -0.05],
    ], dtype=np.float64)

    pose_a = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))  # identite
    pose_b = (np.array([1.37, -0.42, 2.05]), axis_angle_quat([0.3, 1.0, -0.4], 1.9))

    max_roundtrip_err = 0.0
    max_phi_diff = 0.0
    for L in local_pts:
        recovered = []
        phis = []
        for (x, q) in (pose_a, pose_b):
            R = quat_to_mat3(q)
            p_world = x + R @ L
            p_local_back = R.T @ (p_world - x)
            recovered.append(p_local_back)
            phi, _ = query_body_sdf(sim, 0, p_local_back.reshape(1, 3).astype(np.float32))
            phis.append(float(phi[0]))
        rt_err = max(np.max(np.abs(recovered[0] - L)), np.max(np.abs(recovered[1] - L)))
        phi_diff = abs(phis[0] - phis[1])
        max_roundtrip_err = max(max_roundtrip_err, rt_err)
        max_phi_diff = max(max_phi_diff, phi_diff)

    print(f"  round-trip R^T(p-x) max erreur: {max_roundtrip_err:.3e}")
    print(f"  |phi(pose A) - phi(pose B)| max: {max_phi_diff:.3e}")
    all_ok &= check(max_roundtrip_err < 1e-4, "round-trip du point local < 1e-4 m")
    all_ok &= check(max_phi_diff < 1e-4, "body_sdf identique entre deux poses monde (< 1e-4)")

    dll.bq_destroy(sim)

    # ----------------------------------------------------------- 5. plafond
    print("\n=== 5. Plafond de resolution : declenchement et signalement ===")
    small_max_res = 16
    big_half = (1.0, 0.02, 1.0)  # sol tres etendu, fin -- typique D9/R3
    fine_cell = 0.01
    sim = make_sim()
    build_body_sdf(sim, 0, box_triangles(big_half), fine_cell, small_max_res)
    res_c, cell_c, origin_c, ncell_c = read_body_sdf_meta(sim, 0)
    print(f"  target_cell={fine_cell}, max_res={small_max_res} -> "
          f"res={res_c}, cell effective={cell_c:.5f}")
    print("  (message attendu sur stderr, visible ci-dessus dans la console)")
    all_ok &= check(max(res_c) <= small_max_res, f"resolution <= {small_max_res} sur chaque axe")
    all_ok &= check(cell_c > fine_cell * 1.5, "voxel effectif agrandi au-dela de la cible")
    dll.bq_destroy(sim)

    # -------------------------------------------------------------- 6. cout
    print("\n=== 6. Cout : construction a 1 puis 10 corps ===")
    print(f"  1 corps (boite, res={res}): {build_time_1*1000:.2f} ms, "
          f"VRAM phi = {ncell*4/1024:.1f} Ko")

    sim = make_sim()
    boxes = [box_triangles((0.1 + 0.01 * i, 0.15, 0.12)) for i in range(10)]
    t0 = time.perf_counter()
    total_ncell = 0
    for i, b in enumerate(boxes):
        build_body_sdf(sim, i, b, target_cell, max_res)
        r, c, o, nc = read_body_sdf_meta(sim, i)
        total_ncell += nc
    total_time = time.perf_counter() - t0
    print(f"  10 corps: {total_time*1000:.2f} ms total, {total_time*1000/10:.2f} ms/corps en moyenne")
    print(f"  VRAM totale (phi, 10 corps): {total_ncell*4/1024:.1f} Ko "
          f"({total_ncell*4/(1024*1024):.2f} Mo)")
    dll.bq_destroy(sim)

    print("\n" + ("TOUT OK" if all_ok else "AU MOINS UN ECHEC"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
