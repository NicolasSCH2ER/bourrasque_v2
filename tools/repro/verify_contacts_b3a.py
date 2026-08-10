"""M17 / B3a -- verification de la detection de contact corps<->corps
(broadphase + generation, DETECTION SEULEMENT, cf. docs/plan-milestone-17.md
D10, taches B3a).

Charge la DLL directement par ctypes (comme verify_body_sdf.py de B1) --
n'utilise PAS extension/lib.py.

Sections :
  0. Validation du GRADIENT de body_sdf() (jamais verifiee par B1) : erreur
     angulaire max en degres, boite (faces plates vs aretes separement) et
     sphere.
  1. Deux boites qui se recouvrent d'une profondeur connue : contact, depth,
     normale.
  2. Deux boites separees : zero contact.
  3. Paire statique/statique : zero contact, ecartee des la broadphase.
  4. Caisse posee sur un sol (boite statique) : contacts sous la caisse
     seulement ; normale +y (vers le haut, convention solveur Y-up) pour les
     contacts testes contre le SDF du sol, -y (sortante de la caisse) pour
     les contacts testes contre le SDF de la caisse (sens inverse, D10) --
     la normale suit toujours le corps B interroge, pas une convention
     "haut/bas" absolue.
  5. Saturation du tampon de contacts : rapportee via bq_contacts_last_overflow.
  6. Cout par sous-pas a 2 puis 10 corps (ms).

Lancement (interpreteur avec numpy) :
    python tools/repro/verify_contacts_b3a.py
"""

import ctypes
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DLL_PATH = ROOT / "extension" / "bin" / "bourrasque.dll"
sys.path.insert(0, str(ROOT / "extension"))
import rigidbody  # noqa: E402  module pur, pas de bpy


# ---------------------------------------------------------------------------
# Chargement DLL (surface minimale necessaire a ce script)
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


class BqMaterial(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),
        ("rho", ctypes.c_float),
        ("E", ctypes.c_float),
        ("nu", ctypes.c_float),
        ("bulk", ctypes.c_float),
        ("gamma", ctypes.c_float),
        ("friction_angle", ctypes.c_float),
        ("cohesion", ctypes.c_float),
    ]


BQ_MODEL_WATER = 1


class BqRigidBody(ctypes.Structure):
    _fields_ = [
        ("dynamic", ctypes.c_int),
        ("mass", ctypes.c_float),
        ("inv_inertia", ctypes.c_float * 9),
        ("x", ctypes.c_float * 3),
        ("q", ctypes.c_float * 4),
        ("v", ctypes.c_float * 3),
        ("w", ctypes.c_float * 3),
        ("use_gravity", ctypes.c_int),
        ("added_mass", ctypes.c_float),
        ("lock_lin", ctypes.c_int * 3),
        ("lock_ang", ctypes.c_int * 3),
        ("restitution", ctypes.c_float),
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

dll.bq_add_material.argtypes = [ctypes.c_void_p, ctypes.POINTER(BqMaterial)]
dll.bq_add_material.restype = ctypes.c_int
dll.bq_emit_box.argtypes = [ctypes.c_void_p, ctypes.c_int,
                            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                            ctypes.POINTER(ctypes.c_float)]
dll.bq_emit_box.restype = ctypes.c_int
dll.bq_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
dll.bq_step.restype = ctypes.c_int

dll.bq_build_body_sdf.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_float, ctypes.c_int,
]
dll.bq_build_body_sdf.restype = ctypes.c_int

dll.bq_set_body_samples.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
]
dll.bq_set_body_samples.restype = ctypes.c_int

dll.bq_query_body_sdf.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
dll.bq_query_body_sdf.restype = ctypes.c_int

dll.bq_set_collider_bodies.argtypes = [ctypes.c_void_p, ctypes.POINTER(BqRigidBody), ctypes.c_int]
dll.bq_set_collider_bodies.restype = ctypes.c_int

dll.bq_read_contacts.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
dll.bq_read_contacts.restype = ctypes.c_int

dll.bq_contacts_last_overflow.argtypes = [ctypes.c_void_p]
dll.bq_contacts_last_overflow.restype = ctypes.c_int


def check(cond, msg):
    status = "OK" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    return bool(cond)


def make_sim(grid_res=32, cell_size=1.0 / 32.0):
    cfg = BqConfig()
    dll.bq_default_config(ctypes.byref(cfg))
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = grid_res
    cfg.cell_size = cell_size
    sim = dll.bq_create(ctypes.byref(cfg))
    if not sim:
        raise RuntimeError(f"bq_create a echoue: {dll.bq_last_error().decode()}")
    return sim


def add_water(sim):
    mat = BqMaterial(model=BQ_MODEL_WATER, rho=1000.0, bulk=4e4, gamma=3.0)
    mid = dll.bq_add_material(sim, ctypes.byref(mat))
    if mid < 0:
        raise RuntimeError(f"bq_add_material a echoue: {dll.bq_last_error().decode()}")
    return mid


def emit_far_droplet(sim, mid):
    """bq_step() sort tot si aucune particule fluide n'est presente (s->n==0
    || n_mats==0) -- CONSTAT du lot B3a : meme la detection de contact pure
    ne tourne pas sans au moins une particule. Contournement de harnais
    (PAS de la production) : une seule cellule de particules dans un coin du
    DOMAINE (le solveur n'emet rien hors de sa grille), loin des corps
    testes."""
    lo = (ctypes.c_float * 3)(0.82, 0.82, 0.82)
    hi = (ctypes.c_float * 3)(0.85, 0.85, 0.85)
    v0 = (ctypes.c_float * 3)(0.0, 0.0, 0.0)
    n = dll.bq_emit_box(sim, mid, lo, hi, v0)
    if n <= 0:
        raise RuntimeError("emit_far_droplet: aucune particule emise")


def build_body_sdf(sim, body, tri, target_cell, max_res):
    tri = np.ascontiguousarray(tri, dtype=np.float32)
    n_tri = tri.shape[0]
    ptr = tri.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    rc = dll.bq_build_body_sdf(sim, body, ptr, n_tri, target_cell, max_res)
    if rc != 0:
        raise RuntimeError(f"bq_build_body_sdf a echoue: {dll.bq_last_error().decode()}")


def set_body_samples(sim, body, pts):
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    n = pts.shape[0]
    ptr = pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if n > 0 else None
    rc = dll.bq_set_body_samples(sim, body, ptr, n)
    if rc != 0:
        raise RuntimeError(f"bq_set_body_samples a echoue: {dll.bq_last_error().decode()}")


def query_body_sdf(sim, body, pts_local, want_grad=True):
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


def set_bodies(sim, bodies):
    arr = (BqRigidBody * len(bodies))(*bodies)
    rc = dll.bq_set_collider_bodies(sim, arr, len(bodies))
    if rc != 0:
        raise RuntimeError(f"bq_set_collider_bodies a echoue: {dll.bq_last_error().decode()}")


def read_contacts(sim, cap=20000):
    buf = (ctypes.c_float * (cap * 9))()
    n = dll.bq_read_contacts(sim, buf, cap)
    if n < 0:
        raise RuntimeError(f"bq_read_contacts a echoue: {dll.bq_last_error().decode()}")
    n_read = min(n, cap)
    arr = np.frombuffer(buf, dtype=np.float32, count=n_read * 9).reshape(n_read, 9)
    contacts = []
    for row in arr:
        contacts.append({
            "bodyA": int(round(row[0])),
            "bodyB": int(round(row[1])),
            "point": row[2:5].copy(),
            "normal": row[5:8].copy(),
            "depth": float(row[8]),
        })
    return n, contacts  # n = total detecte (peut depasser cap)


def make_rigid_body(x, dynamic=1, mass=1.0, lock_all=False):
    b = BqRigidBody()
    b.dynamic = dynamic
    b.mass = mass if dynamic else 0.0
    for i in range(9):
        b.inv_inertia[i] = 1.0 if i in (0, 4, 8) else 0.0
    for i in range(3):
        b.x[i] = x[i]
        b.v[i] = 0.0
        b.w[i] = 0.0
    b.q[0] = 1.0
    b.q[1] = b.q[2] = b.q[3] = 0.0
    b.use_gravity = 0
    if lock_all:
        for i in range(3):
            b.lock_lin[i] = 1
            b.lock_ang[i] = 1
    return b


# ---------------------------------------------------------------------------
# Geometrie de test
# ---------------------------------------------------------------------------


def box_triangles(half_extent):
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
    tris_idx = np.array(faces, dtype=np.int64)
    verts = v
    return np.array([[verts[a], verts[b], verts[c]] for a, b, c in faces], dtype=np.float32), verts, tris_idx


def box_sdf_analytic(p, half_extent):
    h = np.asarray(half_extent, dtype=np.float64)
    q = np.abs(p) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


def box_analytic_gradient(p, half_extent, eps=1e-6):
    """Gradient numerique (differences centrees, precision double) de
    box_sdf_analytic -- sert de verite terrain, DISTINCTE du gradient de
    body_sdf (grille de voxels) que l'on verifie."""
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    g = np.zeros_like(p)
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = eps
        g[:, k] = (box_sdf_analytic(p + dp, half_extent) - box_sdf_analytic(p - dp, half_extent)) / (2 * eps)
    norm = np.linalg.norm(g, axis=-1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    return g / norm


def box_edge_band(p, half_extent, band):
    """True si p est dans une region ou la normale de la boite est
    AMBIGUE (pas seulement "proche d'une arete geometrique") :

      1. A l'EXTERIEUR sur au moins deux axes a la fois (q = |p|-h > 0 sur
         >=2 composantes) : la feature la plus proche EST une arete/sommet,
         quelle que soit la distance.
      2. A moins de `band` de deux plans de face simultanement (cas
         interieur proche de la surface, meme logique que 1 en plus fin).
      3. RIDGE MEDIAN (axe interieur dominant) : pour un point interieur
         (tous q<0), la face la plus proche est sur l'axe k = argmax(q), et
         le signe de la normale sur cet axe est sign(p_k). Si |p_k| est
         proche de 0 (le point est pres du plan median de symetrie de cet
         axe), ce signe bascule d'un cote a l'autre pour une variation
         infinitesimale de p_k -- exactement le mode de defaillance observe
         sur une boite tres fine selon un axe (demi-etendue 0.10 contre
         0.15/0.25 ici) : le gradient numerique (grille de voxels) et le
         gradient analytique (differences finies exactes) peuvent choisir
         des cotes differents pres de ce plan, sans que ce soit un defaut de
         body_sdf -- l'ambiguite est dans la geometrie elle-meme, pas dans
         le champ."""
    h = np.asarray(half_extent, dtype=np.float64)
    q = np.abs(p) - h
    outside_axes = np.sum(q > 1e-9, axis=-1) >= 2
    near_band = np.sum(np.abs(q) < band, axis=-1) >= 2

    # critere general de quasi-egalite entre les DEUX plus grandes composantes
    # de q (nearest-face et second-nearest-face presque a egalite) : couvre
    # a la fois le ridge median interieur (deux faces opposees du meme axe
    # fin, cf. docstring) et le ridge 3D pres d'un coin ou trois faces sont
    # presque equidistantes -- observe empiriquement sur cette geometrie.
    q_sorted = np.sort(q, axis=-1)[..., ::-1]
    gap = q_sorted[..., 0] - q_sorted[..., 1]
    ridge = gap < band

    return outside_axes | near_band | ridge


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


def grid_face_samples(half_extent, axis, sign, n=7, margin=0.08):
    """Points regulierement repartis sur UNE face d'une boite (repere de
    corps), en retrait de `margin` (fraction du demi-cote) par rapport aux
    aretes -- pour separer proprement "pres d'une face plate" de "pres d'une
    arete" cote generation d'echantillons (complementaire de box_edge_band
    cote analyse)."""
    hx, hy, hz = half_extent
    h = [hx, hy, hz]
    other = [i for i in range(3) if i != axis]
    u = np.linspace(-1 + margin, 1 - margin, n)
    pts = []
    for a in u:
        for b in u:
            p = [0.0, 0.0, 0.0]
            p[axis] = sign * h[axis]
            p[other[0]] = a * h[other[0]]
            p[other[1]] = b * h[other[1]]
            pts.append(p)
    return np.array(pts, dtype=np.float64)


def box_surface_samples(half_extent, n_per_face=7, margin=0.08):
    """Echantillons deterministes sur les 6 faces d'une boite (repere de
    corps) : grille regulieres par face (evite les aretes exactes, cf.
    grid_face_samples) + les 8 coins + les 6 centres de face -- garantit
    qu'un point de contact analytiquement connu (centre de face) est present
    dans le jeu d'echantillons."""
    hx, hy, hz = half_extent
    pts = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            pts.append(grid_face_samples(half_extent, axis, sign, n=n_per_face, margin=margin))
    pts = np.concatenate(pts, axis=0)
    corners = np.array([[sx * hx, sy * hy, sz * hz]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    centers = np.array([
        [hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0], [0, 0, hz], [0, 0, -hz],
    ], dtype=np.float64)
    return np.concatenate([pts, corners, centers], axis=0)


def main():
    all_ok = True

    # ===================================================================
    # 0. Validation du GRADIENT (jamais faite par B1) -- PREREQUIS
    # ===================================================================
    print("=== 0. Validation du gradient de body_sdf() (normales de contact) ===")
    rng = np.random.default_rng(2026)

    half = (0.15, 0.25, 0.10)
    target_cell = 0.01
    max_res = 128
    tri, _, _ = box_triangles(half)
    sim = make_sim()
    build_body_sdf(sim, 0, tri, target_cell, max_res)

    pad = 3.0 * target_cell
    n_pts = 20000
    lo = -np.array(half) - pad
    hi = np.array(half) + pad
    pts = rng.uniform(lo, hi, size=(n_pts, 3))
    # exclut le hors-grille (sentinelle, gradient nul par construction)
    far_mask = np.abs(box_sdf_analytic(pts, half)) > (np.max(half) + pad) * 0.999
    pts = pts[~far_mask]

    phi_num, grad_num = query_body_sdf(sim, 0, pts.astype(np.float32), want_grad=True)
    grad_num = grad_num.astype(np.float64)
    grad_ana = box_analytic_gradient(pts, half)

    gn_num = np.linalg.norm(grad_num, axis=-1)
    valid = gn_num > 1e-6  # exclut les points ou le gradient numerique degenere (loin, plat)
    grad_num_u = grad_num[valid] / gn_num[valid, None]
    cosang = np.clip(np.sum(grad_num_u * grad_ana[valid], axis=-1), -1.0, 1.0)
    ang_deg = np.degrees(np.arccos(cosang))

    edge_mask_all = box_edge_band(pts, half, 3.0 * target_cell)
    edge_mask = edge_mask_all[valid]

    ang_face = ang_deg[~edge_mask]
    ang_edge = ang_deg[edge_mask]
    print(f"  points valides: {valid.sum()}/{n_pts}, pres d'aretes/sommets: {edge_mask.sum()}")
    print(f"  boite -- erreur angulaire max PRES DES FACES PLATES : {np.max(ang_face):.3f} deg "
          f"(mediane {np.median(ang_face):.3f} deg)")
    print(f"  boite -- erreur angulaire max PRES DES ARETES/SOMMETS : {np.max(ang_edge):.3f} deg "
          f"(mediane {np.median(ang_edge):.3f} deg)")
    all_ok &= check(np.max(ang_face) < 5.0, "boite : erreur angulaire max < 5 deg pres des faces plates")
    all_ok &= check(np.median(ang_edge) < 30.0, "boite : erreur angulaire mediane < 30 deg pres des aretes"
                     " (l'arete elle-meme n'a pas de normale unique -- tolerance large volontaire)")
    dll.bq_destroy(sim)

    # sphere
    radius = 0.20
    sph_tri = icosphere(radius, subdiv=4)
    sim = make_sim()
    build_body_sdf(sim, 0, sph_tri, target_cell, max_res)
    pts_s = rng.uniform(-(radius + pad), radius + pad, size=(n_pts, 3))
    far_mask_s = np.abs(np.linalg.norm(pts_s, axis=-1) - radius) > (radius + pad) * 0.999 - radius
    # loin du centre ET a l'exterieur de la grille -> exclu (meme logique que verify_body_sdf)
    r_pts = np.linalg.norm(pts_s, axis=-1)
    far_mask_s = r_pts > (radius + pad) * 0.999
    pts_s = pts_s[~far_mask_s]
    phi_num_s, grad_num_s = query_body_sdf(sim, 0, pts_s.astype(np.float32), want_grad=True)
    grad_num_s = grad_num_s.astype(np.float64)
    r_pts = np.linalg.norm(pts_s, axis=-1)
    near_center = r_pts < 0.02  # normale non definie au centre -- exclu
    grad_ana_s = pts_s / np.maximum(r_pts, 1e-9)[:, None]
    gn_num_s = np.linalg.norm(grad_num_s, axis=-1)
    valid_s = (gn_num_s > 1e-6) & (~near_center)
    grad_num_s_u = grad_num_s[valid_s] / gn_num_s[valid_s, None]
    cosang_s = np.clip(np.sum(grad_num_s_u * grad_ana_s[valid_s], axis=-1), -1.0, 1.0)
    ang_deg_s = np.degrees(np.arccos(cosang_s))
    print(f"  sphere -- erreur angulaire max : {np.max(ang_deg_s):.3f} deg "
          f"(mediane {np.median(ang_deg_s):.3f} deg), points valides {valid_s.sum()}/{n_pts}")
    all_ok &= check(np.max(ang_deg_s) < 5.0, "sphere : erreur angulaire max < 5 deg (pas d'aretes)")
    dll.bq_destroy(sim)

    if not all_ok:
        print("\n!!! GRADIENT DEFAILLANT -- ARRET (inutile de construire la detection dessus) !!!")
        return 1

    # ===================================================================
    # 1. Deux boites qui se recouvrent d'une profondeur connue
    # ===================================================================
    print("\n=== 1. Deux boites en recouvrement -- profondeur et normale connues ===")
    half0 = (0.10, 0.10, 0.10)
    half1 = (0.10, 0.10, 0.10)
    overlap = 0.05
    gap_x = half0[0] + half1[0] - overlap  # 0.15
    pos0 = np.array([0.5, 0.5, 0.5])
    pos1 = pos0 + np.array([gap_x, 0.0, 0.0])

    sim = make_sim()
    mid = add_water(sim)
    emit_far_droplet(sim, mid)
    build_body_sdf(sim, 0, box_triangles(half0)[0], target_cell, max_res)
    build_body_sdf(sim, 1, box_triangles(half1)[0], target_cell, max_res)
    samples0 = box_surface_samples(half0)
    samples1 = box_surface_samples(half1)
    set_body_samples(sim, 0, samples0)
    set_body_samples(sim, 1, samples1)

    b0 = make_rigid_body(pos0, dynamic=1, mass=1.0, lock_all=True)  # gele : broadphase pas ecartee
    b1 = make_rigid_body(pos1, dynamic=0)  # statique reel
    set_bodies(sim, [b0, b1])

    if dll.bq_step(sim, 1.0 / 24.0) < 0:
        raise RuntimeError(dll.bq_last_error().decode())

    n_total, contacts = read_contacts(sim)
    print(f"  contacts detectes ce sous-pas : {n_total}")
    all_ok &= check(n_total > 0, "au moins un contact detecte (boites en recouvrement)")

    # point de contact attendu (centre de la face +x de la boite 0, en monde)
    expected_point = pos0 + np.array([half0[0], 0.0, 0.0])
    best = None
    best_d = 1e9
    for c in contacts:
        d = np.linalg.norm(c["point"] - expected_point)
        if d < best_d:
            best_d = d
            best = c
    if best is not None:
        print(f"  contact le plus proche du point attendu {expected_point} : "
              f"point={best['point']}, dist={best_d:.5f}, depth={best['depth']:.5f}, "
              f"normal={best['normal']}, bodyA={best['bodyA']}, bodyB={best['bodyB']}")
        all_ok &= check(best_d < 1e-3, "contact trouve au point de face-centre attendu")
        all_ok &= check(abs(best["depth"] - overlap) < 3 * (target_cell),
                         f"profondeur ~= chevauchement analytique ({overlap} m), "
                         f"tolerance 3 voxels ({3*target_cell:.4f} m)")
        n = best["normal"]
        all_ok &= check(abs(abs(n[0]) - 1.0) < 0.15 and abs(n[1]) < 0.15 and abs(n[2]) < 0.15,
                         "normale ~alignee sur l'axe x (axe du recouvrement)")
    else:
        all_ok &= check(False, "aucun contact trouve pres du point attendu")

    max_depth = max((c["depth"] for c in contacts), default=0.0)
    print(f"  profondeur MAX observee : {max_depth:.5f} m (chevauchement analytique {overlap} m)")
    dll.bq_destroy(sim)

    # ===================================================================
    # 2. Deux boites separees -- zero contact
    # ===================================================================
    print("\n=== 2. Deux boites separees -- zero contact attendu ===")
    gap = 0.05
    pos1_sep = pos0 + np.array([half0[0] + half1[0] + gap, 0.0, 0.0])
    sim = make_sim()
    mid = add_water(sim)
    emit_far_droplet(sim, mid)
    build_body_sdf(sim, 0, box_triangles(half0)[0], target_cell, max_res)
    build_body_sdf(sim, 1, box_triangles(half1)[0], target_cell, max_res)
    set_body_samples(sim, 0, box_surface_samples(half0))
    set_body_samples(sim, 1, box_surface_samples(half1))
    b0 = make_rigid_body(pos0, dynamic=1, mass=1.0, lock_all=True)
    b1 = make_rigid_body(pos1_sep, dynamic=0)
    set_bodies(sim, [b0, b1])
    if dll.bq_step(sim, 1.0 / 24.0) < 0:
        raise RuntimeError(dll.bq_last_error().decode())
    n_total_sep, contacts_sep = read_contacts(sim)
    print(f"  contacts detectes : {n_total_sep}")
    all_ok &= check(n_total_sep == 0, "zero contact entre boites separees")
    dll.bq_destroy(sim)

    # ===================================================================
    # 3. Paire statique/statique -- ecartee des la broadphase
    # ===================================================================
    print("\n=== 3. Paire statique/statique (AABB en recouvrement) -- zero contact ===")
    sim = make_sim()
    mid = add_water(sim)
    emit_far_droplet(sim, mid)
    build_body_sdf(sim, 0, box_triangles(half0)[0], target_cell, max_res)
    build_body_sdf(sim, 1, box_triangles(half1)[0], target_cell, max_res)
    set_body_samples(sim, 0, box_surface_samples(half0))
    set_body_samples(sim, 1, box_surface_samples(half1))
    b0 = make_rigid_body(pos0, dynamic=0)  # STATIQUE
    b1 = make_rigid_body(pos1, dynamic=0)  # STATIQUE, en recouvrement (meme pose que test 1)
    set_bodies(sim, [b0, b1])
    if dll.bq_step(sim, 1.0 / 24.0) < 0:
        raise RuntimeError(dll.bq_last_error().decode())
    n_total_ss, _ = read_contacts(sim)
    print(f"  contacts detectes (deux statiques en recouvrement d'AABB) : {n_total_ss}")
    all_ok &= check(n_total_ss == 0, "paire statique/statique ecartee -- zero contact malgre le recouvrement")
    dll.bq_destroy(sim)

    # ===================================================================
    # 4. Caisse posee sur un sol -- contacts sous la caisse, normale +y
    # ===================================================================
    print("\n=== 4. Caisse posee sur un sol statique -- normales vers le haut (+y) ===")
    floor_half = (0.5, 0.05, 0.5)
    box_half = (0.08, 0.08, 0.08)
    floor_pos = np.array([0.5, 0.20, 0.5])
    # penetration legere (0.01) pour garantir un contact sans dependre d'un pas de temps
    box_pos = floor_pos + np.array([0.0, floor_half[1] + box_half[1] - 0.01, 0.0])

    sim = make_sim()
    mid = add_water(sim)
    emit_far_droplet(sim, mid)
    build_body_sdf(sim, 0, box_triangles(box_half)[0], target_cell, max_res)
    build_body_sdf(sim, 1, box_triangles(floor_half)[0], target_cell, max_res)
    set_body_samples(sim, 0, box_surface_samples(box_half))
    set_body_samples(sim, 1, box_surface_samples(floor_half, n_per_face=5))
    b0 = make_rigid_body(box_pos, dynamic=1, mass=1.0, lock_all=True)  # caisse, gelee pour la mesure
    b1 = make_rigid_body(floor_pos, dynamic=0)  # sol statique
    set_bodies(sim, [b0, b1])
    if dll.bq_step(sim, 1.0 / 24.0) < 0:
        raise RuntimeError(dll.bq_last_error().decode())
    n_total_f, contacts_f = read_contacts(sim)
    print(f"  contacts detectes : {n_total_f}")
    all_ok &= check(n_total_f > 0, "au moins un contact caisse/sol")
    if contacts_f:
        # La normale est TOUJOURS le gradient sortant du corps B interroge
        # (cf. spec, section 2) -- elle depend donc de QUEL corps est B, pas
        # seulement de la geometrie. Les deux sens sont testes (D10) :
        #   bodyB=1 (sol)   -> normale sortante du sol   -> +y (vers le haut)
        #   bodyB=0 (caisse) -> normale sortante de la caisse au point de
        #                       contact (bas de la caisse) -> -y (vers le bas)
        # Les DEUX sont corrects et attendus -- ce n'est pas une incoherence,
        # c'est la consequence directe de la convention "normale = R_B*grad"
        # combinee au test bidirectionnel.
        to_floor = [c for c in contacts_f if c["bodyB"] == 1]
        to_box = [c for c in contacts_f if c["bodyB"] == 0]
        print(f"  contacts vers le SDF du sol (bodyB=1)   : {len(to_floor)}")
        print(f"  contacts vers le SDF de la caisse (bodyB=0) : {len(to_box)}")
        all_ok &= check(len(to_floor) > 0, "au moins un contact teste contre le SDF du sol")
        all_ok &= check(len(to_box) > 0, "au moins un contact teste contre le SDF de la caisse (sens inverse, D10)")
        if to_floor:
            ny_floor = np.array([c["normal"][1] for c in to_floor])
            print(f"    normale.y min (vers sol)   : {float(np.min(ny_floor)):.4f} (attendu > 0.9)")
            all_ok &= check(float(np.min(ny_floor)) > 0.9, "normales vers le SDF du sol : +y (vers le haut)")
        if to_box:
            ny_box = np.array([c["normal"][1] for c in to_box])
            print(f"    normale.y max (vers caisse): {float(np.max(ny_box)):.4f} (attendu < -0.9)")
            all_ok &= check(float(np.max(ny_box)) < -0.9,
                             "normales vers le SDF de la caisse : -y (sortante de sa face du bas)")

        points_y = np.array([c["point"][1] for c in contacts_f])
        max_pt_y = float(np.max(points_y))
        box_bottom_y = box_pos[1] - box_half[1]
        print(f"  point.y max des contacts : {max_pt_y:.4f} (bas de la caisse = {box_bottom_y:.4f})")
        all_ok &= check(max_pt_y < box_bottom_y + 0.02,
                         "les contacts restent pres de la face du bas de la caisse (pas sur les cotes/dessus)")
    dll.bq_destroy(sim)

    # ===================================================================
    # 5. Saturation du tampon
    # ===================================================================
    print("\n=== 5. Saturation du tampon de contacts (BQ_MAX_CONTACTS) ===")
    dense_half = (0.10, 0.10, 0.10)
    sim = make_sim()
    mid = add_water(sim)
    emit_far_droplet(sim, mid)
    build_body_sdf(sim, 0, box_triangles(dense_half)[0], target_cell, max_res)
    build_body_sdf(sim, 1, box_triangles(dense_half)[0], target_cell, max_res)
    # grille tres dense sur les deux corps, CONCENTRIQUES (recouvrement total)
    dense_samples = box_surface_samples(dense_half, n_per_face=70, margin=0.02)
    print(f"  echantillons/corps : {dense_samples.shape[0]} (x2 corps, x2 sens -> jusqu'a "
          f"{2*dense_samples.shape[0]} contacts potentiels)")
    set_body_samples(sim, 0, dense_samples)
    set_body_samples(sim, 1, dense_samples)
    b0 = make_rigid_body(pos0, dynamic=1, mass=1.0, lock_all=True)
    b1 = make_rigid_body(pos0, dynamic=0)  # meme position -- recouvrement total
    set_bodies(sim, [b0, b1])
    if dll.bq_step(sim, 1.0 / 24.0) < 0:
        raise RuntimeError(dll.bq_last_error().decode())
    n_total_sat, _ = read_contacts(sim, cap=1)
    overflow = dll.bq_contacts_last_overflow(sim)
    print(f"  contacts detectes (total rapporte) : {n_total_sat}, bq_contacts_last_overflow() = {overflow}")
    all_ok &= check(overflow == 1, "saturation effectivement rapportee")
    dll.bq_destroy(sim)

    # ===================================================================
    # 6. Cout par sous-pas : 2 puis 10 corps
    # ===================================================================
    print("\n=== 6. Cout par sous-pas : 2 puis 10 corps ===")

    def bench(n_bodies, n_frames=10):
        sim = make_sim()
        mid = add_water(sim)
        emit_far_droplet(sim, mid)
        bodies = []
        samples_counts = []
        h = (0.08, 0.08, 0.08)
        for i in range(n_bodies):
            build_body_sdf(sim, i, box_triangles(h)[0], target_cell, max_res)
            s = box_surface_samples(h, n_per_face=6)
            set_body_samples(sim, i, s)
            samples_counts.append(s.shape[0])
            pos = np.array([0.5 + 0.02 * i, 0.5, 0.5])  # legerement decales, AABB qui se touchent
            bodies.append(make_rigid_body(pos, dynamic=1, mass=1.0, lock_all=True))
        set_bodies(sim, bodies)
        # chauffe (compile les kernels / premiere allocation lazy eventuelle)
        dll.bq_step(sim, 1.0 / 24.0)
        t0 = time.perf_counter()
        total_substeps = 0
        for _ in range(n_frames):
            ns = dll.bq_step(sim, 1.0 / 24.0)
            if ns < 0:
                raise RuntimeError(dll.bq_last_error().decode())
            total_substeps += ns
        elapsed = time.perf_counter() - t0
        dll.bq_destroy(sim)
        return elapsed, total_substeps

    for nb in (2, 10):
        elapsed, total_substeps = bench(nb)
        ms_per_substep = 1000.0 * elapsed / max(total_substeps, 1)
        print(f"  {nb:2d} corps : {elapsed*1000:.2f} ms / {n_frames if False else 10} frames "
              f"({total_substeps} sous-pas) -> {ms_per_substep:.4f} ms/sous-pas "
              f"(inclut fluide + corps + detection de contact)")

    print("\n" + ("TOUT OK" if all_ok else "AU MOINS UN ECHEC"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
