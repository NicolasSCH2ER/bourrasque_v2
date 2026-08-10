"""M17 / B3b -- verification du solveur de contact corps<->corps (D11) :
impulsions sequentielles, split impulse, warm starting, mise en sommeil, et
correctif du point 0 (bq_step doit simuler des corps SANS aucun fluide).

Charge la DLL directement par ctypes (meme motif que verify_contacts_b3a.py),
avec un BqRigidBody etendu du champ `friction` (ABI 12 -> 13, B3b).

Lancement (interpreteur avec numpy) :
    python tools/repro/verify_contact_solve_b3b.py
"""

import ctypes
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DLL_PATH = ROOT / "extension" / "bin" / "bourrasque.dll"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_contacts_b3a as b3a  # noqa: E402  reutilise geometrie (box_triangles, box_surface_samples)


# ---------------------------------------------------------------------------
# Bindings ctypes (BqRigidBody ETENDU du champ `friction`, B3b)
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
        ("friction", ctypes.c_float),
    ]


dll = ctypes.CDLL(str(DLL_PATH))

dll.bq_abi_version.argtypes = []
dll.bq_abi_version.restype = ctypes.c_int
dll.bq_rigid_body_size.argtypes = []
dll.bq_rigid_body_size.restype = ctypes.c_size_t

dll.bq_default_config.argtypes = [ctypes.POINTER(BqConfig)]
dll.bq_default_config.restype = None
dll.bq_create.argtypes = [ctypes.POINTER(BqConfig)]
dll.bq_create.restype = ctypes.c_void_p
dll.bq_destroy.argtypes = [ctypes.c_void_p]
dll.bq_destroy.restype = None
dll.bq_last_error.argtypes = []
dll.bq_last_error.restype = ctypes.c_char_p

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

dll.bq_set_collider_bodies.argtypes = [ctypes.c_void_p, ctypes.POINTER(BqRigidBody), ctypes.c_int]
dll.bq_set_collider_bodies.restype = ctypes.c_int

dll.bq_read_collider_bodies.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
dll.bq_read_collider_bodies.restype = ctypes.c_int

dll.bq_read_body_sleep.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
dll.bq_read_body_sleep.restype = ctypes.c_int

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


def build_body_sdf(sim, body, tri, target_cell, max_res=64):
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


def set_bodies(sim, bodies):
    arr = (BqRigidBody * len(bodies))(*bodies)
    rc = dll.bq_set_collider_bodies(sim, arr, len(bodies))
    if rc != 0:
        raise RuntimeError(f"bq_set_collider_bodies a echoue: {dll.bq_last_error().decode()}")


def read_bodies(sim, n_bodies):
    buf = (ctypes.c_float * (n_bodies * 13))()
    n = dll.bq_read_collider_bodies(sim, buf)
    assert n == n_bodies
    arr = np.frombuffer(buf, dtype=np.float32, count=n_bodies * 13).reshape(n_bodies, 13)
    out = []
    for row in arr:
        out.append({
            "x": row[0:3].copy(), "q": row[3:7].copy(),
            "v": row[7:10].copy(), "w": row[10:13].copy(),
        })
    return out


def read_sleep(sim, n_bodies):
    buf = (ctypes.c_uint8 * n_bodies)()
    n = dll.bq_read_body_sleep(sim, buf)
    assert n == n_bodies
    return [bool(buf[i]) for i in range(n_bodies)]


def make_rigid_body(x, dynamic=1, mass=1.0, half_extent=(0.05, 0.05, 0.05),
                     restitution=0.0, friction=0.5, use_gravity=1, q=None, v=None):
    b = BqRigidBody()
    b.dynamic = dynamic
    b.mass = mass if dynamic else 0.0
    hx, hy, hz = half_extent
    # inertie d'une boite pleine homogene, DIAGONALE (repere de corps aligne
    # sur les axes de la boite) : I_xx = m/12*(hy^2+hz^2)*4, etc. (demi-etendues)
    ixx = mass / 3.0 * (hy * hy + hz * hz)
    iyy = mass / 3.0 * (hx * hx + hz * hz)
    izz = mass / 3.0 * (hx * hx + hy * hy)
    for i in range(9):
        b.inv_inertia[i] = 0.0
    if dynamic and ixx > 0 and iyy > 0 and izz > 0:
        b.inv_inertia[0] = 1.0 / ixx
        b.inv_inertia[4] = 1.0 / iyy
        b.inv_inertia[8] = 1.0 / izz
    for i in range(3):
        b.x[i] = x[i]
        b.v[i] = v[i] if v is not None else 0.0
        b.w[i] = 0.0
    if q is None:
        b.q[0] = 1.0; b.q[1] = b.q[2] = b.q[3] = 0.0
    else:
        b.q[0], b.q[1], b.q[2], b.q[3] = q
    b.use_gravity = use_gravity
    b.restitution = restitution
    b.friction = friction
    for i in range(3):
        b.lock_lin[i] = 0
        b.lock_ang[i] = 0
    return b


def box_body(sim, body_id, x, half_extent, dynamic, mass=1.0, target_cell=0.01,
             restitution=0.0, friction=0.5, use_gravity=1, q=None, v=None, n_per_face=6):
    tri, _, _ = b3a.box_triangles(half_extent)
    build_body_sdf(sim, body_id, tri, target_cell, max_res=64)
    samples = b3a.box_surface_samples(half_extent, n_per_face=n_per_face)
    set_body_samples(sim, body_id, samples)
    return make_rigid_body(x, dynamic=dynamic, mass=mass, half_extent=half_extent,
                           restitution=restitution, friction=friction,
                           use_gravity=use_gravity, q=q, v=v)


def kinetic_energy(state, mass, half_extent):
    v = state["v"]; w = state["w"]
    hx, hy, hz = half_extent
    ixx = mass / 3.0 * (hy * hy + hz * hz)
    iyy = mass / 3.0 * (hx * hx + hz * hz)
    izz = mass / 3.0 * (hx * hx + hy * hy)
    ke_lin = 0.5 * mass * float(np.dot(v, v))
    ke_ang = 0.5 * (ixx * w[0] ** 2 + iyy * w[1] ** 2 + izz * w[2] ** 2)
    return ke_lin + ke_ang


DT_FRAME = 1.0 / 24.0


def main():
    print(f"ABI = {dll.bq_abi_version()} (attendu 13), sizeof(BqRigidBody) DLL = "
          f"{dll.bq_rigid_body_size()} (ctypes = {ctypes.sizeof(BqRigidBody)})")
    all_ok = ctypes.sizeof(BqRigidBody) == dll.bq_rigid_body_size()
    check(all_ok, "sizeof(BqRigidBody) coherent (struct ctypes bien resynchronisee)")

    # -----------------------------------------------------------------
    print("\n=== 1. REPOS -- caisse lachee sur un sol statique, SANS fluide ===")
    sim = make_sim()
    floor_h = (0.4, 0.02, 0.4)
    box_h = (0.05, 0.05, 0.05)
    target_cell = 0.01
    floor = box_body(sim, 0, (0.5, 0.0, 0.5), floor_h, dynamic=0, target_cell=target_cell, n_per_face=4)
    box = box_body(sim, 1, (0.5, 0.3, 0.5), box_h, dynamic=1, mass=1.0,
                  target_cell=target_cell, restitution=0.0, friction=0.5)
    set_bodies(sim, [floor, box])

    n_frames = 120
    substeps_history = []
    positions = []
    for f in range(n_frames):
        ns = dll.bq_step(sim, DT_FRAME)
        if ns < 0:
            raise RuntimeError(dll.bq_last_error().decode())
        substeps_history.append(ns)
        st = read_bodies(sim, 2)[1]
        positions.append(st["x"].copy())
    states = read_bodies(sim, 2)
    box_state = states[1]
    penetration = (floor_h[1] + box_h[1]) - (box_state["x"][1] - 0.0)
    # jitter : amplitude de |v| sur les 20 dernieres frames (doit etre nulle si endormi)
    jitter_vs = []
    for _ in range(20):
        dll.bq_step(sim, DT_FRAME)
        st = read_bodies(sim, 2)[1]
        jitter_vs.append(float(np.linalg.norm(st["v"])))
    lateral_drift = math.hypot(box_state["x"][0] - 0.5, box_state["x"][2] - 0.5)
    sleep = read_sleep(sim, 2)

    print(f"  penetration residuelle = {penetration * 1000:.4f} mm (voxel corps = {target_cell*1000:.2f} mm)")
    print(f"  frémissement (max |v| sur 20 dernieres frames) = {max(jitter_vs):.6f} m/s")
    print(f"  derive laterale = {lateral_drift * 1000:.4f} mm")
    print(f"  endormi (corps 1) = {sleep[1]}")
    ok1 = check(abs(penetration) < target_cell, "penetration < 1 voxel")
    ok1b = check(max(jitter_vs) < 1e-4, "pas de fremissement une fois endormi")
    ok1c = check(lateral_drift < target_cell, "pas de derive laterale")
    ok1d = check(sleep[1], "corps endormi apres repos prolonge")
    all_ok &= ok1 & ok1b & ok1c & ok1d
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 6. SOMMEIL -- nombre de sous-pas avant endormissement, reveil au choc ===")
    sim = make_sim()
    floor = box_body(sim, 0, (0.5, 0.0, 0.5), floor_h, dynamic=0, target_cell=target_cell, n_per_face=4)
    box = box_body(sim, 1, (0.5, 0.15, 0.5), box_h, dynamic=1, mass=1.0,
                  target_cell=target_cell, restitution=0.0, friction=0.5)
    set_bodies(sim, [floor, box])
    total_substeps = 0
    substeps_to_sleep = None
    for f in range(150):
        ns = dll.bq_step(sim, DT_FRAME)
        total_substeps += ns
        if read_sleep(sim, 2)[1] and substeps_to_sleep is None:
            substeps_to_sleep = total_substeps
            break
    print(f"  sous-pas avant endormissement = {substeps_to_sleep}")
    ok6a = check(substeps_to_sleep is not None and substeps_to_sleep < 150 * 6, "s'endort dans un temps raisonnable")

    # reveil : lache une deuxieme caisse par-dessus
    box2 = box_body(sim, 2, (0.5, 0.5, 0.5), box_h, dynamic=1, mass=1.0,
                    target_cell=target_cell, restitution=0.0, friction=0.5)
    # bq_set_collider_bodies reinitialise TOUS les corps -- il faut relire l'etat
    # actuel du corps 1 (endormi, pose) pour ne pas le faire "retomber".
    st1 = read_bodies(sim, 2)[1]
    box.x[0], box.x[1], box.x[2] = [float(c) for c in st1["x"]]
    box.q[0], box.q[1], box.q[2], box.q[3] = [float(c) for c in st1["q"]]
    set_bodies(sim, [floor, box, box2])
    woke = False
    for f in range(60):
        dll.bq_step(sim, DT_FRAME)
        if not read_sleep(sim, 3)[1]:
            woke = True
            break
    ok6b = check(woke, "reveil effectif au choc d'un second corps")
    all_ok &= ok6a & ok6b
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 2. EMPILEMENT -- trois caisses, derive de hauteur sur plusieurs secondes ===")
    sim = make_sim()
    floor = box_body(sim, 0, (0.5, 0.0, 0.5), floor_h, dynamic=0, target_cell=target_cell, n_per_face=4)
    bodies = [floor]
    gap = 0.002
    for i in range(3):
        y = 2 * box_h[1] * i + box_h[1] + gap * i + 0.01
        bodies.append(box_body(sim, i + 1, (0.5, y, 0.5), box_h, dynamic=1, mass=1.0,
                               target_cell=target_cell, restitution=0.0, friction=0.6))
    set_bodies(sim, bodies)
    for f in range(72):  # ~3s de tassement
        dll.bq_step(sim, DT_FRAME)
    h_start = read_bodies(sim, 4)[3]["x"][1]
    h_track = []
    for f in range(96):  # ~4s supplementaires, mesure de derive
        dll.bq_step(sim, DT_FRAME)
        h_track.append(read_bodies(sim, 4)[3]["x"][1])
    h_drift = max(abs(h - h_start) for h in h_track)
    print(f"  hauteur caisse du haut apres tassement = {h_start*1000:.3f} mm")
    print(f"  derive max de hauteur sur 4s = {h_drift*1000:.4f} mm")
    ok2 = check(h_drift < target_cell, "pile stable, ne respire pas (< 1 voxel de derive)")
    all_ok &= ok2
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 3. ENERGIE -- deux corps se heurtent, restitution 0, sans fluide ===")
    sim = make_sim()
    b1 = box_body(sim, 0, (0.3, 0.5, 0.5), box_h, dynamic=1, mass=1.0, target_cell=target_cell,
                 restitution=0.0, friction=0.0, use_gravity=0, v=(0.5, 0.0, 0.0))
    b2 = box_body(sim, 1, (0.7, 0.5, 0.5), box_h, dynamic=1, mass=1.0, target_cell=target_cell,
                 restitution=0.0, friction=0.0, use_gravity=0, v=(-0.5, 0.0, 0.0))
    set_bodies(sim, [b1, b2])
    e0 = None
    max_ratio = 0.0
    for f in range(48):
        dll.bq_step(sim, DT_FRAME)
        states = read_bodies(sim, 2)
        e = kinetic_energy(states[0], 1.0, box_h) + kinetic_energy(states[1], 1.0, box_h)
        if e0 is None:
            e0 = e
        if e0 > 1e-9:
            max_ratio = max(max_ratio, e / e0)
    print(f"  max E(t)/E(0) = {max_ratio:.6f}")
    ok3 = check(max_ratio <= 1.0 + 1e-3, "l'energie cinetique totale ne croit jamais")
    all_ok &= ok3
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 4. ROTATION -- caisse lachee en biais, se stabilise a plat ===")
    sim = make_sim()
    floor = box_body(sim, 0, (0.5, 0.0, 0.5), floor_h, dynamic=0, target_cell=target_cell, n_per_face=4)
    angle = math.radians(25.0)
    qz = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))  # rotation autour de z
    box = box_body(sim, 1, (0.5, 0.25, 0.5), box_h, dynamic=1, mass=1.0,
                  target_cell=target_cell, restitution=0.0, friction=0.6, q=qz)
    set_bodies(sim, [floor, box])
    for f in range(150):
        dll.bq_step(sim, DT_FRAME)
    st = read_bodies(sim, 2)[1]
    qw, qx, qy, qzz = st["q"]
    # angle de la rotation residuelle (double couverture quaternion : |qw| proche de 1)
    ang_final = 2.0 * math.degrees(math.acos(min(1.0, abs(qw))))
    print(f"  angle final residuel = {ang_final:.3f} deg (parti de {math.degrees(angle):.1f} deg)")
    ok4 = check(ang_final < 5.0, "la caisse se redresse et se stabilise a plat")
    all_ok &= ok4
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 5. FRICTION -- caisse lancee horizontalement, distance d'arret ===")
    def slide_test(mu):
        sim = make_sim()
        floor = box_body(sim, 0, (0.5, 0.0, 0.5), floor_h, dynamic=0, target_cell=target_cell,
                         n_per_face=4, friction=mu)
        box = box_body(sim, 1, (0.15, box_h[1] + 0.005, 0.5), box_h, dynamic=1, mass=1.0,
                      target_cell=target_cell, restitution=0.0, friction=mu, v=(1.0, 0.0, 0.0))
        set_bodies(sim, [floor, box])
        x0 = read_bodies(sim, 2)[1]["x"][0]
        for f in range(72):
            dll.bq_step(sim, DT_FRAME)
        st = read_bodies(sim, 2)[1]
        dll.bq_destroy(sim)
        return abs(st["x"][0] - x0), float(np.linalg.norm(st["v"]))

    dist_mu, v_end_mu = slide_test(0.6)
    dist_0, v_end_0 = slide_test(0.0)
    print(f"  mu=0.6 : distance = {dist_mu*1000:.2f} mm, vitesse finale = {v_end_mu:.4f} m/s")
    print(f"  mu=0.0 : distance = {dist_0*1000:.2f} mm, vitesse finale = {v_end_0:.4f} m/s")
    ok5a = check(v_end_mu < 0.05, "avec friction, la caisse s'arrete")
    ok5b = check(dist_0 > dist_mu * 1.5, "sans friction (mu=0), elle glisse nettement plus loin")
    all_ok &= ok5a & ok5b

    # -----------------------------------------------------------------
    print("\n=== 7. SATURATION -- solveur stable meme avec des contacts omis ===")
    sim = make_sim(grid_res=32, cell_size=1.0 / 32.0)
    floor = box_body(sim, 0, (0.5, 0.0, 0.5), floor_h, dynamic=0, target_cell=target_cell,
                     n_per_face=4)
    bodies = [floor]
    n_stack = 12  # empilement dense pour generer beaucoup de contacts simultanes
    for i in range(n_stack):
        y = 2 * box_h[1] * i + box_h[1] + 0.001 * i + 0.01
        bodies.append(box_body(sim, i + 1, (0.5, y, 0.5), box_h, dynamic=1, mass=1.0,
                               target_cell=target_cell, restitution=0.0, friction=0.6,
                               n_per_face=10))  # densite d'echantillonnage elevee
    set_bodies(sim, bodies)
    overflowed = False
    finite = True
    for f in range(48):
        dll.bq_step(sim, DT_FRAME)
        if dll.bq_contacts_last_overflow(sim):
            overflowed = True
        states = read_bodies(sim, n_stack + 1)
        for st in states:
            if not np.all(np.isfinite(st["x"])) or not np.all(np.isfinite(st["v"])):
                finite = False
    print(f"  overflow observe au moins une fois = {overflowed}")
    ok7 = check(finite, "toutes les positions/vitesses restent finies malgre la saturation")
    all_ok &= ok7
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 8. COUT PAR SOUS-PAS ===")
    def bench(n_bodies, n_frames=10):
        sim = make_sim()
        floor = box_body(sim, 0, (0.5, 0.0, 0.5), floor_h, dynamic=0, target_cell=target_cell, n_per_face=4)
        bodies = [floor]
        for i in range(n_bodies):
            bodies.append(box_body(sim, i + 1, (0.3 + 0.05 * i, 0.3, 0.5), box_h, dynamic=1, mass=1.0,
                                   target_cell=target_cell, restitution=0.0, friction=0.5))
        set_bodies(sim, bodies)
        dll.bq_step(sim, DT_FRAME)  # chauffe
        t0 = time.perf_counter()
        total_substeps = 0
        for _ in range(n_frames):
            ns = dll.bq_step(sim, DT_FRAME)
            total_substeps += ns
        elapsed = time.perf_counter() - t0
        dll.bq_destroy(sim)
        return elapsed, total_substeps

    for nb in (1, 3, 10):
        elapsed, total_substeps = bench(nb)
        ms = 1000.0 * elapsed / max(total_substeps, 1)
        print(f"  {nb:2d} corps dynamiques : {ms:.4f} ms/sous-pas ({total_substeps} sous-pas, sans fluide)")

    print("\n" + ("TOUT OK" if all_ok else "AU MOINS UN ECHEC"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
