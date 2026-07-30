import ctypes as ct
import sys

DLL = r"C:\Users\nicol\Code\bourrasque_v2\build\Release\bourrasque.dll"
lib = ct.CDLL(DLL)


class BqConfig(ct.Structure):
    _fields_ = [
        ("grid_res", ct.c_int),
        ("domain", ct.c_float),
        ("gravity_y", ct.c_float),
        ("cfl", ct.c_float),
        ("ppc_axis", ct.c_int),
        ("max_particles", ct.c_int),
    ]


class BqMaterial(ct.Structure):
    _fields_ = [
        ("model", ct.c_int),
        ("rho", ct.c_float),
        ("E", ct.c_float),
        ("nu", ct.c_float),
        ("bulk", ct.c_float),
        ("gamma", ct.c_float),
    ]


BQ_MODEL_WATER = 1

lib.bq_default_config.argtypes = [ct.POINTER(BqConfig)]
lib.bq_default_config.restype = None

lib.bq_create.argtypes = [ct.POINTER(BqConfig)]
lib.bq_create.restype = ct.c_void_p

lib.bq_destroy.argtypes = [ct.c_void_p]
lib.bq_destroy.restype = None

lib.bq_add_material.argtypes = [ct.c_void_p, ct.POINTER(BqMaterial)]
lib.bq_add_material.restype = ct.c_int

lib.bq_emit_box.argtypes = [ct.c_void_p, ct.c_int,
                             ct.POINTER(ct.c_float), ct.POINTER(ct.c_float),
                             ct.POINTER(ct.c_float)]
lib.bq_emit_box.restype = ct.c_int

lib.bq_emit_points.argtypes = [ct.c_void_p, ct.c_int,
                                ct.POINTER(ct.c_float), ct.c_int,
                                ct.POINTER(ct.c_float)]
lib.bq_emit_points.restype = ct.c_int

lib.bq_step.argtypes = [ct.c_void_p, ct.c_float]
lib.bq_step.restype = ct.c_int

lib.bq_particle_count.argtypes = [ct.c_void_p]
lib.bq_particle_count.restype = ct.c_int

lib.bq_read_positions.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float)]
lib.bq_read_positions.restype = ct.c_int

lib.bq_last_error.argtypes = []
lib.bq_last_error.restype = ct.c_char_p


def make_sim():
    cfg = BqConfig()
    lib.bq_default_config(ct.byref(cfg))
    sim = lib.bq_create(ct.byref(cfg))
    assert sim, "bq_create failed"
    mat = BqMaterial(model=BQ_MODEL_WATER, rho=1000.0, E=0.0, nu=0.0,
                      bulk=4e4, gamma=3.0)
    mid = lib.bq_add_material(sim, ct.byref(mat))
    assert mid == 0, f"bq_add_material failed: {lib.bq_last_error()}"
    return sim, cfg


def f3(x, y, z):
    return (ct.c_float * 3)(x, y, z)


def read_positions(sim, n):
    buf = (ct.c_float * (n * 3))()
    rc = lib.bq_read_positions(sim, buf)
    assert rc >= 0, f"bq_read_positions failed: {lib.bq_last_error()}"
    return list(buf)


def sample_box(lo, hi, spacing):
    pts = []
    x = lo[0] + spacing / 2
    while x < hi[0]:
        y = lo[1] + spacing / 2
        while y < hi[1]:
            z = lo[2] + spacing / 2
            while z < hi[2]:
                pts.append((x, y, z))
                z += spacing
            y += spacing
        x += spacing
    return pts


def step1_equivalence():
    print("=== 1/2: equivalence bq_emit_box vs bq_emit_points (a l'emission) ===")
    lo = (0.10, 0.10, 0.10)
    hi = (0.35, 0.60, 0.90)
    vel = f3(0.0, 0.0, 0.0)

    sim_box, cfg = make_sim()
    n_box = lib.bq_emit_box(sim_box, 0, f3(*lo), f3(*hi), vel)
    assert n_box > 0, f"bq_emit_box failed: {lib.bq_last_error()}"
    pos_box = read_positions(sim_box, n_box)

    dx = cfg.domain / cfg.grid_res
    spacing = dx / cfg.ppc_axis
    pts = sample_box(lo, hi, spacing)
    assert len(pts) == n_box, f"echantillonnage python != {n_box}, obtenu {len(pts)}"

    pos_flat = []
    for p in pts:
        pos_flat.extend(p)
    pos_arr = (ct.c_float * len(pos_flat))(*pos_flat)

    sim_pts, _ = make_sim()
    n_pts = lib.bq_emit_points(sim_pts, 0, pos_arr, len(pts), vel)
    assert n_pts > 0, f"bq_emit_points failed: {lib.bq_last_error()}"
    pos_pts = read_positions(sim_pts, n_pts)

    same_count = (n_box == n_pts)
    identical = (pos_box == pos_pts)
    print(f"n_box={n_box} n_pts={n_pts} same_count={same_count}")
    print(f"positions identiques au bit pres (emission) = {identical}")
    if not identical:
        for i in range(min(len(pos_box), len(pos_pts))):
            if pos_box[i] != pos_pts[i]:
                print(f"  premiere divergence a l'index {i}: box={pos_box[i]!r} pts={pos_pts[i]!r}")
                break
    return sim_box, sim_pts, n_box, n_pts, identical and same_count


def max_abs_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def step2_after_frames(sim_box, sim_pts, n_box, n_pts):
    print()
    print("=== 2/2: apres 10 frames ===")
    for i in range(10):
        rc = lib.bq_step(sim_box, 1.0 / 24.0)
        assert rc >= 0, f"bq_step (box) failed: {lib.bq_last_error()}"
        rc = lib.bq_step(sim_pts, 1.0 / 24.0)
        assert rc >= 0, f"bq_step (pts) failed: {lib.bq_last_error()}"

    pos_box_after = read_positions(sim_box, n_box)
    pos_pts_after = read_positions(sim_pts, n_pts)
    identical = (pos_box_after == pos_pts_after)
    print(f"positions identiques au bit pres apres 10 frames = {identical}")
    if not identical:
        diff = max_abs_diff(pos_box_after, pos_pts_after)
        print(f"  ecart max abs (box vs points) = {diff:g}")

    # controle : deux executions independantes de la MEME voie (bq_emit_box)
    lo = (0.10, 0.10, 0.10)
    hi = (0.35, 0.60, 0.90)
    vel = f3(0.0, 0.0, 0.0)
    sim_box2, _ = make_sim()
    n_box2 = lib.bq_emit_box(sim_box2, 0, f3(*lo), f3(*hi), vel)
    assert n_box2 == n_box
    for i in range(10):
        rc = lib.bq_step(sim_box2, 1.0 / 24.0)
        assert rc >= 0
    pos_box2_after = read_positions(sim_box2, n_box2)
    identical_control = (pos_box_after == pos_box2_after)
    print(f"[controle] deux executions de bq_emit_box seul, identiques au bit pres = {identical_control}")
    if not identical_control:
        diff_control = max_abs_diff(pos_box_after, pos_box2_after)
        print(f"  [controle] ecart max abs (box vs box2) = {diff_control:g}")

    lib.bq_destroy(sim_box2)
    return identical


def step3_validation():
    print()
    print("=== 3/4: validations bq_emit_points ===")
    sim, cfg = make_sim()
    vel = f3(0.0, 0.0, 0.0)

    # point hors domaine
    pos_bad = (ct.c_float * 6)(0.5, 0.5, 0.5,  0.999, 0.5, 0.5)  # 2eme point dehors si bound*dx grand
    rc = lib.bq_emit_points(sim, 0, pos_bad, 2, vel)
    print(f"point hors domaine -> rc={rc} err={lib.bq_last_error().decode()!r}")
    assert rc == -1

    # count = 0
    pos_ok = (ct.c_float * 3)(0.5, 0.5, 0.5)
    rc = lib.bq_emit_points(sim, 0, pos_ok, 0, vel)
    print(f"count=0 -> rc={rc} err={lib.bq_last_error().decode()!r}")
    assert rc == -1

    # pos nul
    rc = lib.bq_emit_points(sim, 0, None, 1, vel)
    print(f"pos NULL -> rc={rc} err={lib.bq_last_error().decode()!r}")
    assert rc == -1

    # depassement de capacite
    small_cfg = BqConfig()
    lib.bq_default_config(ct.byref(small_cfg))
    small_cfg.max_particles = 4
    sim2 = lib.bq_create(ct.byref(small_cfg))
    mat = BqMaterial(model=BQ_MODEL_WATER, rho=1000.0, E=0.0, nu=0.0, bulk=4e4, gamma=3.0)
    mid = lib.bq_add_material(sim2, ct.byref(mat))
    many = [0.5, 0.5, 0.5] * 10
    many_arr = (ct.c_float * len(many))(*many)
    rc = lib.bq_emit_points(sim2, mid, many_arr, 10, vel)
    print(f"capacite depassee -> rc={rc} err={lib.bq_last_error().decode()!r}")
    assert rc == -1
    lib.bq_destroy(sim2)

    lib.bq_destroy(sim)
    print("toutes les validations rejettent correctement, sans crash.")


if __name__ == "__main__":
    sim_box, sim_pts, n_box, n_pts, ok1 = step1_equivalence()
    ok2 = step2_after_frames(sim_box, sim_pts, n_box, n_pts)
    lib.bq_destroy(sim_box)
    lib.bq_destroy(sim_pts)
    step3_validation()
    print()
    print(f"RESUME: equivalence a l'emission = {ok1}, apres 10 frames identiques = {ok2}")
