"""Tests 2, 3, 4 de la verification du domaine rectangulaire (M4).

Utilise ctypes directement sur bourrasque.dll, comme le fera l'extension.
"""
import ctypes as C
import sys

DLL_PATH = r"C:\Users\nicol\Code\bourrasque_v2\build\Release\bourrasque.dll"

class BqConfig(C.Structure):
    _fields_ = [
        ("grid_res", C.c_int * 3),
        ("cell_size", C.c_float),
        ("gravity_y", C.c_float),
        ("cfl", C.c_float),
        ("ppc_axis", C.c_int),
        ("max_particles", C.c_int),
    ]

class BqMaterial(C.Structure):
    _fields_ = [
        ("model", C.c_int),
        ("rho", C.c_float),
        ("E", C.c_float),
        ("nu", C.c_float),
        ("bulk", C.c_float),
        ("gamma", C.c_float),
    ]

BQ_MODEL_ELASTIC = 0
BQ_MODEL_WATER = 1

lib = C.CDLL(DLL_PATH)
lib.bq_default_config.argtypes = [C.POINTER(BqConfig)]
lib.bq_create.argtypes = [C.POINTER(BqConfig)]
lib.bq_create.restype = C.c_void_p
lib.bq_destroy.argtypes = [C.c_void_p]
lib.bq_add_material.argtypes = [C.c_void_p, C.POINTER(BqMaterial)]
lib.bq_emit_box.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float),
                             C.POINTER(C.c_float), C.POINTER(C.c_float)]
lib.bq_step.argtypes = [C.c_void_p, C.c_float]
lib.bq_particle_count.argtypes = [C.c_void_p]
lib.bq_read_positions.argtypes = [C.c_void_p, C.POINTER(C.c_float)]
lib.bq_last_error.restype = C.c_char_p

def f3(*vals):
    return (C.c_float * 3)(*vals)

def make_water(lib, sim):
    mat = BqMaterial(model=BQ_MODEL_WATER, rho=1000.0, E=0, nu=0, bulk=4.0e4, gamma=3.0)
    mid = lib.bq_add_material(sim, C.byref(mat))
    assert mid >= 0, lib.bq_last_error()
    return mid

# ---------------------------------------------------------------- test 2 & 4
def test_walls_and_clamp():
    cfg = BqConfig()
    lib.bq_default_config(C.byref(cfg))
    # domaine allonge 4:1:1 : 128x32x32 cellules, dx = 1/32 -> 4m x 1m x 1m
    cfg.grid_res = (C.c_int * 3)(128, 32, 32)
    cfg.cell_size = 1.0 / 32.0
    cfg.max_particles = 2_000_000

    sim = lib.bq_create(C.byref(cfg))
    assert sim, "bq_create a echoue"
    mid = make_water(lib, sim)

    dx = cfg.cell_size
    bound = 3  # cf. upload_params (p.bound = 3), en cellules
    lo_valid = bound * dx
    hi_x = cfg.grid_res[0] * dx - lo_valid
    hi_y = cfg.grid_res[1] * dx - lo_valid
    hi_z = cfg.grid_res[2] * dx - lo_valid
    print(f"domaine : {cfg.grid_res[0]*dx:.3f} x {cfg.grid_res[1]*dx:.3f} x {cfg.grid_res[2]*dx:.3f} m, "
          f"dx={dx:.5f}, bornes valides x[{lo_valid:.4f},{hi_x:.4f}] "
          f"y[{lo_valid:.4f},{hi_y:.4f}] z[{lo_valid:.4f},{hi_z:.4f}]")

    # --- test 2 : bloc d'eau au repos remplissant presque tout le domaine
    margin = 4 * dx  # marge de securite pour rester dans les bornes valides
    lo = f3(lo_valid + margin, lo_valid + margin, lo_valid + margin)
    hi = f3(hi_x - margin, hi_y - margin, hi_z - margin)
    vel = f3(0, 0, 0)
    n = lib.bq_emit_box(sim, mid, lo, hi, vel)
    assert n > 0, lib.bq_last_error()
    print(f"test2 : {n} particules emises au repos")

    # 60 frames a 24fps
    for fr in range(60):
        sub = lib.bq_step(sim, C.c_float(1.0/24.0))
        assert sub >= 0, lib.bq_last_error()

    n = lib.bq_particle_count(sim)
    pos = (C.c_float * (n*3))()
    lib.bq_read_positions(sim, pos)
    xs = pos[0::3]; ys = pos[1::3]; zs = pos[2::3]

    # marge de tolerance : la couche limite (bound*dx) est la zone ou v=0 est impose
    print("test2 (parois par axe) :")
    faces = {
        "x-": (min(xs), lo_valid),
        "x+": (max(xs), hi_x),
        "y-": (min(ys), lo_valid),
        "y+": (max(ys), hi_y),
        "z-": (min(zs), lo_valid),
        "z+": (max(zs), hi_z),
    }
    all_ok = True
    for name, (val, bound_val) in faces.items():
        if name.endswith("-"):
            ok = val >= bound_val - 1e-4
        else:
            ok = val <= bound_val + 1e-4
        all_ok &= ok
        print(f"  face {name}: extremum={val:.5f} borne={bound_val:.5f} -> {'OK' if ok else 'FUITE'}")
    print(f"test2 resultat global : {'OK' if all_ok else 'ECHEC'}")

    lib.bq_destroy(sim)
    return all_ok

# ---------------------------------------------------------------------- test 4
def test_clamp_high_speed():
    cfg = BqConfig()
    lib.bq_default_config(C.byref(cfg))
    cfg.grid_res = (C.c_int * 3)(128, 32, 32)
    cfg.cell_size = 1.0 / 32.0
    cfg.max_particles = 2_000_000
    sim = lib.bq_create(C.byref(cfg))
    assert sim
    mid = make_water(lib, sim)

    dx = cfg.cell_size
    bound = 3
    lo_valid = bound * dx
    hi_x = cfg.grid_res[0] * dx - lo_valid
    hi_y = cfg.grid_res[1] * dx - lo_valid
    hi_z = cfg.grid_res[2] * dx - lo_valid

    # positions distinctes (>= 0.4 m d'ecart sur au moins un axe) pour eviter
    # tout partage de stencil de grille (support 3x3x3, ~3*dx = 0.094 m) entre
    # particules de vitesses opposees, qui annulerait sinon leur quantite de
    # mouvement sur les noeuds communs.
    pos = (C.c_float * 18)(
        2.0, 0.3, 0.3,   # -> +x
        2.0, 0.3, 0.7,   # -> -x
        1.0, 0.5, 0.3,   # -> +y
        1.0, 0.5, 0.7,   # -> -y
        3.0, 0.3, 0.5,   # -> +z
        3.0, 0.7, 0.5,   # -> -z
    )
    vel = (C.c_float * 18)(
         50, 0, 0,
        -50, 0, 0,
         0, 50, 0,
         0,-50, 0,
         0, 0, 50,
         0, 0,-50,
    )
    lib.bq_emit_points_vel = lib.bq_emit_points_vel
    lib.bq_emit_points_vel.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float),
                                        C.POINTER(C.c_float), C.c_int]
    n = lib.bq_emit_points_vel(sim, mid, pos, vel, 6)
    assert n == 6, lib.bq_last_error()

    # on suit l'extremum atteint sur toutes les frames plutot que la seule
    # position finale : avec gravite active, une particule qui a bien touche
    # la paroi haute peut retomber sous l'effet de g avant la derniere frame.
    # Ce qui nous interesse ici est la borne effectivement atteinte (mesure
    # directe du clamp par axe), pas l'etat final apres un long temps de vol.
    labels = ["+x", "-x", "+y", "-y", "+z", "-z"]
    extremes = [None, None, None, None, None, None]  # max ou min selon signe
    out = (C.c_float * 18)()
    for fr in range(20):
        sub = lib.bq_step(sim, C.c_float(1.0/24.0))
        assert sub >= 0, lib.bq_last_error()
        lib.bq_read_positions(sim, out)
        for i, comp in enumerate([0, 0, 1, 1, 2, 2]):
            v = out[3*i + comp]
            if i % 2 == 0:  # direction positive : on garde le max
                extremes[i] = v if extremes[i] is None else max(extremes[i], v)
            else:  # direction negative : on garde le min
                extremes[i] = v if extremes[i] is None else min(extremes[i], v)

    print("test4 (clamp par axe, vitesse elevee, extremum sur 20 frames) :")
    ok_all = True
    # tolerance = 1 cellule : l'axe y est freine par la gravite entre deux
    # frames echantillonnees, le pic exact peut donc tomber entre deux
    # lectures de position (pas de sous-echantillonnage a chaque substep ici).
    tol = dx
    expected = [
        (hi_x, "x"), (lo_valid, "x"),
        (hi_y, "y"), (lo_valid, "y"),
        (hi_z, "z"), (lo_valid, "z"),
    ]
    for i, (bound_val, axis) in enumerate(expected):
        val = extremes[i]
        ok = abs(val - bound_val) < tol
        ok_all &= ok
        print(f"  particule {labels[i]}: extremum axe {axis} = {val:.5f}, attendu {bound_val:.5f} -> {'OK' if ok else 'ECHEC'}")
    print(f"test4 resultat global : {'OK' if ok_all else 'ECHEC'}")

    lib.bq_destroy(sim)
    return ok_all

# ---------------------------------------------------------------------- test 3
def test_memory():
    cfg_rect = BqConfig()
    lib.bq_default_config(C.byref(cfg_rect))
    cfg_rect.grid_res = (C.c_int * 3)(128, 32, 32)
    cfg_rect.cell_size = 1.0/32.0

    cfg_cube = BqConfig()
    lib.bq_default_config(C.byref(cfg_cube))
    cfg_cube.grid_res = (C.c_int * 3)(128, 128, 128)  # cube englobant
    cfg_cube.cell_size = 1.0/32.0

    ncell_rect = cfg_rect.grid_res[0]*cfg_rect.grid_res[1]*cfg_rect.grid_res[2]
    ncell_cube = cfg_cube.grid_res[0]*cfg_cube.grid_res[1]*cfg_cube.grid_res[2]
    ratio = ncell_cube / ncell_rect
    print(f"test3 (memoire) : ncell rectangulaire (128x32x32) = {ncell_rect}, "
          f"ncell cube englobant (128^3) = {ncell_cube}, ratio = {ratio:.2f}x")
    return ncell_rect, ncell_cube, ratio


if __name__ == "__main__":
    ncell_rect, ncell_cube, ratio = test_memory()
    ok2 = test_walls_and_clamp()
    ok4 = test_clamp_high_speed()
    print()
    print("RESUME")
    print(f"  test2 (parois 6 faces) : {'OK' if ok2 else 'ECHEC'}")
    print(f"  test3 (memoire)        : {ncell_rect} vs {ncell_cube} (ratio {ratio:.2f}x)")
    print(f"  test4 (clamp par axe)  : {'OK' if ok4 else 'ECHEC'}")
    sys.exit(0 if (ok2 and ok4) else 1)
