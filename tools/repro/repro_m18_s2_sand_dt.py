"""M18/S2 -- verifie que dt d'un materiau SAND est fini et vaut
sqrt((lam+2mu)/rho) (D4). C'est le mode d'echec de material_sound_speed avant
correctif : un sable de bulk=0 tombant dans le else "sinon WATER" donnait une
vitesse du son nulle, donc dt = cfl*dx/0 = +inf.

bq_step ne renseigne pas dt directement via l'API publique : on l'infere du
nombre de substeps retourne (dt_substep = frame_dt / substeps), et on compare
a la formule analytique. Comme le nombre de substeps est un entier (ceil),
l'egalite n'est qu'approximative -- borne large volontaire.
"""
import ctypes
import math

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
                ("bulk", ctypes.c_float), ("gamma", ctypes.c_float),
                ("friction_angle", ctypes.c_float), ("cohesion", ctypes.c_float)]


d.bq_default_config.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.restype = ctypes.c_void_p
d.bq_destroy.argtypes = [ctypes.c_void_p]
d.bq_add_material.argtypes = [ctypes.c_void_p, ctypes.POINTER(Mat)]
d.bq_add_material.restype = ctypes.c_int
d.bq_emit_box.argtypes = [ctypes.c_void_p, ctypes.c_int, F, F, F]
d.bq_emit_box.restype = ctypes.c_int
d.bq_step.argtypes = [ctypes.c_void_p, ctypes.c_float]
d.bq_step.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p

E, nu, rho = 3.5e5, 0.3, 1600.0
mu = E / (2.0 * (1.0 + nu))
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
c_expected = math.sqrt((lam + 2.0 * mu) / rho)
print(f"E={E} nu={nu} rho={rho} -> mu={mu:.3f} lam={lam:.3f} c_expected={c_expected:.6f} m/s")

cfg = Cfg()
d.bq_default_config(ctypes.byref(cfg))
sim = d.bq_create(ctypes.byref(cfg))
if not sim:
    raise SystemExit(d.bq_last_error().decode())

mat = Mat(model=2, rho=rho, E=E, nu=nu, bulk=0.0, gamma=0.0,
         friction_angle=35.0, cohesion=0.0)
mid = d.bq_add_material(sim, ctypes.byref(mat))
if mid < 0:
    raise SystemExit("bq_add_material a refuse un SAND legitime: " + d.bq_last_error().decode())

R, dx = cfg.grid_res[0], cfg.cell_size
L = R * dx
lo = (ctypes.c_float * 3)(0.3 * L, 0.3 * L, 0.3 * L)
hi = (ctypes.c_float * 3)(0.5 * L, 0.5 * L, 0.5 * L)
vel = (ctypes.c_float * 3)(0.0, 0.0, 0.0)
n_emit = d.bq_emit_box(sim, mid, lo, hi, vel)
if n_emit < 0:
    raise SystemExit(d.bq_last_error().decode())
print(f"emis {n_emit} particules SAND")

FRAME_DT = 1.0 / 30.0
substeps = d.bq_step(sim, FRAME_DT)
if substeps < 0:
    raise SystemExit(d.bq_last_error().decode())
if substeps == 0:
    raise SystemExit("FAIL: 0 substep -- dt probablement infini ou NaN")

dt_substep = FRAME_DT / substeps
c_max_implied = cfg.cfl * dx / dt_substep
print(f"substeps={substeps} dt_substep={dt_substep:.8f} s")
print(f"c_max implique par dt = cfl*dx/dt_substep = {c_max_implied:.6f} m/s "
      f"(attendu ~{c_expected:.6f} m/s, cfl={cfg.cfl})")

rel_err = abs(c_max_implied - c_expected) / c_expected
print(f"erreur relative = {rel_err * 100:.3f} %")

assert math.isfinite(dt_substep) and dt_substep > 0.0, "dt non fini"
# tolerance large : substeps est un entier (arrondi), + le plancher CCD
# (prev_frame_max_speed) peut faire deriver c_max vers le haut a partir de la
# 2e frame -- on ne teste que la 1ere frame, plancher = 0, comparaison directe
# valide.
assert rel_err < 0.05, f"c_max implique s'ecarte de plus de 5% de l'attendu ({rel_err*100:.2f}%)"
print("OK : dt fini et coherent avec sqrt((lam+2mu)/rho)")

d.bq_destroy(sim)
