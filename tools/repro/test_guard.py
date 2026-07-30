import ctypes as C

dll = C.CDLL(r"C:\Users\nicol\Code\bourrasque_v2\build\Release\bourrasque.dll")

class BqConfig(C.Structure):
    _fields_ = [("grid_res", C.c_int), ("domain", C.c_float),
                ("gravity_y", C.c_float), ("cfl", C.c_float),
                ("ppc_axis", C.c_int), ("max_particles", C.c_int)]

class BqMaterial(C.Structure):
    _fields_ = [("model", C.c_int), ("rho", C.c_float), ("E", C.c_float),
                ("nu", C.c_float), ("bulk", C.c_float), ("gamma", C.c_float)]

dll.bq_default_config.argtypes = [C.POINTER(BqConfig)]
dll.bq_create.restype = C.c_void_p
dll.bq_create.argtypes = [C.POINTER(BqConfig)]
dll.bq_add_material.argtypes = [C.c_void_p, C.POINTER(BqMaterial)]
dll.bq_emit_box.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float),
                             C.POINTER(C.c_float), C.POINTER(C.c_float)]
dll.bq_step.argtypes = [C.c_void_p, C.c_float]
dll.bq_step.restype = C.c_int
dll.bq_last_error.restype = C.c_char_p
dll.bq_destroy.argtypes = [C.c_void_p]

cfg = BqConfig()
dll.bq_default_config(C.byref(cfg))
sim = dll.bq_create(C.byref(cfg))
assert sim, "bq_create failed"

mat = BqMaterial(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=5e4, gamma=7.0)
mid = dll.bq_add_material(sim, C.byref(mat))
print("mat_id =", mid)

def farr(vals):
    return (C.c_float * 3)(*vals)

# --- Cas 1 : boite volontairement hors domaine ---
lo = farr([-0.5, -0.5, -0.5])
hi = farr([0.2, 0.2, 0.2])
vel = farr([0.0, 0.0, 0.0])
ret = dll.bq_emit_box(sim, mid, lo, hi, vel)
print("cas 1 (hors domaine) -> ret =", ret)
print("  message :", dll.bq_last_error().decode())
assert ret == -1, "devrait refuser la boite hors domaine"

# --- Cas 1b : boite degeneree lo >= hi ---
lo2 = farr([0.4, 0.4, 0.4])
hi2 = farr([0.4, 0.6, 0.6])
ret2 = dll.bq_emit_box(sim, mid, lo2, hi2, vel)
print("cas 1b (degeneree) -> ret =", ret2)
print("  message :", dll.bq_last_error().decode())
assert ret2 == -1, "devrait refuser la boite degeneree"

# --- Cas 2 : emission valide, simulation qui pourrait pousser des
#             particules hors domaine pendant le pas de temps ---
lo3 = farr([0.3, 0.3, 0.3])
hi3 = farr([0.6, 0.6, 0.6])
vel3 = farr([5.0, 5.0, 5.0])  # vitesse initiale agressive vers le coin
ret3 = dll.bq_emit_box(sim, mid, lo3, hi3, vel3)
print("cas 2 (valide) -> particules emises =", ret3)
assert ret3 > 0

for f in range(30):
    r = dll.bq_step(sim, 1.0/24.0)
    if r < 0:
        print("bq_step a echoue a la frame", f, ":", dll.bq_last_error().decode())
        break
else:
    print("30 frames simulees sans crash, bq_step retourne toujours >= 0")

dll.bq_destroy(sim)
print("OK: pas de crash, contexte CUDA detruit proprement")
