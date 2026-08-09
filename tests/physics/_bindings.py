"""Harnais ctypes partage pour la suite d'invariants physiques (M12).

Factorise le setup ctypes duplique dans quasiment tous les scripts de
diagnostic ecrits pendant la session precedente (tools/repro/diag_*.py,
scratchpad) : structs Cfg/Mat, declaration des prototypes, plus quelques
primitives communes (masse totale, energie mecanique, volume total) pour ne
pas les redefinir dans chaque fichier de test.

N'importe pas bpy : cette suite pilote directement bourrasque.dll, independante
de Blender (cf. docs/plan-milestone-12.md, D1).
"""
import ctypes
import os

import numpy as np

_DLL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "build", "Release", "bourrasque.dll")
_DLL_PATH = os.path.normpath(_DLL_PATH)

F = ctypes.POINTER(ctypes.c_float)
I = ctypes.POINTER(ctypes.c_int)
U8 = ctypes.POINTER(ctypes.c_uint8)


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
    ]


BQ_MODEL_ELASTIC = 0
BQ_MODEL_WATER = 1


def load_dll():
    """Charge bourrasque.dll et declare les prototypes necessaires a cette
    suite. Un appel par processus de test suffit (ctypes met en cache le
    module par chemin), mais chaque appelant peut le refaire sans cout reel."""
    d = ctypes.CDLL(_DLL_PATH)
    for name, argt, rest in (
            ("bq_abi_version", [], ctypes.c_int),
            ("bq_default_config", [ctypes.POINTER(BqConfig)], None),
            ("bq_create", [ctypes.POINTER(BqConfig)], ctypes.c_void_p),
            ("bq_destroy", [ctypes.c_void_p], None),
            ("bq_add_material", [ctypes.c_void_p, ctypes.POINTER(BqMaterial)], ctypes.c_int),
            ("bq_emit_box", [ctypes.c_void_p, ctypes.c_int, F, F, F], ctypes.c_int),
            ("bq_step", [ctypes.c_void_p, ctypes.c_float], ctypes.c_int),
            ("bq_set_colliders", [ctypes.c_void_p, F, F, F, ctypes.c_int], ctypes.c_int),
            ("bq_particle_count", [ctypes.c_void_p], ctypes.c_int),
            ("bq_read_positions", [ctypes.c_void_p, F], ctypes.c_int),
            ("bq_read_velocities", [ctypes.c_void_p, F], ctypes.c_int),
            ("bq_read_materials", [ctypes.c_void_p, U8], ctypes.c_int),
            ("bq_read_J", [ctypes.c_void_p, F], ctypes.c_int),
    ):
        fn = getattr(d, name)
        fn.argtypes = argt
        if rest is not None:
            fn.restype = rest
    d.bq_last_error.restype = ctypes.c_char_p
    return d


def create_sim(d, grid_res=None, cell_size=None, ppc_axis=None, gravity_y=None,
               max_particles=None):
    """Cree une sim avec la config par defaut, en ecrasant seulement les
    champs fournis. Retourne (sim, cfg) -- cfg reste accessible pour deriver
    spacing/p_vol/p_mass cote appelant."""
    cfg = BqConfig()
    d.bq_default_config(ctypes.byref(cfg))
    if grid_res is not None:
        cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = grid_res
    if cell_size is not None:
        cfg.cell_size = ctypes.c_float(cell_size)
    if ppc_axis is not None:
        cfg.ppc_axis = ppc_axis
    if gravity_y is not None:
        cfg.gravity_y = ctypes.c_float(gravity_y)
    if max_particles is not None:
        cfg.max_particles = max_particles
    sim = d.bq_create(ctypes.byref(cfg))
    if not sim:
        raise RuntimeError(f"bq_create refuse: {d.bq_last_error().decode()}")
    return sim, cfg


def add_water(d, sim, rho=1000.0, bulk=4.0e4, gamma=3.0):
    """Enregistre un materiau WATER standard (memes valeurs que partout
    ailleurs cette session) et retourne son id."""
    mat = BqMaterial(model=BQ_MODEL_WATER, rho=rho, E=0.0, nu=0.0,
                      bulk=bulk, gamma=gamma)
    mat_id = d.bq_add_material(sim, ctypes.byref(mat))
    if mat_id < 0:
        raise RuntimeError(f"bq_add_material refuse: {d.bq_last_error().decode()}")
    return mat_id, rho


def particle_mass(cfg, rho):
    """p_mass = rho * p_vol, p_vol = spacing^3, spacing = cell_size/ppc_axis --
    memes formules que core/src/mlsmpm.cu (bq_add_material). Constant par
    materiau, jamais un champ par particule (cf. plan M12, constat 3)."""
    spacing = cfg.cell_size / cfg.ppc_axis
    p_vol = spacing ** 3
    return rho * p_vol


def emit_box(d, sim, mat_id, lo, hi, vel=(0.0, 0.0, 0.0)):
    n = d.bq_emit_box(sim, mat_id,
                       (ctypes.c_float * 3)(*lo), (ctypes.c_float * 3)(*hi),
                       (ctypes.c_float * 3)(*vel))
    if n < 0:
        raise RuntimeError(f"bq_emit_box refuse: {d.bq_last_error().decode()}")
    return n


def read_positions(d, sim, n):
    buf = np.empty(n * 3, dtype=np.float32)
    d.bq_read_positions(sim, buf.ctypes.data_as(F))
    return buf.reshape(-1, 3)


def read_velocities(d, sim, n):
    buf = np.empty(n * 3, dtype=np.float32)
    d.bq_read_velocities(sim, buf.ctypes.data_as(F))
    return buf.reshape(-1, 3)


def read_J(d, sim, n):
    buf = np.empty(n, dtype=np.float32)
    d.bq_read_J(sim, buf.ctypes.data_as(F))
    return buf


def read_materials(d, sim, n):
    buf = np.empty(n, dtype=np.uint8)
    d.bq_read_materials(sim, buf.ctypes.data_as(U8))
    return buf


def mechanical_energy(vel, pos, p_mass, gravity_y, y0=0.0):
    """Energie mecanique = cinetique + potentielle de gravite, p_mass constant
    (pas un champ par particule, cf. particle_mass). y0 : origine de hauteur
    arbitraire (seules les VARIATIONS d'energie potentielle comptent) --
    utiliser une valeur fixe et coherente sur tout un run, jamais recalculee
    par frame. N'inclut PAS l'energie interne EOS (cf. plan M12, constat 5)."""
    ke = 0.5 * p_mass * np.sum(vel ** 2)
    pe = p_mass * (-gravity_y) * np.sum(pos[:, 1] - y0)
    return float(ke + pe)


def total_volume(J, p_vol):
    """Volume total = somme(V0 * J_i), V0 = p_vol constant par materiau."""
    return float(np.sum(J.astype(np.float64)) * p_vol)


def total_momentum(vel, p_mass):
    return p_mass * np.sum(vel, axis=0).astype(np.float64)
