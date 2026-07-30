"""Mesure 1 : le curl noise REEL (extension.ops._CurlNoise) survit-il au
filtrage P2G du solveur, comme coherent.py l'avait mesure pour son
implementation de reference ?

Reprend exactement la methode de coherent.py (ecart-type de vitesse
residuel post-P2G, sigma=1.0 m/s injecte, frames 1/3/6) mais echantillonne
la vitesse via l'implementation de PRODUCTION (ops._CurlNoise.sample),
importee via le meme stub bpy minimal que extension/tests/test_turbulence.py.
"""
import os
import sys
import types

import numpy as np

REPO = r"C:\Users\nicol\Code\bourrasque_v2"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "extension"))


def _install_bpy_stub():
    if "bpy" in sys.modules:
        return
    bpy = types.ModuleType("bpy")
    bpy_types = types.ModuleType("bpy.types")

    class _StubBase:
        pass

    for name in ("Operator", "PropertyGroup", "Object", "Scene", "Panel", "UIList"):
        setattr(bpy_types, name, _StubBase)

    bpy_props = types.ModuleType("bpy.props")
    for name in ("BoolProperty", "EnumProperty", "FloatProperty",
                 "FloatVectorProperty", "IntProperty", "PointerProperty",
                 "StringProperty"):
        setattr(bpy_props, name, lambda *a, **k: None)

    bpy_app_handlers = types.ModuleType("bpy.app.handlers")
    bpy_app_handlers.persistent = lambda fn: fn
    bpy_app_handlers.load_post = []
    bpy_app = types.ModuleType("bpy.app")
    bpy_app.handlers = bpy_app_handlers
    bpy_app.timers = types.SimpleNamespace(
        register=lambda *a, **k: None, unregister=lambda *a, **k: None,
        is_registered=lambda *a, **k: False,
    )
    bpy_utils = types.ModuleType("bpy.utils")
    bpy_utils.register_class = lambda cls: None
    bpy_utils.unregister_class = lambda cls: None

    bpy.types = bpy_types
    bpy.props = bpy_props
    bpy.app = bpy_app
    bpy.utils = bpy_utils
    bpy.ops = types.SimpleNamespace()

    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = lambda *a, **k: None

    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy_types
    sys.modules["bpy.props"] = bpy_props
    sys.modules["bpy.app"] = bpy_app
    sys.modules["bpy.app.handlers"] = bpy_app_handlers
    sys.modules["bpy.utils"] = bpy_utils
    sys.modules["mathutils"] = mathutils


_install_bpy_stub()
EXT_DIR = os.path.join(REPO, "extension")
if "extension" not in sys.modules:
    pkg = types.ModuleType("extension")
    pkg.__path__ = [EXT_DIR]
    sys.modules["extension"] = pkg
import extension.ops as ops  # noqa: E402
import lib  # noqa: E402


RES, PPC, DOM = 64, 2, 4.0
DX = DOM / RES
SPACING = DX / PPC
DT = 1.0 / 24.0

cfg = lib.default_config()
cfg.grid_res, cfg.ppc_axis, cfg.domain = RES, PPC, DOM
cfg.cfl, cfg.gravity_y = 0.3, 0.0

ax = SPACING / 2 + np.arange(8, 24) * SPACING + 1.5
gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
PTS = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1).astype(np.float32)

for mult in (2, 4, 8):
    L = mult * DX
    lo = PTS.min(axis=0).astype(np.float64) - L
    hi = PTS.max(axis=0).astype(np.float64) + L
    # Construit un _CurlNoise directement avec cell = mult*dx (au lieu de
    # 4*dx code en dur) pour comparer les 3 echelles, comme coherent.py.
    noise = ops._CurlNoise(seed=0, emitter_index=0, lo=lo, hi=hi, dx=mult * DX / 4.0,
                            vel=(0.0, 0.0, 0.0), spacing=SPACING, frame_dt=DT)
    v = noise.sample(PTS.astype(np.float64), 0.0)
    v = v / np.std(v)  # renormalise a sigma = 1.0 m/s injecte (deja ~1 mais par securite)

    with lib.Sim(cfg) as sim:
        m = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
        sim.emit_points_vel(m, PTS, v.astype(np.float32))
        prev = sim.read_positions().copy()
        kept = []
        for k in range(6):
            sim.step(DT)
            cur = sim.read_positions()
            d_ = (cur - prev) / DT
            kept.append(float(np.std(d_ - d_.mean(0))))
            prev = cur.copy()
    print(f"curl PRODUCTION L={mult}*dx : f1={kept[0]:.4f}  f3={kept[2]:.4f}  f6={kept[5]:.4f} m/s")
