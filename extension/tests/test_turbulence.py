"""test_turbulence.py — verification autonome de
`extension.ops._turbulent_emission` (turbulence a l'emission, jalon 4).

`ops.py` importe `bpy` en tete de module (idiome standard pour un operateur
Blender) et n'est donc pas importable tel quel hors Blender. Contrairement a
`display.py` (voir la docstring de `test_display_math.py`, qui contourne le
probleme en testant directement `transform.py`, module bpy-libre dont
`display.py` se contente d'importer une fonction), la logique pure qu'on
veut tester ici (`_turbulent_emission`) est volontairement placee DANS
`ops.py` (decision du jalon : elle vit a cote de `_emit_inflow_sites`, son
seul appelant cote inflow, et ne merite pas un module dedie pour une
fonction unique).

Pour l'importer sans Blender, ce module installe un stub MINIMAL de
`bpy`/`mathutils` dans `sys.modules` — juste assez pour que `ops.py` et
`extension.props` (dont il depend via `from .props import ...`) s'importent
sans erreur — puis enregistre un objet `extension` FACTICE dans
`sys.modules` (pointant, via `__path__`, sur le vrai dossier `extension/`)
avant d'importer `extension.ops` : ce contournement evite l'execution de
`extension/__init__.py`, qui tirerait aussi `overlay.py`/`display.py`/
`ui.py`, dependants de `gpu`/`bmesh`, hors de portee de ce test. Le stub ne
fournit AUCUNE fonctionnalite bpy reelle, seulement les symboles references
a l'IMPORT (annotations de PropertyGroup dans props.py, classes de base
d'operateurs, decorateur `bpy.app.handlers.persistent`).

Executable avec `python extension/tests/test_turbulence.py`.
"""

import os
import sys
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_EXT_DIR, ".."))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _install_bpy_stub():
    """Installe un `bpy`/`mathutils` minimal, seulement si un vrai `bpy`
    n'est pas deja present (ce test reste executable, sans modification,
    depuis un `blender --background --python`, ou le vrai module doit
    primer)."""
    if "bpy" in sys.modules:
        return

    bpy = types.ModuleType("bpy")

    bpy_types = types.ModuleType("bpy.types")

    class _StubBase:
        pass

    bpy_types.Operator = _StubBase
    bpy_types.PropertyGroup = _StubBase
    bpy_types.Object = _StubBase
    bpy_types.Scene = _StubBase
    bpy_types.Panel = _StubBase
    bpy_types.UIList = _StubBase

    bpy_props = types.ModuleType("bpy.props")

    def _prop(*_args, **_kwargs):
        return None

    for name in (
        "BoolProperty",
        "EnumProperty",
        "FloatProperty",
        "FloatVectorProperty",
        "IntProperty",
        "PointerProperty",
        "StringProperty",
    ):
        setattr(bpy_props, name, _prop)

    bpy_app_handlers = types.ModuleType("bpy.app.handlers")
    bpy_app_handlers.persistent = lambda fn: fn
    bpy_app_handlers.load_post = []

    bpy_app = types.ModuleType("bpy.app")
    bpy_app.handlers = bpy_app_handlers
    bpy_app.timers = types.SimpleNamespace(
        register=lambda *a, **k: None,
        unregister=lambda *a, **k: None,
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


def _import_ops():
    """Importe `extension.ops` sans declencher `extension/__init__.py`
    (voir docstring de module)."""
    _install_bpy_stub()

    if "extension" not in sys.modules:
        pkg = types.ModuleType("extension")
        pkg.__path__ = [_EXT_DIR]
        sys.modules["extension"] = pkg

    import extension.ops as ops_module  # noqa: E402

    return ops_module


ops = _import_ops()

_FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append((name, exc))
        print(f"[FAIL] {name}: {exc!r}")
    else:
        print(f"[ OK ] {name}")


# ---------------------------------------------------------------------------
# turbulence == 0 : passe-plat exact
# ---------------------------------------------------------------------------


def test_zero_turbulence_is_exact_passthrough():
    rng = np.random.default_rng(12345)
    points = np.array(
        [[0.1, 0.2, 0.3], [1.0, -1.0, 2.5], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    vel = (0.5, -0.25, 0.0)
    spacing = 0.02
    frame_dt = 1.0 / 24.0

    pos_out, vel_out = ops._turbulent_emission(
        points, vel, 0.0, spacing, frame_dt, rng
    )

    assert pos_out.dtype == np.float32
    assert vel_out.dtype == np.float32
    assert pos_out.shape == points.shape
    assert vel_out.shape == points.shape

    np.testing.assert_array_equal(pos_out, points.astype(np.float32))
    expected_vel = np.broadcast_to(
        np.array(vel, dtype=np.float32), points.shape
    )
    np.testing.assert_array_equal(vel_out, expected_vel)


def test_negative_turbulence_is_also_passthrough():
    # `turbulence` est cliffe a >= 0 cote UI (props.py, min=0.0), mais la
    # fonction pure doit rester sure face a une valeur negative accidentelle
    # (ex. appel direct depuis un test) : meme regle que 0 (`<= 0`).
    rng = np.random.default_rng(1)
    points = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    pos_out, vel_out = ops._turbulent_emission(
        points, (0.0, 0.0, 0.0), -0.5, 0.01, 1.0 / 24.0, rng
    )
    np.testing.assert_array_equal(pos_out, points)
    np.testing.assert_array_equal(vel_out, np.zeros((1, 3), dtype=np.float32))


# ---------------------------------------------------------------------------
# jitter positionnel : ne depasse jamais 0.5 * spacing (borne dure du jalon,
# voir docstring de `_turbulent_emission` : amp_pos = turbulence * 0.4 *
# spacing, turbulence borne a 1.0 cote UI)
# ---------------------------------------------------------------------------


def test_positional_jitter_never_exceeds_half_spacing():
    spacing = 0.0137
    frame_dt = 1.0 / 24.0
    dx = 0.005
    n = 5000
    points = np.zeros((n, 3), dtype=np.float32)
    vel = (0.0, 0.0, 0.0)

    max_abs_diff = 0.0
    for turbulence in (0.05, 0.1, 0.3, 0.5, 1.0):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            noise = ops._CurlNoise(
                seed, 0, points.min(axis=0), points.max(axis=0), dx, vel,
                spacing, frame_dt,
            )
            pos_out, _vel_out = ops._turbulent_emission(
                points, vel, turbulence, spacing, frame_dt, rng, dx, 0.0,
                noise,
            )
            diff = np.abs(pos_out - points)
            max_abs_diff = max(max_abs_diff, float(diff.max()))
            assert diff.max() < 0.5 * spacing, (
                f"jitter positionnel {diff.max():.6f} depasse 0.5*spacing "
                f"({0.5 * spacing:.6f}) pour turbulence={turbulence}, "
                f"seed={seed}"
            )
            # Borne CONCUE (0.4 * spacing, avec une petite tolerance
            # flottante) : plus stricte que 0.5 * spacing, verifie que
            # l'implementation respecte bien la formule documentee, pas
            # seulement l'exigence de non-regression du test d'occupation.
            assert diff.max() <= turbulence * 0.4 * spacing + 1e-6, (
                f"jitter positionnel {diff.max():.6f} depasse "
                f"turbulence*0.4*spacing ({turbulence * 0.4 * spacing:.6f})"
            )

    print(f"  (jitter max observe sur tous les essais : {max_abs_diff:.6f} m)")


def _make_noise(seed=1, emitter_index=0, dx=0.005, vel=(0.0, 0.0, -0.5),
                 spacing=0.02, frame_dt=1.0 / 24.0, lo=(0.0, 0.0, 0.0),
                 hi=(0.1, 0.1, 0.1)):
    """Petit `_CurlNoise` pret a l'emploi pour les tests qui n'exercent pas
    directement sa construction (jitter positionnel, plancher v_ref)."""
    return ops._CurlNoise(seed, emitter_index, lo, hi, dx, vel, spacing, frame_dt)


def test_velocity_noise_uses_reference_floor():
    """A vitesse initiale nulle, le bruit de vitesse doit quand meme etre
    non nul des que `turbulence > 0` (plancher `spacing/frame_dt`, voir
    docstring) : sinon le reglage serait inoperant sur un emetteur a
    vitesse nulle, exactement le defaut que `v_ref` est cense corriger."""
    spacing = 0.02
    frame_dt = 1.0 / 24.0
    dx = 0.005
    rng = np.random.default_rng(7)
    points = np.random.default_rng(3).uniform(0.0, 0.1, size=(2000, 3)).astype(np.float32)
    noise = _make_noise(seed=7, dx=dx, vel=(0.0, 0.0, 0.0), spacing=spacing,
                         frame_dt=frame_dt)
    _pos_out, vel_out = ops._turbulent_emission(
        points, (0.0, 0.0, 0.0), 0.2, spacing, frame_dt, rng, dx, 0.0, noise
    )
    assert np.std(vel_out) > 1e-6


# ---------------------------------------------------------------------------
# _CurlNoise : coherence spatiale du champ (points proches -> vitesses
# proches, points distants de plusieurs L_noise -> vitesses decorrelees)
# ---------------------------------------------------------------------------


def test_curl_noise_spatial_coherence():
    dx = 0.01
    l_noise = 4.0 * dx
    noise = _make_noise(
        seed=5, dx=dx, vel=(0.0, 0.0, 0.0), spacing=0.005,
        frame_dt=1.0 / 24.0, lo=(0.0, 0.0, 0.0), hi=(1.0, 1.0, 1.0),
    )

    rng = np.random.default_rng(0)
    base = rng.uniform(0.2, 0.8, size=(300, 3))
    # deplacement << L_noise : le champ doit rester quasi identique.
    close = base + rng.normal(0.0, 0.05 * l_noise, size=base.shape)
    # deplacement de plusieurs L_noise sur un axe : le champ doit devenir
    # decorrele (tire d'une cellule grossiere differente).
    far = base + np.array([3.0 * l_noise, 0.0, 0.0])

    v_base = noise.sample(base, 0.0)
    v_close = noise.sample(close, 0.0)
    v_far = noise.sample(far, 0.0)

    def mean_cosine(a, b):
        na = np.linalg.norm(a, axis=1)
        nb = np.linalg.norm(b, axis=1)
        mask = (na > 1e-9) & (nb > 1e-9)
        cos = np.sum(a[mask] * b[mask], axis=1) / (na[mask] * nb[mask])
        return float(np.mean(cos))

    cos_close = mean_cosine(v_base, v_close)
    cos_far = mean_cosine(v_base, v_far)

    print(f"  (cos_close={cos_close:.3f}, cos_far={cos_far:.3f})")
    assert cos_close > 0.9, (
        f"deux points separes de << L_noise devraient recevoir des "
        f"vitesses fortement correlees, cos moyen={cos_close:.3f}"
    )
    assert cos_far < 0.3, (
        f"deux points separes de plusieurs L_noise devraient recevoir des "
        f"vitesses decorrelees, cos moyen={cos_far:.3f}"
    )


def main():
    check(
        "turbulence == 0 : passe-plat exact",
        test_zero_turbulence_is_exact_passthrough,
    )
    check(
        "turbulence < 0 : passe-plat exact (garde-fou)",
        test_negative_turbulence_is_also_passthrough,
    )
    check(
        "jitter positionnel borne (< 0.5*spacing, <= 0.4*spacing*turbulence)",
        test_positional_jitter_never_exceeds_half_spacing,
    )
    check(
        "bruit de vitesse non nul a vitesse initiale nulle (plancher v_ref)",
        test_velocity_noise_uses_reference_floor,
    )
    check(
        "_CurlNoise : coherence spatiale (proche correle, lointain decorrele)",
        test_curl_noise_spatial_coherence,
    )

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) EN ECHEC.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
