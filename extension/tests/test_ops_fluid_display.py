"""test_ops_fluid_display.py — verification autonome de
`extension.ops._fluid_display_material` (detection d'idempotence de
`BQ_OT_setup_fluid_display`) et de `extension.ops._fluid_asset_path`
(resolution du chemin de l'asset `assets/fluid_display.blend`, relatif au
fichier de l'extension — meme motif que `lib._DLL_PATH`).

Meme contournement que `test_turbulence.py` (voir sa docstring) : `ops.py`
importe `bpy` en tete de module, ce script installe donc un stub MINIMAL de
`bpy`/`mathutils` dans `sys.modules` avant d'importer `extension.ops`, sans
declencher `extension/__init__.py`.

`BQ_OT_setup_fluid_display.execute` lui-meme (le corps de l'operateur) n'est
PAS exerce ici : il appelle `bpy.data.libraries.load`, hors de portee d'un
stub minimal. Seule la logique pure qu'il delegue —
`_fluid_display_material` (idempotence) et `_fluid_asset_path` (resolution
de chemin) — est testee directement, avec un mock d'objet pour la premiere.

Executable avec `python extension/tests/test_ops_fluid_display.py`.
"""

import os
import sys
import types

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
        "BoolVectorProperty",
        "CollectionProperty",
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

    bpy_data_libraries = types.SimpleNamespace(load=None)
    bpy_data = types.SimpleNamespace(libraries=bpy_data_libraries)
    bpy.data = bpy_data

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
# Mocks minimaux pour `_fluid_display_material`
# ---------------------------------------------------------------------------


class _MockMaterial:
    def __init__(self, name):
        self.name = name


class _MockMeshData:
    def __init__(self, materials):
        self.materials = materials


class _MockObject:
    def __init__(self, materials):
        self.data = _MockMeshData(materials)


# ---------------------------------------------------------------------------
# Idempotence : `_fluid_display_material`
# ---------------------------------------------------------------------------


def test_no_materials_returns_none():
    obj = _MockObject([])
    assert ops._fluid_display_material(obj) is None


def test_unrelated_materials_return_none():
    obj = _MockObject([_MockMaterial("Some Other Material")])
    assert ops._fluid_display_material(obj) is None


def test_none_slot_is_ignored():
    obj = _MockObject([None, _MockMaterial("Some Other Material")])
    assert ops._fluid_display_material(obj) is None


def test_matching_material_is_detected():
    target = _MockMaterial(ops._FLUID_ASSET_MATERIAL)
    obj = _MockObject([_MockMaterial("Other"), target])
    found = ops._fluid_display_material(obj)
    assert found is target


def test_matching_material_detected_among_several_slots():
    other = _MockMaterial("Unrelated")
    target = _MockMaterial(ops._FLUID_ASSET_MATERIAL)
    obj = _MockObject([other, target])
    found = ops._fluid_display_material(obj)
    assert found is target


# ---------------------------------------------------------------------------
# Resolution du chemin de l'asset
# ---------------------------------------------------------------------------


def test_asset_path_is_relative_to_extension_dir():
    path = ops._fluid_asset_path()
    expected = os.path.join(_EXT_DIR, "assets", "fluid_display.blend")
    assert os.path.normcase(os.path.normpath(path)) == os.path.normcase(
        os.path.normpath(expected)
    ), (path, expected)


def test_asset_path_points_to_an_existing_file():
    # L'asset est livre avec l'extension (construit hors-ligne, voir
    # docstring de `ops._fluid_asset_path`) : sa presence sur disque est une
    # garantie du depot, pas seulement de la resolution de chemin.
    path = ops._fluid_asset_path()
    assert os.path.isfile(path), f"asset manquant : {path}"


def main():
    check("aucun materiau -> None", test_no_materials_returns_none)
    check(
        "materiaux non lies au materiau cible -> None",
        test_unrelated_materials_return_none,
    )
    check("slot vide (None) ignore -> None", test_none_slot_is_ignored)
    check(
        "materiau deja assigne -> detecte (idempotence)",
        test_matching_material_is_detected,
    )
    check(
        "materiau cible detecte parmi plusieurs slots",
        test_matching_material_detected_among_several_slots,
    )
    check(
        "chemin de l'asset relatif au dossier de l'extension",
        test_asset_path_is_relative_to_extension_dir,
    )
    check(
        "chemin de l'asset pointe vers un fichier existant",
        test_asset_path_points_to_an_existing_file,
    )

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) EN ECHEC.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
