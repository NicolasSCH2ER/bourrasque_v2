"""test_mesh_ergonomics.py — verification autonome de l'ergonomie M7.1 du
mailleur (`extension.props.mesh_particle_spacing`,
`extension.props.mesh_effective_radii`, `extension.props.mesh_resolution_state`,
`extension.props.mesh_layout` en mode automatique) : les rayons du mailleur
sont desormais des FACTEURS de l'espacement inter-particules, et
`mesh_resolution == 0` declenche une resolution automatique plafonnee en
VRAM (voir CLAUDE.md / spec de la tache).

Meme discipline que `test_domain_transform.py` : `props.py` importe
`bpy`/`mathutils` en tete de module (idiome bpy standard), donc ce script
installe un stub MINIMAL de `bpy` dans `sys.modules` avant d'importer
`extension.props`, pour rester executable sans Blender.

Executable avec `python extension/tests/test_mesh_ergonomics.py`.
"""

import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_EXT_DIR, ".."))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Stub bpy/mathutils minimal, avec un Vector/Matrice REELS (voir
# test_domain_transform.py, meme contournement)
# ---------------------------------------------------------------------------


class _Vector:
    __slots__ = ("x", "y", "z")

    def __init__(self, xyz):
        self.x, self.y, self.z = xyz

    def __iter__(self):
        return iter((self.x, self.y, self.z))


class _AffineMatrix:
    """Matrice affine minimale : translation + mise a l'echelle diagonale,
    suffisante pour placer un cube domaine (`primitive_cube_add` + `scale` +
    `location`, sans rotation)."""

    def __init__(self, translation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)):
        self.translation = translation
        self.scale = scale

    def __matmul__(self, v):
        tx, ty, tz = self.translation
        sx, sy, sz = self.scale
        return _Vector((v.x * sx + tx, v.y * sy + ty, v.z * sz + tz))


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

    mathutils = types.ModuleType("mathutils")
    mathutils.Vector = _Vector

    sys.modules["bpy"] = bpy
    sys.modules["bpy.types"] = bpy_types
    sys.modules["bpy.props"] = bpy_props
    sys.modules["bpy.app"] = bpy_app
    sys.modules["bpy.app.handlers"] = bpy_app_handlers
    sys.modules["bpy.utils"] = bpy_utils
    sys.modules["mathutils"] = mathutils


def _import_props():
    """Importe `extension.props` sans declencher `extension/__init__.py`
    (voir docstring de `test_turbulence.py`)."""
    _install_bpy_stub()

    if "extension" not in sys.modules:
        pkg = types.ModuleType("extension")
        pkg.__path__ = [_EXT_DIR]
        sys.modules["extension"] = pkg

    import extension.props as props_module  # noqa: E402

    return props_module


props = _import_props()

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
# Fixtures : domaine cube, place/mis a l'echelle par `matrix_world`
# ---------------------------------------------------------------------------

_UNIT_CUBE_CORNERS = [
    (x, y, z)
    for x in (-0.5, 0.5)
    for y in (-0.5, 0.5)
    for z in (-0.5, 0.5)
]


class _FakeDomainObject:
    def __init__(self, translation, scale):
        self.bound_box = _UNIT_CUBE_CORNERS
        self.matrix_world = _AffineMatrix(translation, scale)


class _FakeBourrasqueProps:
    def __init__(
        self,
        domain_object,
        grid_res,
        ppc_axis=2,
        mesh_resolution=0,
        mesh_influence_factor=3.0,
        mesh_particle_factor=1.0,
        mesh_collider_offset_factor=0.0,
    ):
        self.domain_object = domain_object
        self.grid_res = grid_res
        self.ppc_axis = ppc_axis
        self.mesh_resolution = mesh_resolution
        self.mesh_influence_factor = mesh_influence_factor
        self.mesh_particle_factor = mesh_particle_factor
        self.mesh_collider_offset_factor = mesh_collider_offset_factor


class _FakeScene:
    def __init__(self, domain_object, **kwargs):
        self.bourrasque = _FakeBourrasqueProps(domain_object, **kwargs)


def _make_scene(extent_world, translation=(0.0, 0.0, 0.0), **kwargs):
    """`extent_world` : etendues MONDE `(ex, ey, ez)` de la boite domaine
    (un cube unitaire mis a l'echelle, comme `test_domain_transform.py`)."""
    obj = _FakeDomainObject(translation, extent_world)
    return _FakeScene(obj, **kwargs)


# ---------------------------------------------------------------------------
# Espacement inter-particules — meme formule que l'emission (dx / ppc_axis)
# ---------------------------------------------------------------------------


def test_particle_spacing_matches_dx_over_ppc():
    scene = _make_scene((2.0, 2.0, 2.0), grid_res=64, ppc_axis=2)
    res, dx = props.domain_resolution(scene)
    spacing = props.mesh_particle_spacing(scene)
    assert spacing is not None
    assert abs(spacing - dx / 2) < 1e-9, f"spacing={spacing}, attendu {dx / 2}"


def test_particle_spacing_none_without_domain():
    scene = _FakeScene(None, grid_res=64)
    assert props.mesh_particle_spacing(scene) is None


# ---------------------------------------------------------------------------
# Rayons effectifs — facteur * espacement
# ---------------------------------------------------------------------------


def test_effective_radii_are_factor_times_spacing():
    scene = _make_scene(
        (2.0, 2.0, 2.0),
        grid_res=64,
        ppc_axis=2,
        mesh_influence_factor=3.0,
        mesh_particle_factor=1.0,
        mesh_collider_offset_factor=-0.5,
    )
    spacing = props.mesh_particle_spacing(scene)
    radii = props.mesh_effective_radii(scene)
    assert radii is not None
    influence_radius, particle_radius, collider_offset = radii

    assert abs(influence_radius - 3.0 * spacing) < 1e-9
    assert abs(particle_radius - 1.0 * spacing) < 1e-9
    assert abs(collider_offset - (-0.5) * spacing) < 1e-9


def test_default_influence_factor_stays_above_spacing_on_large_domain():
    """Le cas reel qui a motive cette tache : sur un domaine de 4 m avec
    grille 64 et ppc_axis 2, l'ancien defaut ABSOLU (0.06 m) etait plus
    petit que l'espacement (0.031 m) — presque, mais le vrai probleme
    apparait a plus grande echelle encore. Ce test verifie l'invariant que
    la conversion en facteur restaure : le rayon d'influence par defaut
    (facteur 3.0) reste un multiple constant de l'espacement, quelle que
    soit la taille du domaine — jamais plus petit que l'espacement lui-meme,
    contrairement a un reglage absolu fixe."""
    for extent in (1.0, 4.0, 20.0):
        scene = _make_scene((extent, extent, extent), grid_res=64, ppc_axis=2)
        spacing = props.mesh_particle_spacing(scene)
        influence_radius, _particle_radius, _collider_offset = props.mesh_effective_radii(
            scene
        )
        assert influence_radius >= spacing, (
            f"extent={extent} : rayon d'influence ({influence_radius}) < "
            f"espacement ({spacing})"
        )


# ---------------------------------------------------------------------------
# Resolution automatique du maillage (mesh_resolution == 0)
# ---------------------------------------------------------------------------


def test_auto_resolution_is_grid_res_times_ppc_axis_when_under_cap():
    scene = _make_scene(
        (1.0, 1.0, 1.0), grid_res=16, ppc_axis=2, mesh_resolution=0
    )
    state = props.mesh_resolution_state(scene)
    assert state is not None
    setting, is_auto, was_capped = state
    assert is_auto is True
    assert was_capped is False
    assert setting == 16 * 2


def test_explicit_resolution_is_never_auto_nor_capped():
    scene = _make_scene(
        (20.0, 20.0, 20.0), grid_res=64, ppc_axis=2, mesh_resolution=512
    )
    state = props.mesh_resolution_state(scene)
    assert state is not None
    setting, is_auto, was_capped = state
    assert is_auto is False
    assert was_capped is False
    assert setting == 512


def test_auto_resolution_capped_under_1_5gb_on_very_resolved_scene():
    """Simulation tres resolue (grande grille, ppc_axis eleve, grand
    domaine) : la resolution automatique du maillage ne doit jamais produire
    une empreinte VRAM estimee au-dela de 1,5 Go — c'est le plafonnement
    demande par la tache, pas seulement l'avertissement a 2 Go qui existe
    deja pour un reglage explicite."""
    scene = _make_scene(
        (20.0, 20.0, 20.0), grid_res=256, ppc_axis=4, mesh_resolution=0
    )
    state = props.mesh_resolution_state(scene)
    assert state is not None
    setting, is_auto, was_capped = state
    assert is_auto is True
    assert was_capped is True

    est_bytes = props.mesh_vram_estimate_bytes(scene)
    cap = int(1.5 * 1024 * 1024 * 1024)
    assert est_bytes <= cap, (
        f"empreinte VRAM automatique ({est_bytes}) depasse le plafond "
        f"({cap}) — reglage retenu={setting}"
    )


def test_mesh_layout_uses_resolved_automatic_setting():
    """`mesh_layout` (deja utilisee par tous les appelants existants) doit
    refleter la resolution AUTOMATIQUE resolue, pas planter ni renvoyer un
    champ degenere, quand `mesh_resolution == 0`."""
    scene = _make_scene(
        (4.0, 4.0, 4.0), grid_res=64, ppc_axis=2, mesh_resolution=0
    )
    layout = props.mesh_layout(scene)
    assert layout is not None
    res, cell_size = layout
    assert min(res) > 1
    assert cell_size > 0


def main():
    check(
        "espacement inter-particules == dx / ppc_axis",
        test_particle_spacing_matches_dx_over_ppc,
    )
    check(
        "espacement inter-particules : None sans domaine",
        test_particle_spacing_none_without_domain,
    )
    check(
        "rayons effectifs == facteur * espacement",
        test_effective_radii_are_factor_times_spacing,
    )
    check(
        "rayon d'influence par defaut >= espacement (1m/4m/20m)",
        test_default_influence_factor_stays_above_spacing_on_large_domain,
    )
    check(
        "resolution automatique == grid_res * ppc_axis sous le plafond",
        test_auto_resolution_is_grid_res_times_ppc_axis_when_under_cap,
    )
    check(
        "resolution explicite : jamais auto, jamais plafonnee",
        test_explicit_resolution_is_never_auto_nor_capped,
    )
    check(
        "resolution automatique plafonnee sous 1.5 Go (scene tres resolue)",
        test_auto_resolution_capped_under_1_5gb_on_very_resolved_scene,
    )
    check(
        "mesh_layout reflete la resolution automatique resolue",
        test_mesh_layout_uses_resolved_automatic_setting,
    )

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) EN ECHEC.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
