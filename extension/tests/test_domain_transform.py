"""test_domain_transform.py — verification autonome de
`extension.props.domain_transform` / `domain_resolution` /
`domain_usable_bounds` sur des domaines NON cubiques (jalon M5, domaine en
pave — voir docs/plan-milestone-5.md).

`props.py` importe `bpy`/`mathutils` en tete de module (idiome standard
d'un module bpy) : comme `test_turbulence.py`, ce script installe un stub
MINIMAL de `bpy` dans `sys.modules` pour rester executable sans Blender,
mais avec un `mathutils.Vector`/matrice de transformation REELS (juste
assez pour multiplier un vecteur par une matrice affine), necessaires ici
car `domain_transform` calcule effectivement une bbox monde via
`obj.matrix_world @ Vector(corner)`.

Executable avec `python extension/tests/test_domain_transform.py`.
"""

import math
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT_DIR = os.path.abspath(os.path.join(_HERE, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_EXT_DIR, ".."))

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Stub bpy/mathutils minimal, avec un Vector/Matrix REELS (voir docstring)
# ---------------------------------------------------------------------------


class _Vector:
    __slots__ = ("x", "y", "z")

    def __init__(self, xyz):
        self.x, self.y, self.z = xyz

    def __iter__(self):
        return iter((self.x, self.y, self.z))


class _AffineMatrix:
    """Matrice affine minimale : translation + mise a l'echelle diagonale,
    suffisante pour placer un cube domaine (`primitive_cube_add` + `scale`
    + `location`, sans rotation) dans les tests ci-dessous."""

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
    (voir docstring de `test_turbulence.py` pour la justification du meme
    contournement)."""
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
# Fixtures : domaine cube unitaire (bound_box [-0.5, 0.5]^3), place/mis a
# l'echelle par `matrix_world` — reproduit fidelement
# `BQ_OT_add_domain`/`primitive_cube_add(size=1.0)` + `obj.scale`.
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
    def __init__(self, domain_object, grid_res):
        self.domain_object = domain_object
        self.grid_res = grid_res


class _FakeScene:
    def __init__(self, domain_object, grid_res):
        self.bourrasque = _FakeBourrasqueProps(domain_object, grid_res)


def _make_scene(extent_world, grid_res=64, translation=(0.0, 0.0, 0.0)):
    """`extent_world` : etendues MONDE `(ex, ey, ez)` de la boite domaine."""
    obj = _FakeDomainObject(translation, extent_world)
    return _FakeScene(obj, grid_res)


# ---------------------------------------------------------------------------
# V1 (rappel) — cube : dx == extent/grid_res sur les 3 axes, res identique
# ---------------------------------------------------------------------------


def test_cubic_domain_isotropic():
    scene = _make_scene((2.0, 2.0, 2.0), grid_res=64)
    origin, size = props.domain_transform(scene)
    res, dx = props.domain_resolution(scene)

    expected_dx = 2.0 / 64
    assert abs(dx - expected_dx) < 1e-9, f"dx={dx}, attendu {expected_dx}"
    assert res[0] == res[1] == res[2], f"res non isotrope sur un cube : {res}"
    assert abs(size[0] - size[1]) < 1e-9 and abs(size[1] - size[2]) < 1e-9, (
        f"size non isotrope sur un cube : {size}"
    )


# ---------------------------------------------------------------------------
# V4 — domaine non cubique general : verifie la formule exacte du jalon
# ---------------------------------------------------------------------------


def test_non_cubic_domain_matches_formula():
    """Domaine d'etendues monde (4, 2, 1) (x, y, z), grid_res=32 : verifie
    `dx`, `res` et `size` calcules a la main d'apres la formule du plan
    (docs/plan-milestone-5.md D3)."""
    extent_world = (4.0, 2.0, 1.0)
    grid_res = 32
    scene = _make_scene(extent_world, grid_res=grid_res)

    origin, size = props.domain_transform(scene)
    res, dx = props.domain_resolution(scene)

    expected_dx = max(extent_world) / grid_res  # 4.0 / 32 = 0.125
    assert abs(dx - expected_dx) < 1e-9, f"dx={dx}, attendu {expected_dx}"

    # size_solver = (extent_x, extent_z, extent_y) (D3, transform.py)
    extent_solver = (extent_world[0], extent_world[2], extent_world[1])
    # ceil (pas round, voir _domain_layout) : sur cet exemple
    # extent_solver[a]/dx = (32, 8, 16), des entiers exacts, donc ceil et
    # round coincident et ce test ne distingue pas les deux formules a lui
    # seul (voir test_artist_box_inside_usable_zone plus bas pour un cas
    # qui les distingue).
    expected_res = tuple(
        max(1, math.ceil(extent_solver[a] / expected_dx)) + 2 * props.SOLVER_STENCIL_BOUND
        for a in range(3)
    )
    assert res == expected_res, f"res={res}, attendu {expected_res}"

    expected_size = tuple(expected_res[a] * expected_dx for a in range(3))
    for a in range(3):
        assert abs(size[a] - expected_size[a]) < 1e-6, (
            f"size[{a}]={size[a]}, attendu {expected_size[a]}"
        )


# ---------------------------------------------------------------------------
# V4 — facteurs de forme extremes : res >= 1 sur chaque axe, marge des deux
# cotes
# ---------------------------------------------------------------------------


def _check_extreme_shape(extent_world, grid_res=64):
    scene = _make_scene(extent_world, grid_res=grid_res)
    res, dx = props.domain_resolution(scene)
    lo, hi = props.domain_usable_bounds(scene)
    _origin, size = props.domain_transform(scene)

    assert dx > 0, f"dx <= 0 pour extent={extent_world}"
    for a in range(3):
        assert res[a] >= 1, f"res[{a}]={res[a]} < 1 pour extent={extent_world}"

    margin = props.SOLVER_STENCIL_BOUND * dx
    for a in range(3):
        assert abs(lo[a] - margin) < 1e-6, (
            f"marge basse axe {a} = {lo[a]}, attendu {margin}"
        )
        assert abs((size[a] - hi[a]) - margin) < 1e-6, (
            f"marge haute axe {a} = {size[a] - hi[a]}, attendu {margin}"
        )


def test_extreme_shape_1_1_16():
    _check_extreme_shape((1.0, 1.0, 16.0))


def test_extreme_shape_16_1_1():
    _check_extreme_shape((16.0, 1.0, 1.0))


def test_extreme_shape_1_16_1():
    _check_extreme_shape((1.0, 16.0, 1.0))


# ---------------------------------------------------------------------------
# Invariant « boite de l'artiste incluse dans la zone utile » (revue M5,
# correctif ceil) : sur des etendues qui ne sont PAS des multiples de dx,
# `round` peut arrondir vers le bas et retrecir la zone utile jusqu'a dx/2
# sous la boite de l'artiste, ce qu'aucun test existant ne verifiait — les
# tests `_check_extreme_shape` ci-dessus ne verifient que `lo == marge` et
# `size - hi == marge`, deux tautologies vraies quelle que soit la valeur de
# `res` (elles decoulent directement de la definition de `marge`, pas du
# choix round/ceil). Ici on verifie la propriete reelle : la boite de
# l'artiste, mappee en espace solveur, tient dans [lo, hi] sur les 3 axes.
# ---------------------------------------------------------------------------


def _check_artist_box_inside_usable_zone(extent_world, grid_res=64):
    scene = _make_scene(extent_world, grid_res=grid_res)
    res, dx = props.domain_resolution(scene)
    lo, hi = props.domain_usable_bounds(scene)

    # size_solver = (extent_x, extent_z, extent_y) (D3, transform.py) : la
    # boite de l'artiste, en espace solveur, mesure exactement `extent_solver`
    # sur chaque axe (voir _domain_layout : origin = bbox_min - marge, donc
    # la boite occupe [marge, marge + extent_solver] avant tout arrondi).
    extent_solver = (extent_world[0], extent_world[2], extent_world[1])

    for a in range(3):
        usable_extent = hi[a] - lo[a]
        overflow = extent_solver[a] - usable_extent
        assert overflow <= 1e-9, (
            f"axe {a} : la boite de l'artiste ({extent_solver[a]:.6f}) "
            f"deborde la zone utile ({usable_extent:.6f}) de {overflow:.6f} "
            f"— extent_world={extent_world}, grid_res={grid_res}, res={res}, dx={dx}"
        )


def test_artist_box_inside_usable_zone_non_multiple_extents():
    # Aucune de ces etendues n'est un multiple exact de dx = max(extent)/64
    # sur les axes non-max : c'est precisement le cas que `round` faisait
    # echouer (arrondi vers le bas -> res trop petit -> zone utile trop
    # petite -> boite de l'artiste qui deborde).
    for extent_world in (
        (1.0, 0.71, 1.0),  # cas de regression mesure dans la revue M5
        (3.0, 1.37, 0.53),
        (5.0, 0.9, 2.2),
        (1.0, 1.0, 0.999),
    ):
        _check_artist_box_inside_usable_zone(extent_world)


def test_artist_box_inside_usable_zone_regression_case():
    """Cas precis rapporte par la revue M5 : (1.0, 0.71, 1.0) a grid_res=64
    donnait res=(70,70,51) avec `round` (deborde de 0.44*dx sur sz) et doit
    donner res=(70,70,52) avec `ceil` (deborde de -0.56*dx, donc contenu)."""
    scene = _make_scene((1.0, 0.71, 1.0), grid_res=64)
    res, dx = props.domain_resolution(scene)
    assert res == (70, 70, 52), f"res={res}, attendu (70, 70, 52)"

    lo, hi = props.domain_usable_bounds(scene)
    extent_solver = (1.0, 1.0, 0.71)
    overflow_sz = extent_solver[2] - (hi[2] - lo[2])
    assert overflow_sz <= 1e-9, f"overflow sz={overflow_sz}, attendu <= 0"
    print(f"      (overflow sz = {overflow_sz / dx:.4f} * dx)")


def main():
    check("cube : dx/res/size isotropes", test_cubic_domain_isotropic)
    check(
        "non cubique (4,2,1) : dx/res/size == formule du plan",
        test_non_cubic_domain_matches_formula,
    )
    check("forme extreme 1:1:16 : res>=1, marge symetrique", test_extreme_shape_1_1_16)
    check("forme extreme 16:1:1 : res>=1, marge symetrique", test_extreme_shape_16_1_1)
    check("forme extreme 1:16:1 : res>=1, marge symetrique", test_extreme_shape_1_16_1)
    check(
        "boite artiste incluse dans zone utile (etendues non multiples de dx)",
        test_artist_box_inside_usable_zone_non_multiple_extents,
    )
    check(
        "boite artiste incluse dans zone utile (cas de regression (1,0.71,1))",
        test_artist_box_inside_usable_zone_regression_case,
    )

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) EN ECHEC.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
