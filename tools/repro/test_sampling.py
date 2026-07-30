"""Verification de extension/sampling.py, a executer avec :
  blender --background --factory-startup --python test_sampling.py

N'importe PAS extension/__init__.py (qui enregistre les classes bpy de
l'addon) : on construit un faux module de package "extension" dont
__path__ pointe vers le dossier reel, puis on importe "extension.sampling"
et "extension.transform" comme sous-modules de ce faux package. Ca laisse
"from .transform import ..." (import relatif dans sampling.py) se resoudre
normalement, sans executer le vrai extension/__init__.py.
"""

import importlib
import math
import sys
import time
import types

EXT_ROOT = r"C:\Users\nicol\Code\bourrasque_v2"

_pkg = types.ModuleType("extension")
_pkg.__path__ = [EXT_ROOT + r"\extension"]
sys.modules["extension"] = _pkg

sampling = importlib.import_module("extension.sampling")
transform = importlib.import_module("extension.transform")

import bpy
import bmesh
import numpy as np

FAILURES = []


def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


ORIGIN = (-5.0, -5.0, -5.0)
SIZE = 10.0
GRID_RES = 64
PPC_AXIS = 2


def lattice_bbox_count(obj):
    """Reproduit le compte AABB (mode bloc) via les fonctions internes de
    sampling.py, pour comparaison avec sample_mesh_interior sur une forme
    dont la bbox coincide avec la forme (le cube)."""
    obj_eval, _dg = sampling._evaluated_object_and_depsgraph(obj)
    dx = SIZE / GRID_RES
    spacing = dx / PPC_AXIS
    lo, hi = sampling._solver_bbox(obj_eval, ORIGIN, SIZE)
    count = 1
    for axis in range(3):
        rng = sampling._lattice_k_range(lo[axis], hi[axis], spacing)
        if rng is None:
            return 0
        k_min, k_max = rng
        count *= (k_max - k_min + 1)
    return count


# ---------------------------------------------------------------------------
# Test 1 : cube
# ---------------------------------------------------------------------------


def test_cube():
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object

    expected = lattice_bbox_count(obj)
    t0 = time.perf_counter()
    pts = sampling.sample_mesh_interior(obj, ORIGIN, SIZE, GRID_RES, PPC_AXIS)
    dt = time.perf_counter() - t0
    n = pts.shape[0]
    ratio = n / expected if expected else 0.0
    check(
        "cube: dtype/forme",
        pts.dtype == np.float32 and pts.ndim == 2 and pts.shape[1] == 3,
        f"dtype={pts.dtype} shape={pts.shape}",
    )
    check(
        "cube: compte proche de l'AABB equivalente",
        0.90 <= ratio <= 1.01,
        f"n={n} expected(bbox)={expected} ratio={ratio:.4f} temps={dt:.3f}s",
    )
    print(f"  cube: n={n} expected={expected} ratio={ratio:.4f} temps={dt:.4f}s")


# ---------------------------------------------------------------------------
# Test 2 : tore (LE test qui compte)
# ---------------------------------------------------------------------------


def test_torus():
    clear_scene()
    major_r = 1.5
    minor_r = 0.4
    bpy.ops.mesh.primitive_torus_add(
        location=(0.0, 0.0, 0.0),
        major_radius=major_r,
        minor_radius=minor_r,
        major_segments=48,
        minor_segments=24,
    )
    obj = bpy.context.active_object

    t0 = time.perf_counter()
    pts = sampling.sample_mesh_interior(obj, ORIGIN, SIZE, GRID_RES, PPC_AXIS)
    dt = time.perf_counter() - t0
    n = pts.shape[0]

    # Points solveur -> monde pour verifier la geometrie.
    world = transform.solver_to_world_array(pts.astype(np.float32), ORIGIN, SIZE)
    # Le tore de bpy.ops.mesh.primitive_torus_add est cree dans le plan XY
    # locale, revolution autour de l'axe Z local, centre a `location`.
    # L'objet n'a pas de rotation ici (matrix_world = translation pure),
    # donc l'axe du tore en monde est l'axe Z passant par location=(0,0,0).
    axis_dist = np.sqrt(world[:, 0] ** 2 + world[:, 1] ** 2)

    inner_radius = major_r - minor_r
    min_dist = float(axis_dist.min()) if n else float("nan")
    check(
        "tore: aucun point dans le trou central",
        n > 0 and min_dist >= inner_radius - 1e-3,
        f"n={n} min(dist a l'axe)={min_dist:.4f} rayon interieur attendu>={inner_radius:.4f}",
    )

    dx = SIZE / GRID_RES
    spacing = dx / PPC_AXIS
    vol_torus = 2.0 * math.pi ** 2 * major_r * minor_r ** 2
    expected_n = vol_torus / spacing ** 3
    ratio = n / expected_n if expected_n else 0.0
    check(
        "tore: compte coherent avec le volume analytique (2*pi^2*R*r^2)",
        0.90 <= ratio <= 1.10,
        f"n={n} expected(volume)={expected_n:.0f} ratio={ratio:.4f}",
    )
    print(f"  tore: n={n} expected(vol)={expected_n:.0f} ratio={ratio:.4f} temps={dt:.4f}s")
    return dt


# ---------------------------------------------------------------------------
# Test 3 : sphere
# ---------------------------------------------------------------------------


def test_sphere():
    clear_scene()
    r = 1.2
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=r, location=(0.0, 0.0, 0.0), segments=48, ring_count=24
    )
    obj = bpy.context.active_object

    t0 = time.perf_counter()
    pts = sampling.sample_mesh_interior(obj, ORIGIN, SIZE, GRID_RES, PPC_AXIS)
    dt = time.perf_counter() - t0
    n = pts.shape[0]

    dx = SIZE / GRID_RES
    spacing = dx / PPC_AXIS
    expected_n = (4.0 / 3.0) * math.pi * r ** 3 / spacing ** 3
    ratio = n / expected_n if expected_n else 0.0
    check(
        "sphere: compte coherent avec (4/3)*pi*r^3",
        0.90 <= ratio <= 1.10,
        f"n={n} expected(volume)={expected_n:.0f} ratio={ratio:.4f}",
    )
    print(f"  sphere: n={n} expected(vol)={expected_n:.0f} ratio={ratio:.4f} temps={dt:.4f}s")
    return dt, pts


# ---------------------------------------------------------------------------
# Test 4 : maillage ouvert
# ---------------------------------------------------------------------------


def test_open_mesh():
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object

    # Supprime une face via bmesh direct (pas de context override necessaire
    # en mode background).
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.faces[0]], context="FACES")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()

    ok, msg = sampling.check_mesh_closed(obj)
    check("maillage ouvert: detecte comme non utilisable", ok is False, f"msg={msg!r}")
    check("maillage ouvert: message non vide", bool(msg))
    print(f"  message: {msg}")

    # Maillage ferme (cube intact) : doit passer.
    clear_scene()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    obj2 = bpy.context.active_object
    ok2, msg2 = sampling.check_mesh_closed(obj2)
    check("cube ferme: utilisable", ok2 is True, f"msg={msg2!r}")
    check("cube ferme: message vide", msg2 == "")


# ---------------------------------------------------------------------------
# Test 5 : alignement sur le reseau
# ---------------------------------------------------------------------------


def test_lattice_alignment():
    clear_scene()
    r = 1.0
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0.31, -0.7, 0.12))
    obj = bpy.context.active_object

    pts = sampling.sample_mesh_interior(obj, ORIGIN, SIZE, GRID_RES, PPC_AXIS)
    dx = SIZE / GRID_RES
    spacing = dx / PPC_AXIS

    # residu = (coord - spacing/2) mod spacing, doit etre ~0 ou ~spacing
    residual = np.mod(pts - spacing / 2.0, spacing)
    dist_to_lattice = np.minimum(residual, spacing - residual)
    max_dev = float(dist_to_lattice.max()) if pts.shape[0] else 0.0
    check(
        "alignement reseau: tous les points a 1e-5 pres de spacing/2 + k*spacing",
        max_dev < 1e-5,
        f"n={pts.shape[0]} deviation max={max_dev:.3e} spacing={spacing:.6f}",
    )


# ---------------------------------------------------------------------------
# Test estimate_mesh_sample_count (rapidite + coherence grossiere)
# ---------------------------------------------------------------------------


def test_estimate():
    clear_scene()
    r = 1.2
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object

    t0 = time.perf_counter()
    est = sampling.estimate_mesh_sample_count(obj, ORIGIN, SIZE, GRID_RES, PPC_AXIS)
    dt = time.perf_counter() - t0

    dx = SIZE / GRID_RES
    spacing = dx / PPC_AXIS
    expected_n = (4.0 / 3.0) * math.pi * r ** 3 / spacing ** 3
    ratio = est / expected_n if expected_n else 0.0
    check(
        "estimate: pas de lancer de rayon (rapide)",
        dt < 0.2,
        f"temps={dt:.4f}s",
    )
    check(
        "estimate: coherent avec le volume analytique",
        0.7 <= ratio <= 1.3,
        f"est={est} expected(volume)={expected_n:.0f} ratio={ratio:.4f}",
    )
    print(f"  estimate sphere: est={est} expected(vol)={expected_n:.0f} ratio={ratio:.4f} temps={dt:.5f}s")


if __name__ == "__main__":
    test_cube()
    dt_torus = test_torus()
    dt_sphere, _sphere_pts = test_sphere()
    test_open_mesh()
    test_lattice_alignment()
    test_estimate()

    print()
    print(f"TEMPS MESURE -- tore: {dt_torus:.4f}s  sphere: {dt_sphere:.4f}s  (grid_res={GRID_RES}, ppc_axis={PPC_AXIS})")
    print()
    if FAILURES:
        print(f"ECHECS ({len(FAILURES)}): {FAILURES}")
        sys.exit(1)
    else:
        print("TOUS LES TESTS PASSENT.")
        sys.exit(0)
