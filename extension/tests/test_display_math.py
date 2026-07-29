"""Tests de la conversion vectorisee espace solveur -> monde (transform.py).

`display.py` importe `bpy` en tete de module (idiome standard pour un
sous-module d'extension Blender) et n'est donc pas importable tel quel hors
Blender ; mais `solver_to_world_array` (la conversion vectorisee qu'il
utilise) vit dans `transform.py`, sans dependance a bpy, et est donc
importee ici DIRECTEMENT depuis le code de production — aucune duplication.
Ce test verifie qu'elle coincide avec `solver_to_world` (version scalaire,
meme module), point par point, a 1e-6 pres.

Executable avec `python extension/tests/test_display_math.py`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transform import solver_to_world, solver_to_world_array  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_matches_scalar_reference():
    origin = (-2.0, 5.0, 1.0)
    size = (4.0, 4.0, 4.0)

    rng = np.random.default_rng(12345)
    points = rng.uniform(low=-10.0, high=10.0, size=(1000, 3)).astype(np.float32)

    vectorized = solver_to_world_array(points, origin, size)

    max_err = 0.0
    for i in range(points.shape[0]):
        scalar = solver_to_world(tuple(points[i]), origin, size)
        err = max(abs(vectorized[i, k] - scalar[k]) for k in range(3))
        max_err = max(max_err, err)

    check(
        "vectorized solver_to_world matches scalar reference (<=1e-6)",
        max_err <= 1e-6,
        f"max_err={max_err}",
    )


def test_matches_scalar_reference_non_cubic():
    """Meme verification que `test_matches_scalar_reference`, mais avec un
    `size` NON cubique (voir docs/plan-milestone-5.md) : la version
    vectorisee doit rester d'accord avec la version scalaire axe par axe,
    pas seulement pour un cube."""
    origin = (-2.0, 5.0, 1.0)
    size = (3.0, 9.0, 15.0)

    rng = np.random.default_rng(2024)
    points = rng.uniform(low=-10.0, high=10.0, size=(1000, 3)).astype(np.float32)

    vectorized = solver_to_world_array(points, origin, size)

    max_err = 0.0
    for i in range(points.shape[0]):
        scalar = solver_to_world(tuple(points[i]), origin, size)
        err = max(abs(vectorized[i, k] - scalar[k]) for k in range(3))
        max_err = max(max_err, err)

    check(
        "vectorized solver_to_world (non cubique) matches scalar reference (<=1e-6)",
        max_err <= 1e-6,
        f"max_err={max_err}",
    )


def test_out_buffer_reused():
    origin = (-2.0, 5.0, 1.0)
    size = (4.0, 4.0, 4.0)
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, -1.0, 2.5]], dtype=np.float32
    )
    out = np.empty_like(points)
    result = solver_to_world_array(points, origin, size, out=out)
    check("out buffer is reused (same identity)", result is out)

    expected = [solver_to_world(tuple(p), origin, size) for p in points]
    max_err = max(
        abs(result[i, k] - expected[i][k]) for i in range(3) for k in range(3)
    )
    check("out buffer content matches scalar reference", max_err <= 1e-6, f"max_err={max_err}")


if __name__ == "__main__":
    test_matches_scalar_reference()
    test_matches_scalar_reference_non_cubic()
    test_out_buffer_reused()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {FAILURES}")
        sys.exit(1)
    else:
        print("Tous les tests sont passes.")
        sys.exit(0)
