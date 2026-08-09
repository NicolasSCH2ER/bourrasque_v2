"""test_foam.py — verification autonome de `extension/foam.py`
(`compute_foam_mask`, masque d'ecume `bq_foam_mask`, docs/plan-milestone-14.md
T4).

Executable sans Blender : `python extension/tests/test_foam.py`.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from foam import compute_foam_mask  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_empty_whitewater_gives_zero_mask():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    ww = np.zeros((0, 3), dtype=np.float32)
    mask = compute_foam_mask(verts, ww, influence_radius=0.5)
    check("mask vide -> zeros", mask.shape == (2,) and np.all(mask == 0.0))


def test_empty_verts_gives_empty_mask():
    verts = np.zeros((0, 3), dtype=np.float32)
    ww = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    mask = compute_foam_mask(verts, ww, influence_radius=0.5)
    check("verts vide -> mask vide", mask.shape == (0,))


def test_far_particle_gives_zero():
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ww = np.array([[100.0, 0.0, 0.0]], dtype=np.float32)
    mask = compute_foam_mask(verts, ww, influence_radius=0.5)
    check("particule hors rayon -> 0", mask[0] == 0.0, f"mask={mask[0]}")


def test_coincident_particle_gives_max_weight():
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ww = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    mask = compute_foam_mask(verts, ww, influence_radius=0.5)
    # s=0 -> w=(1-0)^3=1, une seule particule confondue avec le sommet.
    check(
        "particule confondue -> poids 1.0",
        abs(mask[0] - 1.0) < 1e-6,
        f"mask={mask[0]}",
    )


def test_kernel_matches_formula_at_half_radius():
    r = 2.0
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ww = np.array([[r * 0.5, 0.0, 0.0]], dtype=np.float32)
    mask = compute_foam_mask(verts, ww, influence_radius=r)
    s = 0.5
    expected = (1.0 - s * s) ** 3
    check(
        "noyau k(s)=(1-s^2)^3 respecte a s=0.5",
        abs(mask[0] - expected) < 1e-6,
        f"mask={mask[0]} expected={expected}",
    )


def test_clamped_to_one():
    # Beaucoup de particules confondues avec le sommet : la somme des poids
    # depasse largement 1, doit etre clampee.
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ww = np.zeros((50, 3), dtype=np.float32)
    mask = compute_foam_mask(verts, ww, influence_radius=0.5)
    check("clampe a 1.0", mask[0] == 1.0, f"mask={mask[0]}")


def test_values_in_unit_range_on_random_scene():
    rng = np.random.default_rng(7)
    verts = rng.uniform(-5.0, 5.0, size=(200, 3)).astype(np.float32)
    ww = rng.uniform(-5.0, 5.0, size=(500, 3)).astype(np.float32)
    mask = compute_foam_mask(verts, ww, influence_radius=0.8)
    check(
        "toutes les valeurs dans [0, 1]",
        bool(np.all(mask >= 0.0) and np.all(mask <= 1.0)),
        f"min={mask.min()} max={mask.max()}",
    )
    check("au moins une valeur non nulle (scene dense)", bool(np.any(mask > 0.0)))


def test_zero_radius_gives_zero_mask():
    verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    ww = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    mask = compute_foam_mask(verts, ww, influence_radius=0.0)
    check("rayon nul -> 0 (pas de division par zero)", mask[0] == 0.0)


def test_performance_on_moderate_scene():
    """Mesure honnete du cout KD-tree sur une scene de test raisonnable
    (quelques dizaines de milliers de sommets/particules) — pas une
    assertion stricte de performance, juste un rapport (voir la demande de
    la tache T4)."""
    rng = np.random.default_rng(42)
    n_verts = 40_000
    n_ww = 20_000
    verts = rng.uniform(-10.0, 10.0, size=(n_verts, 3)).astype(np.float32)
    ww = rng.uniform(-10.0, 10.0, size=(n_ww, 3)).astype(np.float32)

    t0 = time.perf_counter()
    mask = compute_foam_mask(verts, ww, influence_radius=0.3)
    elapsed = time.perf_counter() - t0

    check(
        "scene moderee : mask valide dans [0, 1]",
        bool(np.all(mask >= 0.0) and np.all(mask <= 1.0)),
    )
    print(
        f"    -> KD-tree {n_verts} sommets / {n_ww} particules "
        f"whitewater : {elapsed * 1000:.1f} ms"
    )


if __name__ == "__main__":
    test_empty_whitewater_gives_zero_mask()
    test_empty_verts_gives_empty_mask()
    test_far_particle_gives_zero()
    test_coincident_particle_gives_max_weight()
    test_kernel_matches_formula_at_half_radius()
    test_clamped_to_one()
    test_values_in_unit_range_on_random_scene()
    test_zero_radius_gives_zero_mask()
    test_performance_on_moderate_scene()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {FAILURES}")
        sys.exit(1)
    else:
        print("Tous les tests sont passes.")
        sys.exit(0)
