"""Tests de la transformation monde <-> solveur (extension/transform.py).

Purement mathematique : aucune dependance a bpy, executable avec un simple
`python extension/tests/test_transform.py`.

Depuis M5 (domaine en pave, dx uniforme mais nombre de cellules different
par axe), `size` est un TRIPLET `(size_x, size_y, size_z)` en espace
SOLVEUR — voir `transform.py` pour le mapping des axes (`size[0]` <->
etendue monde X, `size[1]` <-> etendue monde Z, `size[2]` <-> etendue
monde Y). Les tests ci-dessous couvrent explicitement des tailles NON
cubiques, le mode d'echec le plus probable du jalon etant une confusion
d'axes silencieuse.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transform import solver_to_world, world_to_solver, world_to_solver_dir  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def approx_eq(a, b, tol=1e-5):
    return all(abs(x - y) < tol for x, y in zip(a, b))


def test_round_trip():
    origin = (-2.0, 5.0, 1.0)
    size = (4.0, 4.0, 4.0)
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (-5.0, 7.5, -1.0),
        (3.3, -2.2, 0.0),
        (100.0, -50.0, 25.0),
    ]
    for p in points:
        s = world_to_solver(p, origin, size)
        back = solver_to_world(s, origin, size)
        check(
            f"round_trip {p}",
            approx_eq(p, back, 1e-5),
            f"got {back}",
        )


def test_round_trip_non_cubic():
    """Aller-retour de coordonnees sur un domaine NON cubique (V5 du jalon
    M5) : `size` est un triplet distinct sur chaque axe, echantillon de
    points couvrant tout le domaine (interieur, coins, hors domaine)."""
    origin = (-3.0, 2.0, -1.0)
    size = (2.0, 6.0, 10.0)  # tres allonge sur size[2] (etendue monde Y)

    points = [
        (0.0, 0.0, 0.0),
        (origin[0] + 0.5, origin[1] + 3.0, origin[2] - 4.0),
        (origin[0] + size[0], origin[1] + size[2], origin[2] + size[1]),
        (-100.0, 250.0, -37.5),  # hors domaine : la fonction reste lineaire
        (origin[0] + size[0] * 0.5, origin[1] + size[2] * 0.5, origin[2] + size[1] * 0.5),
    ]
    max_err = 0.0
    for p in points:
        s = world_to_solver(p, origin, size)
        back = solver_to_world(s, origin, size)
        err = max(abs(a - b) for a, b in zip(p, back))
        max_err = max(max_err, err)
        check(f"round_trip non-cubique {p}", err < 1e-5, f"got {back}, err={err}")
    print(f"  (erreur max aller-retour, domaine non cubique : {max_err:.2e})")


def test_chirality_no_axis_swap():
    origin = (-2.0, 5.0, 1.0)
    size = (4.0, 4.0, 4.0)
    s0 = (0.1, 0.2, 0.3)
    b = solver_to_world(s0, origin, size)
    s_back = world_to_solver(b, origin, size)
    check(
        "chirality: round-trip through world preserves solver point",
        approx_eq(s0, s_back, 1e-5),
        f"got {s_back}",
    )
    naive_swap = (s0[0], s0[2], s0[1])
    check(
        "chirality: not a naive axis swap (0.1,0.2,0.3) != (0.1,0.3,0.2)",
        not approx_eq(s_back, naive_swap, 1e-9),
        f"s_back={s_back} naive_swap={naive_swap}",
    )


def test_determinant():
    """Le determinant de la partie lineaire (`world_to_solver_dir`) est
    independant de `size` (aucune mise a l'echelle, seulement permutation +
    inversion d'un axe) : verifie une seule fois, comme avant M5."""
    ex = world_to_solver_dir((1.0, 0.0, 0.0))
    ey = world_to_solver_dir((0.0, 1.0, 0.0))
    ez = world_to_solver_dir((0.0, 0.0, 1.0))

    # Matrice dont les colonnes sont les images des vecteurs de base.
    m = [
        [ex[0], ey[0], ez[0]],
        [ex[1], ey[1], ez[1]],
        [ex[2], ey[2], ez[2]],
    ]

    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    check("determinant == +1 (chiralite preservee)", abs(det - 1.0) < 1e-9, f"det={det}")


def test_domain_bounds():
    """La bbox monde du domaine [origin, origin+L]^3 (les 8 coins) se
    transforme exactement en [0, L]^3 en espace solveur.

    Note : a cause de l'inversion de l'axe Y (chiralite), le coin "tout
    minimal" du monde (mx, my, mz) ne tombe PAS sur le coin solveur (0,0,0)
    componentwise -- il tombe sur (0, 0, L), et le coin "tout maximal" du
    monde tombe sur (L, L, 0). C'est attendu : seul l'axe sz (issu de by)
    est inverse, sx et sy (issus de bx, bz) restent monotones dans le meme
    sens. L'invariant qui tient reellement est agrege sur les 8 coins :
    l'ensemble de la bbox occupe exactement [0, L]^3 en solveur.
    """
    origin = (-2.0, 5.0, 1.0)
    L = 4.0
    size = (L, L, L)
    mx, my, mz = origin

    world_corners = [
        (mx + dx * L, my + dy * L, mz + dz * L)
        for dx in (0.0, 1.0)
        for dy in (0.0, 1.0)
        for dz in (0.0, 1.0)
    ]
    solver_corners = [world_to_solver(c, origin, size) for c in world_corners]

    lo = tuple(min(c[axis] for c in solver_corners) for axis in range(3))
    hi = tuple(max(c[axis] for c in solver_corners) for axis in range(3))

    check(
        "domain bbox aggregate min -> solver (0,0,0)",
        approx_eq(lo, (0.0, 0.0, 0.0), 1e-5),
        f"got {lo}",
    )
    check(
        "domain bbox aggregate max -> solver (L,L,L)",
        approx_eq(hi, (L, L, L), 1e-5),
        f"got {hi}",
    )

    # Documente explicitement la non-monotonie sur sz (voir docstring).
    all_min_world = origin
    all_max_world = (mx + L, my + L, mz + L)
    s_all_min = world_to_solver(all_min_world, origin, size)
    s_all_max = world_to_solver(all_max_world, origin, size)
    check(
        "chirality note: world all-min corner does NOT map to solver (0,0,0)",
        not approx_eq(s_all_min, (0.0, 0.0, 0.0), 1e-9),
        f"s_all_min={s_all_min} (attendu (0,0,L))",
    )
    check(
        "chirality note: world all-min corner maps to solver (0,0,L)",
        approx_eq(s_all_min, (0.0, 0.0, L), 1e-5),
        f"got {s_all_min}",
    )


def test_domain_bounds_non_cubic():
    """Meme verification que `test_domain_bounds`, mais avec un pave
    solveur NON cubique : la bbox monde du domaine (les 8 coins) doit se
    transformer exactement en `[0, size[0]] x [0, size[1]] x [0, size[2]]`,
    pas dans un cube. C'est le test qui attraperait une confusion entre
    `size[0]`/`size[1]`/`size[2]` dans `world_to_solver`/`solver_to_world`.
    """
    origin = (-2.0, 5.0, 1.0)
    size = (3.0, 7.0, 11.0)
    mx, my, mz = origin

    # Coin monde (bx, by, bz) parcourant les 8 combinaisons de
    # {min, max} sur chaque axe MONDE, avec les etendues monde
    # correspondantes : extent_x = size[0], extent_z = size[1] (etendue
    # monde Z <-> solveur sy), extent_y = size[2] (etendue monde Y <->
    # solveur sz, axe inverse).
    world_corners = [
        (mx + fx * size[0], my + fy * size[2], mz + fz * size[1])
        for fx in (0.0, 1.0)
        for fy in (0.0, 1.0)
        for fz in (0.0, 1.0)
    ]
    solver_corners = [world_to_solver(c, origin, size) for c in world_corners]

    lo = tuple(min(c[axis] for c in solver_corners) for axis in range(3))
    hi = tuple(max(c[axis] for c in solver_corners) for axis in range(3))

    check(
        "domain bbox (non cubique) aggregate min -> solver (0,0,0)",
        approx_eq(lo, (0.0, 0.0, 0.0), 1e-5),
        f"got {lo}",
    )
    check(
        "domain bbox (non cubique) aggregate max -> solver size",
        approx_eq(hi, size, 1e-5),
        f"got {hi}, size={size}",
    )


if __name__ == "__main__":
    test_round_trip()
    test_round_trip_non_cubic()
    test_chirality_no_axis_swap()
    test_determinant()
    test_domain_bounds()
    test_domain_bounds_non_cubic()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {FAILURES}")
        sys.exit(1)
    else:
        print("Tous les tests sont passes.")
        sys.exit(0)
