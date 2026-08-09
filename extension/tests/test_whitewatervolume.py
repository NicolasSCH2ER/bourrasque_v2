"""test_whitewatervolume.py — verification autonome de
`extension/whitewatervolume.py` (`compute_whitewater_volume_display_params`,
parametres de l'affichage volumetrique du whitewater).

Executable sans Blender : `python extension/tests/test_whitewatervolume.py`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from whitewatervolume import compute_whitewater_volume_display_params  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def test_measured_scale():
    # Valeur mesuree lors de la verification de session (influence_radius
    # = 0.0234375 -> bq_size ~= 0.007).
    voxel, density = compute_whitewater_volume_display_params(0.007)
    check(
        "voxel_size a l'echelle mesuree ~= 0.0021",
        abs(voxel - 0.007 * 0.3) < 1e-9,
        f"voxel={voxel}",
    )
    check(
        "density_factor a l'echelle mesuree ~= 400 (2.8/0.007)",
        abs(density - 400.0) < 1e-6,
        f"density={density}",
    )


def test_scales_with_size_10x():
    voxel_small, density_small = compute_whitewater_volume_display_params(0.007)
    voxel_big, density_big = compute_whitewater_volume_display_params(0.07)

    check(
        "voxel_size croit proportionnellement a bq_size_mean (x10)",
        abs(voxel_big - voxel_small * 10.0) < 1e-9,
        f"voxel_small={voxel_small} voxel_big={voxel_big}",
    )
    check(
        "density_factor decroit proportionnellement a l'inverse de bq_size_mean (/10)",
        abs(density_big - density_small / 10.0) < 1e-6,
        f"density_small={density_small} density_big={density_big}",
    )


def test_zero_size_no_zero_division():
    try:
        voxel, density = compute_whitewater_volume_display_params(0.0)
        ok = True
    except ZeroDivisionError:
        ok = False
    check("bq_size_mean=0.0 ne leve pas ZeroDivisionError", ok)
    if ok:
        check("density_factor fini a bq_size_mean=0.0", density < float("inf"))


def test_negative_size_uses_floor():
    # bq_size_mean negatif (cas degenere/impossible en pratique) : le
    # plancher `_MIN_BQ_SIZE` s'applique quand meme, jamais de division par
    # une valeur negative ni nulle.
    voxel, density = compute_whitewater_volume_display_params(-1.0)
    check("voxel_size positif meme pour une entree negative", voxel > 0.0, f"voxel={voxel}")
    check("density_factor positif meme pour une entree negative", density > 0.0, f"density={density}")


def main():
    test_measured_scale()
    test_scales_with_size_10x()
    test_zero_size_no_zero_division()
    test_negative_size_uses_floor()

    if FAILURES:
        print(f"\n{len(FAILURES)} echec(s) : {FAILURES}")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
