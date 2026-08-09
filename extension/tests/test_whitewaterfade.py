"""test_whitewaterfade.py — verification autonome de
`extension/whitewaterfade.py` (`compute_age_fade`, facteur de fondu
`bq_fade`).

Executable sans Blender : `python extension/tests/test_whitewaterfade.py`.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from whitewaterfade import compute_age_fade  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def test_birth_is_zero_with_fade_in():
    age = np.array([0.0], dtype=np.float32)
    type_ = np.array([0], dtype=np.int32)
    fade = compute_age_fade(age, type_, 1.0, 1.0, 1.0, fade_in_frac=0.15, fade_out_frac=0.25)
    check("naissance (t=0) -> fade 0.0", abs(fade[0] - 0.0) < 1e-6, f"fade={fade[0]}")


def test_death_is_zero_with_fade_out():
    age = np.array([1.0], dtype=np.float32)  # t = age/life = 1.0
    type_ = np.array([0], dtype=np.int32)
    fade = compute_age_fade(age, type_, 1.0, 1.0, 1.0, fade_in_frac=0.15, fade_out_frac=0.25)
    check("mort (t=1) -> fade 0.0", abs(fade[0] - 0.0) < 1e-6, f"fade={fade[0]}")


def test_plateau_is_one():
    age = np.array([0.5], dtype=np.float32)  # t = 0.5, dans [0.15, 0.75]
    type_ = np.array([0], dtype=np.int32)
    fade = compute_age_fade(age, type_, 1.0, 1.0, 1.0, fade_in_frac=0.15, fade_out_frac=0.25)
    check("milieu de vie (plateau) -> fade 1.0", abs(fade[0] - 1.0) < 1e-6, f"fade={fade[0]}")


def test_exact_smoothstep_formula_mid_rise():
    life = 1.0
    fade_in = 0.2
    age = np.array([0.1], dtype=np.float32)  # t = 0.1, milieu de la rampe montante (0.2)
    type_ = np.array([1], dtype=np.int32)  # foam
    fade = compute_age_fade(age, type_, 1.0, life, 1.0, fade_in_frac=fade_in, fade_out_frac=0.25)
    tr = 0.1 / fade_in
    expected = _smoothstep(tr)
    check(
        "formule smoothstep exacte en milieu de rampe montante",
        abs(fade[0] - expected) < 1e-6,
        f"fade={fade[0]} expected={expected}",
    )


def test_exact_smoothstep_formula_mid_fall():
    life = 1.0
    fade_out = 0.4
    age = np.array([0.9], dtype=np.float32)  # t = 0.9, dans la rampe descendante [0.6, 1.0]
    type_ = np.array([2], dtype=np.int32)  # bubble
    fade = compute_age_fade(age, type_, 1.0, 1.0, life, fade_in_frac=0.15, fade_out_frac=fade_out)
    fall_start = 1.0 - fade_out
    tf = (0.9 - fall_start) / fade_out
    expected = 1.0 - _smoothstep(tf)
    check(
        "formule smoothstep exacte en milieu de rampe descendante",
        abs(fade[0] - expected) < 1e-6,
        f"fade={fade[0]} expected={expected}",
    )


def test_life_zero_gives_fade_one():
    age = np.array([0.0, 5.0], dtype=np.float32)
    type_ = np.array([0, 1], dtype=np.int32)
    fade = compute_age_fade(age, type_, 0.0, 0.0, 0.0)
    check(
        "duree de vie nulle -> fade 1.0 partout (pas de division par zero)",
        bool(np.all(fade == 1.0)),
        f"fade={fade}",
    )


def test_life_negative_gives_fade_one():
    age = np.array([0.0], dtype=np.float32)
    type_ = np.array([2], dtype=np.int32)
    fade = compute_age_fade(age, type_, 1.0, 1.0, -1.0)
    check("duree de vie negative -> fade 1.0", abs(fade[0] - 1.0) < 1e-6, f"fade={fade[0]}")


def test_per_type_life_selection():
    # meme age absolu, vies differentes par type -> t different -> fade different.
    age = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    type_ = np.array([0, 1, 2], dtype=np.int32)
    life_spray, life_foam, life_bubble = 0.1, 1.0, 10.0
    fade = compute_age_fade(
        age, type_, life_spray, life_foam, life_bubble,
        fade_in_frac=0.5, fade_out_frac=0.5,
    )
    # spray : t = 0.5 (age 0.05 / life 0.1) -> proche du bord de la rampe montante.
    # bubble : t = 0.005 -> tres jeune, fade proche de 0.
    check(
        "vie par type affecte bien t (bulle plus jeune relativement que embrun)",
        fade[2] < fade[0],
        f"fade={fade}",
    )


def test_output_dtype_and_shape():
    age = np.linspace(0.0, 1.0, 10).astype(np.float32)
    type_ = np.zeros(10, dtype=np.int32)
    fade = compute_age_fade(age, type_, 1.0, 1.0, 1.0)
    check("dtype float32", fade.dtype == np.float32)
    check("shape preservee", fade.shape == age.shape)


def test_values_always_in_unit_range():
    rng = np.random.default_rng(3)
    age = rng.uniform(0.0, 3.0, size=500).astype(np.float32)
    type_ = rng.integers(0, 3, size=500).astype(np.int32)
    fade = compute_age_fade(age, type_, 1.0, 2.0, 1.5)
    check(
        "toutes les valeurs dans [0, 1]",
        bool(np.all(fade >= 0.0) and np.all(fade <= 1.0)),
        f"min={fade.min()} max={fade.max()}",
    )


def test_overlap_fades_no_negative_no_nan():
    # fade_in_frac + fade_out_frac >= 1 : chevauchement des deux rampes.
    age = np.linspace(0.0, 1.0, 21).astype(np.float32)
    type_ = np.zeros(21, dtype=np.int32)
    fade = compute_age_fade(age, type_, 1.0, 1.0, 1.0, fade_in_frac=0.7, fade_out_frac=0.6)
    check(
        "chevauchement des rampes : pas de NaN ni de valeur negative",
        bool(np.all(np.isfinite(fade)) and np.all(fade >= 0.0) and np.all(fade <= 1.0)),
        f"fade={fade}",
    )
    check("chevauchement : naissance -> 0", abs(fade[0] - 0.0) < 1e-6, f"fade[0]={fade[0]}")
    check("chevauchement : mort -> 0", abs(fade[-1] - 0.0) < 1e-6, f"fade[-1]={fade[-1]}")


def test_overlap_extreme_fade_in_only():
    # fade_in_frac = 1.0, fade_out_frac = 0.0 -> chevauchement degenere,
    # entierement rampe montante.
    age = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    type_ = np.zeros(3, dtype=np.int32)
    fade = compute_age_fade(age, type_, 1.0, 1.0, 1.0, fade_in_frac=1.0, fade_out_frac=0.0)
    check(
        "fade_in=1.0, fade_out=0.0 -> monotone croissant, sans NaN",
        bool(np.all(np.isfinite(fade))) and fade[0] <= fade[1] <= fade[2],
        f"fade={fade}",
    )


if __name__ == "__main__":
    test_birth_is_zero_with_fade_in()
    test_death_is_zero_with_fade_out()
    test_plateau_is_one()
    test_exact_smoothstep_formula_mid_rise()
    test_exact_smoothstep_formula_mid_fall()
    test_life_zero_gives_fade_one()
    test_life_negative_gives_fade_one()
    test_per_type_life_selection()
    test_output_dtype_and_shape()
    test_values_always_in_unit_range()
    test_overlap_fades_no_negative_no_nan()
    test_overlap_extreme_fade_in_only()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {FAILURES}")
        sys.exit(1)
    else:
        print("Tous les tests sont passes.")
        sys.exit(0)
