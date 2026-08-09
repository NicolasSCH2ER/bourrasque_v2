"""test_whitewatercache.py — verification autonome de
`extension/whitewatercache.py`.

Executable sans Blender : `python extension/tests/test_whitewatercache.py`.
"""

import pathlib
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from whitewatercache import (  # noqa: E402
    CHANNEL_VELOCITY,
    WhitewaterCacheReader,
    WhitewaterCacheWriter,
    WhitewaterProductionParams,
    diff_params,
    read_params,
)

_FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append((name, exc))
        print(f"[FAIL] {name}: {exc!r}")
    else:
        print(f"[ OK ] {name}")


def _params(**overrides):
    base = dict(
        influence_radius=0.06,
        spawn_rate=40.0,
        ta_min=0.1,
        ta_max=0.9,
        ta_weight=1.0,
        wc_min=0.1,
        wc_max=0.9,
        wc_weight=1.0,
        ke_min=0.1,
        ke_max=0.9,
        ke_weight=1.0,
        life_spray=1.5,
        life_foam=3.0,
        life_bubble=2.0,
        drag_spray=0.2,
        drag_foam=0.5,
        buoyancy_bubble=1.0,
        src_frames=120,
        src_n_max=200000,
    )
    base.update(overrides)
    return WhitewaterProductionParams(**base)


def _make_particles(n, seed):
    rng = np.random.default_rng(seed)
    pos = rng.random((n, 3), dtype=np.float32)
    type_ = rng.integers(0, 3, size=(n,), dtype=np.int32)
    size = rng.random((n,), dtype=np.float32)
    age = rng.random((n,), dtype=np.float32)
    return pos, type_, size, age


def test_roundtrip_variable_counts_no_velocity():
    counts = [10, 25, 0, 60, 5]
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "ww.bqw"
        expected = []
        with WhitewaterCacheWriter(path, params) as w:
            for i, n in enumerate(counts):
                pos, type_, size, age = _make_particles(n, seed=i)
                expected.append((pos, type_, size, age))
                w.append_frame(pos, type_, size, age)
        assert w.frames_written == len(counts)

        with WhitewaterCacheReader(path) as r:
            assert r.frame_count == len(counts)
            assert r.has_velocity is False
            assert diff_params(r.params, params) == []
            for i, n in enumerate(counts):
                assert r.particle_count(i) == n
                pos, type_, size, age = r.read_frame(i)
                assert pos.shape == (n, 3)
                assert type_.shape == (n,)
                assert size.shape == (n,)
                assert age.shape == (n,)
                exp_pos, exp_type, exp_size, exp_age = expected[i]
                assert np.array_equal(pos, exp_pos), f"frame {i}: positions differentes"
                assert np.array_equal(type_, exp_type), f"frame {i}: type differents"
                assert np.array_equal(size, exp_size), f"frame {i}: tailles differentes"
                assert np.array_equal(age, exp_age), f"frame {i}: ages differents"


def test_roundtrip_with_velocity():
    counts = [8, 16, 12]
    params = _params(spawn_rate=80.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "ww_vel.bqw"
        expected = []
        with WhitewaterCacheWriter(path, params, velocity=True) as w:
            assert w.has_velocity is True
            for i, n in enumerate(counts):
                pos, type_, size, age = _make_particles(n, seed=100 + i)
                vel = (pos * 2.0 - 1.0).astype(np.float32)
                expected.append((pos, type_, size, age, vel))
                w.append_frame(pos, type_, size, age, velocity=vel)

        with WhitewaterCacheReader(path) as r:
            assert r.has_velocity is True
            for i, n in enumerate(counts):
                pos, type_, size, age, vel = r.read_frame(i)
                exp_pos, exp_type, exp_size, exp_age, exp_vel = expected[i]
                assert np.array_equal(pos, exp_pos)
                assert np.array_equal(type_, exp_type)
                assert np.array_equal(size, exp_size)
                assert np.array_equal(age, exp_age)
                assert np.array_equal(vel, exp_vel), f"frame {i}: vitesses differentes"


def test_empty_frame_in_middle_of_sequence():
    """Aucune particule secondaire active sur une frame doit etre un cas
    normal, y compris entoure de frames non vides."""
    counts = [20, 0, 15]
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "empty_middle.bqw"
        with WhitewaterCacheWriter(path, params) as w:
            for i, n in enumerate(counts):
                pos, type_, size, age = _make_particles(n, seed=200 + i)
                w.append_frame(pos, type_, size, age)

        with WhitewaterCacheReader(path) as r:
            assert r.frame_count == 3
            pos, type_, size, age = r.read_frame(1)
            assert pos.shape == (0, 3)
            assert type_.shape == (0,)
            assert size.shape == (0,)
            assert age.shape == (0,)
            # les frames voisines restent lisibles normalement
            assert r.particle_count(0) == 20
            assert r.particle_count(2) == 15


def test_direct_access_matches_sequential_read():
    n_frames = 30
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "direct_access.bqw"
        expected = []
        with WhitewaterCacheWriter(path, params) as w:
            for i in range(n_frames):
                n = 3 + i
                pos, type_, size, age = _make_particles(n, seed=300 + i)
                expected.append((pos, type_, size, age))
                w.append_frame(pos, type_, size, age)

        # lecture sequentielle
        with WhitewaterCacheReader(path) as r_seq:
            seq_results = [r_seq.read_frame(i) for i in range(n_frames)]

        # lecture directe, dans le desordre, sur un lecteur frais
        with WhitewaterCacheReader(path) as r_direct:
            for i in (29, 0, 15, 7):
                pos, type_, size, age = r_direct.read_frame(i)
                seq_pos, seq_type, seq_size, seq_age = seq_results[i]
                assert np.array_equal(pos, seq_pos), (
                    f"frame {i}: acces direct differe de l'acces sequentiel (positions)"
                )
                assert np.array_equal(type_, seq_type)
                exp_pos, exp_type, exp_size, exp_age = expected[i]
                assert np.array_equal(pos, exp_pos)
                assert np.array_equal(type_, exp_type)
                assert np.array_equal(size, exp_size)
                assert np.array_equal(age, exp_age)


def test_truncated_file_raises():
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "truncated.bqw"
        with WhitewaterCacheWriter(path, params) as w:
            for i in range(3):
                pos, type_, size, age = _make_particles(10 + i, seed=400 + i)
                w.append_frame(pos, type_, size, age)

        size_on_disk = path.stat().st_size
        with open(path, "r+b") as f:
            f.truncate(size_on_disk - 5)

        raised = False
        try:
            WhitewaterCacheReader(path)
        except ValueError:
            raised = True
        assert raised, "WhitewaterCacheReader aurait du lever ValueError sur fichier tronque"


def test_invalid_magic_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "bad_magic.bqw"
        with open(path, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 200)

        raised = False
        try:
            WhitewaterCacheReader(path)
        except ValueError as exc:
            raised = True
            assert "magi" in str(exc).lower()
        assert raised, "WhitewaterCacheReader aurait du lever ValueError sur magie invalide"


def test_truncated_header_raises_on_read_params():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "short_header.bqw"
        with open(path, "wb") as f:
            f.write(b"BQW1" + struct.pack("<i", 1))  # bien trop court

        raised = False
        try:
            read_params(path)
        except ValueError:
            raised = True
        assert raised, "read_params aurait du lever ValueError sur header tronque"


def test_read_params_without_loading_geometry():
    params = _params(influence_radius=0.08, spawn_rate=100.0)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "params_only.bqw"
        with WhitewaterCacheWriter(path, params) as w:
            for i in range(4):
                pos, type_, size, age = _make_particles(50, seed=500 + i)
                w.append_frame(pos, type_, size, age)

        got = read_params(path)
        assert diff_params(got, params) == []


def test_diff_params_reports_changed_fields():
    stored = _params(influence_radius=0.06, spawn_rate=40.0)
    same = _params(influence_radius=0.06, spawn_rate=40.0)
    assert diff_params(stored, same) == []

    changed = _params(influence_radius=0.1, spawn_rate=40.0)
    diffs = diff_params(stored, changed)
    assert set(diffs) == {"influence_radius"}, diffs

    changed_more = _params(influence_radius=0.06, spawn_rate=90.0)
    diffs2 = diff_params(stored, changed_more)
    assert set(diffs2) == {"spawn_rate"}, diffs2

    changed_all = _params(
        influence_radius=0.2,
        spawn_rate=10.0,
        ta_min=0.2,
        ta_max=0.8,
        ta_weight=2.0,
        wc_min=0.2,
        wc_max=0.8,
        wc_weight=2.0,
        ke_min=0.2,
        ke_max=0.8,
        ke_weight=2.0,
        life_spray=9.0,
        life_foam=9.0,
        life_bubble=9.0,
        drag_spray=9.0,
        drag_foam=9.0,
        buoyancy_bubble=9.0,
        src_frames=1,
        src_n_max=1,
    )
    diffs_all = diff_params(stored, changed_all)
    assert set(diffs_all) == set(
        f.name for f in __import__("dataclasses").fields(WhitewaterProductionParams)
    ), diffs_all


def test_diff_params_float32_rounding_is_not_a_false_positive():
    """Les parametres flottants passent par un float32 a l'ecriture ; les
    comparer directement en float64 Python produirait de faux positifs de
    simple imprecision de conversion, pas un vrai changement."""
    value = 0.1  # non representable exactement en binaire
    stored = _params(influence_radius=float(np.float32(value)))
    current = _params(influence_radius=value)
    assert diff_params(stored, current) == []


def test_channel_constant_value():
    # fige la valeur du bit pour eviter une regression silencieuse du
    # format sur disque si quelqu'un renumerote les canaux.
    assert CHANNEL_VELOCITY == 1


def test_append_frame_velocity_channel_mismatch_raises():
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mismatch.bqw"
        pos, type_, size, age = _make_particles(5, seed=1)
        vel = np.zeros((5, 3), dtype=np.float32)

        with WhitewaterCacheWriter(path, params, velocity=True) as w:
            raised = False
            try:
                w.append_frame(pos, type_, size, age)  # pas de vitesses fournies
            except ValueError:
                raised = True
            assert raised, "append_frame aurait du exiger les vitesses"

        path2 = pathlib.Path(tmp) / "mismatch2.bqw"
        with WhitewaterCacheWriter(path2, params, velocity=False) as w:
            raised = False
            try:
                w.append_frame(pos, type_, size, age, velocity=vel)  # canal non declare
            except ValueError:
                raised = True
            assert raised, "append_frame aurait du refuser des vitesses non declarees"


def main():
    check(
        "roundtrip_variable_counts_no_velocity",
        test_roundtrip_variable_counts_no_velocity,
    )
    check("roundtrip_with_velocity", test_roundtrip_with_velocity)
    check("empty_frame_in_middle_of_sequence", test_empty_frame_in_middle_of_sequence)
    check(
        "direct_access_matches_sequential_read",
        test_direct_access_matches_sequential_read,
    )
    check("truncated_file_raises", test_truncated_file_raises)
    check("invalid_magic_raises", test_invalid_magic_raises)
    check(
        "truncated_header_raises_on_read_params",
        test_truncated_header_raises_on_read_params,
    )
    check(
        "read_params_without_loading_geometry",
        test_read_params_without_loading_geometry,
    )
    check("diff_params_reports_changed_fields", test_diff_params_reports_changed_fields)
    check(
        "diff_params_float32_rounding_is_not_a_false_positive",
        test_diff_params_float32_rounding_is_not_a_false_positive,
    )
    check("channel_constant_value", test_channel_constant_value)
    check(
        "append_frame_velocity_channel_mismatch_raises",
        test_append_frame_velocity_channel_mismatch_raises,
    )

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) en echec.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
