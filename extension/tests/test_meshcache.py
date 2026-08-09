"""test_meshcache.py — verification autonome de `extension/meshcache.py`.

Executable sans Blender : `python extension/tests/test_meshcache.py`.
"""

import pathlib
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from meshcache import (  # noqa: E402
    CHANNEL_VERTEX_VELOCITY,
    MeshCacheReader,
    MeshCacheWriter,
    MeshProductionParams,
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
        mesh_res=(64, 64, 64),
        cell_size=0.05,
        influence_radius=0.1,
        particle_radius=0.02,
        collider_offset=0.01,
        smoothing_iters=2,
        min_component_tris=12,
        src_frames=120,
        src_n_max=200000,
    )
    base.update(overrides)
    return MeshProductionParams(**base)


def _make_mesh(n_verts, n_tris, seed):
    rng = np.random.default_rng(seed)
    verts = rng.random((n_verts, 3), dtype=np.float32)
    if n_tris:
        tris = rng.integers(0, max(n_verts, 1), size=(n_tris, 3), dtype=np.int32)
    else:
        tris = np.zeros((0, 3), dtype=np.int32)
    return verts, tris


def test_roundtrip_variable_counts_no_velocity():
    counts = [(10, 4), (25, 12), (0, 0), (60, 30), (5, 1)]
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mesh.bqm"
        expected = []
        with MeshCacheWriter(path, params) as w:
            for i, (nv, nt) in enumerate(counts):
                verts, tris = _make_mesh(nv, nt, seed=i)
                expected.append((verts, tris))
                w.append_frame(verts, tris)
        assert w.frames_written == len(counts)

        with MeshCacheReader(path) as r:
            assert r.frame_count == len(counts)
            assert r.has_vertex_velocity is False
            assert diff_params(r.params, params) == []
            for i, (nv, nt) in enumerate(counts):
                assert r.vertex_count(i) == nv
                assert r.triangle_count(i) == nt
                verts, tris, vel = r.read_frame(i)
                assert vel is None
                assert verts.shape == (nv, 3)
                assert tris.shape == (nt, 3)
                exp_verts, exp_tris = expected[i]
                assert np.array_equal(verts, exp_verts), f"frame {i}: sommets differents"
                assert np.array_equal(tris, exp_tris), f"frame {i}: triangles differents"


def test_roundtrip_with_velocity():
    counts = [(8, 3), (16, 7), (12, 5)]
    params = _params(mesh_res=(32, 48, 32))
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mesh_vel.bqm"
        expected = []
        with MeshCacheWriter(path, params, velocity=True) as w:
            assert w.has_vertex_velocity is True
            for i, (nv, nt) in enumerate(counts):
                verts, tris = _make_mesh(nv, nt, seed=100 + i)
                vel = (verts * 2.0 - 1.0).astype(np.float32)
                expected.append((verts, tris, vel))
                w.append_frame(verts, tris, velocities=vel)

        with MeshCacheReader(path) as r:
            assert r.has_vertex_velocity is True
            for i, (nv, nt) in enumerate(counts):
                verts, tris, vel = r.read_frame(i)
                exp_verts, exp_tris, exp_vel = expected[i]
                assert np.array_equal(verts, exp_verts)
                assert np.array_equal(tris, exp_tris)
                assert vel is not None
                assert np.array_equal(vel, exp_vel), f"frame {i}: vitesses differentes"


def test_empty_frame_in_middle_of_sequence():
    """Un maillage vide (aucun fluide dans le domaine) doit etre un cas
    normal, y compris entoure de frames non vides."""
    counts = [(20, 8), (0, 0), (15, 6)]
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "empty_middle.bqm"
        with MeshCacheWriter(path, params) as w:
            for i, (nv, nt) in enumerate(counts):
                verts, tris = _make_mesh(nv, nt, seed=200 + i)
                w.append_frame(verts, tris)

        with MeshCacheReader(path) as r:
            assert r.frame_count == 3
            verts, tris, vel = r.read_frame(1)
            assert verts.shape == (0, 3)
            assert tris.shape == (0, 3)
            assert vel is None
            # les frames voisines restent lisibles normalement
            assert r.vertex_count(0) == 20
            assert r.vertex_count(2) == 15


def test_direct_access_matches_sequential_read():
    n_frames = 30
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "direct_access.bqm"
        expected = []
        with MeshCacheWriter(path, params) as w:
            for i in range(n_frames):
                nv = 3 + i
                nt = 1 + i
                verts, tris = _make_mesh(nv, nt, seed=300 + i)
                expected.append((verts, tris))
                w.append_frame(verts, tris)

        # lecture sequentielle
        with MeshCacheReader(path) as r_seq:
            seq_results = [r_seq.read_frame(i) for i in range(n_frames)]

        # lecture directe, dans le desordre, sur un lecteur frais
        with MeshCacheReader(path) as r_direct:
            for i in (29, 0, 15, 7):
                verts, tris, _vel = r_direct.read_frame(i)
                seq_verts, seq_tris, _seq_vel = seq_results[i]
                assert np.array_equal(verts, seq_verts), (
                    f"frame {i}: acces direct differe de l'acces sequentiel (sommets)"
                )
                assert np.array_equal(tris, seq_tris), (
                    f"frame {i}: acces direct differe de l'acces sequentiel (triangles)"
                )
                exp_verts, exp_tris = expected[i]
                assert np.array_equal(verts, exp_verts)
                assert np.array_equal(tris, exp_tris)


def test_truncated_file_raises():
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "truncated.bqm"
        with MeshCacheWriter(path, params) as w:
            for i in range(3):
                verts, tris = _make_mesh(10 + i, 4 + i, seed=400 + i)
                w.append_frame(verts, tris)

        size = path.stat().st_size
        with open(path, "r+b") as f:
            f.truncate(size - 5)

        raised = False
        try:
            MeshCacheReader(path)
        except ValueError:
            raised = True
        assert raised, "MeshCacheReader aurait du lever ValueError sur fichier tronque"


def test_invalid_magic_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "bad_magic.bqm"
        with open(path, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 200)

        raised = False
        try:
            MeshCacheReader(path)
        except ValueError as exc:
            raised = True
            assert "magi" in str(exc).lower()
        assert raised, "MeshCacheReader aurait du lever ValueError sur magie invalide"


def test_truncated_header_raises_on_read_params():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "short_header.bqm"
        with open(path, "wb") as f:
            f.write(b"BQM1" + struct.pack("<i", 1))  # bien trop court

        raised = False
        try:
            read_params(path)
        except ValueError:
            raised = True
        assert raised, "read_params aurait du lever ValueError sur header tronque"


def test_read_params_without_loading_geometry():
    params = _params(mesh_res=(128, 96, 128), smoothing_iters=5)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "params_only.bqm"
        with MeshCacheWriter(path, params) as w:
            for i in range(4):
                verts, tris = _make_mesh(50, 20, seed=500 + i)
                w.append_frame(verts, tris)

        got = read_params(path)
        assert diff_params(got, params) == []


def test_diff_params_reports_changed_fields():
    stored = _params(mesh_res=(64, 64, 64), cell_size=0.05, smoothing_iters=2)
    same = _params(mesh_res=(64, 64, 64), cell_size=0.05, smoothing_iters=2)
    assert diff_params(stored, same) == []

    changed = _params(mesh_res=(128, 64, 64), cell_size=0.1, smoothing_iters=2)
    diffs = diff_params(stored, changed)
    assert set(diffs) == {"mesh_res", "cell_size"}, diffs

    changed_all = _params(
        mesh_res=(8, 8, 8),
        cell_size=0.2,
        influence_radius=0.3,
        particle_radius=0.05,
        collider_offset=0.02,
        smoothing_iters=9,
        src_frames=1,
        src_n_max=1,
    )
    diffs_all = diff_params(stored, changed_all)
    assert set(diffs_all) == {
        "mesh_res",
        "cell_size",
        "influence_radius",
        "particle_radius",
        "collider_offset",
        "smoothing_iters",
        "src_frames",
        "src_n_max",
    }, diffs_all


def test_diff_params_float32_rounding_is_not_a_false_positive():
    """Les parametres flottants passent par un float32 a l'ecriture ; les
    comparer directement en float64 Python produirait de faux positifs de
    simple imprecision de conversion, pas un vrai changement."""
    value = 0.1  # non representable exactement en binaire
    stored = _params(cell_size=float(np.float32(value)))
    current = _params(cell_size=value)
    assert diff_params(stored, current) == []


def test_channel_constant_value():
    # fige la valeur du bit pour eviter une regression silencieuse du
    # format sur disque si quelqu'un renumerote les canaux.
    assert CHANNEL_VERTEX_VELOCITY == 1


def test_append_frame_velocity_channel_mismatch_raises():
    params = _params()
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mismatch.bqm"
        verts, tris = _make_mesh(5, 2, seed=1)
        vel = np.zeros((5, 3), dtype=np.float32)

        with MeshCacheWriter(path, params, velocity=True) as w:
            raised = False
            try:
                w.append_frame(verts, tris)  # pas de vitesses fournies
            except ValueError:
                raised = True
            assert raised, "append_frame aurait du exiger les vitesses"

        path2 = pathlib.Path(tmp) / "mismatch2.bqm"
        with MeshCacheWriter(path2, params, velocity=False) as w:
            raised = False
            try:
                w.append_frame(verts, tris, velocities=vel)  # canal non declare
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
