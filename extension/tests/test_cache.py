"""test_cache.py — verification autonome de `extension/cache.py`.

Executable sans Blender : `python extension/tests/test_cache.py`.
"""

import os
import pathlib
import struct
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from cache import CacheReader, CacheWriter, cache_paths, ensure_cache_dir  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

_FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append((name, exc))
        print(f"[FAIL] {name}: {exc!r}")
    else:
        print(f"[ OK ] {name}")


def test_roundtrip():
    n = 100
    n_frames = 5
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "roundtrip.bqd"
        expected = []
        with CacheWriter(path, n) as w:
            for fr in range(n_frames):
                frame = np.full((n, 3), float(fr), dtype=np.float32)
                frame[:, 1] = fr * 10.0  # valeurs distinctes par frame/axe
                expected.append(frame.copy())
                w.append_frame(frame)
        assert w.frames_written == n_frames

        with CacheReader(path) as r:
            assert r.n_particles == n
            assert r.frame_count == n_frames
            for fr in range(n_frames):
                got = r.read_frame(fr)
                assert np.array_equal(got, expected[fr]), f"frame {fr} differe"


def test_interrupted_bake():
    n = 50
    announced_frames = 1000  # jamais atteint : le bake est interrompu
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "interrupted.bqd"
        try:
            with CacheWriter(path, n) as w:
                for fr in range(3):
                    w.append_frame(np.full((n, 3), float(fr), dtype=np.float32))
                raise RuntimeError("annulation simulee du bake")
        except RuntimeError:
            pass

        with CacheReader(path) as r:
            assert r.frame_count == 3, f"frame_count={r.frame_count}, 3 attendu"
            assert r.n_particles == n
            for fr in range(3):
                got = r.read_frame(fr)
                assert np.allclose(got, fr)


def test_malformed_frame_raises():
    n = 10
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "malformed.bqd"
        with CacheWriter(path, n) as w:
            bad = np.zeros((n * 3 - 1,), dtype=np.float32)
            raised = False
            try:
                w.append_frame(bad)
            except ValueError:
                raised = True
            assert raised, "append_frame aurait du lever ValueError"


def test_truncated_file_raises():
    n = 10
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "truncated.bqd"
        with CacheWriter(path, n) as w:
            for fr in range(3):
                w.append_frame(np.full((n, 3), float(fr), dtype=np.float32))

        # tronque le fichier de quelques octets (mord sur la table d'index)
        size = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.truncate(size - 5)

        raised = False
        try:
            CacheReader(path)
        except ValueError:
            raised = True
        assert raised, "CacheReader aurait du lever ValueError sur fichier tronque"


def test_missing_sidecar_returns_none():
    n = 10
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "nosidecar.bqd"
        with CacheWriter(path, n) as w:
            w.append_frame(np.zeros((n, 3), dtype=np.float32))
        with CacheReader(path) as r:
            assert r.read_materials() is None


def test_cache_paths_is_pure():
    """cache_paths ne doit avoir aucun effet de bord sur le disque (voir
    display.refresh, chemin chaud du scrub de timeline)."""
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = pathlib.Path(tmp) / "sub" / "cache"
        bqd_path, mat_path = cache_paths(cache_dir, "foo")
        assert not cache_dir.exists(), "cache_paths ne doit pas creer le dossier"
        assert bqd_path == cache_dir / "foo.bqd"
        assert mat_path == cache_dir / "foo.mat"


def test_ensure_cache_dir_creates_dir():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = pathlib.Path(tmp) / "sub" / "cache"
        assert not cache_dir.exists()
        ensure_cache_dir(cache_dir)
        assert cache_dir.is_dir()
        # Idempotent : un second appel ne doit pas lever.
        ensure_cache_dir(cache_dir)
        assert cache_dir.is_dir()


def test_interop_with_headless_dump():
    dam_bqd = REPO_ROOT / "dam.bqd"
    dam_mat = REPO_ROOT / "dam.mat"
    if not dam_bqd.exists() or not dam_mat.exists():
        raise AssertionError(
            f"dumps headless introuvables ({dam_bqd}, {dam_mat}) — "
            "regenerer avec build\\Release\\bourrasque_headless.exe dam 10 test_dam.bqd"
        )

    with CacheReader(dam_bqd) as r:
        print(f"       dam.bqd: n_particles={r.n_particles}, frame_count={r.frame_count}")
        assert r.n_particles > 0
        assert r.frame_count > 0
        assert r.n_particles == 208896, f"n_particles={r.n_particles}, 208896 attendu"
        assert r.frame_count == 240, f"frame_count={r.frame_count}, 240 attendu"
        assert r.is_variable is False, "dam.bqd est un v1, is_variable doit etre False"
        for k in (0, r.frame_count - 1, r.frame_count // 2):
            assert r.particle_count(k) == 208896

        first = r.read_frame(0)
        last = r.read_frame(r.frame_count - 1)
        for name, frame in (("first", first), ("last", last)):
            assert frame.min() >= 0.0, f"frame {name}: min={frame.min()} < 0"
            assert frame.max() <= 1.0, f"frame {name}: max={frame.max()} > 1"

        mats = r.read_materials()
        assert mats is not None
        assert mats.shape[0] == r.n_particles, (
            f"read_materials: {mats.shape[0]} octets, {r.n_particles} attendus"
        )


def test_v2_variable_count_roundtrip():
    counts = [10, 25, 25, 60, 100, 100]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "variable.bqd"
        expected = []
        with CacheWriter(path) as w:
            for fr, count in enumerate(counts):
                # valeurs distinctes et reconnaissables par frame : l'indice
                # de la particule module 1000 sur x, le numero de frame sur y
                frame = np.empty((count, 3), dtype=np.float32)
                frame[:, 0] = np.arange(count, dtype=np.float32)
                frame[:, 1] = float(fr)
                frame[:, 2] = float(fr) * 100.0 + 1.0
                expected.append(frame)
                w.append_frame(frame)
        assert w.frames_written == len(counts)

        with CacheReader(path) as r:
            assert r.is_variable is True
            assert r.frame_count == len(counts)
            assert r.n_particles == counts[-1]
            for fr, count in enumerate(counts):
                assert r.particle_count(fr) == count, (
                    f"frame {fr}: particle_count={r.particle_count(fr)}, "
                    f"{count} attendu"
                )
                got = r.read_frame(fr)
                assert got.shape == (count, 3)
                assert np.array_equal(got, expected[fr]), f"frame {fr} differe"


def test_v2_interrupted_bake():
    counts = [8, 15, 40]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "v2_interrupted.bqd"
        expected = []
        try:
            with CacheWriter(path) as w:
                for fr, count in enumerate(counts):
                    frame = np.full((count, 3), float(fr) + 1.0, dtype=np.float32)
                    expected.append(frame)
                    w.append_frame(frame)
                raise RuntimeError("annulation simulee du bake v2")
        except RuntimeError:
            pass

        with CacheReader(path) as r:
            assert r.is_variable is True
            assert r.frame_count == 3, f"frame_count={r.frame_count}, 3 attendu"
            for fr, count in enumerate(counts):
                assert r.particle_count(fr) == count
                got = r.read_frame(fr)
                assert np.array_equal(got, expected[fr]), f"frame {fr} differe"


def test_v2_random_access_is_indexed():
    """Lit les frames dans le desordre (49, 0, 25) et verifie l'exactitude,
    pour s'assurer que read_frame passe par la table d'index et ne derive
    pas d'une lecture sequentielle."""
    n_frames = 50
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "random_access.bqd"
        expected = []
        with CacheWriter(path) as w:
            for fr in range(n_frames):
                count = 5 + fr  # compte croissant, distinct par frame
                frame = np.full((count, 3), float(fr), dtype=np.float32)
                expected.append(frame)
                w.append_frame(frame)

        with CacheReader(path) as r:
            for fr in (49, 0, 25):
                got = r.read_frame(fr)
                assert np.array_equal(got, expected[fr]), f"frame {fr} differe"
                assert r.particle_count(fr) == 5 + fr


def test_v2_unknown_magic_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "bad_magic.bqd"
        # magie BQD2 correcte mais version inconnue : doit etre rejete
        # explicitement plutot que mal interprete.
        header = struct.pack("<4siiiq", b"BQD2", 99, 0, 0, 24)
        with open(path, "wb") as f:
            f.write(header)

        raised = False
        try:
            CacheReader(path)
        except ValueError:
            raised = True
        assert raised, "CacheReader aurait du lever ValueError sur version inconnue"


def test_v2_truncated_before_index_raises():
    counts = [4, 6, 9]
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "v2_truncated.bqd"
        with CacheWriter(path) as w:
            for fr, count in enumerate(counts):
                w.append_frame(np.full((count, 3), float(fr), dtype=np.float32))

        # tronque avant que la table d'index (ecrite par close()) ait pu
        # etre relue integralement : coupe juste apres le header.
        with open(path, "r+b") as f:
            f.truncate(30)  # header (24) + un peu de la premiere frame

        raised = False
        try:
            CacheReader(path)
        except ValueError:
            raised = True
        assert raised, (
            "CacheReader aurait du lever ValueError sur fichier tronque "
            "avant la table d'index"
        )


def test_v2_index_off_out_of_bounds_raises():
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "v2_bad_index_off.bqd"
        # header valide en apparence mais index_off pointe hors du fichier
        header = struct.pack("<4siiiq", b"BQD2", 2, 2, 5, 10_000_000)
        with open(path, "wb") as f:
            f.write(header)

        raised = False
        try:
            CacheReader(path)
        except ValueError:
            raised = True
        assert raised, (
            "CacheReader aurait du lever ValueError sur index_off hors bornes"
        )


def main():
    check("roundtrip", test_roundtrip)
    check("interrupted_bake", test_interrupted_bake)
    check("malformed_frame_raises", test_malformed_frame_raises)
    check("truncated_file_raises", test_truncated_file_raises)
    check("missing_sidecar_returns_none", test_missing_sidecar_returns_none)
    check("cache_paths_is_pure", test_cache_paths_is_pure)
    check("ensure_cache_dir_creates_dir", test_ensure_cache_dir_creates_dir)
    check("interop_with_headless_dump", test_interop_with_headless_dump)
    check("v2_variable_count_roundtrip", test_v2_variable_count_roundtrip)
    check("v2_interrupted_bake", test_v2_interrupted_bake)
    check("v2_random_access_is_indexed", test_v2_random_access_is_indexed)
    check("v2_unknown_magic_raises", test_v2_unknown_magic_raises)
    check("v2_truncated_before_index_raises", test_v2_truncated_before_index_raises)
    check("v2_index_off_out_of_bounds_raises", test_v2_index_off_out_of_bounds_raises)

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) en echec.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
