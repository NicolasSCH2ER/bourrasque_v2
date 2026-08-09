"""test_mesher.py — verification autonome de `lib.Mesher` (extension/lib.py).

Executable sans Blender : `python extension/tests/test_mesher.py`, ou via
pytest (`python -m pytest extension/tests`).

Requiert la DLL/so native compilee (`extension/bin/bourrasque.dll` ou
`.so`) et un GPU CUDA compatible, exactement comme `test_sdf_sphere.py` : ce
test parle reellement au coeur via `lib.Mesher`, il ne peut pas tourner sans
lui. Contrairement a `test_sdf_sphere.py`, il est ecrit pour etre IGNORE
proprement (pas en echec) si la DLL est absente du poste qui l'execute (ex.
CI sans build CUDA) : `_require_mesher` leve `unittest.SkipTest` sous
pytest, ou affiche un message et renvoie sans echouer en execution directe.
"""

import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import lib  # noqa: E402

_FAILURES = []


def check(name, fn):
    try:
        fn()
    except unittest.SkipTest as exc:
        print(f"[SKIP] {name}: {exc}")
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append((name, exc))
        print(f"[FAIL] {name}: {exc!r}")
    else:
        print(f"[ OK ] {name}")


def _require_mesher():
    """Leve `unittest.SkipTest` si la DLL native est introuvable, pour que
    ce test soit ignore proprement (pas en echec) sur un poste sans build
    CUDA — meme discipline que le reste de la suite (voir docstring de
    module)."""
    if not pathlib.Path(lib._DLL_PATH).is_file():
        raise unittest.SkipTest(
            f"DLL native introuvable ({lib._DLL_PATH}) : le coeur n'est pas "
            "compile sur ce poste, test ignore"
        )


def _sphere_points(n, radius, center, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return (np.asarray(center, dtype=np.float32) + radius * u).astype(np.float32)


def test_mesher_sphere_produces_bounded_mesh():
    """Un nuage sphere doit produire un maillage non vide dont les indices
    de triangle restent dans les bornes du tableau de sommets — le test le
    plus simple qui exerce reellement `run` -> `counts` -> `read` sans
    dependre d'aucune verite geometrique fine (voir `test_sdf_sphere.py`
    pour la verification quantitative du champ de distance sous-jacent)."""
    _require_mesher()

    cfg = lib.default_mesher_config()
    pts = _sphere_points(n=20000, radius=0.25, center=(0.5, 0.5, 0.5))

    with lib.Mesher(cfg) as mesher:
        mesher.run(pts)
        n_verts, n_tris = mesher.counts()
        assert n_verts > 0, "maillage vide sur un nuage spherique"
        assert n_tris > 0, "maillage vide (aucun triangle) sur un nuage spherique"

        verts, tris = mesher.read()

    assert verts.shape == (n_verts, 3)
    assert verts.dtype == np.float32
    assert tris.shape == (n_tris, 3)
    assert tris.dtype == np.int32

    assert tris.min() >= 0, "indice de triangle negatif"
    assert tris.max() < n_verts, (
        f"indice de triangle hors bornes ({tris.max()}) pour {n_verts} sommets"
    )

    # Les sommets reconstruits doivent rester au voisinage de la sphere
    # d'origine (grossierement, a quelques cellules pres) : un maillage
    # place n'importe ou dans le domaine signalerait un bug de conversion
    # d'espace ou de configuration, pas seulement un defaut de forme fine.
    dist_to_center = np.linalg.norm(verts - np.array([0.5, 0.5, 0.5]), axis=1)
    assert abs(float(dist_to_center.mean()) - 0.25) < 4.0 * cfg.cell_size


def test_mesher_empty_cloud_gives_empty_mesh():
    """Un nuage vide (n=0) est un cas NORMAL (voir `bq_mesher_run`,
    `bourrasque.h`) : le maillage resultant doit etre vide, pas une
    erreur."""
    _require_mesher()

    cfg = lib.default_mesher_config()
    empty = np.empty((0, 3), dtype=np.float32)

    with lib.Mesher(cfg) as mesher:
        mesher.run(empty)
        n_verts, n_tris = mesher.counts()
        assert n_verts == 0
        assert n_tris == 0

        verts, tris = mesher.read()

    assert verts.shape == (0, 3)
    assert tris.shape == (0, 3)


def test_mesher_crops_against_collider_half_space():
    """Un solide sous `y = 0.25` (champ de distance signee du collider,
    echantillonne AUX NOEUDS -- meme convention que `Sim.read_sdf`, voir
    `bq_mesher_set_collider_sdf`) doit rogner la surface reconstruite :
    aucun sommet ne doit descendre sous `y = 0.25` (a une cellule pres),
    meme quand le nuage de particules deborde largement sous ce plan
    (docs/plan-milestone-7.md, decision D5, critere V8)."""
    _require_mesher()

    cfg = lib.default_mesher_config()
    res = tuple(int(r) for r in cfg.grid_res)
    cell_size = float(cfg.cell_size)

    # Demi-espace solide sous y = 0.25 : negatif = interieur du solide
    # (convention du solveur), echantillonne AUX NOEUDS (valeur a (i,j,k) =
    # j*cell_size - 0.25, independante de i et k).
    j = np.arange(res[1], dtype=np.float32) * cell_size - 0.25
    sdf = np.broadcast_to(j[None, :, None], res).astype(np.float32)

    # Pave de particules qui deborde largement sous le plan solide
    # (y in [0.10, 0.55]), pour verifier que le rognage agit reellement.
    rng = np.random.default_rng(0)
    n = 20000
    pts = np.empty((n, 3), dtype=np.float32)
    pts[:, 0] = rng.uniform(0.35, 0.65, n).astype(np.float32)
    pts[:, 1] = rng.uniform(0.10, 0.55, n).astype(np.float32)
    pts[:, 2] = rng.uniform(0.35, 0.65, n).astype(np.float32)

    with lib.Mesher(cfg) as mesher:
        mesher.set_collider_sdf(sdf, res, cell_size)
        mesher.run(pts)
        n_verts, n_tris = mesher.counts()
        assert n_verts > 0, "maillage vide alors qu'un nuage epais est fourni"
        assert n_tris > 0
        verts, tris = mesher.read()

    assert verts.shape == (n_verts, 3)
    min_y = float(verts[:, 1].min())
    assert min_y >= 0.25 - cell_size, (
        f"un sommet descend a y={min_y:.5f}, sous le plan solide (0.25) de "
        f"plus d'une cellule ({cell_size:.5f})"
    )


def main():
    check(
        "mesher_sphere_produces_bounded_mesh",
        test_mesher_sphere_produces_bounded_mesh,
    )
    check(
        "mesher_empty_cloud_gives_empty_mesh",
        test_mesher_empty_cloud_gives_empty_mesh,
    )
    check(
        "mesher_crops_against_collider_half_space",
        test_mesher_crops_against_collider_half_space,
    )

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) en echec.")
        sys.exit(1)
    print("\nTous les tests sont passes (ou ignores).")


if __name__ == "__main__":
    main()
