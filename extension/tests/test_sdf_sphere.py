"""test_sdf_sphere.py — verification du champ de distance signee d'un
collider contre une VERITE ANALYTIQUE (une sphere).

Executable sans Blender : `python extension/tests/test_sdf_sphere.py`. Ne
depend PAS de bpy (la sphere est generee ici a la main, en pur
numpy/python — pas via `bpy.ops.mesh.primitive_ico_sphere_add`), mais
requiert la DLL/so native compilee (`extension/bin/bourrasque.dll` ou
`.so`) et un GPU CUDA compatible, exactement comme `lib.py` execute en
`__main__` : ce test parle reellement au coeur via `lib.Sim`, il ne peut
pas tourner sans lui.

Pour une sphere de centre `c` et de rayon `r`, le champ de distance signee
EXACT vaut `||x - c|| - r` en tout point `x` : negatif a l'interieur,
positif a l'exterieur, nul sur la surface. On compare ce champ analytique,
evalue aux noeuds de la grille du solveur (`i*dx`, meme convention que
`core/src/mlsmpm.cu::k_sdf_unsigned`), au champ que
`Sim.read_sdf` renvoie apres avoir transmis une icosphere triangulee au
coeur via `Sim.set_colliders`.

Tolerances : une sphere triangulee est un POLYEDRE, pas une sphere — l'ecart
entre les deux (la "fleche", ou sagitta, de chaque facette) est de l'ordre
de `edge_length^2 / (8*rayon)`, pas zero. Ce test calcule cette fleche a
partir du maillage REELLEMENT genere (pas d'une valeur ecrite en dur) et
en deduit :
  - une bande "ambigue" pres de la surface exacte, dans laquelle le signe
    du champ n'a pas de verite unique (le polyedre peut etre du "mauvais"
    cote de la sphere ideale a cet endroit precis sans que ce soit un bug) :
    exclue EXPLICITEMENT du test de signe, jamais silencieusement — voir
    l'assertion sur `n_ambiguous` qui borne sa taille et la rend visible ;
  - une tolerance de magnitude dans la bande proche de la surface, qui
    prend un multiple de sureté de la fleche (l'ecart facette/sphere,
    dominant) et y ajoute une marge `2e-3*dx` pour le decalage
    `BQ_SDF_NODE_EPS` (1e-3*dx) que le coeur applique au point
    d'echantillonnage pour eviter qu'un noeud tombe exactement sur une
    face (voir `core/src/mlsmpm.cu::k_sdf_unsigned`).

Le signe est verifie sur TOUS les noeuds SAUF ceux de la bande
ambigue (dont la taille est elle-meme bornee et rapportee) ; la magnitude
est verifiee sur la bande proche de la surface (`|distance analytique| <
3*dx`), conformement a la spec.
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import lib  # noqa: E402

_FAILURES = []


def check(name, fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        _FAILURES.append((name, exc))
        print(f"[FAIL] {name}: {exc!r}")
    else:
        print(f"[ OK ] {name}")


# ---------------------------------------------------------------------------
# Icosphere generee a la main (pas de dependance a bpy)
# ---------------------------------------------------------------------------


def _icosahedron():
    """12 sommets / 20 faces d'un icosahedre regulier centre a l'origine,
    sommets projetes sur la sphere unite. Liste de faces "standard"
    (Kahler, "Creating an icosphere mesh in code") — son orientation
    (normales sortantes ou non) n'est PAS supposee correcte ici : voir
    `icosphere_triangles`, qui la verifie et la corrige au besoin plutot
    que de s'y fier."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    verts = [np.array(v, dtype=np.float64) for v in verts]
    verts = [v / np.linalg.norm(v) for v in verts]
    return verts, faces


def _subdivide(verts, faces):
    """Une passe de subdivision (chaque triangle -> 4), sommets milieux
    projetes sur la sphere unite et dedupliques via un cache par arete
    (sans dedup, chaque sommet milieu serait duplique jusqu'a 6 fois)."""
    cache = {}

    def midpoint(i, j):
        key = (min(i, j), max(i, j))
        if key in cache:
            return cache[key]
        m = (verts[i] + verts[j]) / 2.0
        m = m / np.linalg.norm(m)
        verts.append(m)
        idx = len(verts) - 1
        cache[key] = idx
        return idx

    new_faces = []
    for a, b, c in faces:
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.append((a, ab, ca))
        new_faces.append((b, bc, ab))
        new_faces.append((c, ca, bc))
        new_faces.append((ab, bc, ca))
    return verts, new_faces


def icosphere_triangles(center, radius, subdivisions):
    """Triangles `(n, 3, 3)` float64 d'une icosphere de `subdivisions`
    niveaux de subdivision, centree en `center` de rayon `radius`.

    Orientation corrigee via le volume signe (formule des tetraedres
    `(origine, v0, v1, v2)`, meme formule que
    `sampling._evaluated_world_volume_and_bbox`) : le coeur determine le
    signe du champ de distance a partir de la normale
    `(v1-v0) x (v2-v0)` de chaque triangle (voir
    `core/src/mlsmpm.cu::k_sdf_unsigned`), qui doit pointer VERS
    L'EXTERIEUR pour que "interieur" -> signe negatif. On ne suppose pas
    que la liste de faces de `_icosahedron` a la bonne orientation : on la
    verifie et on la corrige si besoin, pour que ce test reste correct
    meme si cette hypothese s'averait fausse.
    """
    verts, faces = _icosahedron()
    for _ in range(subdivisions):
        verts, faces = _subdivide(verts, faces)

    verts_arr = np.array(verts, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int64)
    tris_unit = verts_arr[faces_arr]  # (n, 3, 3), sur la sphere unite

    v0 = tris_unit[:, 0]
    v1 = tris_unit[:, 1]
    v2 = tris_unit[:, 2]
    signed_vol = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2)))) / 6.0
    if signed_vol < 0.0:
        tris_unit = tris_unit[:, (0, 2, 1), :]

    tris_world = tris_unit * radius + np.array(center, dtype=np.float64)
    return tris_world


def _max_edge_length(tris_unit_radius):
    """Longueur d'arete maximale (rayon reel) sur `tris_unit_radius`
    `(n, 3, 3)`, pour estimer la fleche de la triangulation."""
    e0 = np.linalg.norm(tris_unit_radius[:, 0] - tris_unit_radius[:, 1], axis=1)
    e1 = np.linalg.norm(tris_unit_radius[:, 1] - tris_unit_radius[:, 2], axis=1)
    e2 = np.linalg.norm(tris_unit_radius[:, 2] - tris_unit_radius[:, 0], axis=1)
    return float(max(e0.max(), e1.max(), e2.max()))


# ---------------------------------------------------------------------------
# Test principal
# ---------------------------------------------------------------------------


def test_sphere_sdf_sign_and_magnitude():
    subdivisions = 4  # 20*4^4 = 5120 triangles : fleche << dx (voir plus bas)
    grid_res = 32
    dx = 1.0 / grid_res
    center = (0.5, 0.5, 0.5)
    radius = 0.28  # marge confortable avec les bords du domaine [0, 1]^3

    tris = icosphere_triangles(center, radius, subdivisions)
    max_edge = _max_edge_length(tris - np.array(center))
    sagitta = max_edge ** 2 / (8.0 * radius)
    print(
        f"  icosphere : {tris.shape[0]} triangles, arete max {max_edge:.5f}, "
        f"fleche estimee {sagitta:.6f} (dx={dx:.5f})"
    )
    # La fleche doit rester tres petite devant dx pour que la bande ambigue
    # (voir plus bas) ne concerne qu'une poignee de noeuds ; sinon ce
    # test ne serait pas discriminant. Verifie l'hypothese plutot que de la
    # supposer silencieusement.
    assert sagitta < 0.05 * dx, (
        f"fleche de la triangulation ({sagitta:.6f}) trop grande devant dx "
        f"({dx:.5f}) : augmenter `subdivisions`"
    )

    cfg = lib.default_config()
    cfg.grid_res[0] = grid_res
    cfg.grid_res[1] = grid_res
    cfg.grid_res[2] = grid_res
    cfg.cell_size = dx

    n_tri = tris.shape[0]
    vel = np.zeros_like(tris, dtype=np.float32)
    fric = np.zeros(n_tri, dtype=np.float32)

    with lib.Sim(cfg) as sim:
        # bq_set_colliders exige qu'au moins un materiau existe (voir
        # commentaire de `bq_set_colliders` dans mlsmpm.cu) ; sans rapport
        # avec le collider lui-meme, purement une contrainte d'ordre d'appel.
        sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4e4, gamma=3.0)
        sim.set_colliders(tris, vel, fric)
        sdf = sim.read_sdf()

    assert sdf.shape == (grid_res, grid_res, grid_res), sdf.shape

    # Champ analytique, evalue aux noeuds de la grille : i*dx sur chaque
    # axe, meme convention que k_sdf_unsigned. `read_sdf` renvoie un
    # tableau (res0,res1,res2) dont l'indexation [i,j,k] correspond a
    # l'identifiant de noeud id=(i*res.y+j)*res.z+k du coeur (ordre C,
    # dernier axe le plus rapide) : `indexing="ij"` reproduit exactement
    # cette disposition.
    idx = np.arange(grid_res)
    coord = idx * dx
    gx, gy, gz = np.meshgrid(coord, coord, coord, indexing="ij")
    dist_to_center = np.sqrt(
        (gx - center[0]) ** 2 + (gy - center[1]) ** 2 + (gz - center[2]) ** 2
    )
    analytic = dist_to_center - radius

    # Bande ambigue autour de la surface EXACTE : un noeud dont la
    # distance analytique tombe sous cette marge peut, a cause de la
    # facettisation (polyedre, pas une sphere exacte), se retrouver du
    # "mauvais" cote de la surface ideale sans que ce soit un defaut du
    # coeur -- exclue explicitement, jamais silencieusement (voir
    # l'assertion sur sa taille juste apres).
    sign_margin = max(4.0 * sagitta, 1e-6)
    ambiguous = np.abs(analytic) < sign_margin
    n_ambiguous = int(np.sum(ambiguous))
    n_total = int(analytic.size)
    print(
        f"  bande ambigue (|distance| < {sign_margin:.6f}) : "
        f"{n_ambiguous}/{n_total} noeuds"
    )
    assert n_ambiguous < 0.05 * n_total, (
        f"trop de noeuds dans la bande ambigue ({n_ambiguous}/{n_total}) : "
        "la triangulation ou la resolution de grille choisie rend ce test "
        "peu discriminant, revoir `subdivisions`/`grid_res`"
    )

    expected_sign = np.sign(analytic)
    actual_sign = np.sign(sdf.astype(np.float64))
    mismatched = (~ambiguous) & (expected_sign != actual_sign)
    n_mismatch = int(np.sum(mismatched))
    if n_mismatch:
        bad_idx = np.argwhere(mismatched)[:5]
        details = [
            f"{tuple(int(v) for v in ix)}: analytic={analytic[tuple(ix)]:.5f} "
            f"sdf={sdf[tuple(ix)]:.5f}"
            for ix in bad_idx
        ]
        raise AssertionError(
            f"{n_mismatch} noeud(s) hors bande ambigue avec un signe "
            f"incorrect (premieres : {details})"
        )

    # Magnitude dans la bande proche de la surface (spec : "la magnitude
    # dans la bande proche de la surface"). La tolerance couvre deux sources
    # d'ecart reelles, pas une marge arbitraire :
    #  - la fleche (sagitta) de la triangulation, terme dominant, avec un
    #    facteur de surete 2x (mesure : l'ecart observe depasse la fleche
    #    brute d'environ 50%, cf. rapport) ;
    #  - `2e-3*dx`, une marge pour le decalage `BQ_SDF_NODE_EPS = 1e-3*dx`
    #    que le coeur applique au point d'echantillonnage (voir
    #    `k_sdf_unsigned`), qui n'est PAS modelise par le champ analytique
    #    de ce test.
    # Le terme `0.5*dx` d'avant le passage a l'echantillonnage aux noeuds
    # n'a plus lieu d'etre : il compensait un decalage d'un demi-pas entre
    # les points analytiques (centre de cellule) et les points du coeur
    # (centre de cellule) qui n'existe plus maintenant que les deux cotes
    # de la comparaison utilisent i*dx.
    near_band = np.abs(analytic) < 3.0 * dx
    tol = 2.0 * sagitta + 2e-3 * dx
    diff = np.abs(sdf.astype(np.float64)[near_band] - analytic[near_band])
    max_diff = float(diff.max()) if diff.size else 0.0
    print(
        f"  bande proche (|distance| < {3.0 * dx:.5f}) : {int(near_band.sum())} "
        f"noeuds, ecart max {max_diff:.6f} (tolerance {tol:.6f})"
    )
    assert max_diff < tol, (
        f"ecart de magnitude trop grand dans la bande proche de la surface : "
        f"{max_diff:.6f} (tolerance {tol:.6f} = 2*fleche {2.0 * sagitta:.6f} + "
        f"2e-3*dx {2e-3 * dx:.6f})"
    )


def main():
    check("sphere_sdf_sign_and_magnitude", test_sphere_sdf_sign_and_magnitude)

    if _FAILURES:
        print(f"\n{len(_FAILURES)} test(s) en echec.")
        sys.exit(1)
    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
