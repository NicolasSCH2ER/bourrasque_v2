"""test_rigidbody.py — verification autonome de `extension/rigidbody.py`
(proprietes massiques, echantillonnage de surface, quaternions et
transformations pour les colliders rigides dynamiques, jalon M17).

Tests analytiques (cube, pave, sphere approchee), pas des captures de la
sortie du code -- voir la spec de la tache A2/M17.

Executable sans Blender : `python extension/tests/test_rigidbody.py`, ou
via pytest (`pytest extension/tests/test_rigidbody.py`).
"""

import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rigidbody import (  # noqa: E402
    compose_body_transform,
    decompose_loc_rot,
    mass_properties,
    quat_from_matrix,
    quat_normalize,
    quat_to_matrix,
    surface_samples,
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


# ---------------------------------------------------------------------------
# Geometrie de test : boite (cube/pave), icosaedre subdivise (approx sphere)
# ---------------------------------------------------------------------------


def _box_mesh(sx, sy, sz, center=(0.0, 0.0, 0.0)):
    """Boite de dimensions (sx, sy, sz) centree sur `center`, triangulee,
    normales sortantes (orientation CCW vue de l'exterieur)."""
    cx, cy, cz = center
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    verts = np.array(
        [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ],
        dtype=np.float64,
    )
    # Chaque face : deux triangles, normale sortante.
    tris = np.array(
        [
            [0, 3, 2], [0, 2, 1],  # bas (z-)
            [4, 5, 6], [4, 6, 7],  # haut (z+)
            [0, 1, 5], [0, 5, 4],  # y-
            [2, 3, 7], [2, 7, 6],  # y+
            [0, 4, 7], [0, 7, 3],  # x-
            [1, 2, 6], [1, 6, 5],  # x+
        ],
        dtype=np.int64,
    )
    return verts, tris


def _icosphere(radius, subdivisions):
    """Icosaedre unite subdivise `subdivisions` fois puis projete sur la
    sphere de rayon `radius` -- approximation standard d'une sphere par un
    maillage triangule a normales sortantes."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    verts /= np.linalg.norm(verts[0])
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    verts = list(verts)
    midpoint_cache = {}

    def _midpoint(i1, i2):
        key = (min(i1, i2), max(i1, i2))
        if key in midpoint_cache:
            return midpoint_cache[key]
        mid = (verts[i1] + verts[i2]) / 2.0
        mid = mid / np.linalg.norm(mid)
        verts.append(mid)
        idx = len(verts) - 1
        midpoint_cache[key] = idx
        return idx

    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache.clear()
        for a, b, c in faces:
            ab = _midpoint(a, b)
            bc = _midpoint(b, c)
            ca = _midpoint(c, a)
            new_faces.append([a, ab, ca])
            new_faces.append([b, bc, ab])
            new_faces.append([c, ca, bc])
            new_faces.append([ab, bc, ca])
        faces = new_faces

    verts_arr = np.array(verts, dtype=np.float64) * radius
    tris_arr = np.array(faces, dtype=np.int64)
    return verts_arr, tris_arr


# ---------------------------------------------------------------------------
# mass_properties -- cube
# ---------------------------------------------------------------------------


def test_cube_volume_mass_com_inertia():
    a = 2.0
    rho = 3.0
    verts, tris = _box_mesh(a, a, a)
    volume, mass, com, inertia = mass_properties(verts, tris, rho)

    assert abs(volume - a**3) < 1e-9, f"volume={volume} attendu={a**3}"
    assert abs(mass - rho * a**3) < 1e-9
    assert np.allclose(com, [0.0, 0.0, 0.0], atol=1e-9), f"com={com}"

    expected_diag = mass * a * a / 6.0
    expected = np.diag([expected_diag, expected_diag, expected_diag])
    assert np.allclose(inertia, expected, atol=1e-7 * expected_diag), (
        f"inertia=\n{inertia}\nattendu diag={expected_diag}"
    )


def test_cube_translated_same_inertia_correct_com():
    """Le meme cube, translate hors de l'origine : le centre de masse suit
    la translation, mais l'inertie (rapportee au centre de masse) est
    INCHANGEE -- ce test attrape l'erreur classique du theoreme de Huygens
    oublie (inertie qui varie avec la position alors qu'elle ne devrait
    pas, une fois rapportee au COM)."""
    a = 2.0
    rho = 3.0
    offset = np.array([5.0, -3.0, 10.0])
    verts0, tris = _box_mesh(a, a, a)
    verts1, _ = _box_mesh(a, a, a, center=offset)

    _, _, com0, inertia0 = mass_properties(verts0, tris, rho)
    _, _, com1, inertia1 = mass_properties(verts1, tris, rho)

    assert np.allclose(com1, com0 + offset, atol=1e-9), f"com1={com1}"
    assert np.allclose(inertia1, inertia0, atol=1e-6), (
        f"inertie modifiee par une simple translation :\n{inertia0}\nvs\n{inertia1}"
    )


def test_box_anisotropic_inertia():
    a, b, c = 2.0, 3.0, 4.0
    rho = 1.5
    verts, tris = _box_mesh(a, b, c)
    volume, mass, com, inertia = mass_properties(verts, tris, rho)

    assert abs(volume - a * b * c) < 1e-9
    assert np.allclose(com, [0.0, 0.0, 0.0], atol=1e-9)

    expected = np.diag(
        [
            mass * (b * b + c * c) / 12.0,
            mass * (a * a + c * c) / 12.0,
            mass * (a * a + b * b) / 12.0,
        ]
    )
    assert np.allclose(inertia, expected, rtol=1e-6), f"inertia=\n{inertia}\nattendu=\n{expected}"

    # Pas de produits d'inertie hors diagonale pour une boite alignee aux
    # axes et centree.
    off_diag = inertia - np.diag(np.diag(inertia))
    assert np.max(np.abs(off_diag)) < 1e-6 * np.max(expected)


def test_flipped_normals_gives_same_result():
    a, b, c = 2.0, 3.0, 1.5
    rho = 2.0
    verts, tris = _box_mesh(a, b, c)
    tris_flipped = tris[:, [0, 2, 1]]  # inverse l'orientation de chaque triangle

    v0, m0, com0, i0 = mass_properties(verts, tris, rho)
    v1, m1, com1, i1 = mass_properties(verts, tris_flipped, rho)

    assert abs(v0 - v1) < 1e-9, f"volume: {v0} vs {v1}"
    assert abs(m0 - m1) < 1e-9
    assert np.allclose(com0, com1, atol=1e-9)
    assert np.allclose(i0, i1, atol=1e-6), f"inertie differente selon l'orientation:\n{i0}\nvs\n{i1}"
    assert v1 > 0.0, "le volume doit rester positif meme normales inversees"


def test_sphere_inertia_approx():
    """Icosaedre subdivise 3 fois (1280 faces) : approximation raisonnable
    d'une sphere, tolerance a quelques % justifiee par la discretisation
    (le maillage a un volume legerement INFERIEUR a la sphere ideale,
    l'icosphere etant inscrite -- on borne l'ecart plutot que d'exiger une
    egalite exacte, impossible avec un maillage a faces planes)."""
    r = 2.5
    rho = 4.0
    verts, tris = _icosphere(r, subdivisions=3)
    volume, mass, com, inertia = mass_properties(verts, tris, rho)

    expected_volume = 4.0 / 3.0 * math.pi * r**3
    rel_vol_err = abs(volume - expected_volume) / expected_volume
    assert rel_vol_err < 0.02, f"volume icosphere: erreur relative {rel_vol_err:.4f}"

    assert np.allclose(com, [0.0, 0.0, 0.0], atol=1e-6 * r)

    expected_diag = 2.0 / 5.0 * mass * r * r
    diag_vals = np.diag(inertia)
    rel_err = np.abs(diag_vals - expected_diag) / expected_diag
    print(f"    -> inertie icosphere: diag={diag_vals}, attendu={expected_diag:.6g}, "
          f"erreur relative max={rel_err.max():.4%}")
    assert np.all(rel_err < 0.02), f"erreur relative trop grande: {rel_err}"

    off_diag = inertia - np.diag(diag_vals)
    assert np.max(np.abs(off_diag)) < 0.02 * expected_diag


def test_degenerate_mesh_raises():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    tris = np.array([[0, 1, 2]], dtype=np.int64)  # un seul triangle plat, volume nul
    raised = False
    try:
        mass_properties(verts, tris, 1.0)
    except ValueError:
        raised = True
    assert raised, "un maillage plat/degenere aurait du lever ValueError"


def test_empty_mesh_raises():
    verts = np.zeros((0, 3), dtype=np.float64)
    tris = np.zeros((0, 3), dtype=np.int64)
    raised = False
    try:
        mass_properties(verts, tris, 1.0)
    except ValueError:
        raised = True
    assert raised, "un maillage vide aurait du lever ValueError"


# ---------------------------------------------------------------------------
# surface_samples
# ---------------------------------------------------------------------------


def test_surface_samples_on_surface_and_includes_vertices():
    verts, tris = _box_mesh(2.0, 2.0, 2.0)
    points = surface_samples(verts, tris, target_spacing=0.3, max_samples=5000)

    # Tous les sommets du maillage sont presents en tete.
    assert points.shape[0] >= verts.shape[0]
    assert np.allclose(points[: verts.shape[0]], verts)

    # Chaque point doit etre sur une face du cube (une des coordonnees
    # egale a +-1 en valeur absolue, les deux autres dans [-1, 1]).
    on_surface = np.any(np.isclose(np.abs(points), 1.0, atol=1e-9), axis=1)
    in_bounds = np.all(np.abs(points) <= 1.0 + 1e-9, axis=1)
    assert np.all(on_surface & in_bounds), "des points ne sont pas exactement sur la surface du cube"


def test_surface_samples_respects_cap():
    # subdivisions=1 -> 42 sommets, largement sous le plafond choisi : le
    # plafond ne peut etre respecte QUE si le budget de sommets tient
    # dedans (voir la docstring de `surface_samples` pour le cas limite
    # inverse, non teste ici).
    verts, tris = _icosphere(1.0, subdivisions=1)
    max_samples = 500
    points = surface_samples(verts, tris, target_spacing=0.01, max_samples=max_samples)
    assert points.shape[0] <= max_samples, f"{points.shape[0]} > {max_samples}"
    assert points.shape[0] >= verts.shape[0]


def test_surface_samples_vertex_count_exceeding_cap_is_documented_exception():
    """Cas limite : si `max_samples` est inferieur au nombre de sommets du
    maillage lui-meme, la garantie d'inclusion des sommets prime sur le
    plafond (voir docstring de `surface_samples`) -- le tableau renvoye
    depasse alors `max_samples`, ce n'est pas un bug."""
    verts, tris = _icosphere(1.0, subdivisions=3)  # 642 sommets
    max_samples = 500
    points = surface_samples(verts, tris, target_spacing=0.01, max_samples=max_samples)
    assert points.shape[0] == verts.shape[0], (
        "budget aleatoire nul attendu, seuls les sommets sont renvoyes"
    )
    assert points.shape[0] > max_samples


def test_surface_samples_deterministic():
    verts, tris = _icosphere(1.0, subdivisions=2)
    p1 = surface_samples(verts, tris, target_spacing=0.05, max_samples=2000)
    p2 = surface_samples(verts, tris, target_spacing=0.05, max_samples=2000)
    assert np.array_equal(p1, p2), "deux appels identiques doivent produire le meme resultat"


def test_surface_samples_empty_mesh():
    verts = np.zeros((0, 3), dtype=np.float64)
    tris = np.zeros((0, 3), dtype=np.int64)
    points = surface_samples(verts, tris, target_spacing=0.1, max_samples=100)
    assert points.shape == (0, 3)


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------


def test_quat_identity_gives_identity_matrix():
    R = quat_to_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_quat_roundtrip_random_rotations():
    rng = np.random.default_rng(123)
    for _ in range(200):
        q = quat_normalize(rng.normal(size=4))
        R = quat_to_matrix(q)
        q2 = quat_from_matrix(R)
        R2 = quat_to_matrix(q2)
        assert np.allclose(R, R2, atol=1e-9), f"roundtrip matrice divergent:\n{R}\nvs\n{R2}"


def test_quat_roundtrip_180_degrees_each_axis():
    """Cas qui casse la formule naive `sqrt(1+trace)` : une rotation de 180
    degres a une trace de -1 sur l'axe de rotation."""
    for axis in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
        angle = math.pi
        half = angle / 2.0
        q = np.array([math.cos(half), *(axis * math.sin(half))])
        R = quat_to_matrix(q)
        q2 = quat_from_matrix(R)
        R2 = quat_to_matrix(q2)
        assert np.allclose(R, R2, atol=1e-9), f"axe={axis}: roundtrip 180deg divergent"
        assert not np.any(np.isnan(q2)), f"axe={axis}: NaN produit par la formule naive"


def test_quat_normalize():
    q = quat_normalize(np.array([2.0, 0.0, 0.0, 0.0]))
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# compose_body_transform / decompose_loc_rot
# ---------------------------------------------------------------------------


def test_compose_identity_state_gives_exact_m0():
    rng = np.random.default_rng(7)
    m0 = np.eye(4)
    m0[:3, :3] = quat_to_matrix(quat_normalize(rng.normal(size=4)))
    m0[:3, 3] = rng.uniform(-10, 10, size=3)
    com0 = rng.uniform(-5, 5, size=3)

    identity_q = np.array([1.0, 0.0, 0.0, 0.0])
    m = compose_body_transform(com0, identity_q, com0, m0)

    assert np.array_equal(m, m0), "l'etat identite doit reproduire EXACTEMENT m0"


def test_compose_translation_only():
    m0 = np.eye(4)
    com0 = np.array([1.0, 2.0, 3.0])
    x = com0 + np.array([10.0, 0.0, 0.0])
    identity_q = np.array([1.0, 0.0, 0.0, 0.0])

    m = compose_body_transform(x, identity_q, com0, m0)
    assert np.allclose(m[:3, 3], [10.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(m[:3, :3], np.eye(3), atol=1e-12)


def test_decompose_loc_rot_roundtrip():
    rng = np.random.default_rng(42)
    q = quat_normalize(rng.normal(size=4))
    loc = rng.uniform(-10, 10, size=3)
    R = quat_to_matrix(q)
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = loc

    loc2, quat2, ok = decompose_loc_rot(m)
    assert ok is True
    assert np.allclose(loc2, loc, atol=1e-9)
    R2 = quat_to_matrix(quat2)
    assert np.allclose(R2, R, atol=1e-9)


def test_decompose_loc_rot_detects_non_uniform_scale():
    m = np.eye(4)
    m[0, 0] = 2.0
    m[1, 1] = 1.0
    m[2, 2] = 1.0
    _, _, ok = decompose_loc_rot(m)
    assert ok is False, "une echelle non uniforme doit etre detectee"


def test_decompose_loc_rot_uniform_scale_is_ok():
    """Une echelle UNIFORME (meme facteur sur les 3 axes) reste une
    rotation "a echelle uniforme pres" au sens de la spec : le quaternion
    extrait est exact (seule l'information d'echelle est perdue, pas la
    rotation elle-meme), donc `uniform_scale_ok` doit rester True -- au
    contraire d'une echelle non uniforme ou d'un cisaillement (voir les
    tests dedies ci-dessous)."""
    q = quat_normalize(np.array([0.4, -0.3, 0.6, 0.2]))
    R = quat_to_matrix(q)
    m = np.eye(4)
    m[:3, :3] = R * 2.5  # echelle uniforme x2.5 appliquee a une vraie rotation
    m[:3, 3] = [1.0, 2.0, 3.0]

    loc, quat2, ok = decompose_loc_rot(m)
    assert ok is True
    assert np.allclose(loc, [1.0, 2.0, 3.0])
    R2 = quat_to_matrix(quat2)
    assert np.allclose(R2, R, atol=1e-9), "la rotation extraite doit ignorer le facteur d'echelle"


def test_decompose_loc_rot_detects_shear():
    m = np.eye(4)
    m[0, 1] = 0.5  # cisaillement
    _, _, ok = decompose_loc_rot(m)
    assert ok is False, "un cisaillement doit etre detecte"


def test_decompose_loc_rot_pure_rotation_ok():
    q = quat_normalize(np.array([0.3, 0.5, -0.2, 0.7]))
    R = quat_to_matrix(q)
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = [1.0, -2.0, 3.0]
    _, _, ok = decompose_loc_rot(m)
    assert ok is True


if __name__ == "__main__":
    check("test_cube_volume_mass_com_inertia", test_cube_volume_mass_com_inertia)
    check("test_cube_translated_same_inertia_correct_com", test_cube_translated_same_inertia_correct_com)
    check("test_box_anisotropic_inertia", test_box_anisotropic_inertia)
    check("test_flipped_normals_gives_same_result", test_flipped_normals_gives_same_result)
    check("test_sphere_inertia_approx", test_sphere_inertia_approx)
    check("test_degenerate_mesh_raises", test_degenerate_mesh_raises)
    check("test_empty_mesh_raises", test_empty_mesh_raises)

    check("test_surface_samples_on_surface_and_includes_vertices", test_surface_samples_on_surface_and_includes_vertices)
    check("test_surface_samples_respects_cap", test_surface_samples_respects_cap)
    check("test_surface_samples_vertex_count_exceeding_cap_is_documented_exception", test_surface_samples_vertex_count_exceeding_cap_is_documented_exception)
    check("test_surface_samples_deterministic", test_surface_samples_deterministic)
    check("test_surface_samples_empty_mesh", test_surface_samples_empty_mesh)

    check("test_quat_identity_gives_identity_matrix", test_quat_identity_gives_identity_matrix)
    check("test_quat_roundtrip_random_rotations", test_quat_roundtrip_random_rotations)
    check("test_quat_roundtrip_180_degrees_each_axis", test_quat_roundtrip_180_degrees_each_axis)
    check("test_quat_normalize", test_quat_normalize)

    check("test_compose_identity_state_gives_exact_m0", test_compose_identity_state_gives_exact_m0)
    check("test_compose_translation_only", test_compose_translation_only)
    check("test_decompose_loc_rot_roundtrip", test_decompose_loc_rot_roundtrip)
    check("test_decompose_loc_rot_detects_non_uniform_scale", test_decompose_loc_rot_detects_non_uniform_scale)
    check("test_decompose_loc_rot_uniform_scale_is_ok", test_decompose_loc_rot_uniform_scale_is_ok)
    check("test_decompose_loc_rot_detects_shear", test_decompose_loc_rot_detects_shear)
    check("test_decompose_loc_rot_pure_rotation_ok", test_decompose_loc_rot_pure_rotation_ok)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} test(s) failed:")
        for name, exc in _FAILURES:
            print(f"  - {name}: {exc!r}")
        sys.exit(1)
    else:
        print("Tous les tests sont passes.")
        sys.exit(0)
