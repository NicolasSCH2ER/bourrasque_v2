"""rigidbody.py — calcul pur cote Python pour les colliders rigides
dynamiques (jalon M17) : proprietes massiques d'un maillage, echantillonnage
de sa surface pour le solveur de contact, et algebre de quaternions/
transformations necessaire pour composer et decomposer l'etat rigide.

Aucune dependance a `bpy` : ce module est pur, importable et testable en
dehors de Blender (voir `extension/tests/test_rigidbody.py`), au meme titre
que `materials.py`/`foam.py` dont il reprend la convention. Le coeur CUDA
portera l'etat rigide (position `x`, quaternion `q`) et l'integration ; ce
module se contente d'alimenter ce solveur (proprietes massiques, points de
contact) et d'exploiter son resultat (composition/decomposition de la
matrice monde pour poser les keyframes) — aucune integration temporelle ici.

Convention de quaternion imposee, partagee avec le coeur CUDA : `(w, x, y,
z)`, comme `Object.rotation_quaternion` de Blender. Une divergence de
convention ici produirait un bug de rotation tres penible a diagnostiquer
(rotation qui "tremble" ou part dans le mauvais sens), donc c'est ecrit en
toutes lettres a chaque signature concernee plutot que suppose implicite.
"""

import numpy as np

__all__ = (
    "mass_properties",
    "surface_samples",
    "quat_normalize",
    "quat_to_matrix",
    "quat_from_matrix",
    "compose_body_transform",
    "decompose_loc_rot",
)

# Graine fixe de l'echantillonnage de surface : le solveur de contact
# apparie les impulsions d'un sous-pas a l'autre PAR INDICE d'echantillon
# (voir docstring de `surface_samples`) -- un jeu de points qui change d'un
# appel a l'autre casserait cet appariement silencieusement.
_SURFACE_SAMPLE_SEED = 0xB0117A5

# Tolerances utilisees par `decompose_loc_rot` pour juger une matrice
# "rotation a echelle uniforme pres" -- volontairement laches (1e-4) car la
# matrice qu'on decompose vient d'une multiplication de matrices monde
# Blender (float32 a l'origine, promues en float64), pas d'une rotation pure
# construite analytiquement : une tolerance a la precision machine
# declencherait des faux positifs sur des scenes parfaitement valides.
_SCALE_UNIFORM_EPS = 1e-4
_ORTHOGONALITY_EPS = 1e-4


# ---------------------------------------------------------------------------
# Proprietes massiques
# ---------------------------------------------------------------------------


def mass_properties(verts, tris, density):
    """Volume, masse, centre de masse et tenseur d'inertie 3x3 (au centre de
    masse, dans les memes axes que `verts`) d'un maillage FERME `(verts,
    tris)`, de densite uniforme `density` (kg/m3).

    `verts` : `(V, 3)`. `tris` : `(T, 3)` indices dans `verts`.

    Methode : decomposition du volume en tetraedres signes `(origine, v0,
    v1, v2)` sur chaque triangle -- meme principe que
    `sampling.py:_evaluated_world_volume_and_bbox` (volume = somme de
    `v0.(v1 x v2)/6`), prolonge ici aux moments d'ordre 1 (premier moment,
    donne le centre de masse) et d'ordre 2 (tenseur d'inertie), par
    integration analytique exacte sur chaque tetraedre plutot que par
    quadrature approchee -- entierement vectorise numpy, aucune boucle sur
    les triangles.

    Pour un tetraedre de sommets `(0, A, B, C)` (apex a l'origine), le
    tenseur du second moment `S = integrale(p p^T dV)` vaut, avec `V` le
    volume SIGNE du tetraedre :
        S = V * [ (1/10)(AA^T+BB^T+CC^T)
                  + (1/20)(AB^T+BA^T+AC^T+CA^T+BC^T+CB^T) ]
    (integration du parametrage barycentrique standard du simplexe ; les
    poids 1/10 et 1/20 sortent de `integrale(s^2)=1/60`,
    `integrale(st)=1/120` sur le simplexe unite, multiplies par le
    jacobien `6V`). Le tenseur d'inertie s'en deduit par
    `I = trace(S)*Id - S`, puis le theoreme de Huygens ramene `I` du repere
    a l'origine au centre de masse.

    Orientation : si le volume signe total ressort negatif (normales
    inversees), TOUTES les integrales (volume, premier moment, second
    moment) sont retournees du meme signe correcteur -- pas seulement le
    volume -- pour ne jamais produire un volume positif avec un centre de
    masse ou une inertie faux (les trois integrales partagent le meme
    facteur de signe par construction, une correction partielle serait
    incoherente).

    Leve `ValueError` si le volume est nul ou degenere (maillage ouvert,
    plat, ou vide) : c'est a l'appelant de nommer l'objet Blender fautif
    dans le message d'erreur, cette fonction ne connait que des tableaux.

    Renvoie `(volume, mass, com, inertia)` : `volume` (float, toujours
    positif), `mass` (float), `com` `(3,)`, `inertia` `(3, 3)`.
    """
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)

    if verts.shape[0] == 0 or tris.shape[0] == 0:
        raise ValueError(
            "mass_properties: maillage vide (aucun sommet ou aucun triangle)"
        )

    A = verts[tris[:, 0]]
    B = verts[tris[:, 1]]
    C = verts[tris[:, 2]]

    # Volume signe de chaque tetraedre (origine, A, B, C).
    tet_vol = np.einsum("ti,ti->t", A, np.cross(B, C)) / 6.0
    total_signed_vol = float(np.sum(tet_vol))

    # Seuil de degenerescence relatif a l'echelle du maillage (une epsilon
    # absolue serait fausse aussi bien pour un grain de sable que pour un
    # iceberg) : le cube de la diagonale de la bbox est l'echelle naturelle
    # d'un volume.
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    vol_eps = max(bbox_diag, 1e-9) ** 3 * 1e-9
    if abs(total_signed_vol) < vol_eps:
        raise ValueError(
            f"mass_properties: volume nul ou degenere (volume signe="
            f"{total_signed_vol:.6g}, seuil={vol_eps:.6g}) -- maillage "
            "probablement ouvert ou plat"
        )

    sign = 1.0 if total_signed_vol >= 0.0 else -1.0
    volume = sign * total_signed_vol

    # Premier moment (integrale de p dV), tetraedre a apex 0 :
    # M1_tet = V_tet * (A+B+C)/4.
    m1 = sign * np.sum(tet_vol[:, None] * (A + B + C) / 4.0, axis=0)
    com = m1 / volume

    def _outer_sum(P, Q):
        return np.einsum("ti,tj,t->ij", P, Q, tet_vol)

    diag_terms = _outer_sum(A, A) + _outer_sum(B, B) + _outer_sum(C, C)
    cross_terms = (
        _outer_sum(A, B)
        + _outer_sum(B, A)
        + _outer_sum(A, C)
        + _outer_sum(C, A)
        + _outer_sum(B, C)
        + _outer_sum(C, B)
    )
    second_moment_origin = sign * (diag_terms / 10.0 + cross_terms / 20.0)

    inertia_origin = np.trace(second_moment_origin) * np.eye(3) - second_moment_origin

    mass = density * volume
    inertia_origin_mass = density * inertia_origin

    # Huygens-Steiner : I_origine = I_com + m*(|c|^2 Id - c c^T), donc
    # I_com = I_origine - m*(|c|^2 Id - c c^T).
    shift = mass * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    inertia = inertia_origin_mass - shift

    return volume, mass, com, inertia


# ---------------------------------------------------------------------------
# Echantillonnage de surface
# ---------------------------------------------------------------------------


def surface_samples(verts, tris, target_spacing, max_samples):
    """Nuage de points repartis sur la surface `(verts, tris)`, dans le
    meme repere que `verts`, destine au solveur de contact rigide.

    Les sommets du maillage sont TOUJOURS inclus en tete du tableau renvoye
    (ils portent les coins de la geometrie, que l'echantillonnage aleatoire
    barycentrique rate systematiquement -- un coin a une probabilite nulle
    d'etre pioche par un tirage uniforme sur des triangles). Le reste des
    points est tire aleatoirement sur les triangles, un triangle etant
    choisi avec une probabilite proportionnelle a son aire (methode
    barycentrique standard `u=1-sqrt(r1), v=r2*sqrt(r1)` pour une
    distribution uniforme sur le triangle), avec une densite visant
    `target_spacing` entre points (nombre de points ~= aire_totale /
    target_spacing^2).

    Le nombre total est plafonne a `max_samples` : si la densite visee
    depasserait le plafond, le nombre de points ALEATOIRES est reduit en
    amont (pas de sous-tirage a posteriori) -- comme le tirage aleatoire
    est deja proportionnel a l'aire, en reduire le nombre ne biaise pas la
    repartition vers un bout du maillage, elle reste uniforme, juste moins
    dense. Si `max_samples` est inferieur au nombre de sommets du maillage
    lui-meme, le plafond ne peut pas etre respecte sans violer la garantie
    d'inclusion des sommets : cette fonction privilegie alors l'inclusion
    des sommets (le tableau renvoye peut depasser `max_samples` dans ce cas
    limite) -- c'est a l'appelant de garantir `max_samples >= V` en amont
    s'il veut une garantie stricte de plafond.

    Deterministe (graine fixe, voir `_SURFACE_SAMPLE_SEED`) : deux appels
    avec les memes arguments renvoient EXACTEMENT le meme tableau. Le
    solveur de contact apparie les impulsions d'un sous-pas a l'autre par
    indice d'echantillon ; un jeu de points qui change d'un appel a l'autre
    casserait cet appariement.

    Renvoie `(N, 3)` float64, avec `N = V + n_random` (voir ci-dessus pour
    le cas limite). Maillage vide -> tableau `(0, 3)`.
    """
    verts = np.asarray(verts, dtype=np.float64)
    tris = np.asarray(tris, dtype=np.int64)
    n_verts = verts.shape[0]

    if n_verts == 0 or tris.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    A = verts[tris[:, 0]]
    B = verts[tris[:, 1]]
    C = verts[tris[:, 2]]
    tri_areas = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
    total_area = float(np.sum(tri_areas))

    rng = np.random.default_rng(_SURFACE_SAMPLE_SEED)

    n_random = 0
    if total_area > 0.0 and target_spacing > 0.0:
        n_random = int(np.ceil(total_area / (target_spacing**2)))

    # Budget reserve aux points aleatoires : les sommets priment toujours
    # (voir docstring), donc on reduit d'abord n_random plutot que de
    # sous-tirer un ensemble deja construit.
    remaining_budget = max(max_samples - n_verts, 0)
    n_random = min(n_random, remaining_budget)

    if n_random > 0:
        tri_probs = tri_areas / total_area
        tri_idx = rng.choice(tris.shape[0], size=n_random, p=tri_probs)
        r1 = rng.random(n_random)
        r2 = rng.random(n_random)
        sqrt_r1 = np.sqrt(r1)
        u = 1.0 - sqrt_r1
        v = r2 * sqrt_r1
        w = 1.0 - u - v
        random_pts = (
            u[:, None] * A[tri_idx]
            + v[:, None] * B[tri_idx]
            + w[:, None] * C[tri_idx]
        )
    else:
        random_pts = np.zeros((0, 3), dtype=np.float64)

    return np.concatenate([verts, random_pts], axis=0)


# ---------------------------------------------------------------------------
# Quaternions et transformations
# ---------------------------------------------------------------------------


def quat_normalize(q):
    """Normalise un quaternion `(w, x, y, z)`. Un quaternion de norme quasi
    nulle (degenere, ne devrait jamais survenir sur un etat rigide valide)
    retombe sur l'identite `(1, 0, 0, 0)` plutot que de produire des NaN
    qui se propageraient silencieusement jusqu'a une keyframe posee."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


def quat_to_matrix(q):
    """Matrice de rotation `(3, 3)` d'un quaternion `(w, x, y, z)` --
    normalise en interne (accepte un quaternion legerement derive par
    l'integration numerique du solveur sans que l'appelant ait a y
    penser)."""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_from_matrix(R):
    """Quaternion `(w, x, y, z)` d'une matrice de rotation `(3, 3)`, par la
    methode de Shepperd (choix de la composante de plus grande magnitude
    parmi `w, x, y, z` avant de diviser). La formule naive en
    `sqrt(1+trace)` perd toute precision quand la trace approche -1 (angle
    proche de 180 degres) : diviser par un `sqrt` qui approche 0 amplifie
    le bruit numerique jusqu'a rendre le resultat inutilisable. Shepperd
    choisit systematiquement la branche dont le denominateur reste loin de
    zero."""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def compose_body_transform(x, q, com0, m0):
    """Matrice monde `(4, 4)` d'un objet dont le corps rigide est a l'etat
    `(x, q)` (position et orientation du solveur) :

        M = Translate(x) @ R(q) @ Translate(-com0) @ m0

    ou `com0` est le centre de masse INITIAL en espace monde et `m0` la
    matrice monde INITIALE de l'objet `(4, 4)`. Intuition : on ramene
    l'objet a son centre de masse (`Translate(-com0) @ m0`), on lui
    applique la rotation rigide autour de ce centre, puis on le replace a
    la position courante du centre de masse (`Translate(x)`). C'est cette
    matrice que l'appelant decompose (`decompose_loc_rot`) pour poser les
    keyframes.

    A l'etat identite (`x == com0`, `q` identite), le resultat est
    EXACTEMENT `m0` (a bit pres, sans erreur d'arrondi introduite par cette
    fonction) : `Translate(com0) @ Translate(-com0)` s'annule exactement en
    arithmetique flottante car les deux translations partagent les memes
    valeurs de `x`/`com0` -- c'est l'invariant qui garantit qu'un corps au
    repos ne saute pas a la premiere frame."""
    x = np.asarray(x, dtype=np.float64)
    com0 = np.asarray(com0, dtype=np.float64)
    m0 = np.asarray(m0, dtype=np.float64)

    R = quat_to_matrix(q)

    translate_x = np.eye(4, dtype=np.float64)
    translate_x[:3, 3] = x

    rotate = np.eye(4, dtype=np.float64)
    rotate[:3, :3] = R

    translate_neg_com0 = np.eye(4, dtype=np.float64)
    translate_neg_com0[:3, 3] = -com0

    return translate_x @ rotate @ translate_neg_com0 @ m0


def decompose_loc_rot(m):
    """Decompose une matrice monde `(4, 4)` en translation `loc` `(3,)` et
    quaternion `quat` `(4,)` (`w, x, y, z`).

    `uniform_scale_ok` est `False` si la partie lineaire `m[:3, :3]` n'est
    pas une rotation a echelle uniforme pres (echelle non uniforme, ou
    cisaillement, ou reflexion) : dans ce cas, `quat` est tout de meme
    calcule (a partir de la colonne normalisee la plus proche d'une base
    orthonormee) pour ne jamais lever d'exception -- l'appelant en fera un
    avertissement plutot qu'un blocage, car un objet a l'echelle non
    uniforme reste affichable, juste avec une decomposition approximative.

    Methode de detection : normalise chaque colonne de `m[:3, :3]` (donne
    l'echelle par axe), verifie que les trois normes sont egales (echelle
    uniforme) ET que la matrice normalisee est bien orthogonale a
    determinant positif (pas de cisaillement, pas de reflexion)."""
    m = np.asarray(m, dtype=np.float64)
    loc = m[:3, 3].copy()
    linear = m[:3, :3]

    col_norms = np.linalg.norm(linear, axis=0)
    uniform_scale_ok = True

    if np.any(col_norms < 1e-12):
        # Colonne(s) degeneree(s) (echelle nulle sur un axe) : pas de base
        # orthonormee a en extraire, la rotation resultante sera arbitraire.
        uniform_scale_ok = False
        col_norms = np.where(col_norms < 1e-12, 1.0, col_norms)

    R = linear / col_norms[np.newaxis, :]

    if uniform_scale_ok:
        scale_ref = max(float(col_norms.max()), 1.0)
        if (col_norms.max() - col_norms.min()) > _SCALE_UNIFORM_EPS * scale_ref:
            uniform_scale_ok = False

    if uniform_scale_ok:
        ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
        if ortho_err > _ORTHOGONALITY_EPS:
            uniform_scale_ok = False

    if uniform_scale_ok and np.linalg.det(R) < 0.0:
        # Reflexion (determinant negatif) : pas une rotation, meme si les
        # colonnes sont orthonormees.
        uniform_scale_ok = False

    quat = quat_from_matrix(R)
    return loc, quat, uniform_scale_ok
