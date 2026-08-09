"""foam.py — calcul pur du masque d'ecume par sommet (`bq_foam_mask`).

Aucune dependance a `bpy` : ce module est pur, importable et testable en
dehors de Blender (voir `extension/tests/test_foam.py`), au meme titre que
`transform.py`/`meshcache.py` dont il reprend la convention.

Contexte (docs/plan-milestone-14.md, section 3, decisions D3/D4) : le masque
d'ecume est une densite de proximite au whitewater, calculee cote PYTHON
(pas de nouveau canal CUDA/API C — `mesher.cu` et `whitewater.cu` restent
deliberement independants, la correlation se fait ici, ou les deux jeux de
donnees issus du bake sont deja disponibles) plutot que par un nouveau
couplage GPU entre les deux modules.

Noyau reutilise TEL QUEL : `k(s) = (1 - s^2)^3` pour `s = d/R < 1`, 0
au-dela (meme convention que `k_zhu_bridson_field`/l'echantillonnage
whitewater dans `mesher.cu`/`whitewater.cu`) — pas un nouveau noyau invente
cote Python.

Depend de scipy (`cKDTree`), fourni via un wheel bundle dans l'extension
(`extension/wheels/`, declare dans `blender_manifest.toml`) puisque
l'environnement Python embarque de Blender ne l'inclut pas nativement.
"""

import numpy as np
from scipy.spatial import cKDTree

__all__ = ("compute_foam_mask",)


def compute_foam_mask(mesh_verts, whitewater_pos, influence_radius):
    """Renvoie un tableau `(n_verts,)` float32 dans `[0, 1]` : la densite de
    proximite au whitewater pour chaque sommet de `mesh_verts`.

    `mesh_verts` (n_verts, 3) et `whitewater_pos` (n_ww, 3) doivent etre
    exprimes dans le MEME espace (en pratique l'espace SOLVEUR, avant
    transformation monde, puisque `influence_radius` est une grandeur
    solveur — voir `props.mesh_effective_radii`) : appliquer cette fonction
    a des positions deja converties en espace monde donnerait un rayon
    d'influence incoherent avec les distances mesurees.

    Pour chaque sommet, la densite est la somme des poids de noyau
    `w = (1 - (d/influence_radius)^2)^3` de toutes les particules whitewater
    a distance `d < influence_radius` (meme noyau que le reste du projet,
    voir docstring du module), CLAMPEE a 1.0 : une somme de poids peut
    depasser 1 des que plusieurs particules whitewater sont proches d'un
    meme sommet (poids non normalises par construction, chacun valant au
    plus 1 pour une particule confondue avec le sommet) — `min(1.0, sum_w)`
    est la normalisation la plus simple qui preserve a la fois le zero
    (aucune particule proche) et le plein (au moins une particule proche
    ou plusieurs a distance moderee), au prix de saturer l'information de
    densite tres locale au-dela du seuil (pas de distinction entre "une
    particule collee" et "dix particules collees") — juge suffisant pour un
    usage de masque de shading, qui n'a pas besoin de resolution fine
    au-dela de la saturation.

    Renvoie un tableau de zeros si `whitewater_pos` est vide ou si
    `mesh_verts` est vide (cas NORMAUX, jamais d'exception) : c'est
    l'appelant (`display.py`) qui a deja etabli ce cas via son propre
    parcours de cas normaux (cache absent, frame hors plage, etc.), cette
    fonction se contente de rester coherente avec un appel a vide.
    """
    n_verts = mesh_verts.shape[0]
    mask = np.zeros(n_verts, dtype=np.float32)

    if n_verts == 0 or whitewater_pos.shape[0] == 0 or influence_radius <= 0.0:
        return mask

    verts = np.ascontiguousarray(mesh_verts, dtype=np.float64)
    ww = np.ascontiguousarray(whitewater_pos, dtype=np.float64)

    tree = cKDTree(ww)
    # `query_ball_point` (vectorise sur `verts`) renvoie, pour chaque
    # sommet, la liste des indices de particules whitewater a distance
    # < influence_radius — un seul parcours KD-tree plutot qu'une boucle
    # O(n_verts * n_ww) naive.
    neighbor_lists = tree.query_ball_point(verts, r=float(influence_radius))

    inv_r = 1.0 / float(influence_radius)
    for i, neighbors in enumerate(neighbor_lists):
        if not neighbors:
            continue
        d = np.linalg.norm(ww[neighbors] - verts[i], axis=1)
        s = d * inv_r
        w = (1.0 - s * s) ** 3
        mask[i] = min(1.0, float(np.sum(w)))

    return mask
