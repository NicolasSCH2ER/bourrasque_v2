"""Echantillonnage de l'interieur d'un maillage sur le reseau du solveur.

Aujourd'hui un emetteur Bourrasque est sa boite englobante (`bq_emit_box`,
voir `props.py::emitter_bounds_solver` / `estimate_particle_count`). Ce
module calcule, cote Blender, quels points du reseau du solveur tombent a
l'INTERIEUR de la forme reelle d'un maillage, pour les passer ensuite a
`bq_emit_points` (voir `core/include/bourrasque.h` et
`docs/plan-milestone-4.md`, decision D3/D4).

Repere de travail retenu : le test d'intersection rayon/maillage
(`BVHTree.ray_cast`) se fait entierement en coordonnees MONDE, sur le
maillage EVALUE (modificateurs appliques) de l'objet. Attention :
`BVHTree.FromObject` construit son arbre en espace LOCAL de l'objet (sans
appliquer `matrix_world`), ce n'est PAS le repere monde malgre les
apparences sur un objet a l'origine sans transformation. Le BVH est donc
construit ici via `BVHTree.FromPolygons` a partir des sommets du maillage
evalue explicitement transformes par `matrix_world`, pour que le meme
repere (monde) serve a la fois a la geometrie et aux rayons lances. Le
reseau de points candidats, en revanche, est genere directement en espace
SOLVEUR (voir plus bas pourquoi), puis chaque candidat est converti en
monde juste avant son lancer de rayon via `transform.solver_to_world_array`.
La fonction publique renvoie des points en espace solveur (float32), le
repere attendu par `Sim.emit_box` / l'appel a venir de `bq_emit_points`.

Pourquoi generer le reseau en espace solveur et pas en espace monde : le
solveur echantillonne son reseau a pas `spacing = dx / ppc_axis` (`dx` =
taille de maille, uniforme sur les trois axes, voir
`props.domain_resolution`), aux positions `spacing/2 + k*spacing` sur
CHAQUE axe solveur, dans le pave `[0, size[0]] x [0, size[1]] x [0,
size[2]]` (voir `bq_emit_box`, `core/src/mlsmpm.cu`). Pour qu'un emetteur
en forme de maillage produise
une densite coherente avec un emetteur en mode bloc, ses particules
doivent tomber sur ce meme reseau GLOBAL (ancre a l'origine du domaine),
pas sur un reseau ancre au coin de la bbox de l'objet. Generer les
candidats directement en espace solveur est aussi la maniere la plus
directe de garantir cet alignement a la precision flottante pres.

Methode d'interieur imposee : parite de rayons (BVH), PAS
`closest_point_on_mesh` + signe de la normale — cette derniere est plus
rapide mais fausse sur tout maillage non convexe (tore, tuyau,
personnage), precisement le cas d'usage vise. Voir
`docs/plan-milestone-4.md`, decision D4.
"""

import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .transform import solver_to_world_array, world_to_solver

__all__ = (
    "sample_mesh_interior",
    "check_mesh_closed",
    "estimate_mesh_sample_count",
    "evaluated_world_mesh",
    "evaluated_world_triangles",
)

# Direction de rayon deliberement non alignee sur les axes : un rayon parti
# exactement le long d'un axe a une probabilite bien plus elevee de raser
# une face ou de passer exactement par une arete, ce qui fausse la parite
# (double comptage ou intersection manquee). Cf. plan-milestone-4.md D4.
_RAY_DIR = Vector((0.5773, 0.5774, 0.5775)).normalized()

# Nombre maximal d'intersections comptees par rayon avant d'abandonner :
# filet de securite contre un maillage pathologique (auto-intersections
# degenerees, boucle infinie de rebonds numeriques).
_MAX_RAY_BOUNCES = 128


# ---------------------------------------------------------------------------
# Geometrie evaluee partagee
# ---------------------------------------------------------------------------


def _evaluated_object_and_depsgraph(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    return obj_eval, depsgraph


def _world_bbox_corners(obj_eval):
    """Les 8 coins de la bbox de `obj_eval` (deja evalue), en monde."""
    mat = obj_eval.matrix_world
    return [mat @ Vector(c) for c in obj_eval.bound_box]


def _solver_bbox(obj_eval, origin, size):
    """Bbox de `obj_eval` en espace solveur, `(lo, hi)` avec `lo <= hi` sur
    chaque axe (la conversion `world_to_solver` inverse l'axe sy monde ->
    sz solveur, voir `transform.py` et `props.py::emitter_bounds_solver`,
    dont la logique est reprise ici sur le maillage EVALUE plutot que sur
    `obj.bound_box` brut)."""
    corners_world = _world_bbox_corners(obj_eval)
    corners_solver = [
        world_to_solver((c.x, c.y, c.z), origin, size) for c in corners_world
    ]
    xs = [c[0] for c in corners_solver]
    ys = [c[1] for c in corners_solver]
    zs = [c[2] for c in corners_solver]
    lo = (min(xs), min(ys), min(zs))
    hi = (max(xs), max(ys), max(zs))
    return lo, hi


def _lattice_k_range(lo, hi, spacing):
    """Plage d'indices `k` (inclusive) tels que `spacing/2 + k*spacing`
    tombe dans `[lo, hi)`, `k` etant borne a `>= 0` puisque le reseau du
    solveur est ancre a l'origine du domaine (voir docstring du module).
    Renvoie `None` si la plage est vide."""
    if spacing <= 0.0 or hi <= lo:
        return None
    k_min = max(0, math.ceil((lo - spacing / 2.0) / spacing))
    k_max = math.floor((hi - spacing / 2.0) / spacing)
    if k_max < k_min:
        return None
    return int(k_min), int(k_max)


def _lattice_candidates_solver(lo, hi, spacing):
    """Points du reseau solveur global contenus dans la bbox `[lo, hi]`,
    en `(n, 3)` float64. Tableau vide (mais de forme `(0, 3)`) si aucun
    point ne tombe dans la bbox sur au moins un axe."""
    ranges = [_lattice_k_range(lo[a], hi[a], spacing) for a in range(3)]
    if any(r is None for r in ranges):
        return np.empty((0, 3), dtype=np.float64)

    axes = []
    for k_min, k_max in ranges:
        ks = np.arange(k_min, k_max + 1, dtype=np.float64)
        axes.append(spacing / 2.0 + ks * spacing)

    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)


def evaluated_world_mesh(obj_eval):
    """Sommets et triangles du maillage evalue de `obj_eval`, en espace
    MONDE : `(verts_world, tris)`, `verts_world` `(n_verts, 3)` float64,
    `tris` `(n_tris, 3)` int64 (indices dans `verts_world`).

    Brique de base PARTAGEE par `_world_bvh_from_evaluated` (BVH pour
    l'emission par maillage, voir docstring du module) et par
    `ops.BQ_OT_bake._update_colliders` (extraction des triangles de
    colliders, y compris animes, frame par frame) : c'est le seul endroit
    qui sait extraire un maillage evalue en espace monde (`to_mesh()`,
    `calc_loop_triangles()`, `foreach_get`, `matrix_world`,
    `to_mesh_clear()`), pour ne pas dupliquer cette logique.

    Chiralite : si `matrix_world` a un determinant NEGATIF (echelle
    negative sur un ou trois axes, modificateur Mirror — tres courant),
    l'orientation des faces est INVERSEE une fois transformee en espace
    monde (une reflexion inverse le sens du produit vectoriel des aretes
    d'un triangle). Le signe du champ de distance du coeur depend de cette
    orientation (normale = (v1-v0) x (v2-v0), voir
    `core/src/mlsmpm.cu::k_sdf_unsigned`) : sans correction, un collider en
    miroir produirait un champ de signe inverse (interieur vu comme
    exterieur). On retablit ici l'orientation en permutant deux sommets de
    chaque triangle (colonnes 1 et 2 de `tris`), au point UNIQUE de
    production des triangles monde, pour que tous les consommateurs
    (colliders, BVH d'emission par maillage) en beneficient sans avoir a y
    penser individuellement. Le test d'interieur par parite de rayons
    (`_count_ray_hits`) est lui insensible a l'orientation des faces (il ne
    depend que du nombre d'intersections, pas de leur sens) : cette
    correction ne change donc rien pour l'emission par maillage, seulement
    pour le signe du champ de distance des colliders.

    Tableaux vides (formes `(0, 3)`) si le maillage evalue n'a ni sommet ni
    triangle.
    """
    mesh = obj_eval.to_mesh()
    try:
        mesh.calc_loop_triangles()
        n_verts = len(mesh.vertices)
        n_tris = len(mesh.loop_triangles)
        if n_verts == 0 or n_tris == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.int64),
            )

        verts_local = np.empty(n_verts * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", verts_local)
        verts_local = verts_local.reshape(-1, 3)

        tris = np.empty(n_tris * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3)

        mat = np.array(obj_eval.matrix_world, dtype=np.float64)
        verts_world = verts_local @ mat[:3, :3].T + mat[:3, 3]

        if obj_eval.matrix_world.determinant() < 0.0:
            tris = tris[:, (0, 2, 1)]
    finally:
        obj_eval.to_mesh_clear()

    return verts_world, tris


def evaluated_world_triangles(obj_eval):
    """Triangles du maillage evalue de `obj_eval`, en espace MONDE,
    `(n_tri, 3, 3)` float64 (triangle, sommet, xyz), orientation
    corrigee pour la chiralite (voir `evaluated_world_mesh`). N'a
    aujourd'hui aucun appelant dans l'extension elle-meme : utilisee par
    `tests/test_sampling_triangles.py` pour verifier `evaluated_world_mesh`
    sous une forme plus directement comparable (triangles deja indexes,
    plutot que sommets + indices). Conservee comme utilitaire public au-dessus
    de `evaluated_world_mesh` (source de verite unique de l'extraction, voir
    sa docstring) pour tout appelant futur qui prefererait cette forme.
    """
    verts_world, tris = evaluated_world_mesh(obj_eval)
    if tris.shape[0] == 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    return verts_world[tris]


def _world_bvh_from_evaluated(obj_eval):
    """Construit un `BVHTree` en espace MONDE a partir du maillage evalue
    de `obj_eval` : `BVHTree.FromObject` construit son arbre en espace
    LOCAL (sans appliquer `matrix_world`), ce qui desaccorde geometrie et
    rayons des que l'objet est deplace/tourne/mis a l'echelle (voir
    docstring du module). On reutilise donc `evaluated_world_mesh`, qui a
    deja transforme les sommets par `matrix_world`, pour construire l'arbre
    via `BVHTree.FromPolygons`."""
    verts_world, tris = evaluated_world_mesh(obj_eval)
    if verts_world.shape[0] == 0 or tris.shape[0] == 0:
        return None

    return BVHTree.FromPolygons(
        [Vector(v) for v in verts_world], tris.tolist(), all_triangles=True
    )


# ---------------------------------------------------------------------------
# Test d'interieur par parite de rayons
# ---------------------------------------------------------------------------


def _count_ray_hits(bvh, start, direction, epsilon):
    """Nombre d'intersections du rayon `start + t*direction` avec `bvh`,
    en relancant depuis chaque point d'impact (decale de `epsilon` le long
    du rayon pour eviter de re-toucher la meme face). `BVHTree.ray_cast` ne
    renvoie que la PREMIERE intersection, d'ou la boucle."""
    count = 0
    cur = start
    for _ in range(_MAX_RAY_BOUNCES):
        loc, _normal, _index, _dist = bvh.ray_cast(cur, direction)
        if loc is None:
            break
        count += 1
        cur = loc + direction * epsilon
    return count


def sample_mesh_interior(obj, origin, size, dx, ppc_axis):
    """Points du reseau du solveur contenus dans le volume REEL de `obj`
    (pas seulement sa bbox), en espace SOLVEUR, `(n, 3)` float32.

    `obj` doit etre un objet MESH. Le maillage evalue (modificateurs
    appliques) est utilise pour le BVH et pour la bbox de pre-filtrage.
    Ne teste par lancer de rayon que les points du reseau contenus dans
    cette bbox : c'est le seul filtrage gratuit possible avant la boucle de
    rayons, qui domine le cout de la fonction (voir docstring du module).

    `dx` est la taille de maille (uniforme, voir `props.domain_resolution`) :
    grandeur primitive depuis M5, plutot que `size / grid_res`.

    A n'appeler qu'une fois par bake (ou par emission INFLOW) : le cout est
    de l'ordre d'un rayon (potentiellement plusieurs rebonds) par point
    candidat, pas d'une operation par frame.
    """
    obj_eval, _depsgraph = _evaluated_object_and_depsgraph(obj)

    spacing = dx / ppc_axis
    if spacing <= 0.0:
        return np.empty((0, 3), dtype=np.float32)

    lo, hi = _solver_bbox(obj_eval, origin, size)
    candidates_solver = _lattice_candidates_solver(lo, hi, spacing)
    n = candidates_solver.shape[0]
    if n == 0:
        return np.empty((0, 3), dtype=np.float32)

    bvh = _world_bvh_from_evaluated(obj_eval)
    if bvh is None:
        return np.empty((0, 3), dtype=np.float32)

    # Rayon caracteristique du maillage : sert a choisir un decalage de
    # relance (epsilon) petit devant la geometrie mais grand devant le
    # bruit flottant, quelle que soit l'echelle de l'objet.
    corners_world = _world_bbox_corners(obj_eval)
    xs = [c.x for c in corners_world]
    ys = [c.y for c in corners_world]
    zs = [c.z for c in corners_world]
    diag = math.sqrt(
        (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2
    )
    epsilon = max(diag * 1e-5, 1e-6)

    world_coords = solver_to_world_array(
        candidates_solver.astype(np.float32), origin, size
    )

    inside = np.zeros(n, dtype=bool)
    for i in range(n):
        start = Vector((float(world_coords[i, 0]), float(world_coords[i, 1]), float(world_coords[i, 2])))
        hits = _count_ray_hits(bvh, start, _RAY_DIR, epsilon)
        inside[i] = (hits % 2) == 1

    return candidates_solver[inside].astype(np.float32)


# ---------------------------------------------------------------------------
# Detection de maillage ouvert / non-manifold
# ---------------------------------------------------------------------------


def check_mesh_closed(obj):
    """`(utilisable, message)` : `utilisable` est `False` si la parite de
    rayons sur `obj` serait ambigue (maillage ouvert ou non-manifold), et
    `message` est alors un avertissement en francais directement affichable
    dans le panneau. `message` vaut `""` si `utilisable` est `True`.

    Une arete est consideree normale si elle borde EXACTEMENT deux faces ;
    une arete isolee (`is_loose`, 0 face) ou de bord (1 face) rend le
    maillage ouvert, une arete partagee par 3 faces ou plus le rend
    non-manifold — dans les deux cas, un rayon peut traverser un "trou"
    sans que le nombre d'intersections change de parite au bon endroit.
    """
    if obj.type != "MESH":
        return False, "L'objet n'est pas un maillage."

    obj_eval, _depsgraph = _evaluated_object_and_depsgraph(obj)
    mesh = obj_eval.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.edges.ensure_lookup_table()

        open_edges = 0
        non_manifold_edges = 0
        for edge in bm.edges:
            n_faces = len(edge.link_faces)
            if n_faces < 2:
                open_edges += 1
            elif n_faces > 2:
                non_manifold_edges += 1
        bm.free()
    finally:
        obj_eval.to_mesh_clear()

    if open_edges == 0 and non_manifold_edges == 0:
        return True, ""

    if open_edges and non_manifold_edges:
        message = (
            f"Maillage non ferme : {open_edges} arete(s) de bord et "
            f"{non_manifold_edges} arete(s) non-manifold detectee(s). Le "
            "test d'interieur par lancer de rayons ne peut pas donner un "
            "resultat fiable. Bouchez les trous (Face > Combler la grille "
            "ou Ponter les boucles d'aretes) et corrigez la geometrie "
            "non-manifold (Maillage > Nettoyer > Rendre manifold) avant "
            "d'utiliser cet objet comme emetteur en mode Maillage."
        )
    elif open_edges:
        message = (
            f"Maillage ouvert : {open_edges} arete(s) de bord detectee(s) "
            "(au moins un trou dans la surface). Le test d'interieur par "
            "lancer de rayons produira un nuage de points troue ou "
            "aberrant. Bouchez le maillage (Face > Combler la grille ou "
            "Ponter les boucles d'aretes) avant de l'utiliser comme "
            "emetteur en mode Maillage."
        )
    else:
        message = (
            f"Maillage non-manifold : {non_manifold_edges} arete(s) "
            "partagee(s) par plus de deux faces. Le test d'interieur par "
            "lancer de rayons peut donner un resultat aberrant. Nettoyez "
            "la geometrie (Maillage > Nettoyer > Rendre manifold ou "
            "Fusionner par distance) avant de l'utiliser comme emetteur en "
            "mode Maillage."
        )
    return False, message


# ---------------------------------------------------------------------------
# Estimation rapide (sans lancer de rayons), pour le redessin du panneau
# ---------------------------------------------------------------------------


def _evaluated_world_volume_and_bbox(obj_eval):
    """Volume signe (valeur absolue) du maillage evalue de `obj_eval` en
    espace MONDE, et le volume de sa bbox monde, tous deux calcules de
    facon vectorisee (numpy) : aucune boucle Python par sommet ou par
    triangle, indispensable puisque cette fonction est appelee a chaque
    redessin du panneau.

    Le volume est la somme des volumes signes des tetraedres
    `(origine, v0, v1, v2)` sur chaque triangle de la surface — formule
    standard, valable uniquement si le maillage est ferme (un maillage
    ouvert donne un volume sans signification physique ; c'est a
    `check_mesh_closed` de prevenir l'artiste, pas a cette fonction de s'en
    proteger)."""
    mesh = obj_eval.to_mesh()
    try:
        mesh.calc_loop_triangles()
        n_verts = len(mesh.vertices)
        n_tris = len(mesh.loop_triangles)
        if n_verts == 0 or n_tris == 0:
            return 0.0, 0.0

        verts_local = np.empty(n_verts * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", verts_local)
        verts_local = verts_local.reshape(-1, 3)

        tris = np.empty(n_tris * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(-1, 3)

        mat = np.array(obj_eval.matrix_world, dtype=np.float64)
        verts_world = verts_local @ mat[:3, :3].T + mat[:3, 3]

        v0 = verts_world[tris[:, 0]]
        v1 = verts_world[tris[:, 1]]
        v2 = verts_world[tris[:, 2]]
        signed_vol = float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2)))) / 6.0
        mesh_volume = abs(signed_vol)

        mins = verts_world.min(axis=0)
        maxs = verts_world.max(axis=0)
        extent = maxs - mins
        bbox_volume = float(extent[0] * extent[1] * extent[2])
    finally:
        obj_eval.to_mesh_clear()

    return mesh_volume, bbox_volume


def estimate_mesh_sample_count(obj, origin, size, dx, ppc_axis):
    """Estimation RAPIDE (sans lancer un seul rayon) du nombre de points
    que renverrait `sample_mesh_interior(obj, ...)`.

    APPROXIMATION : compte les points du reseau dans la bbox de `obj`
    (comme un emetteur en mode bloc) puis applique le rapport `volume du
    maillage / volume de la bbox`, sous l'hypothese (fausse en general,
    mais suffisante pour un affichage indicatif) que les points a
    l'interieur du maillage sont distribues dans la bbox avec la meme
    densite que le reseau global. Sur une forme tres concave (tore fin
    dans une grosse bbox, par exemple), l'estimation peut s'ecarter
    sensiblement du compte reel — c'est attendu et acceptable pour un
    indicateur de panneau, pas pour un compte exact.

    `dx` est la taille de maille (uniforme, voir `props.domain_resolution`) :
    grandeur primitive depuis M5, plutot que `size / grid_res`.

    Volume nul (maillage ouvert, degenere) -> renvoie 0.
    """
    obj_eval, _depsgraph = _evaluated_object_and_depsgraph(obj)

    spacing = dx / ppc_axis
    if spacing <= 0.0:
        return 0

    lo, hi = _solver_bbox(obj_eval, origin, size)
    bbox_count = 1
    for axis in range(3):
        rng = _lattice_k_range(lo[axis], hi[axis], spacing)
        if rng is None:
            return 0
        k_min, k_max = rng
        bbox_count *= (k_max - k_min + 1)

    mesh_volume, bbox_volume = _evaluated_world_volume_and_bbox(obj_eval)
    if bbox_volume <= 0.0 or mesh_volume <= 0.0:
        return 0

    ratio = min(mesh_volume / bbox_volume, 1.0)
    return int(round(bbox_count * ratio))
