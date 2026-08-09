"""display.py — affichage des particules dans le viewport et relecture du
cache `.bqd` au scrub de la timeline.

Le maillage affiche (`Bourrasque_Particles`) n'a que des sommets, aucune
arete ni face : chaque sommet est une particule, sa position est ecrite
directement depuis le cache converti en espace monde (voir `transform.py`,
source de verite pour la formule de conversion).

Ce module cote frontiere bpy/CUDA ne parle jamais a `lib.py` : il ne lit que
le cache `.bqd`/`.mat` via `cache.py`, ecrit par le bake (`ops.py`).
"""

import os

import bpy
import numpy as np

from . import cache, meshcache, whitewatercache
from .foam import compute_foam_mask
from .props import domain_transform, mesh_cache_path, mesh_effective_radii, whitewater_cache_path
from .transform import solver_to_world_array, solver_to_world_dir_array
from .whitewaterfade import compute_age_fade
from .whitewatervolume import compute_whitewater_volume_display_params

__all__ = (
    "ensure_particle_object",
    "write_material_attribute",
    "refresh",
    "clear_particle_object",
    "ensure_mesh_object",
    "refresh_mesh",
    "clear_mesh_object",
    "ensure_whitewater_object",
    "refresh_whitewater",
    "clear_whitewater_object",
    "register",
    "unregister",
)

_OBJECT_NAME = "Bourrasque_Particles"
_MATERIAL_ATTR = "bq_material"
_MESH_OBJECT_NAME = "Bourrasque_Mesh"
_MESH_FOAM_ATTR = "bq_foam_mask"
_WHITEWATER_OBJECT_NAME = "Bourrasque_Whitewater"
_WW_TYPE_ATTR = "bq_type"
_WW_SIZE_ATTR = "bq_size"
_WW_AGE_ATTR = "bq_age"
_WW_FADE_ATTR = "bq_fade"
# Nom EXACT reconnu nativement par Cycles pour le motion blur par attribut
# (mesh/point cloud) : voir docstring de `refresh_whitewater`.
_WW_VELOCITY_ATTR = "velocity"

# Asset volumetrique (jalon "affichage volumetrique du whitewater", voir
# `ops.py` section correspondante) : nom du groupe de noeuds Geometry Nodes,
# du materiau, du socket expose du groupe et du noeud Value dans le
# materiau qui recoit les parametres calcules par `whitewatervolume.py` a
# chaque frame — voir `_refresh_whitewater_volume_display`.
_WW_VOLUME_NODE_GROUP = "BQ Whitewater Volume"
_WW_VOLUME_MATERIAL = "BQ_Whitewater_Volume"
_WW_VOLUME_VOXEL_SOCKET_NAME = "Voxel Size"
_WW_VOLUME_DENSITY_NODE = "Value.DensityFactor"

# ---------------------------------------------------------------------------
# Etat module-level : cache d'un CacheReader ouvert et des buffers de lecture
# ---------------------------------------------------------------------------
# `refresh` est le chemin chaud du scrub de timeline (potentiellement 200k
# particules, appele a chaque frame). Rouvrir le fichier et reallouer les
# buffers a chaque appel couterait cher : on garde le dernier CacheReader
# ouvert tant que le chemin et l'horodatage du fichier ne changent pas, et on
# reutilise les buffers numpy d'une frame a l'autre.

_reader_state = {"path": None, "mtime": None, "reader": None}
_pos_buffer = None  # (n_max, 3) float32, espace solveur — reutilise entre appels
_world_buffer = None  # (n_max, 3) float32, espace monde — reutilise entre appels

# Position de repli (monde), exprimee comme un offset multiple de `size` par
# rapport a `origin` (le coin min du pave domaine) : sur chaque frame,
# `fallback = origin + _FALLBACK_WORLD_OFFSET * size`. Retenu : un coin
# diagonalement oppose au domaine, a une distance d'un cote du pave dans les
# trois directions negatives — donc juste a cote du domaine, ni au sein de
# son volume ni projete a une distance arbitrairement grande.
#
# Pourquoi pas l'origine du monde ou une valeur "infinie" (1e6, etc.) :
# l'objet n'a aucune arete/face, ces sommets sont donc rendus comme des
# points nus par le viewport (voir NOTE plus bas sur `point_size`) ; AUCUNE
# position finie ne les rend invisibles au sens strict — un artiste qui
# dezoome suffisamment ou active "View All" (Home) verra un nuage de points
# a cote du domaine. Une position "infinie" serait pire : elle ferait
# exploser la bounding box de l'objet (recalcul de `View All`, du clipping
# de camera, d'un eventuel bake de Geometry Nodes en aval) pour un gain nul
# en discretion. Ce choix minimise donc le rayon de la geometrie fantome
# (elle reste a l'echelle du domaine) au prix d'une limitation connue et
# assumee : les sommets replies restent, en toute rigueur, visibles.
# Rendre les points excedentaires reellement invisibles demanderait de
# masquer des sommets individuels via un modificateur (Geometry Nodes), hors
# perimetre de ce jalon (voir la meme note plus bas a propos de
# `point_size`).
_FALLBACK_WORLD_OFFSET = (-1.0, -1.0, -1.0)

# Type d'attribut entier retenu pour `bq_material`, determine paresseusement
# a la premiere ecriture et memorise pour ne pas retenter a chaque frame.
_int_attr_type = None


# ---------------------------------------------------------------------------
# ensure_particle_object
# ---------------------------------------------------------------------------


def ensure_particle_object(context, n_max):
    """Cree ou reutilise l'objet maillage `Bourrasque_Particles`.

    `n_max` sommets, aucune arete ni face. Avec un cache v2 (compte de
    particules variable d'une frame a l'autre), `n_max` est le compte de la
    DERNIERE frame de la simulation : le maillage est alloue une seule fois
    a cette taille maximale, et `refresh` replie les sommets excedentaires
    d'une frame donnee sur une position de repli plutot que de redimensionner
    le maillage a chaque frame (voir docstring de `refresh`).

    Reutilise l'objet existant tel quel si le compte de sommets correspond
    deja (pour ne pas perdre l'attribut materiau inutilement) ; reconstruit
    sa geometrie sinon. Ne cree jamais de doublon : un seul objet de ce nom
    existe a la fois.

    Reste selectionnable dans le viewport (l'artiste doit pouvoir le
    cliquer pour l'inspecter).
    """
    scene = context.scene
    obj = bpy.data.objects.get(_OBJECT_NAME)

    if obj is not None and obj.type != "MESH":
        # Un objet non-maillage porte deja ce nom (cas degenere, ex. import
        # externe) : on le pousse de cote plutot que de lui voler son nom
        # silencieusement.
        obj.name = f"{obj.name}.stale"
        obj = None

    if obj is None:
        mesh = bpy.data.meshes.new(_OBJECT_NAME)
        mesh.vertices.add(n_max)
        obj = bpy.data.objects.new(_OBJECT_NAME, mesh)
    elif len(obj.data.vertices) != n_max:
        mesh = obj.data
        mesh.clear_geometry()
        mesh.vertices.add(n_max)

    if obj.name not in scene.collection.all_objects:
        scene.collection.objects.link(obj)

    return obj


def clear_particle_object():
    """Vide la geometrie de `Bourrasque_Particles` s'il existe (0 sommet).

    Appele quand le cache disque est supprime (`bq.free_cache`) : sans ca,
    la derniere frame bakee resterait affichee dans le viewport alors que
    le fichier `.bqd` qui la sous-tend n'existe plus. Ne supprime pas
    l'objet lui-meme (evite de perturber une eventuelle selection/liens).
    """
    obj = bpy.data.objects.get(_OBJECT_NAME)
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    if len(mesh.vertices) == 0:
        return
    mesh.clear_geometry()
    mesh.update()
    obj.update_tag()


# ---------------------------------------------------------------------------
# write_material_attribute
# ---------------------------------------------------------------------------


def _ensure_material_attr_type(mesh):
    """Determine, une fois pour toutes, le type d'attribut entier a
    utiliser pour `bq_material`.

    `INT8` est le type le plus compact (les ids materiau sont deja stockes
    en uint8 dans le cache), mais sa disponibilite comme type d'attribut de
    maillage cree via `mesh.attributes.new` depend de la version de
    Blender. On sonde en creant puis supprimant un attribut probe ; en cas
    d'echec on replie sur `INT` (4 octets). Resultat memorise au niveau du
    module pour ne pas resonder a chaque frame.
    """
    global _int_attr_type
    if _int_attr_type is not None:
        return _int_attr_type

    probe_name = "_bq_material_probe"
    try:
        probe = mesh.attributes.new(name=probe_name, type="INT8", domain="POINT")
        mesh.attributes.remove(probe)
        _int_attr_type = "INT8"
    except (RuntimeError, TypeError):
        _int_attr_type = "INT"
    return _int_attr_type


def write_material_attribute(obj, mat_array):
    """Ecrit les ids materiau dans l'attribut `bq_material` (domaine POINT).

    Recree l'attribut s'il existe deja avec un domaine ou un type
    incompatible. `mat_array` peut etre `None` (pas de sidecar `.mat`) :
    dans ce cas, ne fait rien.
    """
    if mat_array is None:
        return

    mesh = obj.data
    arr = np.asarray(mat_array)
    n = len(mesh.vertices)
    if arr.size != n:
        raise ValueError(
            f"write_material_attribute: {arr.size} materiaux fournis, "
            f"{n} attendus (nombre de sommets de l'objet)"
        )

    attr_type = _ensure_material_attr_type(mesh)

    attr = mesh.attributes.get(_MATERIAL_ATTR)
    if attr is not None and (attr.domain != "POINT" or attr.data_type != attr_type):
        mesh.attributes.remove(attr)
        attr = None
    if attr is None:
        attr = mesh.attributes.new(name=_MATERIAL_ATTR, type=attr_type, domain="POINT")

    dtype = np.int8 if attr_type == "INT8" else np.int32
    attr.data.foreach_set("value", np.ascontiguousarray(arr, dtype=dtype))
    mesh.update()


# ---------------------------------------------------------------------------
# refresh — chemin chaud, appele a chaque changement de frame
# ---------------------------------------------------------------------------


def _open_reader_if_needed(bqd_path):
    """Renvoie un CacheReader valide pour `bqd_path`, en rouvrant seulement
    si le chemin ou l'horodatage du fichier a change. Renvoie `None` si le
    fichier est illisible."""
    global _pos_buffer, _world_buffer

    try:
        mtime = os.path.getmtime(bqd_path)
    except OSError:
        return None

    reader = _reader_state["reader"]
    if (
        reader is not None
        and _reader_state["path"] == bqd_path
        and _reader_state["mtime"] == mtime
    ):
        return reader

    if reader is not None:
        reader.close()

    try:
        reader = cache.CacheReader(bqd_path)
    except (OSError, ValueError):
        _reader_state["reader"] = None
        _reader_state["path"] = None
        _reader_state["mtime"] = None
        return None

    _reader_state["reader"] = reader
    _reader_state["path"] = bqd_path
    _reader_state["mtime"] = mtime
    # Le fichier a change (nouveau bake) : les buffers d'une eventuelle
    # simulation precedente ne sont plus fiables (compte de particules
    # different).
    _pos_buffer = None
    _world_buffer = None
    return reader


def refresh(scene):
    """Recharge la frame courante depuis le cache et met a jour la
    geometrie de `Bourrasque_Particles`.

    Avec un cache v2 (compte variable d'une frame a l'autre), le maillage
    est dimensionne une fois pour toutes a `n_max` (compte de la derniere
    frame, voir `ensure_particle_object`) : redimensionner la geometrie a
    chaque frame couterait cher au scrub d'un inflow (c'est exactement le
    chemin chaud optimise au jalon precedent). Une frame de `count < n_max`
    particules ecrit donc les `count` premieres positions normalement, puis
    replie les `n_max - count` sommets restants sur `_FALLBACK_WORLD_OFFSET`
    relatif au coin du domaine (voir la constante pour la justification de
    ce choix et ses limites) plutot que de les laisser a leur position
    d'une frame anterieure.

    Ne leve jamais : appelee a chaque changement de frame, une exception y
    transformerait le scrub de la timeline en avalanche d'erreurs console.
    Toute condition anormale (cache absent, index hors plage, objet
    manquant) se solde par un retour silencieux, sans modifier la geometrie
    existante.
    """
    global _pos_buffer, _world_buffer

    props = scene.bourrasque

    try:
        cache_dir = bpy.path.abspath(props.cache_dir)
        bqd_path, _ = cache.cache_paths(cache_dir, scene.name)
        bqd_path = str(bqd_path)
    except OSError:
        return

    if not os.path.isfile(bqd_path):
        return

    reader = _open_reader_if_needed(bqd_path)
    if reader is None:
        return

    index = scene.frame_current - props.frame_start
    if not (0 <= index < reader.frame_count):
        return

    transform = domain_transform(scene)
    if transform is None:
        return
    origin, size = transform

    obj = bpy.data.objects.get(_OBJECT_NAME)
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    n_max = reader.n_particles
    if len(mesh.vertices) != n_max:
        return

    try:
        _pos_buffer = reader.read_frame(index, out=_pos_buffer)
    except (OSError, ValueError, IndexError):
        return

    count = _pos_buffer.shape[0]

    if _world_buffer is None or _world_buffer.shape != (n_max, 3):
        _world_buffer = np.empty((n_max, 3), dtype=np.float32)

    solver_to_world_array(_pos_buffer, origin, size, out=_world_buffer[:count])

    if count < n_max:
        # `size` est desormais un triplet en espace SOLVEUR : size[0] ->
        # etendue monde X, size[1] -> etendue monde Z, size[2] -> etendue
        # monde Y (voir transform.py). L'offset de repli est exprime en
        # espace MONDE, donc chaque composante monde doit utiliser
        # l'etendue solveur qui lui correspond, pas un scalaire unique
        # (valable uniquement pour un domaine cubique avant M5).
        fallback = (
            origin[0] + _FALLBACK_WORLD_OFFSET[0] * size[0],
            origin[1] + _FALLBACK_WORLD_OFFSET[1] * size[2],
            origin[2] + _FALLBACK_WORLD_OFFSET[2] * size[1],
        )
        _world_buffer[count:n_max] = fallback

    mesh.vertices.foreach_set("co", _world_buffer.ravel())
    mesh.update()
    obj.update_tag()


# ---------------------------------------------------------------------------
# Maillage de surface (jalon M7) — meme frontiere que les particules :
# ce module ne parle jamais a `lib.py`, il ne fait que relire le cache
# `.bqm` (ecrit par `ops.BQ_OT_bake_mesh`) via `meshcache.py`.
# ---------------------------------------------------------------------------
#
# Contrairement a `Bourrasque_Particles` (compte de sommets FIXE, `n_max`,
# les sommets excedentaires d'une frame etant replies sur une position de
# repli), le nombre de sommets ET de triangles du maillage varie
# arbitrairement d'une frame a l'autre (topologie d'isosurface). La
# geometrie est donc reellement RECONSTRUITE a chaque frame
# (`mesh.clear_geometry()` puis `vertices.add`/`loops.add`/`polygons.add`)
# plutot que redimensionnee au maximum observe puis repliee : c'est ce qui
# gere nativement un compte variable, sur le meme OBJET persistant (jamais
# de creation/destruction d'objet par frame, voir docs/plan-milestone-7.md,
# D8). Un maillage vide (0 sommet) est un cas NORMAL, obtenu naturellement
# par un `clear_geometry()` qui n'est suivi d'aucun ajout.

_mesh_reader_state = {"path": None, "mtime": None, "reader": None}


def ensure_mesh_object(scene):
    """Cree ou reutilise l'objet maillage `Bourrasque_Mesh`.

    Contrairement a `ensure_particle_object`, ne dimensionne rien a la
    creation : la topologie variable d'une frame a l'autre est geree par
    `refresh_mesh`, qui reconstruit entierement la geometrie a chaque appel
    (voir docstring de section). Ne cree jamais de doublon.
    """
    obj = bpy.data.objects.get(_MESH_OBJECT_NAME)

    if obj is not None and obj.type != "MESH":
        obj.name = f"{obj.name}.stale"
        obj = None

    if obj is None:
        mesh = bpy.data.meshes.new(_MESH_OBJECT_NAME)
        obj = bpy.data.objects.new(_MESH_OBJECT_NAME, mesh)

    if obj.name not in scene.collection.all_objects:
        scene.collection.objects.link(obj)

    return obj


def clear_mesh_object():
    """Vide la geometrie de `Bourrasque_Mesh` s'il existe (0 sommet, 0
    face). Meme role que `clear_particle_object` (voir sa docstring) : ne
    supprime pas l'objet lui-meme."""
    obj = bpy.data.objects.get(_MESH_OBJECT_NAME)
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    if len(mesh.vertices) == 0 and len(mesh.polygons) == 0:
        return
    mesh.clear_geometry()
    mesh.update()
    obj.update_tag()


def _open_mesh_reader_if_needed(bqm_path):
    """Meme discipline que `_open_reader_if_needed` (particules), pour le
    cache de maillage : ne rouvre que si le chemin ou l'horodatage du
    fichier a change. Renvoie `None` si le fichier est illisible."""
    try:
        mtime = os.path.getmtime(bqm_path)
    except OSError:
        return None

    reader = _mesh_reader_state["reader"]
    if (
        reader is not None
        and _mesh_reader_state["path"] == bqm_path
        and _mesh_reader_state["mtime"] == mtime
    ):
        return reader

    if reader is not None:
        reader.close()

    try:
        reader = meshcache.MeshCacheReader(bqm_path)
    except (OSError, ValueError):
        _mesh_reader_state["reader"] = None
        _mesh_reader_state["path"] = None
        _mesh_reader_state["mtime"] = None
        return None

    _mesh_reader_state["reader"] = reader
    _mesh_reader_state["path"] = bqm_path
    _mesh_reader_state["mtime"] = mtime
    return reader


def _compute_mesh_foam_mask(scene, index, verts_solver):
    """Calcule le masque d'ecume par sommet pour la MEME frame `index` que
    celle affichee par `refresh_mesh` (docs/plan-milestone-14.md, D3/D4).

    Ouvre le cache `.bqw` (whitewater) a la demande, comme `refresh_mesh`
    ouvre `.bqm` — jamais de prechargement. Toute condition anormale (cache
    absent, index hors plage, aucune particule whitewater active, rayon
    d'influence indisponible) est un cas NORMAL qui renvoie un tableau de
    zeros de la meme longueur que `verts_solver`, jamais une exception (meme
    discipline que le reste de ce module — voir docstring de `refresh_mesh`).

    `verts_solver` doit etre en espace SOLVEUR (avant conversion monde) :
    `influence_radius` (`mesh_effective_radii`) est une grandeur solveur, cf.
    docstring de `foam.compute_foam_mask`.
    """
    n_verts = verts_solver.shape[0]

    try:
        cache_dir = bpy.path.abspath(scene.bourrasque.cache_dir)
        bqw_path = str(whitewater_cache_path(cache_dir, scene.name))
    except OSError:
        return np.zeros(n_verts, dtype=np.float32)

    if not os.path.isfile(bqw_path):
        return np.zeros(n_verts, dtype=np.float32)

    reader = _open_whitewater_reader_if_needed(bqw_path)
    if reader is None:
        return np.zeros(n_verts, dtype=np.float32)

    if not (0 <= index < reader.frame_count):
        return np.zeros(n_verts, dtype=np.float32)

    radii = mesh_effective_radii(scene)
    if radii is None:
        return np.zeros(n_verts, dtype=np.float32)
    influence_radius = radii[0]

    try:
        ww_pos = reader.read_frame(index)[0]
    except (OSError, ValueError, IndexError):
        return np.zeros(n_verts, dtype=np.float32)

    return compute_foam_mask(verts_solver, ww_pos, influence_radius)


def _ensure_mesh_foam_attr(mesh):
    """Cree, si necessaire, l'attribut de domaine POINT `bq_foam_mask`
    (FLOAT) — meme pattern d'idempotence que `_ensure_whitewater_attrs`."""
    attr = mesh.attributes.get(_MESH_FOAM_ATTR)
    if attr is not None and (attr.domain != "POINT" or attr.data_type != "FLOAT"):
        mesh.attributes.remove(attr)
        attr = None
    if attr is None:
        mesh.attributes.new(name=_MESH_FOAM_ATTR, type="FLOAT", domain="POINT")


def refresh_mesh(scene):
    """Recharge la frame courante depuis le cache `.bqm` et reconstruit la
    geometrie de `Bourrasque_Mesh` — lecture A LA DEMANDE, une frame a la
    fois, jamais de prechargement (voir docstring de section).

    Une frame absente du `.bqm` (index hors plage, cache absent, cache pas
    encore bake) ou un maillage vide (0 sommet) sont des cas NORMAUX : la
    geometrie de l'objet est simplement videe, sans erreur ni avertissement
    (voir docs/plan-milestone-7.md, D8). Comme `refresh` (particules), cette
    fonction ne leve jamais : appelee a chaque changement de frame, une
    exception y transformerait le scrub de la timeline en avalanche
    d'erreurs console.

    Ecrit egalement l'attribut de domaine POINT `bq_foam_mask` (FLOAT, voir
    `_compute_mesh_foam_mask`/`foam.compute_foam_mask`) : une densite de
    proximite au whitewater de la MEME frame, en `[0, 1]`, pour permettre a
    l'artiste de melanger une texture d'ecume au shading
    (docs/plan-milestone-14.md, D3/D4).
    """
    props = scene.bourrasque

    try:
        cache_dir = bpy.path.abspath(props.cache_dir)
        bqm_path = str(mesh_cache_path(cache_dir, scene.name))
    except OSError:
        return

    if not os.path.isfile(bqm_path):
        clear_mesh_object()
        return

    reader = _open_mesh_reader_if_needed(bqm_path)
    if reader is None:
        clear_mesh_object()
        return

    index = scene.frame_current - props.frame_start
    if not (0 <= index < reader.frame_count):
        clear_mesh_object()
        return

    transform = domain_transform(scene)
    if transform is None:
        return
    origin, size = transform

    try:
        verts, tris, _vel = reader.read_frame(index)
    except (OSError, ValueError, IndexError):
        return

    obj = ensure_mesh_object(scene)
    mesh = obj.data

    n_verts = verts.shape[0]
    n_tris = tris.shape[0]

    mesh.clear_geometry()
    if n_verts == 0:
        # Maillage vide (aucun fluide dans le domaine sur cette frame) : cas
        # NORMAL, l'objet reste vide, rien de plus a faire.
        mesh.update()
        obj.update_tag()
        return

    world_verts = solver_to_world_array(verts, origin, size)

    mesh.vertices.add(n_verts)
    mesh.vertices.foreach_set("co", world_verts.ravel())

    # Masque d'ecume (docs/plan-milestone-14.md, D3/D4) : calcule en espace
    # SOLVEUR (`verts`, avant conversion monde), MEME index de frame que le
    # mesh, jamais d'exception (voir `_compute_mesh_foam_mask`).
    foam_mask = _compute_mesh_foam_mask(scene, index, verts)
    _ensure_mesh_foam_attr(mesh)
    mesh.attributes[_MESH_FOAM_ATTR].data.foreach_set(
        "value", np.ascontiguousarray(foam_mask, dtype=np.float32)
    )

    if n_tris > 0:
        mesh.loops.add(n_tris * 3)
        mesh.polygons.add(n_tris)
        mesh.loops.foreach_set("vertex_index", tris.ravel())
        loop_start = (np.arange(n_tris, dtype=np.int32) * 3)
        loop_total = np.full(n_tris, 3, dtype=np.int32)
        mesh.polygons.foreach_set("loop_start", loop_start)
        mesh.polygons.foreach_set("loop_total", loop_total)

    mesh.update(calc_edges=True)
    obj.update_tag()


# ---------------------------------------------------------------------------
# Nuage de points whitewater (jalon M8) — meme frontiere que les particules
# et le maillage : ce module ne parle jamais a `lib.py`, il ne fait que
# relire le cache `.bqw` (ecrit par `ops.BQ_OT_bake_whitewater`) via
# `whitewatercache.py`.
# ---------------------------------------------------------------------------
#
# Un objet mesh, un vertex par particule secondaire ACTIVE a la frame
# courante — AUCUNE arete ni face (docs/plan-milestone-8.md, D9) : l'artiste
# instancie lui-meme via Geometry Nodes, rien de plus n'est fourni par
# l'add-on. Comme `Bourrasque_Mesh` (et contrairement a `Bourrasque_
# Particles`), le nombre de sommets varie arbitrairement d'une frame a
# l'autre (particules generees/mortes) : la geometrie est donc reellement
# RECONSTRUITE a chaque frame (`mesh.clear_geometry()` puis `vertices.add`)
# plutot que redimensionnee au maximum observe puis repliee, sur le meme
# OBJET persistant (jamais de creation/destruction d'objet par frame). Zero
# particule active est un cas NORMAL (mesh vide), pas une erreur.

_whitewater_reader_state = {"path": None, "mtime": None, "reader": None}


def ensure_whitewater_object(scene):
    """Cree ou reutilise l'objet maillage `Bourrasque_Whitewater`.

    Meme discipline que `ensure_mesh_object` (voir sa docstring) : ne
    dimensionne rien a la creation, la topologie variable d'une frame a
    l'autre est geree par `refresh_whitewater`. Ne cree jamais de doublon.
    """
    obj = bpy.data.objects.get(_WHITEWATER_OBJECT_NAME)

    if obj is not None and obj.type != "MESH":
        obj.name = f"{obj.name}.stale"
        obj = None

    if obj is None:
        mesh = bpy.data.meshes.new(_WHITEWATER_OBJECT_NAME)
        obj = bpy.data.objects.new(_WHITEWATER_OBJECT_NAME, mesh)

    if obj.name not in scene.collection.all_objects:
        scene.collection.objects.link(obj)

    return obj


def clear_whitewater_object():
    """Vide la geometrie de `Bourrasque_Whitewater` s'il existe (0 sommet).
    Meme role que `clear_particle_object`/`clear_mesh_object` (voir leurs
    docstrings) : ne supprime pas l'objet lui-meme."""
    obj = bpy.data.objects.get(_WHITEWATER_OBJECT_NAME)
    if obj is None or obj.type != "MESH":
        return
    mesh = obj.data
    if len(mesh.vertices) == 0:
        return
    mesh.clear_geometry()
    mesh.update()
    obj.update_tag()


def _open_whitewater_reader_if_needed(bqw_path):
    """Meme discipline que `_open_reader_if_needed`/
    `_open_mesh_reader_if_needed` : ne rouvre que si le chemin ou
    l'horodatage du fichier a change. Renvoie `None` si le fichier est
    illisible."""
    try:
        mtime = os.path.getmtime(bqw_path)
    except OSError:
        return None

    reader = _whitewater_reader_state["reader"]
    if (
        reader is not None
        and _whitewater_reader_state["path"] == bqw_path
        and _whitewater_reader_state["mtime"] == mtime
    ):
        return reader

    if reader is not None:
        reader.close()

    try:
        reader = whitewatercache.WhitewaterCacheReader(bqw_path)
    except (OSError, ValueError):
        _whitewater_reader_state["reader"] = None
        _whitewater_reader_state["path"] = None
        _whitewater_reader_state["mtime"] = None
        return None

    _whitewater_reader_state["reader"] = reader
    _whitewater_reader_state["path"] = bqw_path
    _whitewater_reader_state["mtime"] = mtime
    return reader


def _ensure_whitewater_attrs(mesh, with_velocity):
    """Cree, si necessaire, les attributs de domaine POINT `bq_type`
    (INT), `bq_size` (FLOAT), `bq_age` (FLOAT) et `bq_fade` (FLOAT) — voir
    docs/plan-milestone-8.md, D9, et `whitewaterfade.compute_age_fade` pour
    `bq_fade`. Idempotent : ne recree pas un attribut deja present avec le
    bon domaine/type.

    `with_velocity` controle en plus l'attribut `velocity` (FLOAT_VECTOR,
    motion blur Cycles) : cree si `True` (canal vitesse present dans le
    `.bqw` courant), SUPPRIME si present mais `False` (ancien cache sans
    vitesse rebake par-dessus un cache qui en avait un — un attribut
    `velocity` perime laisserait Cycles flouter avec une vitesse obsolete,
    silencieusement fausse)."""
    for name, attr_type in (
        (_WW_TYPE_ATTR, "INT"),
        (_WW_SIZE_ATTR, "FLOAT"),
        (_WW_AGE_ATTR, "FLOAT"),
        (_WW_FADE_ATTR, "FLOAT"),
    ):
        attr = mesh.attributes.get(name)
        if attr is not None and (attr.domain != "POINT" or attr.data_type != attr_type):
            mesh.attributes.remove(attr)
            attr = None
        if attr is None:
            mesh.attributes.new(name=name, type=attr_type, domain="POINT")

    vel_attr = mesh.attributes.get(_WW_VELOCITY_ATTR)
    if with_velocity:
        if vel_attr is not None and (
            vel_attr.domain != "POINT" or vel_attr.data_type != "FLOAT_VECTOR"
        ):
            mesh.attributes.remove(vel_attr)
            vel_attr = None
        if vel_attr is None:
            mesh.attributes.new(
                name=_WW_VELOCITY_ATTR, type="FLOAT_VECTOR", domain="POINT"
            )
    elif vel_attr is not None:
        mesh.attributes.remove(vel_attr)


def _whitewater_volume_input_identifier(node_group, socket_name):
    """Renvoie l'identifiant RNA (ex. `"Socket_2"`) du socket d'ENTREE nomme
    `socket_name` sur l'interface d'un groupe de noeuds, ou `None` s'il
    n'existe pas (asset d'une version anterieure a l'exposition du socket,
    ou groupe de noeuds absent/renomme).

    Utilise `node_group.interface.items_tree` (API d'interface de groupe de
    noeuds Blender 4.x/5.x) — PAS l'ancienne API `node_group.inputs`,
    retiree.
    """
    if node_group is None:
        return None
    for item in node_group.interface.items_tree:
        if (
            getattr(item, "item_type", None) == "SOCKET"
            and getattr(item, "in_out", None) == "INPUT"
            and getattr(item, "name", None) == socket_name
        ):
            return item.identifier
    return None


def _find_whitewater_volume_modifier(obj):
    """Renvoie le modificateur Geometry Nodes de `obj` qui utilise le
    groupe de noeuds `BQ Whitewater Volume`, ou `None` s'il n'y en a aucun
    (affichage volumetrique jamais configure sur cet objet — voir
    `ops.BQ_OT_setup_whitewater_display_volume`)."""
    for mod in obj.modifiers:
        if mod.type != "NODES":
            continue
        node_group = mod.node_group
        if node_group is not None and node_group.name == _WW_VOLUME_NODE_GROUP:
            return mod
    return None


def _refresh_whitewater_volume_display(obj, size_arr):
    """Derive et injecte, a partir de `size_arr` (le tableau `bq_size` LU
    DEPUIS LE CACHE `.bqw` de la frame courante, AVANT le multiplicateur
    cosmetique `ww_size_mult`), les deux parametres qui rendent l'affichage
    volumetrique du whitewater visible a l'echelle REELLE de la scene (voir
    `whitewatervolume.py` pour le detail/la provenance des constantes) :

    - Le voxel du noeud `Points to Volume`, via le socket expose `Voxel
      Size` du groupe de noeuds `BQ Whitewater Volume` (ecrit sur le
      MODIFICATEUR, pas sur le groupe de noeuds partage — chaque objet peut
      donc avoir sa propre echelle sans affecter les autres utilisateurs du
      meme groupe).
    - Le multiplicateur de densite du materiau `BQ_Whitewater_Volume`
      (`Value.DensityFactor`, ecrit directement sur le noeud — ce materiau
      n'est pas instancie par objet, contrairement au groupe de noeuds : un
      seul jeu de particules whitewater existe par scene dans ce jalon).

    Ne fait rien (silencieusement) si le modificateur volumetrique n'est
    pas present sur `obj`, si `size_arr` est vide, ou si le socket/materiau/
    noeud attendu est introuvable (asset absent, jamais configure, ou
    modifie manuellement par l'artiste) — meme discipline "jamais lever" que
    le reste de `refresh_whitewater`.
    """
    modifier = _find_whitewater_volume_modifier(obj)
    if modifier is None:
        return

    if size_arr.size == 0:
        return

    bq_size_mean = float(np.mean(size_arr))
    voxel_size, density_factor = compute_whitewater_volume_display_params(bq_size_mean)

    identifier = _whitewater_volume_input_identifier(
        modifier.node_group, _WW_VOLUME_VOXEL_SOCKET_NAME
    )
    if identifier is not None:
        try:
            setattr(getattr(modifier.properties.inputs, identifier), "value", voxel_size)
        except AttributeError:
            pass

    material = bpy.data.materials.get(_WW_VOLUME_MATERIAL)
    if material is None or material.node_tree is None:
        return
    density_node = material.node_tree.nodes.get(_WW_VOLUME_DENSITY_NODE)
    if density_node is None:
        return
    density_node.outputs[0].default_value = density_factor


def refresh_whitewater(scene):
    """Recharge la frame courante depuis le cache `.bqw` et reconstruit la
    geometrie de `Bourrasque_Whitewater` — lecture A LA DEMANDE, une frame
    a la fois, jamais de prechargement (meme discipline que
    `refresh_mesh`).

    Une frame absente du `.bqw` (index hors plage, cache absent, cache pas
    encore bake) ou 0 particule active sont des cas NORMAUX : la geometrie
    de l'objet est simplement videe, sans erreur ni avertissement (voir
    docs/plan-milestone-8.md, D9). Ne leve jamais : appelee a chaque
    changement de frame, une exception y transformerait le scrub de la
    timeline en avalanche d'erreurs console.

    Ecrit egalement l'attribut de domaine POINT `bq_fade` (FLOAT,
    `whitewaterfade.compute_age_fade`) : un facteur `[0, 1]` calcule a
    partir de `bq_age`/`bq_type` et des durees de vie par regime stockees
    dans l'en-tete du cache (`reader.params`), consomme par le groupe de
    noeuds `BQ Whitewater Display` pour faire fondre l'echelle affichee a
    la naissance/mort d'une particule plutot que de la faire apparaitre/
    disparaitre d'un coup ("effet pop").

    `bq_size` ecrit sur l'objet est le tableau du cache multiplie par
    `scene.bourrasque.ww_size_mult` : un multiplicateur PUREMENT
    COSMETIQUE (defaut 1.0, aucun changement de comportement), qui
    n'affecte que l'affichage — le cache `.bqw` lui-meme n'est jamais
    modifie.
    """
    props = scene.bourrasque

    try:
        cache_dir = bpy.path.abspath(props.cache_dir)
        bqw_path = str(whitewater_cache_path(cache_dir, scene.name))
    except OSError:
        return

    if not os.path.isfile(bqw_path):
        clear_whitewater_object()
        return

    reader = _open_whitewater_reader_if_needed(bqw_path)
    if reader is None:
        clear_whitewater_object()
        return

    index = scene.frame_current - props.frame_start
    if not (0 <= index < reader.frame_count):
        clear_whitewater_object()
        return

    transform = domain_transform(scene)
    if transform is None:
        return
    origin, size = transform

    try:
        frame = reader.read_frame(index)
    except (OSError, ValueError, IndexError):
        return
    pos, type_, size_arr, age = frame[:4]
    # `frame` a un 5eme element (vitesse) uniquement si le canal vitesse est
    # present dans ce `.bqw` (voir `WhitewaterCacheReader.read_frame`) : un
    # cache bake avant l'introduction du canal vitesse n'en a pas, cas
    # NORMAL (pas d'exception, motion blur simplement indisponible sur ce
    # cache — voir docstring de `refresh_whitewater`).
    vel = frame[4] if reader.has_velocity else None

    obj = ensure_whitewater_object(scene)
    mesh = obj.data

    # Affichage volumetrique (voir `_refresh_whitewater_volume_display`) :
    # calcule et injecte independamment de `n` (une frame a 0 particule
    # active laisse simplement `size_arr` vide, geree par le garde-fou de
    # la fonction), et independamment de la reconstruction du maillage de
    # points ci-dessous — les deux affichages coexistent (voir docstring de
    # section `ops.py`).
    _refresh_whitewater_volume_display(obj, size_arr)

    n = pos.shape[0]

    mesh.clear_geometry()
    if n == 0:
        # Aucune particule secondaire active sur cette frame : cas NORMAL,
        # l'objet reste vide, rien de plus a faire.
        mesh.update()
        obj.update_tag()
        return

    world_pos = solver_to_world_array(pos, origin, size)

    mesh.vertices.add(n)
    mesh.vertices.foreach_set("co", world_pos.ravel())

    fade = compute_age_fade(
        age,
        type_,
        reader.params.life_spray,
        reader.params.life_foam,
        reader.params.life_bubble,
    )

    # Multiplicateur cosmetique (defaut 1.0 -> comportement identique a
    # avant) : n'affecte que l'attribut Blender ecrit ici, jamais le cache
    # `.bqw` (voir docstring de `refresh_whitewater`).
    size_display = np.ascontiguousarray(size_arr, dtype=np.float32) * np.float32(
        props.ww_size_mult
    )

    _ensure_whitewater_attrs(mesh, with_velocity=vel is not None)
    mesh.attributes[_WW_TYPE_ATTR].data.foreach_set(
        "value", np.ascontiguousarray(type_, dtype=np.int32)
    )
    mesh.attributes[_WW_SIZE_ATTR].data.foreach_set("value", size_display)
    mesh.attributes[_WW_AGE_ATTR].data.foreach_set(
        "value", np.ascontiguousarray(age, dtype=np.float32)
    )
    mesh.attributes[_WW_FADE_ATTR].data.foreach_set(
        "value", np.ascontiguousarray(fade, dtype=np.float32)
    )
    if vel is not None:
        # Vitesse = un VECTEUR, pas un point : seule la partie
        # rotation/echelle de la transformation solveur -> monde s'applique,
        # JAMAIS la translation (piege documente dans
        # `solver_to_world_dir_array`) — contrairement a `pos` ci-dessus,
        # convertie via `solver_to_world_array` (avec translation).
        world_vel = solver_to_world_dir_array(vel)
        mesh.attributes[_WW_VELOCITY_ATTR].data.foreach_set(
            "vector", world_vel.ravel()
        )

    mesh.update()
    obj.update_tag()


# ---------------------------------------------------------------------------
# Handler frame_change_post
# ---------------------------------------------------------------------------
#
# NOTE — taille des points (`scene.bourrasque.point_size`) :
# aucun cablage n'est fait ici. Un maillage reduit a des sommets nus
# (aucune arete/face) s'affiche en mode Objet comme des points de taille
# fixe geree par le viewport ; il n'existe pas, sans Geometry Nodes (hors
# perimetre de ce jalon) ni passage a un objet `PointCloud` (hors perimetre
# de la spec, qui demande explicitement un objet MESH), de reglage par
# objet exposant une taille de point en mode Objet dans l'API Blender
# 5.1/5.2 standard. La seule option existante — le reglage de theme
# `preferences.themes[...].view_3d.vertex_size` — est une preference
# utilisateur globale, ne s'applique qu'en mode Edition, et la modifier a
# chaque refresh degraderait l'edition d'autres maillages sans rapport avec
# la simulation. Plutot que de cabler `point_size` sur ce levier trompeur,
# la propriete reste inerte pour ce jalon ; un jalon Geometry Nodes ulterieur
# est le bon endroit pour une taille de point pilotable par objet.


@bpy.app.handlers.persistent
def _bq_frame_change_post(scene, depsgraph):
    refresh(scene)
    refresh_mesh(scene)
    refresh_whitewater(scene)


def _remove_existing_handlers():
    # Filtre par nom de fonction plutot que par identite d'objet : un
    # rechargement du module change l'identite de `_bq_frame_change_post`
    # sans changer son nom, et un doublon d'identite passerait a travers un
    # filtre par `is`.
    for fn in list(bpy.app.handlers.frame_change_post):
        if fn.__name__ == _bq_frame_change_post.__name__:
            bpy.app.handlers.frame_change_post.remove(fn)


def close_particle_reader():
    """Referme le CacheReader de particules eventuellement garde ouvert
    (scrub de la timeline) et remet son etat a zero.

    A appeler AVANT toute suppression du fichier `.bqd` sur disque (cf.
    `BQ_OT_free_cache` dans ops.py) : sur Windows, un descripteur de fichier
    encore ouvert empeche `os.remove` (PermissionError, WinError 32) — sans
    cet appel, vider le cache echoue des qu'on a scrubbe la timeline au moins
    une fois depuis le dernier bake."""
    reader = _reader_state["reader"]
    if reader is not None:
        reader.close()
    _reader_state["reader"] = None
    _reader_state["path"] = None
    _reader_state["mtime"] = None


def close_mesh_reader():
    """Meme role que `close_particle_reader`, pour le cache de maillage
    (`.bqm`) — a appeler avant `os.remove` dans `BQ_OT_free_mesh_cache`."""
    mesh_reader = _mesh_reader_state["reader"]
    if mesh_reader is not None:
        mesh_reader.close()
    _mesh_reader_state["reader"] = None
    _mesh_reader_state["path"] = None
    _mesh_reader_state["mtime"] = None


def close_whitewater_reader():
    """Meme role que `close_particle_reader`, pour le cache de whitewater
    (`.bqw`) — a appeler avant `os.remove` dans `BQ_OT_free_whitewater_cache`."""
    whitewater_reader = _whitewater_reader_state["reader"]
    if whitewater_reader is not None:
        whitewater_reader.close()
    _whitewater_reader_state["reader"] = None
    _whitewater_reader_state["path"] = None
    _whitewater_reader_state["mtime"] = None


def register():
    _remove_existing_handlers()
    bpy.app.handlers.frame_change_post.append(_bq_frame_change_post)


def unregister():
    _remove_existing_handlers()

    # Referme les CacheReader eventuellement gardes ouverts pour ne pas
    # laisser de descripteur de fichier fuir au dela du cycle de vie de
    # l'extension (memes fonctions que celles appelees avant un os.remove
    # dans ops.py, cf. leurs docstrings).
    close_particle_reader()
    close_mesh_reader()
    close_whitewater_reader()
