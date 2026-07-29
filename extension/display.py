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

from . import cache
from .props import domain_transform
from .transform import solver_to_world_array

__all__ = (
    "ensure_particle_object",
    "write_material_attribute",
    "refresh",
    "clear_particle_object",
    "register",
    "unregister",
)

_OBJECT_NAME = "Bourrasque_Particles"
_MATERIAL_ATTR = "bq_material"

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


def _remove_existing_handlers():
    # Filtre par nom de fonction plutot que par identite d'objet : un
    # rechargement du module change l'identite de `_bq_frame_change_post`
    # sans changer son nom, et un doublon d'identite passerait a travers un
    # filtre par `is`.
    for fn in list(bpy.app.handlers.frame_change_post):
        if fn.__name__ == _bq_frame_change_post.__name__:
            bpy.app.handlers.frame_change_post.remove(fn)


def register():
    _remove_existing_handlers()
    bpy.app.handlers.frame_change_post.append(_bq_frame_change_post)


def unregister():
    _remove_existing_handlers()

    # Referme le CacheReader eventuellement garde ouvert pour ne pas laisser
    # de descripteur de fichier fuir au dela du cycle de vie de l'extension.
    reader = _reader_state["reader"]
    if reader is not None:
        reader.close()
    _reader_state["reader"] = None
    _reader_state["path"] = None
    _reader_state["mtime"] = None
