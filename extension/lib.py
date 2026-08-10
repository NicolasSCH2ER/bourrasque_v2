"""lib.py — binding ctypes de l'API C plate de Bourrasque.

C'est le SEUL module de l'extension qui parle directement a la DLL native
(bourrasque.dll / bourrasque.so). Tout le reste de l'extension (operateurs,
panneaux bpy) doit passer par la classe `Sim` definie ici plutot que par des
appels ctypes bruts.

Aucun import de `bpy` : ce module doit rester testable hors de Blender avec
`python lib.py`.

Choix numpy vs ctypes pur : on utilise numpy pour les buffers de sortie
(`read_positions` / `read_materials`). Numpy est disponible dans l'interpreteur
embarque par Blender depuis longtemps, et un `numpy.ndarray` float32 contigu
s'ecrit directement dans un fichier binaire (`ndarray.tofile` / `.tobytes()`)
et se passe tel quel a `bpy.types.Attribute.foreach_set` / `foreach_set` sur un
mesh, qui acceptent nativement des buffers numpy. Ca evite aussi une
reallocation Python par frame pendant un bake : on peut reutiliser le meme
ndarray via le parametre `out`.

Le chargement de la DLL est PARESSEUX (`load()` appele au premier usage) :
si on chargeait au moment de l'import, l'extension entiere echouerait a
s'enregistrer dans Blender des que la DLL est absente (build pas encore
lance) ou incompatible (runtime CUDA introuvable depuis le processus hote).
"""

import ctypes
import os

import numpy as np

# ---------------------------------------------------------------------------
# Erreurs
# ---------------------------------------------------------------------------


class BourrasqueError(Exception):
    """Erreur remontee par le solveur natif ou par le chargement de la DLL."""


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BQ_MODEL_ELASTIC = 0
BQ_MODEL_WATER = 1
BQ_MODEL_SAND = 2
BQ_MAX_MATERIALS = 8

# Doit rester synchronise avec la macro BQ_ABI_VERSION de core/include/bourrasque.h.
BQ_ABI_VERSION = 13

# Champs de bits de BqMesherConfig.channels (cf. core/include/bourrasque.h) :
# un seul canal existe pour l'instant, la vitesse par sommet (motion blur).
BQ_MESHER_CHANNEL_VERTEX_VELOCITY = 1 << 0

# BqWhitewaterType (cf. core/include/bourrasque.h, section whitewater).
BQ_WW_SPRAY = 0
BQ_WW_FOAM = 1
BQ_WW_BUBBLE = 2

_DLL_NAME = "bourrasque.dll" if os.name == "nt" else "libbourrasque.so"
_DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", _DLL_NAME)


# ---------------------------------------------------------------------------
# Structures ctypes — ordre et types EXACTEMENT ceux de core/include/bourrasque.h
# ---------------------------------------------------------------------------


class BqConfig(ctypes.Structure):
    _fields_ = [
        ("grid_res", ctypes.c_int * 3),
        ("cell_size", ctypes.c_float),
        ("gravity_y", ctypes.c_float),
        ("cfl", ctypes.c_float),
        ("ppc_axis", ctypes.c_int),
        ("max_particles", ctypes.c_int),
    ]


class BqMaterial(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),
        ("rho", ctypes.c_float),
        ("E", ctypes.c_float),
        ("nu", ctypes.c_float),
        ("bulk", ctypes.c_float),
        ("gamma", ctypes.c_float),
        # SAND uniquement (Drucker-Prager, jalon M18) : angle de frottement
        # interne en DEGRES, et cohesion (0 = sable sec).
        ("friction_angle", ctypes.c_float),
        ("cohesion", ctypes.c_float),
    ]


class BqSim(ctypes.Structure):
    """Handle opaque cote C (struct BqSim). On ne dereference jamais son
    contenu depuis Python ; on ne manipule que des pointeurs dessus."""


BqSimPtr = ctypes.POINTER(BqSim)


class BqMesherConfig(ctypes.Structure):
    _fields_ = [
        ("grid_res", ctypes.c_int * 3),
        ("cell_size", ctypes.c_float),
        ("influence_radius", ctypes.c_float),
        ("particle_radius", ctypes.c_float),
        ("collider_offset", ctypes.c_float),
        ("smoothing_iters", ctypes.c_int),
        ("min_component_tris", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class BqMesher(ctypes.Structure):
    """Handle opaque cote C (struct BqMesher). Meme discipline que BqSim :
    jamais dereference depuis Python, seulement passe par pointeur."""


BqMesherPtr = ctypes.POINTER(BqMesher)


class BqWhitewaterConfig(ctypes.Structure):
    _fields_ = [
        ("max_particles", ctypes.c_int),
        ("influence_radius", ctypes.c_float),
        ("gravity_y", ctypes.c_float),
        ("ta_min", ctypes.c_float),
        ("ta_max", ctypes.c_float),
        ("ta_weight", ctypes.c_float),
        ("wc_min", ctypes.c_float),
        ("wc_max", ctypes.c_float),
        ("wc_weight", ctypes.c_float),
        ("ke_min", ctypes.c_float),
        ("ke_max", ctypes.c_float),
        ("ke_weight", ctypes.c_float),
        ("spawn_rate", ctypes.c_float),
        ("life_spray", ctypes.c_float),
        ("life_foam", ctypes.c_float),
        ("life_bubble", ctypes.c_float),
        ("drag_spray", ctypes.c_float),
        ("drag_foam", ctypes.c_float),
        ("buoyancy_bubble", ctypes.c_float),
        ("grid_res", ctypes.c_int * 3),
        ("cell_size", ctypes.c_float),
    ]


class BqWhitewater(ctypes.Structure):
    """Handle opaque cote C (struct BqWhitewater), etat persistant entre
    appels (cf. bourrasque.h, section whitewater). Meme discipline que BqSim
    et BqMesher : jamais dereference depuis Python, seulement passe par
    pointeur."""


BqWhitewaterPtr = ctypes.POINTER(BqWhitewater)


class BqRigidBody(ctypes.Structure):
    """Etat d'un corps rigide (jalon M17, phase A) — ordre et types
    EXACTEMENT ceux de `BqRigidBody` dans `core/include/bourrasque.h`.

    `inv_inertia` est l'inverse du tenseur d'inertie AU CENTRE DE MASSE, en
    repere de corps, range-majeur (9 floats). `q` est `(w, x, y, z)`, meme
    convention que `rigidbody.py` et `Object.rotation_quaternion`.
    """

    _fields_ = [
        ("dynamic", ctypes.c_int),
        ("mass", ctypes.c_float),
        ("inv_inertia", ctypes.c_float * 9),
        ("x", ctypes.c_float * 3),
        ("q", ctypes.c_float * 4),
        ("v", ctypes.c_float * 3),
        ("w", ctypes.c_float * 3),
        ("use_gravity", ctypes.c_int),
        ("added_mass", ctypes.c_float),
        ("lock_lin", ctypes.c_int * 3),
        ("lock_ang", ctypes.c_int * 3),
        ("restitution", ctypes.c_float),
        # Contact corps-corps (M17 phase B) : coefficient de Coulomb du corps.
        # Le coefficient d'une PAIRE est la moyenne geometrique des deux,
        # sqrt(muA * muB), cote coeur.
        ("friction", ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# Chargement paresseux de la DLL
# ---------------------------------------------------------------------------

_dll = None


def load():
    """Charge la DLL native et declare les prototypes de toutes les fonctions.

    Idempotent : les appels suivants renvoient l'instance deja chargee.
    Leve BourrasqueError avec un message actionnable pour un artiste si le
    chargement echoue.
    """
    global _dll
    if _dll is not None:
        return _dll

    if not os.path.isfile(_DLL_PATH):
        raise BourrasqueError(
            "Le moteur de simulation Bourrasque est introuvable "
            f"({_DLL_PATH}).\n"
            "Il faut d'abord compiler le solveur (lancer le build CMake du "
            "projet) avant d'utiliser l'extension."
        )

    try:
        dll = ctypes.CDLL(_DLL_PATH)
    except OSError as exc:
        raise BourrasqueError(
            "Le moteur de simulation Bourrasque n'a pas pu etre charge "
            f"({_DLL_PATH}).\n"
            "C'est probablement du a un pilote NVIDIA ou un runtime CUDA "
            "manquant ou incompatible sur cette machine : verifiez que la "
            "carte graphique NVIDIA et ses pilotes sont a jour.\n"
            f"Detail technique : {exc}"
        )

    _check_abi_compat(dll)

    _declare_prototypes(dll)
    _dll = dll
    return _dll


def _check_abi_compat(dll):
    """Verifie que la DLL chargee est compatible avec ce code Python AVANT
    toute declaration de prototype ou tout appel.

    Ce garde-fou existe parce que ctypes garde une bibliotheque native
    chargee dans le processus une fois `CDLL()` appele : reinstaller
    l'extension (ou meme la desactiver/reactiver dans Blender) ne decharge
    PAS l'ancienne DLL tant que le processus Blender tourne encore. Sans ce
    controle, une DLL perimee est relue avec la mauvaise disposition de
    struct (ex. `BqConfig` passee de 24 a 32 octets entre deux versions), ce
    qui produit des valeurs aberrantes (ex. `max_particles` lu a l'offset de
    `cfl`) et des diagnostics completement trompeurs plus loin dans la chaine
    (ex. "memoire insuffisante" sur un `cudaMalloc` de 99 Gio).
    """
    try:
        dll.bq_abi_version.restype = ctypes.c_int
        dll.bq_abi_version.argtypes = []
        native_abi = dll.bq_abi_version()
    except AttributeError:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est perimee : elle "
            "ne fournit meme pas la fonction de verification de version "
            "d'ABI (bq_abi_version), introduite depuis.\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque.\n"
            f"Version d'ABI attendue par cette extension : {BQ_ABI_VERSION}."
        )

    if native_abi != BQ_ABI_VERSION:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est incompatible "
            "avec cette version de l'extension "
            f"(version d'ABI native = {native_abi}, "
            f"version d'ABI attendue = {BQ_ABI_VERSION}).\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque."
        )

    try:
        dll.bq_config_size.restype = ctypes.c_int
        dll.bq_config_size.argtypes = []
        native_config_size = dll.bq_config_size()
    except AttributeError:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est perimee : elle "
            "ne fournit meme pas la fonction de verification de taille de "
            "configuration (bq_config_size), introduite depuis.\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque."
        )

    expected_config_size = ctypes.sizeof(BqConfig)
    if native_config_size != expected_config_size:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est incompatible "
            "avec cette version de l'extension : la structure de "
            "configuration (BqConfig) n'a pas la meme taille cote natif et "
            f"cote Python (taille native = {native_config_size} octets, "
            f"taille attendue = {expected_config_size} octets).\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque.\n"
            "Sans redemarrage, la configuration serait relue avec de "
            "mauvais decalages memoire, ce qui produit des valeurs "
            "aberrantes (ex. un nombre de particules delirant) et des "
            "diagnostics trompeurs (ex. 'memoire insuffisante')."
        )

    try:
        dll.bq_mesher_config_size.restype = ctypes.c_int
        dll.bq_mesher_config_size.argtypes = []
        native_mesher_config_size = dll.bq_mesher_config_size()
    except AttributeError:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est perimee : elle "
            "ne fournit meme pas la fonction de verification de taille de "
            "configuration du mailleur (bq_mesher_config_size), introduite "
            "depuis.\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque."
        )

    expected_mesher_config_size = ctypes.sizeof(BqMesherConfig)
    if native_mesher_config_size != expected_mesher_config_size:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est incompatible "
            "avec cette version de l'extension : la structure de "
            "configuration du mailleur (BqMesherConfig) n'a pas la meme "
            "taille cote natif et cote Python (taille native = "
            f"{native_mesher_config_size} octets, taille attendue = "
            f"{expected_mesher_config_size} octets).\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque.\n"
            "Sans redemarrage, la configuration serait relue avec de "
            "mauvais decalages memoire, ce qui produit des valeurs "
            "aberrantes et des diagnostics trompeurs."
        )

    try:
        dll.bq_whitewater_config_size.restype = ctypes.c_int
        dll.bq_whitewater_config_size.argtypes = []
        native_whitewater_config_size = dll.bq_whitewater_config_size()
    except AttributeError:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est perimee : elle "
            "ne fournit meme pas la fonction de verification de taille de "
            "configuration du whitewater (bq_whitewater_config_size), "
            "introduite depuis.\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque."
        )

    expected_whitewater_config_size = ctypes.sizeof(BqWhitewaterConfig)
    if native_whitewater_config_size != expected_whitewater_config_size:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est incompatible "
            "avec cette version de l'extension : la structure de "
            "configuration du whitewater (BqWhitewaterConfig) n'a pas la "
            "meme taille cote natif et cote Python (taille native = "
            f"{native_whitewater_config_size} octets, taille attendue = "
            f"{expected_whitewater_config_size} octets).\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque.\n"
            "Sans redemarrage, la configuration serait relue avec de "
            "mauvais decalages memoire, ce qui produit des valeurs "
            "aberrantes et des diagnostics trompeurs."
        )

    try:
        dll.bq_rigid_body_size.restype = ctypes.c_size_t
        dll.bq_rigid_body_size.argtypes = []
        native_rigid_body_size = dll.bq_rigid_body_size()
    except AttributeError:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est perimee : elle "
            "ne fournit meme pas la fonction de verification de taille des "
            "corps rigides (bq_rigid_body_size), introduite depuis.\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque."
        )

    expected_rigid_body_size = ctypes.sizeof(BqRigidBody)
    if native_rigid_body_size != expected_rigid_body_size:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est incompatible "
            "avec cette version de l'extension : la structure de corps "
            "rigide (BqRigidBody) n'a pas la meme taille cote natif et cote "
            f"Python (taille native = {native_rigid_body_size} octets, "
            f"taille attendue = {expected_rigid_body_size} octets).\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque.\n"
            "Sans redemarrage, la configuration serait relue avec de "
            "mauvais decalages memoire, ce qui produit des valeurs "
            "aberrantes et des diagnostics trompeurs."
        )


def _declare_prototypes(dll):
    dll.bq_default_config.argtypes = [ctypes.POINTER(BqConfig)]
    dll.bq_default_config.restype = None

    dll.bq_create.argtypes = [ctypes.POINTER(BqConfig)]
    dll.bq_create.restype = BqSimPtr

    dll.bq_destroy.argtypes = [BqSimPtr]
    dll.bq_destroy.restype = None

    dll.bq_add_material.argtypes = [BqSimPtr, ctypes.POINTER(BqMaterial)]
    dll.bq_add_material.restype = ctypes.c_int

    dll.bq_emit_box.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.bq_emit_box.restype = ctypes.c_int

    dll.bq_emit_points.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.bq_emit_points.restype = ctypes.c_int

    dll.bq_emit_points_vel.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    dll.bq_emit_points_vel.restype = ctypes.c_int

    dll.bq_set_colliders.argtypes = [
        BqSimPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    dll.bq_set_colliders.restype = ctypes.c_int

    dll.bq_step.argtypes = [BqSimPtr, ctypes.c_float]
    dll.bq_step.restype = ctypes.c_int

    dll.bq_particle_count.argtypes = [BqSimPtr]
    dll.bq_particle_count.restype = ctypes.c_int

    dll.bq_read_positions.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_read_positions.restype = ctypes.c_int

    dll.bq_read_velocities.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_read_velocities.restype = ctypes.c_int

    dll.bq_read_materials.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_uint8)]
    dll.bq_read_materials.restype = ctypes.c_int

    dll.bq_read_sdf.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_read_sdf.restype = ctypes.c_int

    dll.bq_read_cnrm.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_read_cnrm.restype = ctypes.c_int

    dll.bq_last_error.argtypes = []
    dll.bq_last_error.restype = ctypes.c_char_p

    # -- corps rigides --------------------------------------------------

    dll.bq_rigid_body_size.argtypes = []
    dll.bq_rigid_body_size.restype = ctypes.c_size_t

    dll.bq_set_collider_bodies.argtypes = [
        BqSimPtr,
        ctypes.POINTER(BqRigidBody),
        ctypes.c_int,
    ]
    dll.bq_set_collider_bodies.restype = ctypes.c_int

    # -- contact corps <-> corps (M17, phase B) ---------------------------

    dll.bq_build_body_sdf.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_int,
    ]
    dll.bq_build_body_sdf.restype = ctypes.c_int

    dll.bq_set_body_samples.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    dll.bq_set_body_samples.restype = ctypes.c_int

    dll.bq_read_contacts.argtypes = [
        BqSimPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    dll.bq_read_contacts.restype = ctypes.c_int

    dll.bq_contacts_last_overflow.argtypes = [BqSimPtr]
    dll.bq_contacts_last_overflow.restype = ctypes.c_int

    dll.bq_read_body_sleep.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_uint8)]
    dll.bq_read_body_sleep.restype = ctypes.c_int

    dll.bq_set_body_pose.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.bq_set_body_pose.restype = ctypes.c_int

    dll.bq_read_collider_bodies.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_read_collider_bodies.restype = ctypes.c_int

    dll.bq_read_collider_wrench.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_read_collider_wrench.restype = ctypes.c_int

    # -- mailleur -----------------------------------------------------

    dll.bq_mesher_default_config.argtypes = [ctypes.POINTER(BqMesherConfig)]
    dll.bq_mesher_default_config.restype = None

    dll.bq_mesher_vram_estimate.argtypes = [
        ctypes.POINTER(BqMesherConfig),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
    ]
    dll.bq_mesher_vram_estimate.restype = ctypes.c_int

    dll.bq_mesher_create.argtypes = [ctypes.POINTER(BqMesherConfig)]
    dll.bq_mesher_create.restype = BqMesherPtr

    dll.bq_mesher_destroy.argtypes = [BqMesherPtr]
    dll.bq_mesher_destroy.restype = None

    dll.bq_mesher_run.argtypes = [
        BqMesherPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    dll.bq_mesher_run.restype = ctypes.c_int

    dll.bq_mesher_read_field.argtypes = [BqMesherPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_mesher_read_field.restype = ctypes.c_int

    dll.bq_mesher_counts.argtypes = [
        BqMesherPtr,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.bq_mesher_counts.restype = ctypes.c_int

    dll.bq_mesher_read.argtypes = [
        BqMesherPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.bq_mesher_read.restype = ctypes.c_int

    dll.bq_mesher_set_collider_sdf.argtypes = [
        BqMesherPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_float,
    ]
    dll.bq_mesher_set_collider_sdf.restype = ctypes.c_int

    # -- whitewater -----------------------------------------------------

    dll.bq_whitewater_default_config.argtypes = [ctypes.POINTER(BqWhitewaterConfig)]
    dll.bq_whitewater_default_config.restype = None

    dll.bq_whitewater_vram_estimate.argtypes = [
        ctypes.POINTER(BqWhitewaterConfig),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int64),
    ]
    dll.bq_whitewater_vram_estimate.restype = ctypes.c_int

    dll.bq_whitewater_create.argtypes = [ctypes.POINTER(BqWhitewaterConfig)]
    dll.bq_whitewater_create.restype = BqWhitewaterPtr

    dll.bq_whitewater_destroy.argtypes = [BqWhitewaterPtr]
    dll.bq_whitewater_destroy.restype = None

    dll.bq_whitewater_step.argtypes = [
        BqWhitewaterPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
    ]
    dll.bq_whitewater_step.restype = ctypes.c_int

    dll.bq_whitewater_count.argtypes = [BqWhitewaterPtr]
    dll.bq_whitewater_count.restype = ctypes.c_int

    dll.bq_whitewater_read.argtypes = [
        BqWhitewaterPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.bq_whitewater_read.restype = ctypes.c_int

    dll.bq_whitewater_last_refused.argtypes = [BqWhitewaterPtr]
    dll.bq_whitewater_last_refused.restype = ctypes.c_int

    dll.bq_whitewater_set_collider_sdf.argtypes = [
        BqWhitewaterPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_float,
    ]
    dll.bq_whitewater_set_collider_sdf.restype = ctypes.c_int

    dll.bq_whitewater_set_collider_cnrm.argtypes = [
        BqWhitewaterPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_float,
    ]
    dll.bq_whitewater_set_collider_cnrm.restype = ctypes.c_int


def _last_error(dll):
    msg = dll.bq_last_error()
    if msg is None:
        return "(pas de detail disponible)"
    return msg.decode("utf-8", errors="replace")


def default_config():
    """Retourne un BqConfig rempli par bq_default_config."""
    dll = load()
    cfg = BqConfig()
    dll.bq_default_config(ctypes.byref(cfg))
    return cfg


def default_mesher_config():
    """Retourne un BqMesherConfig rempli par bq_mesher_default_config."""
    dll = load()
    cfg = BqMesherConfig()
    dll.bq_mesher_default_config(ctypes.byref(cfg))
    return cfg


def mesher_vram_estimate(config, n_particles=0):
    """Empreinte VRAM (octets) du champ de maillage seul (sans le champ
    collider ni les tampons du marching cubes), pour la configuration et le
    nombre de particules donnes. Ne rien alloue cote GPU (voir
    `bq_mesher_vram_estimate`, `bourrasque.h`)."""
    dll = load()
    out = ctypes.c_int64(0)
    code = dll.bq_mesher_vram_estimate(
        ctypes.byref(config), int(n_particles), ctypes.byref(out)
    )
    if code < 0:
        raise BourrasqueError(_last_error(dll))
    return int(out.value)


def default_whitewater_config():
    """Retourne un BqWhitewaterConfig rempli par bq_whitewater_default_config."""
    dll = load()
    cfg = BqWhitewaterConfig()
    dll.bq_whitewater_default_config(ctypes.byref(cfg))
    return cfg


def whitewater_vram_estimate(config, n_fluid_hint=0):
    """Empreinte VRAM (octets) du whitewater a max_particles et
    n_fluid_hint donnes, pour la configuration donnee. N'alloue rien cote
    GPU (voir `bq_whitewater_vram_estimate`, `bourrasque.h`)."""
    dll = load()
    out = ctypes.c_int64(0)
    code = dll.bq_whitewater_vram_estimate(
        ctypes.byref(config), int(n_fluid_hint), ctypes.byref(out)
    )
    if code < 0:
        raise BourrasqueError(_last_error(dll))
    return int(out.value)


def _vec3(seq):
    arr = (ctypes.c_float * 3)(*seq)
    return arr


# ---------------------------------------------------------------------------
# API pythonique
# ---------------------------------------------------------------------------


class Sim:
    """Encapsule un BqSim* et expose une API pythonique au-dessus de l'API C.

    Supporte le protocole de contexte : `with Sim(cfg) as sim: ...` garantit
    la liberation de la memoire GPU meme si une exception survient pendant
    le bake.
    """

    def __init__(self, config):
        self._dll = load()
        # Memorise la resolution de grille (espace hote, pas cote coeur) :
        # necessaire pour dimensionner le buffer de `read_sdf`, seule
        # methode dont la taille du resultat ne se lit pas via
        # `particle_count`.
        self._grid_res = (
            int(config.grid_res[0]),
            int(config.grid_res[1]),
            int(config.grid_res[2]),
        )
        self._handle = self._dll.bq_create(ctypes.byref(config))
        if not self._handle:
            raise BourrasqueError(
                "Impossible de creer la simulation Bourrasque : "
                f"{_last_error(self._dll)}"
            )
        # Nombre de corps rigides declares (voir set_collider_bodies) : 0
        # tant qu'aucun corps n'a ete declare.
        self._n_bodies = 0

    def _check(self, code):
        if code < 0:
            raise BourrasqueError(_last_error(self._dll))
        return code

    def add_material(self, model, rho, E=0.0, nu=0.0, bulk=0.0, gamma=0.0,
                     friction_angle=0.0, cohesion=0.0):
        """`friction_angle` (degres) et `cohesion` ne concernent que
        `BQ_MODEL_SAND` ; les laisser a zero pour les autres modeles."""
        mat = BqMaterial(model=model, rho=rho, E=E, nu=nu, bulk=bulk, gamma=gamma,
                         friction_angle=friction_angle, cohesion=cohesion)
        return self._check(self._dll.bq_add_material(self._handle, ctypes.byref(mat)))

    def emit_box(self, mat_id, lo, hi, vel=(0.0, 0.0, 0.0)):
        return self._check(
            self._dll.bq_emit_box(
                self._handle, mat_id, _vec3(lo), _vec3(hi), _vec3(vel)
            )
        )

    def emit_points(self, mat_id, positions, vel=(0.0, 0.0, 0.0)):
        """Emet des particules a des positions explicites (espace solveur).

        `positions` doit etre convertible en ndarray `(n, 3)`. On force
        `np.ascontiguousarray(..., dtype=np.float32)` avant de prendre le
        pointeur : un tableau non contigu (ex. une vue transposee ou un
        slice avec un pas) ou en float64 (le dtype par defaut de numpy)
        passerait silencieusement des octets errones a `bq_emit_points`,
        qui attend un buffer `float32` C-contigu de `count*3` elements.
        """
        arr = np.ascontiguousarray(positions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"emit_points: positions doit etre de forme (n, 3), "
                f"recu {arr.shape}"
            )
        count = arr.shape[0]
        if count == 0:
            return 0
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._check(
            self._dll.bq_emit_points(
                self._handle, mat_id, ptr, count, _vec3(vel)
            )
        )

    def emit_points_vel(self, mat_id, positions, velocities):
        """Emet des particules a des positions explicites, chacune avec sa
        propre vitesse (espace solveur).

        `positions` et `velocities` doivent etre convertibles en ndarray
        `(n, 3)`, meme n pour les deux. Meme justification que `emit_points`
        pour le `np.ascontiguousarray(..., dtype=np.float32)` applique aux
        DEUX tableaux : un tableau non contigu (ex. une vue transposee ou un
        slice avec un pas) ou en float64 (le dtype par defaut de numpy)
        passerait silencieusement des octets errones a `bq_emit_points_vel`,
        qui attend deux buffers `float32` C-contigus de `count*3` elements
        chacun.
        """
        pos_arr = np.ascontiguousarray(positions, dtype=np.float32)
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
            raise ValueError(
                f"emit_points_vel: positions doit etre de forme (n, 3), "
                f"recu {pos_arr.shape}"
            )
        vel_arr = np.ascontiguousarray(velocities, dtype=np.float32)
        if vel_arr.ndim != 2 or vel_arr.shape[1] != 3:
            raise ValueError(
                f"emit_points_vel: velocities doit etre de forme (n, 3), "
                f"recu {vel_arr.shape}"
            )
        if pos_arr.shape[0] != vel_arr.shape[0]:
            raise ValueError(
                "emit_points_vel: positions et velocities doivent avoir le "
                f"meme nombre de lignes (recu {pos_arr.shape[0]} et "
                f"{vel_arr.shape[0]})"
            )
        count = pos_arr.shape[0]
        if count == 0:
            return 0
        pos_ptr = pos_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vel_ptr = vel_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._check(
            self._dll.bq_emit_points_vel(
                self._handle, mat_id, pos_ptr, vel_ptr, count
            )
        )

    def set_colliders(self, triangles, velocities, frictions, tri_body=None):
        """Remplace l'ensemble des colliders du solveur (espace solveur).

        `triangles` et `velocities` doivent etre convertibles en ndarray
        `(n_tri, 3, 3)` (triangle, sommet, xyz) de MEME forme : `velocities`
        est la vitesse par SOMMET, pas par triangle. `frictions` doit etre
        convertible en ndarray `(n_tri,)`, un coefficient par triangle.
        Meme discipline `np.ascontiguousarray(..., dtype=np.float32)` que
        `emit_points_vel` sur les trois tableaux : un tableau non contigu ou
        en float64 (le dtype par defaut de numpy) passerait silencieusement
        des octets errones a `bq_set_colliders`, qui attend des buffers
        `float32` C-contigus.

        `tri_body` : `None` (defaut) passe `NULL` au coeur — tous les
        triangles sont alors attribues au corps 0 s'il existe des corps
        declares (voir `set_collider_bodies`), ou a aucun corps sinon (voir
        `bq_set_colliders`, `bourrasque.h`). Sinon, convertible en ndarray
        `(n_tri,)` int32, l'indice de corps proprietaire de chaque triangle.
        Les appelants qui ne declarent jamais de corps (bake de maillage,
        bake de whitewater — des `Sim` legeres qui ne construisent qu'un
        champ de distance statique) doivent laisser ce parametre a `None`.

        `n_tri == 0` est un appel VALIDE (pas court-circuite cote Python,
        contrairement a `emit_points`/`emit_points_vel`) : c'est la
        convention du coeur pour effacer les colliders (voir
        `bourrasque.h`), a appeler explicitement des qu'un bake n'a plus de
        collider a transmettre pour la frame courante.
        """
        tri_arr = np.ascontiguousarray(triangles, dtype=np.float32)
        if tri_arr.ndim != 3 or tri_arr.shape[1:] != (3, 3):
            raise ValueError(
                f"set_colliders: triangles doit etre de forme (n, 3, 3), "
                f"recu {tri_arr.shape}"
            )
        vel_arr = np.ascontiguousarray(velocities, dtype=np.float32)
        if vel_arr.shape != tri_arr.shape:
            raise ValueError(
                "set_colliders: velocities doit avoir la meme forme que "
                f"triangles ({tri_arr.shape}), recu {vel_arr.shape}"
            )
        fric_arr = np.ascontiguousarray(frictions, dtype=np.float32)
        if fric_arr.ndim != 1 or fric_arr.shape[0] != tri_arr.shape[0]:
            raise ValueError(
                f"set_colliders: frictions doit etre de forme (n,) avec "
                f"n == {tri_arr.shape[0]} (nombre de triangles), recu "
                f"{fric_arr.shape}"
            )

        n_tri = tri_arr.shape[0]
        tri_ptr = tri_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vel_ptr = vel_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        fric_ptr = fric_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        if tri_body is None:
            body_ptr = None
        else:
            body_arr = np.ascontiguousarray(tri_body, dtype=np.int32)
            if body_arr.ndim != 1 or body_arr.shape[0] != n_tri:
                raise ValueError(
                    f"set_colliders: tri_body doit etre de forme (n,) avec "
                    f"n == {n_tri} (nombre de triangles), recu "
                    f"{body_arr.shape}"
                )
            body_ptr = body_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        return self._check(
            self._dll.bq_set_colliders(
                self._handle, tri_ptr, vel_ptr, fric_ptr, body_ptr, n_tri
            )
        )

    def set_collider_bodies(self, bodies):
        """Declare l'ensemble des corps rigides et REINITIALISE leur etat
        (voir `bq_set_collider_bodies`, `bourrasque.h`) — a appeler une
        seule fois au debut du bake, jamais par frame.

        `bodies` : sequence de `BqRigidBody` (longueur `n_bodies`, plafond
        de 64 cote coeur). `len(bodies) == 0` efface tous les corps.
        Memorise `n_bodies` pour dimensionner les buffers de
        `read_collider_bodies`/`read_collider_wrench`.
        """
        bodies = list(bodies)
        n_bodies = len(bodies)
        self._n_bodies = n_bodies
        if n_bodies == 0:
            bodies_ptr = None
        else:
            arr = (BqRigidBody * n_bodies)(*bodies)
            bodies_ptr = ctypes.cast(arr, ctypes.POINTER(BqRigidBody))
        return self._check(
            self._dll.bq_set_collider_bodies(self._handle, bodies_ptr, n_bodies)
        )

    def build_body_sdf(self, body, triangles, target_cell, max_res=128):
        """Construit et stocke le champ de distance signee LOCAL du corps
        `body`, a partir de ses triangles de REPOS exprimes en repere de
        corps (origine au centre de masse) — voir `bq_build_body_sdf`.

        A appeler UNE SEULE FOIS par corps, au debut du bake : le corps
        etant rigide, ce champ ne change jamais, une requete monde se
        ramene a `p_local = R^T (p - x)` puis une lecture trilineaire.

        `triangles` : ndarray `(n_tri, 3, 3)` float32, meme mise en forme
        que `set_colliders`. `target_cell` : taille de voxel visee (le `dx`
        du domaine, pour que la finesse du contact suive celle de la
        simulation) ; `max_res` plafonne la resolution par axe — au-dela le
        coeur agrandit le voxel et le signale sur stderr.
        """
        arr = np.ascontiguousarray(triangles, dtype=np.float32)
        n_tri = arr.shape[0]
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if n_tri else None
        return self._check(
            self._dll.bq_build_body_sdf(
                self._handle, body, ptr, n_tri, float(target_cell), int(max_res)
            )
        )

    def set_body_samples(self, body, points):
        """Points d'echantillonnage de SURFACE du corps `body`, en repere de
        corps (voir `bq_set_body_samples`). Produits par
        `extension.rigidbody.surface_samples`, qui est DETERMINISTE : le
        solveur de contact apparie ses impulsions d'un sous-pas a l'autre par
        indice d'echantillon (warm starting), un jeu de points qui changerait
        d'un appel a l'autre casserait la stabilite des empilements.

        `points` : ndarray `(n, 3)`. `n == 0` efface les echantillons.
        """
        arr = np.ascontiguousarray(points, dtype=np.float32)
        n = arr.shape[0]
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if n else None
        return self._check(self._dll.bq_set_body_samples(self._handle, body, ptr, n))

    def set_body_pose(self, body, x, q, v=(0.0, 0.0, 0.0), w=(0.0, 0.0, 0.0)):
        """Met a jour la pose ET la vitesse d'un corps CINEMATIQUE
        (`dynamic == 0`) — voir `bq_set_body_pose`. A appeler par frame pour
        un collider anime : rien d'autre ne fait avancer son `x`/`q`, et son
        SDF de contact resterait sinon fige a la pose de la premiere frame.

        Le coeur REFUSE un corps dynamique (sa pose appartient au solveur) et
        un indice hors bornes. Contrairement a `set_collider_bodies`, cet
        appel ne touche ni l'etat des autres corps, ni les compteurs de
        sommeil, ni le cache de warm starting du contact — c'est toute sa
        raison d'etre.

        `v` et `w` comptent autant que la pose : le contact travaille sur la
        vitesse RELATIVE. Sans elles, un mur qui avance ne transfere rien, il
        penetre puis se fait repousser par la correction de position. Les
        calculer par difference finie entre frames, comme la vitesse par
        sommet des colliders animes pour le fluide.

        `q` est en convention `(w, x, y, z)`, normalise defensivement cote
        coeur.
        """
        return self._check(
            self._dll.bq_set_body_pose(
                self._handle,
                int(body),
                _vec3(x),
                (ctypes.c_float * 4)(*[float(c) for c in q]),
                _vec3(v),
                _vec3(w),
            )
        )

    def contact_count(self):
        """Nombre de contacts detectes au DERNIER sous-pas (sans les copier)."""
        return self._check(self._dll.bq_read_contacts(self._handle, None, 0))

    def read_contacts(self, max_contacts=4096):
        """Diagnostic : ndarray float32 `(k, 9)` — corps A, corps B (stockes
        comme flottants), point[3], normale[3], profondeur. `k` est borne par
        `max_contacts`. Voir `bq_read_contacts`."""
        out = np.empty((max_contacts, 9), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        total = self._check(
            self._dll.bq_read_contacts(self._handle, ptr, max_contacts)
        )
        return out[: min(total, max_contacts)]

    def contacts_last_overflow(self):
        """1 si le tampon de contacts a sature au dernier sous-pas. A
        verifier plutot que de laisser une perte de contacts invisible —
        meme discipline que `whitewater.last_refused`."""
        return self._check(self._dll.bq_contacts_last_overflow(self._handle))

    def read_body_sleep(self, out=None):
        """Etat de sommeil des corps, ndarray uint8 `(n_bodies,)` : 1 =
        endormi. Sans la mise en sommeil, un tas de debris fremit
        indefiniment — defaut tres visible au rendu."""
        n = getattr(self, "_n_bodies", 0)
        if out is None or out.shape != (n,) or out.dtype != np.uint8:
            out = np.empty((n,), dtype=np.uint8)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) if n else None
        self._check(self._dll.bq_read_body_sleep(self._handle, ptr))
        return out

    def read_collider_bodies(self, out=None):
        """Renvoie l'etat courant des corps, ndarray float32 de forme
        `(n_bodies, 13)` : `x[3], q[4], v[3], w[3]` par corps, meme ordre
        que `BqRigidBody` (voir `bq_read_collider_bodies`, `bourrasque.h`).
        `n_bodies` est celui du dernier `set_collider_bodies`."""
        n = getattr(self, "_n_bodies", 0)
        shape = (n, 13)
        if out is None or out.shape != shape or out.dtype != np.float32:
            out = np.empty(shape, dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if n else None
        self._check(self._dll.bq_read_collider_bodies(self._handle, ptr))
        return out

    def read_collider_wrench(self, out=None):
        """Renvoie le wrench recolte par le DERNIER sous-pas effectue,
        ndarray float32 de forme `(n_bodies, 7)` : impulsion lineaire [3],
        impulsion angulaire (couple) [3], masse de fluide en contact [1]
        (voir `bq_read_collider_wrench`, `bourrasque.h`). Diagnostic — ne
        PAS integrer sur la frame, chaque sous-pas remet l'accumulateur a
        zero."""
        n = getattr(self, "_n_bodies", 0)
        shape = (n, 7)
        if out is None or out.shape != shape or out.dtype != np.float32:
            out = np.empty(shape, dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if n else None
        self._check(self._dll.bq_read_collider_wrench(self._handle, ptr))
        return out

    def step(self, frame_dt):
        return self._check(self._dll.bq_step(self._handle, frame_dt))

    @property
    def particle_count(self):
        return self._check(self._dll.bq_particle_count(self._handle))

    def read_positions(self, out=None):
        """Renvoie un ndarray float32 de forme (n, 3) rempli in-place si
        `out` est fourni (reutilisation entre frames pendant un bake)."""
        n = self.particle_count
        if out is None or out.shape != (n, 3) or out.dtype != np.float32:
            out = np.empty((n, 3), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._check(self._dll.bq_read_positions(self._handle, ptr))
        return out

    def read_velocities(self, out=None):
        """Renvoie un ndarray float32 de forme (n, 3), meme convention que
        `read_positions` (reutilisation de `out` entre frames pendant un bake)."""
        n = self.particle_count
        if out is None or out.shape != (n, 3) or out.dtype != np.float32:
            out = np.empty((n, 3), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._check(self._dll.bq_read_velocities(self._handle, ptr))
        return out

    def read_materials(self, out=None):
        """Renvoie un ndarray uint8 de forme (n,) rempli in-place si `out`
        est fourni."""
        n = self.particle_count
        if out is None or out.shape != (n,) or out.dtype != np.uint8:
            out = np.empty((n,), dtype=np.uint8)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._check(self._dll.bq_read_materials(self._handle, ptr))
        return out

    def read_sdf(self, out=None):
        """Renvoie le champ de distance signee courant, ndarray float32 de
        forme `(grid_res[0], grid_res[1], grid_res[2])` (espace SOLVEUR),
        rempli in-place si `out` est fourni. Echantillonne AUX NOEUDS de la
        grille (`i*dx` sur chaque axe, meme convention que `k_grid_update`),
        pas au centre des cellules.

        Diagnostic et validation (voir `bq_read_sdf`, `bourrasque.h`) :
        permet de verifier depuis Python qu'un collider produit bien un
        champ coherent (signe negatif a l'interieur, distance nulle sur la
        surface), sans avoir a en deduire l'etat indirectement via les
        positions de particules.
        """
        shape = self._grid_res
        if out is None or out.shape != shape or out.dtype != np.float32:
            out = np.empty(shape, dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._check(self._dll.bq_read_sdf(self._handle, ptr))
        return out

    def read_cnrm(self, out=None):
        """Renvoie le champ de normale de contact courant, ndarray float32 de
        forme `(grid_res[0], grid_res[1], grid_res[2], 4)` (espace SOLVEUR),
        rempli in-place si `out` est fourni. Echantillonne AUX NOEUDS de la
        grille, meme convention que `read_sdf`. Chaque cellule fournit 4
        floats : x,y,z la normale de contact unitaire (direction point-le-
        plus-proche-sur-triangle -> cellule, exacte, calculee sur le maillage
        collider brut), w la distance NON signee jusqu'au triangle le plus
        proche.

        Diagnostic et validation (voir `bq_read_cnrm`, `bourrasque.h`), et
        source destinee a etre retransmise telle quelle a
        `Whitewater.set_collider_cnrm`.
        """
        shape = self._grid_res + (4,)
        if out is None or out.shape != shape or out.dtype != np.float32:
            out = np.empty(shape, dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._check(self._dll.bq_read_cnrm(self._handle, ptr))
        return out

    def destroy(self):
        """Libere la memoire GPU. Idempotent."""
        if getattr(self, "_handle", None):
            self._dll.bq_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    def __del__(self):
        # Filet de securite : ne pas s'appuyer dessus pour du code sensible,
        # mais evite une fuite GPU silencieuse si destroy() n'a pas ete
        # appele explicitement.
        try:
            self.destroy()
        except Exception:
            pass


class Mesher:
    """Encapsule un BqMesher* et expose une API pythonique au-dessus de
    l'API C du mailleur (voir `BqMesherConfig`, `bourrasque.h`, section
    mailleur).

    Independant de tout `Sim` : consomme un nuage de positions, d'ou qu'il
    vienne (etat vivant d'une simulation ou cache `.bqd` relu) — c'est ce qui
    rend le bake de maillage modulaire (voir docs/plan-milestone-7.md, D7) et
    testable sur un nuage synthetique sans simuler.

    Supporte le protocole de contexte comme `Sim` : `with Mesher(cfg) as
    mesher: ...` garantit la liberation de la memoire GPU meme si une
    exception survient pendant le bake de maillage.
    """

    def __init__(self, config):
        self._dll = load()
        self._handle = self._dll.bq_mesher_create(ctypes.byref(config))
        if not self._handle:
            raise BourrasqueError(
                "Impossible de creer le mailleur Bourrasque : "
                f"{_last_error(self._dll)}"
            )

    def _check(self, code):
        if code < 0:
            raise BourrasqueError(_last_error(self._dll))
        return code

    def run(self, positions):
        """Reconstruit le champ de maillage a partir de `positions`
        (convertible en ndarray `(n, 3)` float32, ou tableau ctypes deja
        contigu — meme discipline que `Sim.emit_points`).

        `n == 0` est un appel VALIDE (aucun fluide dans le domaine sur cette
        frame) : le maillage resultant sera vide (voir `counts`/`read`).
        """
        arr = np.ascontiguousarray(positions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"Mesher.run: positions doit etre de forme (n, 3), "
                f"recu {arr.shape}"
            )
        n = arr.shape[0]
        if n == 0:
            ptr = None
        else:
            ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._check(self._dll.bq_mesher_run(self._handle, ptr, n))

    def counts(self):
        """Renvoie `(n_verts, n_tris)` du dernier `run()`. Un maillage vide
        (0, 0) est un resultat NORMAL (champ sans fluide)."""
        n_verts = ctypes.c_int(0)
        n_tris = ctypes.c_int(0)
        self._check(
            self._dll.bq_mesher_counts(
                self._handle, ctypes.byref(n_verts), ctypes.byref(n_tris)
            )
        )
        return (int(n_verts.value), int(n_tris.value))

    def read(self):
        """Renvoie `(verts, tris)` du dernier `run()` : `verts` un ndarray
        `(n_verts, 3)` float32, `tris` un ndarray `(n_tris, 3)` int32 (indices
        de sommets DEDUPLIQUES, voir `bq_mesher_read`). `(n_verts, n_tris) ==
        (0, 0)` (maillage vide) renvoie deux tableaux vides, pas une erreur.

        Le canal de vitesse par sommet (`vel` de `bq_mesher_read`) n'est pas
        expose ici : il reste ignore par le coeur pour l'instant (voir la
        docstring de `bq_mesher_read`, `bourrasque.h`), passer NULL est le
        seul usage sur, `Mesher.read` ne le sollicite donc jamais.
        """
        n_verts, n_tris = self.counts()

        verts = np.empty((n_verts, 3), dtype=np.float32)
        tris = np.empty((n_tris, 3), dtype=np.int32)

        verts_ptr = (
            verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if n_verts
            else None
        )
        tris_ptr = (
            tris.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            if n_tris
            else None
        )
        self._check(
            self._dll.bq_mesher_read(self._handle, verts_ptr, tris_ptr, None)
        )
        return verts, tris

    def set_collider_sdf(self, sdf, res, cell_size):
        """Fournit le champ de distance signee des colliders au mailleur,
        pour rognage AVANT polygonisation (voir `bq_mesher_set_collider_sdf`,
        `bourrasque.h`, et docs/plan-milestone-7.md D5).

        `sdf` : convertible en ndarray de `res[0]*res[1]*res[2]` float32,
        range en ordre C (`idx = (i*res[1]+j)*res[2]+k`) — EXACTEMENT la
        forme renvoyee par `Sim.read_sdf()`, echantillonnee AUX NOEUDS de sa
        grille (meme convention que le coeur, cf. sa docstring) : aucune
        conversion n'est appliquee ici, le tableau est transmis tel quel.
        `res` : triplet d'entiers, la resolution DE CE CHAMP (peut differer
        de la resolution du champ de maillage). `cell_size` : le pas de
        cette grille collider (unites monde).

        `sdf=None` efface le collider courant (equivalent a passer `NULL`
        cote C).
        """
        if sdf is None:
            res_arr = (ctypes.c_int * 3)(0, 0, 0)
            return self._check(
                self._dll.bq_mesher_set_collider_sdf(
                    self._handle, None, res_arr, ctypes.c_float(0.0)
                )
            )

        arr = np.ascontiguousarray(sdf, dtype=np.float32)
        expected = int(res[0]) * int(res[1]) * int(res[2])
        if arr.size != expected:
            raise ValueError(
                f"set_collider_sdf: {arr.size} valeurs fournies, "
                f"{expected} attendues pour res={tuple(int(r) for r in res)}"
            )
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        res_arr = (ctypes.c_int * 3)(int(res[0]), int(res[1]), int(res[2]))
        return self._check(
            self._dll.bq_mesher_set_collider_sdf(
                self._handle, ptr, res_arr, ctypes.c_float(cell_size)
            )
        )

    def destroy(self):
        """Libere la memoire GPU. Idempotent."""
        if getattr(self, "_handle", None):
            self._dll.bq_mesher_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    def __del__(self):
        # Filet de securite, meme discipline que Sim.__del__.
        try:
            self.destroy()
        except Exception:
            pass


class Whitewater:
    """Encapsule un BqWhitewater* et expose une API pythonique au-dessus de
    l'API C du whitewater (voir `BqWhitewaterConfig`, `bourrasque.h`, section
    whitewater).

    Contrairement a `Mesher`, cette classe porte un ETAT PERSISTANT entre
    appels : `step()` doit etre appele une fois par frame, dans l'ordre
    croissant, sans saut (voir la docstring de `bq_whitewater_step`,
    `bourrasque.h`, et docs/plan-milestone-8.md D5). Aucun garde-fou de
    sequentialite n'est fait ici, comme cote natif.

    Supporte le protocole de contexte comme `Sim`/`Mesher` : `with
    Whitewater(cfg) as ww: ...` garantit la liberation de la memoire GPU
    meme si une exception survient pendant le bake.
    """

    def __init__(self, cfg=None):
        self._dll = load()
        if cfg is None:
            cfg = default_whitewater_config()
        self._handle = self._dll.bq_whitewater_create(ctypes.byref(cfg))
        if not self._handle:
            raise BourrasqueError(
                "Impossible de creer le whitewater Bourrasque : "
                f"{_last_error(self._dll)}"
            )

    def _check(self, code):
        if code < 0:
            raise BourrasqueError(_last_error(self._dll))
        return code

    def step(self, pos, vel, dt):
        """Avance d'UNE frame (voir contrat de sequentialite dans la
        docstring de la classe).

        `pos`/`vel` : particules fluides de CETTE frame, convertibles en
        ndarray `(n, 3)` float32, meme n pour les deux (espace solveur,
        memoire hote — meme discipline que `Mesher.run`/`Sim.emit_points_vel`
        pour la contiguite et le dtype).
        """
        pos_arr = np.ascontiguousarray(pos, dtype=np.float32)
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
            raise ValueError(
                f"Whitewater.step: pos doit etre de forme (n, 3), "
                f"recu {pos_arr.shape}"
            )
        vel_arr = np.ascontiguousarray(vel, dtype=np.float32)
        if vel_arr.ndim != 2 or vel_arr.shape[1] != 3:
            raise ValueError(
                f"Whitewater.step: vel doit etre de forme (n, 3), "
                f"recu {vel_arr.shape}"
            )
        if pos_arr.shape[0] != vel_arr.shape[0]:
            raise ValueError(
                "Whitewater.step: pos et vel doivent avoir le meme nombre "
                f"de lignes (recu {pos_arr.shape[0]} et {vel_arr.shape[0]})"
            )
        n = pos_arr.shape[0]
        if n == 0:
            pos_ptr = None
            vel_ptr = None
        else:
            pos_ptr = pos_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            vel_ptr = vel_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._check(
            self._dll.bq_whitewater_step(
                self._handle, pos_ptr, vel_ptr, n, float(dt)
            )
        )

    def count(self):
        """Renvoie le nombre de particules secondaires actives."""
        return self._check(self._dll.bq_whitewater_count(self._handle))

    def last_refused(self):
        """Renvoie le nombre de candidates refusees au dernier `step()`
        faute de capacite (voir `bq_whitewater_last_refused`, `bourrasque.h`,
        et docs/plan-milestone-8.md, risque 4)."""
        return self._check(self._dll.bq_whitewater_last_refused(self._handle))

    def set_collider_sdf(self, sdf, res, cell_size):
        """Fournit le champ de distance signee des colliders au whitewater,
        MEME CONVENTION EXACTE que `Mesher.set_collider_sdf` (voir
        `bq_whitewater_set_collider_sdf`, `bourrasque.h`, section whitewater).

        `sdf` : convertible en ndarray de `res[0]*res[1]*res[2]` float32,
        range en ordre C (`idx = (i*res[1]+j)*res[2]+k`) — EXACTEMENT la
        forme renvoyee par `Sim.read_sdf()`, echantillonnee AUX NOEUDS de sa
        grille (meme convention que le coeur, cf. sa docstring) : aucune
        conversion n'est appliquee ici, le tableau est transmis tel quel.
        `res` : triplet d'entiers, la resolution DE CE CHAMP (peut differer
        de la resolution de la grille whitewater). `cell_size` : le pas de
        cette grille collider (unites monde).

        `sdf=None` efface le collider courant (equivalent a passer `NULL`
        cote C) ; dans ce cas, seule la borne de domaine
        (`grid_res`/`cell_size` de `BqWhitewaterConfig`) contraint les
        particules.
        """
        if sdf is None:
            res_arr = (ctypes.c_int * 3)(0, 0, 0)
            return self._check(
                self._dll.bq_whitewater_set_collider_sdf(
                    self._handle, None, res_arr, ctypes.c_float(0.0)
                )
            )

        arr = np.ascontiguousarray(sdf, dtype=np.float32)
        expected = int(res[0]) * int(res[1]) * int(res[2])
        if arr.size != expected:
            raise ValueError(
                f"set_collider_sdf: {arr.size} valeurs fournies, "
                f"{expected} attendues pour res={tuple(int(r) for r in res)}"
            )
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        res_arr = (ctypes.c_int * 3)(int(res[0]), int(res[1]), int(res[2]))
        return self._check(
            self._dll.bq_whitewater_set_collider_sdf(
                self._handle, ptr, res_arr, ctypes.c_float(cell_size)
            )
        )

    def set_collider_cnrm(self, cnrm, res, cell_size):
        """Fournit le champ de normale de contact des colliders au
        whitewater, MEME CONVENTION EXACTE que `Sim.read_cnrm` (voir
        `bq_whitewater_set_collider_cnrm`, `bourrasque.h`, section
        whitewater).

        `cnrm` : convertible en ndarray de `res[0]*res[1]*res[2]*4` float32,
        range en ordre C (`idx = (i*res[1]+j)*res[2]+k`, 4 floats par
        cellule) — EXACTEMENT la forme renvoyee par `Sim.read_cnrm()`,
        echantillonnee AUX NOEUDS de sa grille : aucune conversion n'est
        appliquee ici, le tableau est transmis tel quel. `res` : triplet
        d'entiers, la resolution DE CE CHAMP (peut differer de la resolution
        de la grille whitewater). `cell_size` : le pas de cette grille
        collider (unites monde).

        `cnrm=None` efface le champ courant (equivalent a passer `NULL` cote
        C) ; dans ce cas, `bq_whitewater_step` retombe sur le calcul par
        differences finies a partir du seul `collider_sdf`.
        """
        if cnrm is None:
            res_arr = (ctypes.c_int * 3)(0, 0, 0)
            return self._check(
                self._dll.bq_whitewater_set_collider_cnrm(
                    self._handle, None, res_arr, ctypes.c_float(0.0)
                )
            )

        arr = np.ascontiguousarray(cnrm, dtype=np.float32)
        expected = int(res[0]) * int(res[1]) * int(res[2]) * 4
        if arr.size != expected:
            raise ValueError(
                f"set_collider_cnrm: {arr.size} valeurs fournies, "
                f"{expected} attendues pour res={tuple(int(r) for r in res)}"
            )
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        res_arr = (ctypes.c_int * 3)(int(res[0]), int(res[1]), int(res[2]))
        return self._check(
            self._dll.bq_whitewater_set_collider_cnrm(
                self._handle, ptr, res_arr, ctypes.c_float(cell_size)
            )
        )

    def read(self):
        """Renvoie `(pos, type, size, age, vel)` de l'etat courant : `pos` un
        ndarray `(n, 3)` float32, `type` un ndarray `(n,)` int32
        (`BQ_WW_SPRAY`/`BQ_WW_FOAM`/`BQ_WW_BUBBLE`), `size`/`age` des
        ndarray `(n,)` float32, `vel` un ndarray `(n, 3)` float32 (vitesse
        courante, espace solveur, memes unites que `pos`). `n == 0` (aucune
        particule active) renvoie des tableaux vides, pas une erreur."""
        n = self.count()

        pos = np.empty((n, 3), dtype=np.float32)
        type_ = np.empty((n,), dtype=np.int32)
        size = np.empty((n,), dtype=np.float32)
        age = np.empty((n,), dtype=np.float32)
        vel = np.empty((n, 3), dtype=np.float32)

        if n:
            pos_ptr = pos.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            type_ptr = type_.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            size_ptr = size.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            age_ptr = age.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            vel_ptr = vel.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            pos_ptr = type_ptr = size_ptr = age_ptr = vel_ptr = None

        self._check(
            self._dll.bq_whitewater_read(
                self._handle, pos_ptr, type_ptr, size_ptr, age_ptr, vel_ptr
            )
        )
        return pos, type_, size, age, vel

    def close(self):
        """Libere la memoire GPU. Idempotent."""
        if getattr(self, "_handle", None):
            self._dll.bq_whitewater_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        # Filet de securite, meme discipline que Sim.__del__/Mesher.__del__.
        try:
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Verification autonome : `python lib.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = default_config()
    print(
        "config par defaut :",
        "grid_res=(%d,%d,%d) cell_size=%.5f gravity_y=%.3f cfl=%.3f ppc_axis=%d max_particles=%d"
        % (
            cfg.grid_res[0],
            cfg.grid_res[1],
            cfg.grid_res[2],
            cfg.cell_size,
            cfg.gravity_y,
            cfg.cfl,
            cfg.ppc_axis,
            cfg.max_particles,
        ),
    )

    with Sim(cfg) as sim:
        mat_id = sim.add_material(BQ_MODEL_WATER, rho=1000.0, bulk=4e4, gamma=3.0)
        print("materiau eau cree, id =", mat_id)

        n_emit = sim.emit_box(
            mat_id,
            lo=(0.10, 0.10, 0.10),
            hi=(0.35, 0.60, 0.90),
        )
        print("particules emises :", n_emit)
        print("particle_count :", sim.particle_count)

        for i in range(3):
            substeps = sim.step(1.0 / 24.0)
            print("step %d -> %d substeps" % (i, substeps))

        pos = sim.read_positions()
        print("positions shape :", pos.shape, pos.dtype)
        print("min :", pos.min(axis=0))
        print("max :", pos.max(axis=0))

    print("OK — simulation detruite proprement.")
