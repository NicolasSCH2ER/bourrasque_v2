"""PropertyGroups de l'extension Bourrasque et presets materiaux.

Ce module ne fait QUE de la declaration de proprietes et du calcul derive
(transformation monde <-> solveur, estimation du nombre de particules). Aucun
operateur, aucun panneau, aucun appel a la DLL : voir ops.py, ui.py, lib.py.

La logique de transformation pure (sans dependance bpy) vit dans
`extension.transform`, pour rester testable hors Blender.
"""

import math

import bpy
import mathutils
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, PropertyGroup, Scene

from .transform import solver_to_world, world_to_solver, world_to_solver_dir

__all__ = (
    "BqObjectProps",
    "BqSceneProps",
    "iter_elements",
    "domain_transform",
    "domain_resolution",
    "domain_usable_bounds",
    "world_to_solver",
    "solver_to_world",
    "world_to_solver_dir",
    "emitter_bounds_solver",
    "emitter_overflow",
    "estimate_particle_count",
    "estimate_inflow_count",
    "classes",
    "register",
    "unregister",
)


# ---------------------------------------------------------------------------
# Presets materiaux
# ---------------------------------------------------------------------------
# Valeurs exactement celles des scenes de reference de core/headless/main.cpp
# (scenes "jelly" et "dam") : ne pas les modifier sans mettre a jour le
# headless en meme temps, sous peine de casser la parite V4 du jalon.

_PRESET_WATER = {"model": "WATER", "rho": 1000.0, "bulk": 4.0e4, "gamma": 3.0}
_PRESET_JELLY = {"model": "ELASTIC", "rho": 1000.0, "young": 5.0e4, "poisson": 0.2}


def _apply_preset(obj_props, values):
    for key, value in values.items():
        setattr(obj_props, key, value)


def _on_preset_update(self, context):
    if self.preset == "WATER":
        _apply_preset(self, _PRESET_WATER)
    elif self.preset == "JELLY":
        _apply_preset(self, _PRESET_JELLY)
    # CUSTOM : ne touche a rien.


def _poll_domain_object(self, obj):
    return obj.type == "MESH"


class BqObjectProps(PropertyGroup):
    """Role et parametres materiau, stockes par objet."""

    role: EnumProperty(
        name="Role",
        items=(
            ("NONE", "Aucun", "Cet objet n'a aucun role dans la simulation"),
            ("DOMAIN", "Domaine", "Definit le domaine de la simulation"),
            ("EMITTER", "Émetteur", "Emet des particules dans la simulation"),
        ),
        default="NONE",
    )

    model: EnumProperty(
        name="Modele",
        items=(
            ("ELASTIC", "Élastique", "Modele corotationnel (E, nu)"),
            ("WATER", "Eau", "EOS de Tait (bulk, gamma)"),
        ),
        default="WATER",
    )

    rho: FloatProperty(
        name="Densite",
        description="Densite du materiau (kg/m^3)",
        default=1000.0,
        min=1e-6,
        unit="NONE",
    )

    young: FloatProperty(
        name="Module de Young",
        description="Module de Young E (modele elastique)",
        default=5.0e4,
        min=1e-6,
    )

    poisson: FloatProperty(
        name="Coefficient de Poisson",
        description="Coefficient de Poisson nu (modele elastique)",
        default=0.2,
        min=0.0,
        max=0.49,
    )

    bulk: FloatProperty(
        name="Module de compressibilite",
        description="Module de compressibilite k (modele eau)",
        default=4.0e4,
        min=1e-6,
    )

    gamma: FloatProperty(
        name="Exposant de Tait",
        description="Exposant de Tait (modele eau)",
        default=3.0,
        min=1.0,
        max=7.0,
    )

    initial_velocity: FloatVectorProperty(
        name="Vitesse initiale",
        description="Vitesse initiale, exprimee dans l'espace monde Blender",
        size=3,
        subtype="VELOCITY",
        default=(0.0, 0.0, 0.0),
    )

    turbulence: FloatProperty(
        name="Turbulence",
        description=(
            "Perturbation aleatoire des positions et vitesses a l'emission. "
            "0 donne un ecoulement parfaitement lisse (reseau regulier, "
            "vitesse uniforme) ; la valeur agit comme une fraction de la "
            "vitesse caracteristique de l'emission"
        ),
        default=0.0,
        min=0.0,
        soft_max=1.0,
        max=1.0,
    )

    turbulence_seed: IntProperty(
        name="Graine",
        description=(
            "Graine du generateur aleatoire de turbulence : deux graines "
            "differentes donnent deux motifs de turbulence differents pour "
            "des reglages identiques, la meme graine redonne le meme bake"
        ),
        default=0,
        min=0,
    )

    emit_mode: EnumProperty(
        name="Mode d'émission",
        description="Comment la matiere de cet emetteur est introduite dans la simulation",
        items=(
            (
                "BLOCK",
                "Bloc",
                "Toute la matiere est emise d'un coup, a la premiere frame",
            ),
            (
                "INFLOW",
                "Continu",
                "La matiere est emise en continu pendant toute la simulation",
            ),
        ),
        default="BLOCK",
    )

    emit_source: EnumProperty(
        name="Source d'émission",
        description="Forme utilisee pour determiner ou la matiere est emise",
        items=(
            (
                "BOUNDS",
                "Boîte englobante",
                "La forme d'emission est la boite englobante (AABB) de l'objet",
            ),
            (
                "MESH",
                "Forme du maillage",
                "La forme d'emission suit la geometrie du maillage",
            ),
        ),
        default="BOUNDS",
    )

    preset: EnumProperty(
        name="Preset",
        items=(
            ("WATER", "Eau", "Preset eau (modele Tait)"),
            ("JELLY", "Gelée", "Preset gelee (modele elastique)"),
            ("CUSTOM", "Personnalisé", "Parametres personnalises"),
        ),
        default="WATER",
        update=_on_preset_update,
    )


class BqSceneProps(PropertyGroup):
    """Reglages du solveur et de la simulation, stockes par scene."""

    domain_object: PointerProperty(
        name="Domaine",
        description="Objet maillage dont la bounding box definit le domaine",
        type=Object,
        poll=_poll_domain_object,
    )

    grid_res: IntProperty(
        name="Resolution de grille",
        description=(
            "Résolution appliquée au plus grand axe du domaine ; la "
            "résolution des deux autres axes en découle par arrondi"
        ),
        default=64,
        min=16,
        max=256,
    )

    ppc_axis: IntProperty(
        name="Particules/cellule/axe",
        default=2,
        min=1,
        max=4,
    )

    gravity: FloatProperty(
        name="Gravite",
        description="Gravite verticale en monde Blender (sur Z)",
        default=-9.8,
    )

    cfl: FloatProperty(
        name="CFL",
        default=0.3,
        min=0.05,
        max=1.0,
    )

    max_particles: IntProperty(
        name="Particules max",
        default=2000000,
        min=1,
    )

    frame_start: IntProperty(
        name="Frame de debut",
        default=1,
    )

    frame_end: IntProperty(
        name="Frame de fin",
        default=120,
    )

    cache_dir: StringProperty(
        name="Dossier de cache",
        subtype="DIR_PATH",
        default="//bourrasque_cache/",
    )

    # Convention : cet index indexe `scene.objects`, PAS la liste filtree
    # renvoyee par `iter_elements`. `BQ_UL_elements.filter_items` (ui.py) ne
    # fait que masquer/reordonner l'AFFICHAGE d'un `template_list` construit
    # sur `scene.objects` ; l'index actif d'un `template_list` reste tou-
    # jours relatif a la collection passee en argument, non filtree. Tout
    # code qui lit `active_element_index` doit donc resoudre l'objet via
    # `scene.objects[index]`.
    active_element_index: IntProperty(
        name="Index de selection",
        default=0,
    )

    point_size: FloatProperty(
        name="Taille des points",
        default=0.01,
        min=0.0,
    )

    color_by_material: BoolProperty(
        name="Colorer par materiau",
        default=True,
    )

    is_baking: BoolProperty(
        name="Bake en cours",
        default=False,
    )

    bake_progress: FloatProperty(
        name="Progression du bake",
        default=0.0,
        min=0.0,
        max=1.0,
    )

    baked_frames: IntProperty(
        name="Frames bakees",
        default=0,
    )


# ---------------------------------------------------------------------------
# Fonctions utilitaires (module-level : source de verite unique pour l'UI et
# les operateurs)
# ---------------------------------------------------------------------------


def iter_elements(scene):
    """Itere sur tous les objets de `scene` dont `role != NONE`.

    Ordre : le domaine d'abord (s'il existe), puis les emetteurs, dans
    l'ordre du nom.
    """
    domain = []
    emitters = []
    for obj in scene.objects:
        role = obj.bourrasque.role
        if role == "DOMAIN":
            domain.append(obj)
        elif role == "EMITTER":
            emitters.append(obj)
    domain.sort(key=lambda o: o.name)
    emitters.sort(key=lambda o: o.name)
    return domain + emitters


# Demi-largeur du stencil B-spline quadratique du MLS-MPM (`p.bound` dans
# `core/src/mlsmpm.cu::upload_params`, et le clamp de `k_g2p`) : une
# particule a moins de `SOLVER_STENCIL_BOUND` noeuds du bord du pave solveur
# lirait la grille hors bornes. Valeur figee cote solveur (coeur CUDA, hors
# perimetre de ce module) ; dupliquee ici uniquement pour le calcul du
# mapping boite-artiste -> pave-solveur, voir `domain_transform`.
SOLVER_STENCIL_BOUND = 3


def _stencil_margin(dx):
    """Marge (en espace solveur, meme unite que `dx`) occupee par le
    stencil du MLS-MPM sur chaque face du pave SOLVEUR — source de verite
    unique reutilisee par `domain_transform`/`_domain_layout`,
    `domain_usable_bounds` et `emitter_overflow`, pour ne jamais la
    calculer deux fois avec des valeurs incoherentes.

    Depuis M5, `dx` (taille de maille, UNIFORME sur les trois axes) est la
    grandeur primitive du domaine ; la marge vaut simplement
    `SOLVER_STENCIL_BOUND * dx`, la meme sur les six faces du pave puisque
    `dx` ne varie pas par axe (voir docs/plan-milestone-5.md, decision D1).
    """
    return SOLVER_STENCIL_BOUND * dx


def _domain_layout(scene):
    """Calcule `(origin, size, res, dx)` pour l'objet domaine de `scene` —
    source de verite unique reutilisee par `domain_transform`,
    `domain_resolution` et `domain_usable_bounds`, pour ne calculer la bbox
    de l'objet domaine et le pas de grille qu'une seule fois.

    `size` et `res` sont des triplets en espace SOLVEUR (`(x, y, z)`
    solveur — voir `transform.py` pour le mapping des axes monde <->
    solveur). `dx` est la taille de maille, UNIFORME sur les trois axes.

    L'artiste regle une seule resolution `R` (`scene.bourrasque.grid_res`),
    appliquee au PLUS GRAND axe de sa boite (en espace monde, equivalent en
    espace solveur puisque le mapping ne fait que permuter/inverser les
    axes sans les mettre a l'echelle) :

        dx      = max(extent_monde) / R
        res[i]  = max(1, ceil(extent_solveur[i] / dx)) + 2*SOLVER_STENCIL_BOUND
        size[i] = res[i] * dx
        origin  = bbox_min_monde - marge de stencil (SOLVER_STENCIL_BOUND * dx)

    Le solveur impose des parois separantes sur une marge de
    `SOLVER_STENCIL_BOUND` noeuds (stencil du MLS-MPM, structurellement
    necessaire, voir `core/src/mlsmpm.cu`), qui rend cette bande du pave
    inutilisable. Plutot que de laisser cette marge grignoter la boite que
    l'artiste a placee, on agrandit le pave du solveur de la marge sur
    chaque face, pour que la boite de l'artiste devienne exactement la zone
    utile (voir `domain_usable_bounds`).

    Renvoie `None` si aucun objet domaine n'est defini.
    """
    obj = scene.bourrasque.domain_object
    if obj is None:
        return None

    mat = obj.matrix_world
    corners = [mat @ mathutils.Vector(c) for c in obj.bound_box]

    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]

    min_corner = (min(xs), min(ys), min(zs))
    max_corner = (max(xs), max(ys), max(zs))
    extent_world = (
        max_corner[0] - min_corner[0],
        max_corner[1] - min_corner[1],
        max_corner[2] - min_corner[2],
    )
    # Mapping monde -> solveur (voir transform.py, docs/plan-milestone-5.md
    # D3) : solveur X <- monde X, solveur Y <- monde Z, solveur Z <- monde Y
    # (inverse, mais l'ETENDUE est la meme quel que soit le sens).
    extent_solver = (extent_world[0], extent_world[2], extent_world[1])

    grid_res = scene.bourrasque.grid_res
    max_extent = max(extent_world)

    if max_extent <= 0:
        # Boite degeneree (dimension nulle) : aucune formule de dx n'a de
        # sens. On renvoie un pave degenere (size nulle) plutot que de
        # lever ou de diviser par zero ici ; c'est `BQ_OT_bake._validate`
        # (ops.py) qui refuse explicitement un domaine de taille nulle
        # avant le bake.
        dx = 0.0
        res = tuple(2 * SOLVER_STENCIL_BOUND for _ in range(3))
    else:
        dx = max_extent / grid_res
        # ceil, pas round : round peut arrondir vers le bas, ce qui rend
        # la zone utile (voir domain_usable_bounds) plus PETITE que la
        # boite de l'artiste — jusqu'a dx/2 de moins sur un axe — et un
        # emetteur cale sur cette boite serait alors refuse par
        # emitter_overflow alors que l'artiste n'a rien fait de mal. ceil
        # garantit `res[a]*dx >= extent_solver[a] + 2*marge`, donc que la
        # boite de l'artiste reste toujours entierement contenue dans la
        # zone utile, au prix d'au plus une cellule de plus par axe. Sur
        # un domaine cubique, extent/dx est entier et ceil == round : ce
        # choix ne change rien au cas deja valide (V1).
        res = tuple(
            max(1, math.ceil(extent_solver[a] / dx)) + 2 * SOLVER_STENCIL_BOUND
            for a in range(3)
        )

    size = tuple(res[a] * dx for a in range(3))
    marge = _stencil_margin(dx)

    origin = (
        min_corner[0] - marge,
        min_corner[1] - marge,
        min_corner[2] - marge,
    )
    return (origin, size, res, dx)


def domain_transform(scene):
    """Renvoie `(origin, size)` decrivant le pave du SOLVEUR pour l'objet
    domaine de `scene` — PAS la boite de l'artiste. `size` est un triplet
    `(size_x, size_y, size_z)` en espace solveur (voir `_domain_layout` et
    `transform.py` pour le mapping des axes).

    Renvoie `None` si aucun objet domaine n'est defini.
    """
    layout = _domain_layout(scene)
    if layout is None:
        return None
    origin, size, _res, _dx = layout
    return (origin, size)


def domain_resolution(scene):
    """Renvoie `(res, dx)` : `res` est le triplet `(res_x, res_y, res_z)`
    du nombre de cellules par axe du pave SOLVEUR (celui effectivement
    alloue par le solveur, marge de stencil incluse), `dx` est la taille de
    maille (uniforme sur les trois axes). C'est desormais `dx` la grandeur
    primitive (voir `_domain_layout`) : tout code qui calculait `dx = size
    / grid_res` doit passer par cette fonction plutot que refaire le calcul.

    Renvoie `None` si aucun objet domaine n'est defini.
    """
    layout = _domain_layout(scene)
    if layout is None:
        return None
    _origin, _size, res, dx = layout
    return (res, dx)


def domain_usable_bounds(scene):
    """Renvoie `(lo, hi)`, les bornes de la zone UTILE du pave solveur en
    espace solveur — c'est-a-dire la boite que l'artiste a placee, mappee
    dans `[marge, size - marge]` par axe (voir `_domain_layout`). `lo` et
    `hi` sont desormais des triplets (le pave n'est plus isotrope) : `lo`
    vaut `(marge, marge, marge)` (`dx` uniforme, la marge est la meme sur
    les trois axes), `hi` vaut `(size[0]-marge, size[1]-marge,
    size[2]-marge)`. Renvoie `None` si aucun domaine n'est defini.
    """
    layout = _domain_layout(scene)
    if layout is None:
        return None
    _origin, size, _res, dx = layout
    marge = _stencil_margin(dx)
    lo = (marge, marge, marge)
    hi = tuple(size[a] - marge for a in range(3))
    return (lo, hi)


def emitter_bounds_solver(obj, origin, size):
    """Renvoie `(lo, hi)`, la bbox monde de `obj` convertie en espace solveur.

    Chaque composante est reordonnee pour que `lo[i] <= hi[i]` : la
    conversion `world_to_solver` inverse l'axe Y monde, donc min et max
    s'echangent sur l'axe solveur correspondant (sz).
    """
    mat = obj.matrix_world
    corners_world = [mat @ mathutils.Vector(c) for c in obj.bound_box]
    corners_solver = [
        world_to_solver((c.x, c.y, c.z), origin, size) for c in corners_world
    ]

    xs = [c[0] for c in corners_solver]
    ys = [c[1] for c in corners_solver]
    zs = [c[2] for c in corners_solver]

    lo = (min(xs), min(ys), min(zs))
    hi = (max(xs), max(ys), max(zs))
    return (lo, hi)


def emitter_overflow(obj, origin, size, dx):
    """Detecte un emetteur trop proche ou hors du bord de la zone UTILE du
    domaine (la boite que l'artiste a placee, pas le pave du solveur
    elargi — voir `domain_transform` / `domain_usable_bounds`).

    Le kernel `k_p2g` du solveur n'a pas de test de bornes : un emetteur
    dont la boite sort du pave SOLVEUR produit des coordonnees hors
    `[0, size[axis]]` et une ecriture hors buffer GPU, qui tue le contexte
    CUDA de tout le processus Blender. `domain_transform` a deja agrandi le
    pave solveur de la marge du stencil (`SOLVER_STENCIL_BOUND` noeuds) de
    chaque cote pour que la boite de l'artiste tombe exactement sur la zone
    utile `[margin, size[axis] - margin]` ; c'est CETTE marge qu'on
    reutilise ici (`domain_usable_bounds`), pour ne pas la retrancher une
    seconde fois et faire perdre a l'artiste une bande qu'il a deja
    recuperee. `dx` (taille de maille, uniforme) est desormais la grandeur
    primitive : la marge est la meme sur les six faces.

    Renvoie `None` si `obj` est valide, sinon `(fully_outside,
    offending_axes, margin)` ou `offending_axes` est une liste de noms
    d'axes SOLVEUR (`sx`, `sy`, `sz` — voir `transform.py`) et
    `fully_outside` indique si l'emetteur est entierement hors du domaine
    sur au moins un de ces axes (par opposition a un simple debordement
    partiel).
    """
    margin = _stencil_margin(dx)
    lo, hi = emitter_bounds_solver(obj, origin, size)
    axis_names = ("sx", "sy", "sz")
    offending = []
    fully_outside = False

    # Tolerance flottante : un emetteur dont la boite COINCIDE exactement
    # avec celle du domaine (cas le plus courant en pratique : un artiste
    # cale volontairement son emetteur sur les parois) donne des bornes
    # solveur calculees par deux chemins geometriques legerement differents
    # (`emitter_bounds_solver` sur l'objet emetteur vs. `domain_transform`/
    # `_stencil_margin` sur l'objet domaine) : elles ne sont egales qu'a
    # l'epsilon flottant pres. Sans tolerance, l'arrondi peut faire declarer
    # "hors domaine" un emetteur pourtant exactement cale sur ses bords —
    # precisement le cas que ce correctif doit accepter.
    eps = max(1e-6, 1e-6 * max(size))
    for axis in range(3):
        s = size[axis]
        if hi[axis] <= margin - eps or lo[axis] >= s - margin + eps:
            fully_outside = True
            offending.append(axis_names[axis])
        elif lo[axis] < margin - eps or hi[axis] > s - margin + eps:
            offending.append(axis_names[axis])
    if not offending:
        return None
    return (fully_outside, offending, margin)


def _mesh_block_count(obj, origin, size, dx, ppc_axis):
    """Compte pour un emetteur BLOCK/MESH, via `extension.sampling`.

    `extension.sampling` est ecrit en parallele par un autre lot : ce module
    peut etre absent (`ImportError`) ou momentanement incomplet/en panne
    (toute autre exception). Dans les deux cas, on renvoie `None` pour que
    l'appelant retombe sur l'estimation par bbox — ce panneau est redessine
    a chaque interaction UI, il ne doit jamais planter a cause d'un module
    externe encore en chantier.
    """
    try:
        from .sampling import estimate_mesh_sample_count
    except ImportError:
        return None
    try:
        return estimate_mesh_sample_count(obj, origin, size, dx, ppc_axis)
    except Exception:
        return None


def estimate_inflow_count(
    obj, origin, size, dx, ppc_axis, duration, gravity=0.0
):
    """Nombre de particules qu'un emetteur INFLOW emettrait sur `duration`
    secondes de simulation.

    Regle d'emission retenue (celle implementee cote bake, voir
    `ops._emit_inflow_sites`) : le volume COMPLET de l'emetteur est
    ensemence des la premiere frame (test d'occupation contre un nuage vide
    au depart), puis maintenu SATURE — un site se reemet des que la
    particule qui l'occupait s'en est eloignee. Le total estime a donc DEUX
    termes : le remplissage initial du volume, puis le regime etabli
    (renouvellement continu au rythme ou la matiere s'ecoule).

    Le regime etabli est approxime par l'aire de la section transverse de
    la boite englobante (espace solveur) de l'emetteur, projetee sur le
    plan perpendiculaire a `v`. Pour un pave aligne sur les axes d'aretes
    `(ex, ey, ez)`, l'aire de cette silhouette vaut :

        aire = (ex*ey*|vz| + ey*ez*|vx| + ex*ez*|vy|) / |v|

    (formule standard de l'aire projetee d'un AABB ; c'est aussi exactement
    le volume balaye par seconde divise par `|v|`, ce qui rend le compte du
    regime etabli coherent : particules/s = aire * |v| / spacing^2 =
    volume_balaye_par_s / spacing^3, la densite attendue de la grille
    d'emission). Le remplissage initial est approxime, dans le meme esprit,
    par le volume de la boite englobante divise par `spacing^3` : le nuage
    de sites couvre TOUT le volume de l'emetteur (plus de notion de couche
    aval), donc c'est bien ce volume entier, pas une simple tranche, qui
    est ensemence des la premiere frame.

    Approximation par boite englobante retenue (regime etabli ET
    remplissage initial) car elle ne depend d'aucune hypothese sur
    l'orientation de l'emetteur ou de la vitesse, et ne necessite aucun
    rayon (fonction appelee a chaque redessin du panneau). Ne tient pas
    compte de `emit_source` : meme si `emit_source == "MESH"`, la geometrie
    exacte n'est pas connue sans echantillonnage — approximation jugee
    suffisante pour prevenir un depassement de `max_particles`.

    Vitesse nulle : le regime etabli est nul (aucun renouvellement), mais
    le remplissage initial reste du (l'emetteur sature son volume une fois
    puis s'arrete, voir docstring de `ops._InflowState`).

    Regime d'emission REEL (avec plafond de conservation du nombre, voir
    `ops._emit_inflow_sites`) : le volume de l'emetteur est desormais
    maintenu a la densite de repos, jamais comprime au-dela — le nombre de
    particules emises par frame est plafonne au deficit exact
    (`n_sites - n_occupants`). Consequence directe pour cette estimation :
    le site reemis a chaque frame n'est PAS necessairement celui qui vient
    de se liberer a la face d'ENTREE de l'emetteur (le plafond replace la
    matiere manquante a un site VACANT choisi au hasard n'importe ou dans
    le volume, voir `ops._emit_inflow_sites`) — l'ensemencement est donc
    VOLUMIQUE UNIFORME, pas une injection a la face amont.

    Correction de gravite (regime etabli uniquement) : le debit en regime
    etabli est le flux de matiere QUITTANT le volume de l'emetteur, pas
    celui y ENTRANT. Or `|v|` (la vitesse nominale reglee par l'artiste) est
    la vitesse d'ENTREE dans ce volume ; la matiere accelere sous gravite
    pendant qu'elle le TRAVERSE, donc c'est la vitesse de SORTIE qui
    gouverne le debit reel — mais PAS la vitesse de sortie d'une particule
    qui aurait parcouru la longueur COMPLETE `L = volume / aire` (`aire`
    etant la meme aire projetee que ci-dessus) : puisque le reapprovisionnement
    est volumique uniforme (voir ci-dessus), une particule fraichement
    reemise se trouve en moyenne au MILIEU du volume, pas a la face
    d'entree — sa distance moyenne restante jusqu'a la sortie est `L/2`, pas
    `L`. La formule de Torricelli (conservation de l'energie sous
    acceleration constante) s'applique donc sur `L/2` :

        g_par = gravite (espace solveur) . u,  u = v / |v|
        v_out = sqrt(max(|v|^2, |v|^2 + 2 * g_par * (L/2)))

    Ce facteur 1/2 N'EST PAS un ajustement empirique : c'est la consequence
    directe et necessaire du regime d'ensemencement volumique uniforme
    decrit ci-dessus (distance moyenne a parcourir = moitie de la
    traversee complete). Verifie par mesure (bake reel avec le plafond en
    place, emetteur BOUNDS, plusieurs gravites et durees, y compris a
    domaine agrandi pour eviter tout contact avec le fond) : ratio
    estime/reel dans [1.08, 1.22] sous gravite (jamais < 1, jamais > 1.3),
    contre [1.45, 1.53] avec la longueur complete `L` (surestimation trop
    large) et une derive sous 1.0 (SOUS-estimation, inacceptable pour un
    garde-fou) avec la moyenne cinematique `(|v|+v_out(L))/2` a mesure que
    la duree du bake grandit.

    Le `max(...)` est un plancher deliberement conservateur : si la gravite
    FREINE l'ecoulement (`g_par < 0`), on retombe sur `|v|` nominal plutot
    que sur une racine d'un nombre potentiellement negatif (ecoulement qui
    s'arreterait avant la sortie, hors du modele simple retenu ici). Pour un
    garde-fou contre `max_particles`, surestimer est le bon sens de
    l'erreur — mieux vaut un chiffre trop haut qu'un depassement silencieux
    en cours de bake. Le terme de remplissage initial (`initial_fill`
    ci-dessus) n'est pas concerne : c'est un volume fixe, independant de la
    vitesse de traversee.

    A gravite nulle (`g_par = 0`), `v_out` degenere en `|v|` quel que soit
    le denominateur (`L` ou `L/2`) : ce cas n'est pas ameliore par cette
    correction (l'estimation y reste a son ratio historique, ~1.19 mesure),
    la correction n'agit que sur la partie gravitationnelle du modele.
    """
    spacing = dx / ppc_axis
    if spacing <= 0 or duration <= 0:
        return 0

    lo, hi = emitter_bounds_solver(obj, origin, size)
    ex = hi[0] - lo[0]
    ey = hi[1] - lo[1]
    ez = hi[2] - lo[2]

    initial_fill = max(0, math.ceil(ex * ey * ez / (spacing**3) - 0.5))

    v_solver = world_to_solver_dir(tuple(obj.bourrasque.initial_velocity))
    speed = math.sqrt(sum(c * c for c in v_solver))
    if speed < 1e-6:
        return int(initial_fill)

    vx, vy, vz = v_solver
    area = (ex * ey * abs(vz) + ey * ez * abs(vx) + ex * ez * abs(vy)) / speed
    particles_per_layer = max(0, math.ceil(area / (spacing * spacing) - 0.5))

    # Vitesse de sortie du volume de l'emetteur, sous gravite (voir
    # docstring) : longueur MOYENNE restant a parcourir pour une particule
    # fraichement reemise, sous plafond de conservation du nombre.
    # Reapprovisionnement volumique uniforme (le plafond replace la matiere
    # manquante a un site vacant n'importe ou dans le volume, pas
    # uniquement a la face d'entree) : cette distance moyenne est L/2, la
    # moitie de la traversee complete `L = volume / aire`. Formule de
    # Torricelli sur cette demi-longueur, avec plancher conservateur sur
    # `speed`.
    volume = ex * ey * ez
    if area > 0.0:
        traversal_length = volume / area
        uy = vy / speed  # composante Y de u = v / |v| (espace solveur)
        g_par = gravity * uy  # gravite solveur = (0, gravity, 0) . u
        v_out = math.sqrt(max(speed * speed, speed * speed + 2.0 * g_par * (traversal_length / 2.0)))
    else:
        # `area` nulle (emetteur degenere, aire projetee nulle) : le
        # produit `particles_per_layer * n_layers` est de toute facon nul
        # via `particles_per_layer`, la valeur de `v_out` n'a pas d'impact.
        v_out = speed

    n_layers = v_out * duration / spacing
    steady_state = particles_per_layer * n_layers

    return int(initial_fill + steady_state)


def estimate_particle_count(scene):
    """Nombre total de particules qui seraient emises, tous emetteurs
    confondus, sur la duree totale du bake.

    Pour un emetteur BLOCK : le solveur emet sur une grille reguliere de pas
    `dx / ppc_axis` (`dx` = taille de maille, uniforme, voir
    `domain_resolution`) ; c'est le produit sur les 3 axes du nombre
    d'echantillons tenant dans la boite (ou, en mode `emit_source ==
    "MESH"`, l'estimation par rapport de volumes de
    `extension.sampling.estimate_mesh_sample_count`, si ce module est
    disponible). Pour un emetteur INFLOW, voir `estimate_inflow_count`.
    Renvoie 0 si le domaine n'est pas defini.

    Reproduit exactement le compte du solveur (`bq_emit_box`) pour le cas
    BLOCK/BOUNDS, qui echantillonne par
    `for (x = lo + step/2; x < hi; x += step)`. Le nombre d'iterations de
    cette boucle est `ceil(extent/step - 0.5)` (0 si negatif ou nul) :
    `max(0, int)` seul (utilise auparavant) est un plancher qui peut
    sous-estimer d'une particule par axe.
    """
    transform = domain_transform(scene)
    if transform is None:
        return 0
    origin, size = transform
    resolution = domain_resolution(scene)
    if resolution is None:
        return 0
    _res, dx = resolution

    props = scene.bourrasque
    step = dx / props.ppc_axis
    if step <= 0:
        return 0

    fps_base = scene.render.fps_base or 1.0
    fps = scene.render.fps / fps_base if fps_base else 0.0
    n_frames = max(0, props.frame_end - props.frame_start + 1)
    duration = n_frames / fps if fps > 0 else 0.0

    total = 0
    for obj in scene.objects:
        obj_props = obj.bourrasque
        if obj_props.role != "EMITTER":
            continue

        if obj_props.emit_mode == "INFLOW":
            total += estimate_inflow_count(
                obj,
                origin,
                size,
                dx,
                props.ppc_axis,
                duration,
                props.gravity,
            )
            continue

        # BLOCK.
        count = None
        if obj_props.emit_source == "MESH":
            count = _mesh_block_count(
                obj, origin, size, dx, props.ppc_axis
            )
        if count is None:
            lo, hi = emitter_bounds_solver(obj, origin, size)
            count = 1
            for axis in range(3):
                extent = hi[axis] - lo[axis]
                n = max(0, math.ceil(extent / step - 0.5))
                count *= n
        # Un emetteur BLOCK plus petit que le pas du reseau (ou dont
        # l'estimation MESH par rapport de volumes arrondit a 0) n'est plus
        # refuse au bake : `BQ_OT_bake.invoke` (ops.py) emet alors une
        # UNIQUE particule au centre de l'emetteur. L'estimation doit
        # refleter ce plancher, sinon elle contredirait le resultat reel du
        # bake (0 annonce, 1 obtenu). Ne s'applique qu'a BLOCK : un INFLOW
        # dont le nuage de sites est vide reste refuse (voir ops.py), la
        # notion de "centre" n'ayant pas de sens pour un flux continu.
        count = max(count, 1)
        total += count
    return total


# ---------------------------------------------------------------------------
# Enregistrement
# ---------------------------------------------------------------------------

classes = (
    BqObjectProps,
    BqSceneProps,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    Object.bourrasque = PointerProperty(type=BqObjectProps)
    Scene.bourrasque = PointerProperty(type=BqSceneProps)


def unregister():
    del Scene.bourrasque
    del Object.bourrasque
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
