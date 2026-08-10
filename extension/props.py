"""PropertyGroups de l'extension Bourrasque et presets materiaux.

Ce module ne fait QUE de la declaration de proprietes et du calcul derive
(transformation monde <-> solveur, estimation du nombre de particules). Aucun
operateur, aucun panneau, aucun appel a la DLL : voir ops.py, ui.py, lib.py.

La logique de transformation pure (sans dependance bpy) vit dans
`extension.transform`, pour rester testable hors Blender.
"""

import math
import pathlib

import bpy
import mathutils
from bpy.props import (
    BoolProperty,
    BoolVectorProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Object, PropertyGroup, Scene

from . import lib
from .materials import unique_name
from .transform import solver_to_world, world_to_solver, world_to_solver_dir

__all__ = (
    "BqMaterialProps",
    "BqObjectProps",
    "BqSceneProps",
    "iter_elements",
    "material_usage",
    "used_material_slot_count",
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
    "collider_triangle_count",
    "mesh_layout",
    "mesh_resolution_state",
    "mesh_particle_spacing",
    "mesh_effective_radii",
    "mesh_vram_estimate_bytes",
    "mesh_vram_warning_threshold_bytes",
    "mesh_cache_path",
    "whitewater_cache_path",
    "whitewater_config_from_scene",
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
# Sable sec, sans cohesion (jalon M18, D7) : la cohesion cote coeur est
# REFUSEE pour SAND aujourd'hui (`bq_add_material`), donc aucun preset ne
# peut en proposer une valeur non nulle.
_PRESET_SAND = {
    "model": "SAND",
    "rho": 1600.0,
    "young": 3.5e5,
    "poisson": 0.3,
    "friction_angle": 35.0,
}


def _apply_preset(obj_props, values):
    for key, value in values.items():
        setattr(obj_props, key, value)


def _on_preset_update(self, context):
    if self.preset == "WATER":
        _apply_preset(self, _PRESET_WATER)
    elif self.preset == "JELLY":
        _apply_preset(self, _PRESET_JELLY)
    elif self.preset == "SAND":
        _apply_preset(self, _PRESET_SAND)
    # CUSTOM : ne touche a rien.


# Modele constitutif que chaque preset IMPLIQUE (derive de `_PRESET_WATER` /
# `_PRESET_JELLY` / `_PRESET_SAND` ci-dessus, source de verite de ces
# valeurs).
_PRESET_IMPLIED_MODEL = {
    "WATER": _PRESET_WATER["model"],
    "JELLY": _PRESET_JELLY["model"],
    "SAND": _PRESET_SAND["model"],
}


def _on_model_update(self, context):
    """Callback `update` de `model` : bascule `preset` sur « Personnalisé »
    des que le modele choisi ne correspond plus a celui qu'implique le preset
    courant.

    Sans ce garde-fou, `preset` et `model` derivent silencieusement l'un de
    l'autre : c'est exactement ce qui s'est produit dans l'ancienne UI, ou un
    materiau pouvait porter `preset == "WATER"` et `model == "ELASTIC"`. La
    migration M16 baptisait alors « Eau » un materiau elastique. Le libelle a
    ete rendu robuste en aval (`materials._label_for_emitter`), mais la CAUSE
    est ici : deux champs qui se contredisent sans que rien ne le signale.
    Aucune valeur physique n'est touchee — seule l'etiquette `preset` est
    remise a une valeur honnete."""
    implied = _PRESET_IMPLIED_MODEL.get(self.preset)
    if implied is not None and implied != self.model:
        self.preset = "CUSTOM"


def _poll_domain_object(self, obj):
    return obj.type == "MESH"


def _on_material_name_update(self, context):
    """Callback `update` de `BqMaterialProps.name` : force l'unicite du nom
    au sein de la bibliotheque de la scene, puis propage le renommage a tous
    les emetteurs qui referencaient l'ancien nom (voir docstring de
    `BqMaterialProps.name_prev` pour pourquoi ce champ existe).

    Piege a eviter (et evite ici) : reassigner `self.name` depuis ce meme
    callback re-declenche ce callback (recursion). Le motif retenu est de
    ne reecrire `self.name` QUE si l'unicite l'exige reellement (la valeur
    calculee differe de la valeur courante) et de sortir aussitot : l'appel
    recursif voit alors une valeur deja unique, ne reecrit plus rien, et
    c'est LUI qui effectue la propagation et la mise a jour de `name_prev`
    (avec le VRAI `name_prev`, capture avant toute reecriture). Sans ce
    `return` immediat, l'appel exterieur continuerait avec un `self.name`
    perime (celui d'avant le suffixage) et propagerait le mauvais nom.
    """
    scene = self.id_data
    if not isinstance(scene, Scene):
        # `id_data` est l'ID proprietaire de la donnee RNA (ici la Scene qui
        # possede `scene.bourrasque.materials`) ; ce garde-fou n'est la que
        # pour rester defensif si ce PropertyGroup venait un jour a etre
        # instancie hors de ce contexte precis.
        return

    existing = [
        m.name
        for m in scene.bourrasque.materials
        if m.as_pointer() != self.as_pointer()
    ]
    unique = unique_name(self.name, existing)
    if unique != self.name:
        self.name = unique
        return

    old_name = self.name_prev
    if old_name and old_name != self.name:
        for obj in scene.objects:
            if obj.bourrasque.material_name == old_name:
                obj.bourrasque.material_name = self.name
    self.name_prev = self.name


class BqMaterialProps(PropertyGroup):
    """Un materiau nomme de la bibliotheque de la scene (`BqSceneProps.materials`).

    Regroupe les parametres physiques auparavant recopies sur chaque objet
    emetteur (voir la section DEPRECIEE de `BqObjectProps` ci-dessous) : les
    emetteurs referencent desormais un materiau de cette bibliotheque par son
    nom (`BqObjectProps.material_name`) plutot que de porter leurs propres
    valeurs.
    """

    name: StringProperty(
        name="Nom",
        description="Nom du materiau, unique au sein de la scene",
        default="Matériau",
        update=_on_material_name_update,
    )

    # Memorise le nom precedent pour que `_on_material_name_update` sache
    # quels emetteurs (dont `material_name == name_prev`) doivent suivre le
    # renommage : Blender ne fournit PAS l'ancienne valeur d'une propriete
    # dans son callback `update`, seule la nouvelle est visible via `self`.
    # Cache de l'UI (HIDDEN) : ce n'est pas un reglage, uniquement un etat
    # interne de suivi.
    name_prev: StringProperty(
        name="Nom precedent",
        options={"HIDDEN"},
        default="",
    )

    model: EnumProperty(
        name="Modele",
        items=(
            ("ELASTIC", "Élastique", "Modele corotationnel (E, nu)"),
            ("WATER", "Eau", "EOS de Tait (bulk, gamma)"),
            ("SAND", "Sable", "Modele elastoplastique de Drucker-Prager (E, nu, angle de frottement)"),
        ),
        default="WATER",
        update=_on_model_update,
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
        description="Module de Young E (modele elastique/sable)",
        default=5.0e4,
        min=1e-6,
    )

    poisson: FloatProperty(
        name="Coefficient de Poisson",
        description="Coefficient de Poisson nu (modele elastique/sable)",
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

    friction_angle: FloatProperty(
        name="Angle de frottement",
        description=(
            "Angle de frottement interne, en degrés (modèle sable) : "
            "commande la pente du tas au repos. Le cœur refuse toute "
            "valeur hors de l'intervalle ouvert ]0°, 90°["
        ),
        default=35.0,
        min=1e-3,
        max=89.999,
        subtype="NONE",
        unit="NONE",
    )

    cohesion: FloatProperty(
        name="Cohésion",
        description=(
            "Cohésion du sable (modèle sable) — NON IMPLÉMENTÉE côté "
            "solveur : le cœur refuse aujourd'hui tout matériau sable "
            "avec une cohésion non nulle. Volontairement absente de "
            "l'interface (voir ui.py) ; laisser à 0"
        ),
        default=0.0,
        min=0.0,
    )

    preset: EnumProperty(
        name="Preset",
        items=(
            ("WATER", "Eau", "Preset eau (modele Tait)"),
            ("JELLY", "Gelée", "Preset gelee (modele elastique)"),
            ("SAND", "Sable", "Preset sable (modele de Drucker-Prager)"),
            ("CUSTOM", "Personnalisé", "Parametres personnalises"),
        ),
        default="WATER",
        update=_on_preset_update,
    )

    # Couleur d'identification du materiau dans les listes UI (bibliotheque,
    # `prop_search` des emetteurs) et dans l'overlay des emetteurs du
    # viewport (`overlay.py`). Ne colore PAS le nuage de particules : ce
    # pipeline-la n'existe pas. `color_by_material` sur `BqSceneProps` n'est
    # lu nulle part dans la codebase — c'est un reglage annonce mais jamais
    # implemente, ne pas le presenter comme actif.
    viewport_color: FloatVectorProperty(
        name="Couleur",
        description="Couleur d'identification de ce materiau dans les listes de l'interface",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.1, 0.4, 0.9, 1.0),
    )


class BqObjectProps(PropertyGroup):
    """Role et parametres materiau, stockes par objet."""

    role: EnumProperty(
        name="Role",
        items=(
            ("NONE", "Aucun", "Cet objet n'a aucun role dans la simulation"),
            ("DOMAIN", "Domaine", "Definit le domaine de la simulation"),
            ("EMITTER", "Émetteur", "Emet des particules dans la simulation"),
            (
                "COLLIDER",
                "Collider",
                "Obstacle solide avec lequel le fluide interagit",
            ),
        ),
        default="NONE",
    )

    material_name: StringProperty(
        name="Materiau",
        description=(
            "Nom du materiau de la bibliotheque de la scene utilise par cet "
            "emetteur (voir `BqSceneProps.materials`) ; edite via "
            "`layout.prop_search` cote UI"
        ),
        default="",
    )

    # -- DEPRECIE (jalon bibliotheque de materiaux) ----------------------
    #
    # Les champs materiau ci-dessous vivaient auparavant directement sur
    # l'objet emetteur ; ils sont REMPLACES par `material_name` ci-dessus,
    # qui reference un `BqMaterialProps` de `BqSceneProps.materials`. On ne
    # les supprime PAS : ils restent la seule source de verite pour migrer
    # les .blend existants vers la bibliotheque (operateur de migration,
    # ecrit ailleurs, qui les lit puis les vide/ignore). `ui.py` ne les
    # affiche plus.

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

    # -- Fin section depreciee --------------------------------------------

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

    friction: FloatProperty(
        name="Friction",
        description=(
            "Frottement du fluide contre cet obstacle : 0 laisse le fluide "
            "glisser librement le long de sa surface, 1 le fait y adhérer"
        ),
        default=0.2,
        min=0.0,
        max=1.0,
    )

    # -- Colliders dynamiques (jalon M17, phase A) ------------------------

    dynamic: BoolProperty(
        name="Dynamique",
        description="Poussé par le fluide (corps rigide libre, 6 degrés de liberté)",
        default=False,
    )

    density: FloatProperty(
        name="Densité",
        description=(
            "Densité du corps (kg/m³) : gouverne s'il flotte ou coule face "
            "au fluide environnant"
        ),
        default=500.0,
        min=0.0001,
        soft_max=5000.0,
    )

    use_gravity: BoolProperty(
        name="Gravité",
        description="Ce corps dynamique subit la gravité",
        default=True,
    )

    added_mass: FloatProperty(
        name="Masse ajoutée",
        description=(
            "Amortit un corps très léger devant la masse de fluide en "
            "contact pour éviter la divergence du couplage explicite, au "
            "prix d'une perte de quantité de mouvement (mesuré : ~21 % de "
            "dérive à 1.0, contre 10⁻⁶ à 0.0). Ce n'est pas un réglage de "
            "confort mais une soupape : à laisser à 0 sauf si le corps "
            "diverge visiblement"
        ),
        default=0.0,
        min=0.0,
        max=1.0,
    )

    lock_location: BoolVectorProperty(
        name="Verrouiller position",
        description="Bloque la translation de ce corps dynamique sur les axes monde cochés",
        size=3,
        default=(False, False, False),
    )

    lock_rotation: BoolVectorProperty(
        name="Verrouiller rotation",
        description="Bloque la rotation de ce corps dynamique autour des axes monde cochés",
        size=3,
        default=(False, False, False),
    )

    restitution: FloatProperty(
        name="Restitution",
        description="Élasticité du contact corps à corps (0 = aucun rebond, 1 = rebond parfait)",
        default=0.0,
        min=0.0,
        max=1.0,
    )


class BqSceneProps(PropertyGroup):
    """Reglages du solveur et de la simulation, stockes par scene."""

    # -- Bibliotheque de materiaux (jalon bibliotheque de materiaux) -----
    #
    # Materiaux nommes, references par les emetteurs via
    # `BqObjectProps.material_name` plutot que recopies sur chaque objet
    # (voir la section DEPRECIEE de `BqObjectProps`). `BqMaterialProps` doit
    # etre enregistre AVANT `BqSceneProps` (voir `classes` en bas de ce
    # fichier), Blender exigeant qu'un PropertyGroup reference existe deja
    # au moment de l'enregistrement du PropertyGroup qui le contient.

    materials: CollectionProperty(
        name="Materiaux",
        description="Bibliotheque de materiaux nommes de la scene",
        type=BqMaterialProps,
    )

    active_material_index: IntProperty(
        name="Index du materiau actif",
        default=0,
    )

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

    # -- Maillage de surface (jalon M7) --------------------------------
    #
    # La resolution du maillage est INDEPENDANTE de la resolution de
    # simulation (`grid_res` ci-dessus) — decision D3 du plan M7 : l'artiste
    # peut vouloir un maillage plus fin (ou plus grossier) que la grille du
    # solveur. Meme convention de reglage que `grid_res` (un entier unique,
    # applique au plus grand axe du domaine) pour rester coherent avec la
    # facon dont la resolution de simulation est deja exposee.

    mesh_resolution: IntProperty(
        name="Resolution du maillage",
        description=(
            "Resolution du champ de reconstruction de surface, appliquee "
            "au plus grand axe du domaine (independante de la resolution "
            "de simulation) ; la resolution des deux autres axes en "
            "decoule par arrondi. 0 = automatique (une cellule de "
            "maillage par espacement inter-particules, plafonnee pour "
            "rester sous l'empreinte memoire recommandee — voir le "
            "panneau ci-dessous). Voir l'empreinte memoire estimee "
            "ci-dessous avant d'augmenter cette valeur"
        ),
        default=0,
        min=0,
        soft_max=320,
        max=1024,
    )

    mesh_influence_factor: FloatProperty(
        name="Facteur de rayon d'influence",
        description=(
            "Rayon d'influence R de la reconstruction Zhu-Bridson, exprime "
            "comme un multiple de l'espacement inter-particules de la "
            "scene : au-dela de cette distance, une particule n'a plus "
            "d'effet sur la surface reconstruite. Monter lisse davantage et "
            "sert les eclaboussures, ou la densite locale est plus faible ; "
            "descendre gagne du detail mais rapproche du plancher"
        ),
        default=3.0,
        # Borne basse relevee a 2.5 (ancienne valeur : 1.0). Premiere mesure,
        # sans jitter -- nappe au repos de 430 592 particules, composantes
        # connexes du maillage :
        #   facteur 0.8 -> 88 composantes, 1.0 -> 2, 1.5 et au-dela -> 1
        # Ce plancher tenait sur un artefact du cas de test : le reseau
        # d'emission est PARFAITEMENT regulier, ce qui aligne par coincidence
        # de nombreux voisins exactement a la distance R et masque l'absence
        # de vraie marge. Le jitter positionnel ajoute au mailleur pour
        # eliminer le motif de vaguelettes visible sur une nappe au repos
        # (cf. mesher.cu, k_zhu_bridson_field) casse cette regularite -- et
        # revele le probleme plutot qu'il ne le cree : re-mesure avec jitter,
        #   facteur 1.0 -> 55 composantes, 1.5 -> 19, 2.0 -> 6, 2.75+ -> 1
        # (cf. valide_lissage_jitter.py, scratchpad de la session qui a
        # introduit le jitter). Le plancher est donc remonte a 2.5 : proche
        # de la ou la connexite redevient franche, avec un peu de marge sous
        # le defaut (3.0) pour laisser un peu de detail accessible sans
        # rouvrir la fragmentation. Ne pas redescendre sans re-mesurer les
        # deux ensemble (facteur ET amplitude de jitter).
        min=2.5,
        soft_max=6.0,
    )

    mesh_particle_factor: FloatProperty(
        name="Facteur de rayon de particule",
        description=(
            "Rayon de particule r utilise par la reconstruction "
            "Zhu-Bridson, exprime comme un multiple de l'espacement "
            "inter-particules de la scene"
        ),
        default=1.0,
        min=0.1,
        soft_max=3.0,
    )

    mesh_collider_offset_factor: FloatProperty(
        name="Facteur de decalage collider",
        description=(
            "Decalage applique au rognage de la surface contre les "
            "colliders, exprime comme un multiple de l'espacement "
            "inter-particules de la scene : compense le fait que la "
            "surface reconstruite deborde toujours d'environ un demi-rayon "
            "de noyau au-dela des particules. 0 place deja la surface "
            "exactement sur le plan du collider"
        ),
        default=0.0,
        soft_min=-2.0,
        soft_max=2.0,
    )

    mesh_smoothing_iters: IntProperty(
        name="Iterations de lissage",
        description=(
            "Nombre d'iterations de lissage Taubin (lambda|mu) du champ, "
            "pour attenuer les artefacts de Zhu-Bridson dans les zones "
            "concaves sans faire retrecir la surface au fil des passes "
            "(contrairement a un lissage laplacien pur) — un lissage trop "
            "fort efface quand meme le detail voulu"
        ),
        default=0,
        min=0,
        soft_max=30,
    )

    mesh_min_component_tris: IntProperty(
        name="Triangles min. par composante",
        description=(
            "Supprime les fragments de maillage deconnectes du corps "
            "principal du fluide en dessous de ce nombre de triangles : "
            "la reconstruction locale peut produire jusqu'a plus d'un "
            "millier de ces ilots microscopiques sur une scene "
            "d'eclaboussure, du bruit visuel plutot que du fluide utile. "
            "La composante principale du fluide n'est jamais supprimee, "
            "quel que soit le reglage. 0 desactive le filtrage"
        ),
        default=50,
        min=0,
        soft_max=200,
    )

    is_baking_mesh: BoolProperty(
        name="Bake de maillage en cours",
        default=False,
    )

    bake_mesh_progress: FloatProperty(
        name="Progression du bake de maillage",
        default=0.0,
        min=0.0,
        max=1.0,
    )

    baked_mesh_frames: IntProperty(
        name="Frames de maillage bakees",
        default=0,
    )

    # -- Whitewater (jalon M8) ------------------------------------------
    #
    # Miroir des champs de `lib.BqWhitewaterConfig` (voir bourrasque.h,
    # section whitewater), a DEUX exceptions pres, actees par
    # docs/plan-milestone-8.md (D2) : `influence_radius` n'est PAS un
    # reglage ici, il reutilise `mesh_effective_radii(scene)` (le rayon deja
    # calcule pour le maillage) ; `gravity_y` n'est pas un reglage ici non
    # plus, il reutilise `scene.bourrasque.gravity` (la gravite de la
    # simulation principale). Voir `whitewater_config_from_scene` plus bas,
    # qui assemble ces deux exceptions avec les reglages ci-dessous.
    #
    # Valeurs par defaut : celles de `bq_whitewater_default_config`
    # (core/src/whitewater.cu). Ce sont des POINTS DE DEPART, PAS des
    # valeurs calibrees (cf. docs/plan-milestone-8.md, risque 1) : aucune
    # description ci-dessous ne pretend le contraire.

    ww_max_particles: IntProperty(
        name="Particules secondaires max",
        description=(
            "Capacite active maximale de particules secondaires (spray, "
            "foam, bulles), garde-fou VRAM : au-dela, les nouvelles "
            "candidates sont refusees plutot que d'ecraser des particules "
            "existantes ou de faire deborder la memoire"
        ),
        default=200000,
        min=1,
    )

    ww_ta_min: FloatProperty(
        name="Air piégé — seuil bas",
        description=(
            "Borne basse de normalisation du potentiel d'air piégé (les "
            "particules fluides dont les voisines s'ecartent fortement en "
            "vitesse, signe de bulles d'air entrainees) : en dessous, le "
            "potentiel est traite comme nul"
        ),
        default=2.0,
    )
    ww_ta_max: FloatProperty(
        name="Air piégé — seuil haut",
        description=(
            "Borne haute de normalisation du potentiel d'air piégé : "
            "au-dela, le potentiel est traite comme maximal (1)"
        ),
        default=8.0,
    )
    ww_ta_weight: FloatProperty(
        name="Air piégé — poids",
        description="Poids du potentiel d'air piégé dans le potentiel de génération combiné",
        default=1.0,
        min=0.0,
    )

    ww_wc_min: FloatProperty(
        name="Crête de vague — seuil bas",
        description=(
            "Borne basse de normalisation du potentiel de crête de vague "
            "(convexité locale de la surface combinée à une vitesse "
            "sortante) : en dessous, le potentiel est traite comme nul"
        ),
        default=1.0,
    )
    ww_wc_max: FloatProperty(
        name="Crête de vague — seuil haut",
        description=(
            "Borne haute de normalisation du potentiel de crête de vague : "
            "au-dela, le potentiel est traite comme maximal (1)"
        ),
        default=5.0,
    )
    ww_wc_weight: FloatProperty(
        name="Crête de vague — poids",
        description="Poids du potentiel de crête de vague dans le potentiel de génération combiné",
        default=1.0,
        min=0.0,
    )

    ww_ke_min: FloatProperty(
        name="Énergie cinétique — seuil bas",
        description=(
            "Borne basse de normalisation du potentiel d'énergie "
            "cinétique de la particule fluide : en dessous, le potentiel "
            "est traite comme nul"
        ),
        default=1.0,
    )
    ww_ke_max: FloatProperty(
        name="Énergie cinétique — seuil haut",
        description=(
            "Borne haute de normalisation du potentiel d'énergie "
            "cinétique : au-dela, le potentiel est traite comme maximal (1)"
        ),
        default=10.0,
    )
    ww_ke_weight: FloatProperty(
        name="Énergie cinétique — poids",
        description="Poids du potentiel d'énergie cinétique dans le potentiel de génération combiné",
        default=1.0,
        min=0.0,
    )

    ww_spawn_rate: FloatProperty(
        name="Taux de génération",
        description=(
            "Particules secondaires generees par seconde quand le "
            "potentiel de génération combiné vaut 1 (maximal)"
        ),
        default=50.0,
        min=0.0,
    )

    ww_life_spray: FloatProperty(
        name="Durée de vie — embruns",
        description="Durée de vie moyenne (secondes) d'une particule de régime embrun (spray)",
        default=1.0,
        min=0.0,
    )
    ww_life_foam: FloatProperty(
        name="Durée de vie — écume",
        description="Durée de vie moyenne (secondes) d'une particule de régime écume (foam)",
        default=2.0,
        min=0.0,
    )
    ww_life_bubble: FloatProperty(
        name="Durée de vie — bulles",
        description="Durée de vie moyenne (secondes) d'une particule de régime bulle (bubble)",
        default=1.5,
        min=0.0,
    )

    ww_drag_spray: FloatProperty(
        name="Traînée — embruns",
        description=(
            "Coefficient de traînée quadratique appliqué à l'advection "
            "balistique des embruns (spray), insensible à la vitesse du "
            "fluide"
        ),
        default=0.1,
        min=0.0,
    )
    ww_drag_foam: FloatProperty(
        name="Traînée — écume/bulles",
        description=(
            "Coefficient de relaxation vers la vitesse ambiante du fluide, "
            "appliqué à l'écume (foam) et aux bulles (bubble)"
        ),
        default=3.0,
        min=0.0,
    )

    ww_buoyancy_bubble: FloatProperty(
        name="Flottabilité — bulles",
        description=(
            "Accélération verticale supplémentaire appliquée aux bulles "
            "(bubble), au-delà de la vitesse ambiante du fluide"
        ),
        default=2.0,
    )

    ww_size_mult: FloatProperty(
        name="Multiplicateur de taille (affichage)",
        description=(
            "Multiplicateur PUREMENT COSMÉTIQUE appliqué à la taille des "
            "particules secondaires affichées (attribut bq_size). N'affecte "
            "ni la génération ni la physique (le rayon d'influence utilisé "
            "par le solveur reste inchangé) ; le cache .bqw n'est jamais "
            "modifié, seul l'affichage l'est"
        ),
        default=1.0,
        min=0.1,
        max=3.0,
    )

    is_baking_whitewater: BoolProperty(
        name="Bake de whitewater en cours",
        default=False,
    )

    bake_whitewater_progress: FloatProperty(
        name="Progression du bake de whitewater",
        default=0.0,
        min=0.0,
        max=1.0,
    )

    baked_whitewater_frames: IntProperty(
        name="Frames de whitewater bakees",
        default=0,
    )

    baked_whitewater_max_refused: IntProperty(
        name="Candidates refusées (max observé)",
        description=(
            "Maximum, sur toute la durée du dernier bake whitewater, du "
            "nombre de particules secondaires qui auraient dû être "
            "générées mais ont été refusées faute de capacité (voir "
            "« Particules secondaires max ») — diagnostic, pas une erreur"
        ),
        default=0,
        min=0,
    )


# ---------------------------------------------------------------------------
# Fonctions utilitaires (module-level : source de verite unique pour l'UI et
# les operateurs)
# ---------------------------------------------------------------------------


def iter_elements(scene):
    """Itere sur tous les objets de `scene` dont `role != NONE`.

    Ordre : le domaine d'abord (s'il existe), puis les emetteurs, puis les
    colliders, chaque groupe trie dans l'ordre du nom.
    """
    domain = []
    emitters = []
    colliders = []
    for obj in scene.objects:
        role = obj.bourrasque.role
        if role == "DOMAIN":
            domain.append(obj)
        elif role == "EMITTER":
            emitters.append(obj)
        elif role == "COLLIDER":
            colliders.append(obj)
    domain.sort(key=lambda o: o.name)
    emitters.sort(key=lambda o: o.name)
    colliders.sort(key=lambda o: o.name)
    return domain + emitters + colliders


def material_usage(scene):
    """Recense l'usage de la bibliotheque de materiaux de `scene` par ses
    emetteurs, en un seul passage sur `scene.objects` (O(nb objets), aucun
    `to_mesh()` : appelable sans cout depuis un `draw()` de panneau, redessine
    plusieurs fois par seconde).

    Renvoie un tuple `(usage, dangling)` plutot qu'un dict a cle magique,
    pour que les deux cas (materiau connu vs. reference orpheline) restent
    distinguables sans comparaison de chaine cote appelant :

    - `usage` : `{nom_materiau: [objets emetteurs]}`, UNE entree par materiau
      de `scene.bourrasque.materials` (y compris les materiaux sans aucun
      emetteur, avec une liste vide — utile pour que `ui.py` puisse afficher
      « inutilise » sans requete separee).
    - `dangling` : liste des objets emetteurs dont `material_name` est soit
      vide (aucun materiau assigne), soit ne correspond a AUCUN materiau de
      la bibliotheque (reference pendante, par exemple apres suppression du
      materiau referencé) — ces deux cas sont traites ensemble, `ui.py`
      distingue au besoin en relisant `obj.bourrasque.material_name`.
    """
    usage = {mat.name: [] for mat in scene.bourrasque.materials}
    dangling = []
    for obj in scene.objects:
        if obj.bourrasque.role != "EMITTER":
            continue
        name = obj.bourrasque.material_name
        bucket = usage.get(name) if name else None
        if bucket is None:
            dangling.append(obj)
        else:
            bucket.append(obj)
    return usage, dangling


def used_material_slot_count(scene):
    """Nombre d'emplacements materiau que le bake consommera reellement pour
    `scene`, a comparer a `lib.BQ_MAX_MATERIALS`.

    Compte les materiaux utilises **dedupliques par `materials.material_key`**,
    EXACTEMENT comme `BQ_OT_bake._validate` — et non le nombre de NOMS
    distincts utilises. La difference est observable : deux materiaux nommes
    differemment mais physiquement identiques occupent deux lignes dans la
    bibliotheque et un seul emplacement solveur. Compter les noms faisait
    afficher a l'UI un depassement de capacite alors que le bake passait
    (ecart 2 de la revue d'architecture M16).

    Source de verite unique pour l'avertissement de capacite : `ui.py` doit
    appeler cette fonction plutot que recompter de son cote.
    """
    from . import materials as _materials

    usage, _dangling = material_usage(scene)
    keys = set()
    for mat in scene.bourrasque.materials:
        if not usage.get(mat.name):
            continue  # materiau inutilise : n'occupe aucun emplacement
        keys.add(
            _materials.material_key(
                {
                    "model": mat.model,
                    "rho": mat.rho,
                    "young": mat.young,
                    "poisson": mat.poisson,
                    "bulk": mat.bulk,
                    "gamma": mat.gamma,
                    "friction_angle": mat.friction_angle,
                }
            )
        )
    return len(keys)


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


# ---------------------------------------------------------------------------
# Maillage de surface (jalon M7)
# ---------------------------------------------------------------------------
#
# Empreinte VRAM RETENUE POUR LE GARDE-FOU UI : 36 octets par cellule du
# champ de maillage (docs/plan-milestone-7.md, decision D2, correction du
# 2026-08-02 apres implementation) — le champ Zhu-Bridson lui-meme (4
# octets/cellule, en gather) PLUS les tampons du marching cubes (32
# octets/cellule de plus, drapeaux et sommes prefixes sur les aretes et les
# cubes), qui dominent l'empreinte totale. C'est un CALCUL PUR (pas d'appel
# a `bq_mesher_vram_estimate`, qui ne compte que le champ seul, cf. sa
# docstring dans bourrasque.h) : l'UI doit pouvoir avertir l'artiste avant
# meme que la DLL native soit chargeable, avant tout bake.
_MESH_BYTES_PER_CELL = 36
_MESH_VRAM_WARNING_BYTES = 2 * 1024 * 1024 * 1024  # 2 Go (cf. plan M7, R1)

# Plafond de la resolution AUTOMATIQUE du maillage (`mesh_resolution == 0`,
# ergonomie M7.1) : distinct de `_MESH_VRAM_WARNING_BYTES` ci-dessus, qui
# reste un simple AVERTISSEMENT visuel applicable a un reglage EXPLICITE
# (jamais plafonne en silence, voir docstring de `mesh_resolution_state`).
# Ce plafond-ci, lui, borne activement la valeur choisie automatiquement,
# pour qu'un domaine large ne fasse pas exploser l'empreinte VRAM sans
# qu'aucun reglage n'ait ete touche.
_MESH_AUTO_VRAM_CAP_BYTES = int(1.5 * 1024 * 1024 * 1024)  # 1,5 Go


def _mesh_layout_for_setting(size, mesh_res_setting):
    """Coeur pur de `mesh_layout` : `res`/`cell_size` pour un reglage de
    resolution DEJA RESOLU (entier positif, plus de sens « 0 = auto » ici
    — voir `mesh_resolution_state`). Factorise pour etre reutilisee par le
    plafonnement VRAM de la resolution automatique (`_cap_auto_mesh_resolution`),
    qui doit evaluer plusieurs reglages candidats sans passer par `_domain_layout`
    a chaque fois.
    """
    max_extent = max(size)
    if max_extent <= 0 or mesh_res_setting <= 0:
        return (tuple(1 for _ in range(3)), 0.0)

    cell_size = max_extent / mesh_res_setting
    res = tuple(max(1, math.ceil(size[a] / cell_size)) for a in range(3))
    return (res, cell_size)


def _mesh_bytes_for_setting(size, mesh_res_setting):
    """Empreinte VRAM (octets) du champ de maillage pour un reglage de
    resolution donne, meme formule que `mesh_vram_estimate_bytes`."""
    res, _cell_size = _mesh_layout_for_setting(size, mesh_res_setting)
    return res[0] * res[1] * res[2] * _MESH_BYTES_PER_CELL


def _cap_auto_mesh_resolution(size, auto_setting):
    """Reduit `auto_setting` (reglage de resolution automatique, avant
    plafonnement) jusqu'a ce que son empreinte VRAM tienne sous
    `_MESH_AUTO_VRAM_CAP_BYTES`, sans jamais renvoyer moins de 1.

    L'empreinte croit approximativement au CUBE du reglage (le nombre de
    cellules du champ est un produit sur les 3 axes) : une estimation
    directe par racine cubique donne un point de depart proche de la
    solution, affine ensuite par decrement pas a pas (le `ceil` par axe de
    `_mesh_layout_for_setting` interdit une formule fermee exacte).
    """
    if auto_setting <= 1:
        return max(1, auto_setting)

    n_bytes = _mesh_bytes_for_setting(size, auto_setting)
    if n_bytes <= _MESH_AUTO_VRAM_CAP_BYTES:
        return auto_setting

    factor = (_MESH_AUTO_VRAM_CAP_BYTES / n_bytes) ** (1.0 / 3.0)
    setting = max(1, int(math.floor(auto_setting * factor)))
    n_bytes = _mesh_bytes_for_setting(size, setting)
    while n_bytes > _MESH_AUTO_VRAM_CAP_BYTES and setting > 1:
        setting -= 1
        n_bytes = _mesh_bytes_for_setting(size, setting)
    return setting


def mesh_resolution_state(scene):
    """Renvoie `(setting, is_auto, was_capped)` : `setting` est le reglage
    de resolution EFFECTIVEMENT utilise par `mesh_layout` (deja resolu si
    `scene.bourrasque.mesh_resolution == 0`, voir ci-dessous), `is_auto`
    indique si ce reglage vient du mode automatique, `was_capped` si le
    plafond VRAM (`_MESH_AUTO_VRAM_CAP_BYTES`) a du le reduire par rapport
    a la valeur automatique brute.

    `mesh_resolution == 0` signifie AUTOMATIQUE (ergonomie M7.1, defaut) :
    la resolution naturelle est celle qui donne UNE CELLULE DE MAILLAGE PAR
    ESPACEMENT INTER-PARTICULES, c'est-a-dire `grid_res * ppc_axis` (meme
    convention « applique au plus grand axe » que `grid_res` lui-meme, voir
    `_domain_layout`) — plafonnee pour que l'empreinte VRAM estimee
    (`_MESH_BYTES_PER_CELL` par cellule) reste sous 1,5 Go.

    Un reglage EXPLICITE (non nul) n'est JAMAIS plafonne ici : c'est un
    choix assume de l'artiste, seul l'avertissement visuel de `ui.py`
    (`mesh_vram_estimate_bytes` / `mesh_vram_warning_threshold_bytes`,
    seuil 2 Go) continue de s'appliquer.

    Renvoie `None` si aucun domaine n'est defini.
    """
    layout = _domain_layout(scene)
    if layout is None:
        return None
    _origin, size, _res, _dx = layout

    props = scene.bourrasque
    setting = props.mesh_resolution
    if setting > 0:
        return (setting, False, False)

    auto_setting = max(1, props.grid_res * props.ppc_axis)
    capped_setting = _cap_auto_mesh_resolution(size, auto_setting)
    return (capped_setting, True, capped_setting < auto_setting)


def mesh_layout(scene):
    """Renvoie `(res, cell_size)` pour le champ de maillage de `scene` —
    `res` le triplet `(res_x, res_y, res_z)` de cellules par axe, `cell_size`
    la taille de cellule du champ (uniforme sur les trois axes).

    INDEPENDANT de `domain_resolution` (grille de simulation) — decision D3
    du plan M7 : `scene.bourrasque.mesh_resolution` est un reglage a part,
    applique au plus grand axe du pave SOLVEUR (meme convention que
    `grid_res`, voir `_domain_layout`) pour dimensionner le champ du
    mailleur, RESOLU en une valeur automatique plafonnee si son reglage
    vaut 0 (voir `mesh_resolution_state`, ergonomie M7.1). Le champ couvre
    exactement le meme pave solveur que la simulation (`origin`/`size` de
    `domain_transform`), puisque c'est dans cet espace que vivent les
    positions du `.bqd` relu par le mailleur ; aucune marge de stencil
    MLS-MPM n'est necessaire ici (le mailleur n'a pas de contrainte de
    stencil de transfert grille<->particule).

    Renvoie `None` si aucun domaine n'est defini. Tous les appelants
    existants (avant l'ajout du mode automatique) profitent de la
    resolution sans changement : la signature de cette fonction n'a pas
    change.
    """
    layout = _domain_layout(scene)
    if layout is None:
        return None
    _origin, size, _res, _dx = layout

    state = mesh_resolution_state(scene)
    mesh_res_setting = state[0]
    return _mesh_layout_for_setting(size, mesh_res_setting)


def mesh_particle_spacing(scene):
    """Espacement inter-particules effectif de `scene` : `dx / ppc_axis`,
    la MEME formule que celle deja utilisee pour l'emission
    (`estimate_particle_count`, `estimate_inflow_count`) et pour la
    resolution automatique du maillage ci-dessus — source de verite unique,
    ne pas la reimplementer ailleurs.

    Renvoie `None` si aucun domaine n'est defini.
    """
    resolution = domain_resolution(scene)
    if resolution is None:
        return None
    _res, dx = resolution
    ppc_axis = scene.bourrasque.ppc_axis
    if ppc_axis <= 0:
        return None
    return dx / ppc_axis


def mesh_effective_radii(scene):
    """Renvoie `(influence_radius, particle_radius, collider_offset)`, les
    valeurs ABSOLUES (unites solveur) effectivement transmises au mailleur
    — chaque facteur reglable par l'artiste (`mesh_influence_factor`,
    `mesh_particle_factor`, `mesh_collider_offset_factor`, voir Travail 1
    de l'ergonomie M7.1) multiplie par `mesh_particle_spacing(scene)`.

    C'est cette fonction, et NON les facteurs bruts de `scene.bourrasque`,
    que `ops.py` doit lire pour remplir `bq_mesher_config` et
    `meshcache.MeshProductionParams` : le format `.bqm` continue de stocker
    des valeurs absolues (voir `meshcache.py`), pour que
    `meshcache.diff_params` invalide correctement le cache quand un
    FACTEUR change (la valeur absolue qui en decoule change aussi).

    Renvoie `None` si aucun domaine n'est defini.
    """
    spacing = mesh_particle_spacing(scene)
    if spacing is None:
        return None
    props = scene.bourrasque
    influence_radius = props.mesh_influence_factor * spacing
    particle_radius = props.mesh_particle_factor * spacing
    collider_offset = props.mesh_collider_offset_factor * spacing
    return (influence_radius, particle_radius, collider_offset)


def mesh_vram_estimate_bytes(scene):
    """Empreinte VRAM estimee (octets) du champ de maillage pour les
    reglages courants de `scene`, garde-fou UI (voir `_MESH_BYTES_PER_CELL`)
    — a afficher et a comparer a `_MESH_VRAM_WARNING_BYTES` AVANT tout bake,
    pas decouverte par un echec d'allocation CUDA. Renvoie 0 si aucun
    domaine n'est defini.
    """
    layout = mesh_layout(scene)
    if layout is None:
        return 0
    res, _cell_size = layout
    n_cells = res[0] * res[1] * res[2]
    return n_cells * _MESH_BYTES_PER_CELL


def mesh_vram_warning_threshold_bytes():
    """Seuil (octets) au-dela duquel `ui.py` avertit visiblement l'artiste
    (voir `_MESH_VRAM_WARNING_BYTES`)."""
    return _MESH_VRAM_WARNING_BYTES


def mesh_cache_path(cache_dir, name):
    """Construit le chemin `.bqm` pour une simulation `name`, meme
    convention que `cache.cache_paths` (meme dossier, meme nom de base,
    extension `.bqm`). Fonction PURE, aucun effet de bord sur le disque —
    meme discipline que `cache.cache_paths`, appelable sur le chemin chaud
    du scrub de timeline (`display.py`)."""
    return pathlib.Path(cache_dir) / f"{name}.bqm"


def whitewater_cache_path(cache_dir, name):
    """Construit le chemin `.bqw` pour une simulation `name`, meme
    convention que `cache_paths`/`mesh_cache_path` (meme dossier, meme nom
    de base, extension `.bqw`). Fonction PURE, aucun effet de bord sur le
    disque."""
    return pathlib.Path(cache_dir) / f"{name}.bqw"


def whitewater_config_from_scene(scene):
    """Construit un `lib.BqWhitewaterConfig` depuis les reglages whitewater
    de `scene.bourrasque` (voir la section « Whitewater » de `BqSceneProps`
    ci-dessus), meme motif que la construction inline de `BqMesherConfig`
    pour le maillage (voir `ops.BQ_OT_bake_mesh.invoke`).

    `influence_radius` et `gravity_y` NE SONT PAS lus sur des reglages
    whitewater dedies (il n'en existe pas, voir docs/plan-milestone-8.md,
    D2/D6 et la docstring de `BqSceneProps`) : `influence_radius` reutilise
    `mesh_effective_radii(scene)` (le rayon deja calcule pour le maillage),
    `gravity_y` reutilise `scene.bourrasque.gravity` (la gravite de la
    simulation principale).

    Renvoie `None` si aucun domaine n'est defini (`mesh_effective_radii`
    renvoie alors `None`).
    """
    radii = mesh_effective_radii(scene)
    if radii is None:
        return None
    influence_radius, _particle_radius, _collider_offset = radii

    props = scene.bourrasque
    cfg = lib.default_whitewater_config()
    cfg.max_particles = props.ww_max_particles
    cfg.influence_radius = influence_radius
    cfg.gravity_y = props.gravity
    cfg.ta_min = props.ww_ta_min
    cfg.ta_max = props.ww_ta_max
    cfg.ta_weight = props.ww_ta_weight
    cfg.wc_min = props.ww_wc_min
    cfg.wc_max = props.ww_wc_max
    cfg.wc_weight = props.ww_wc_weight
    cfg.ke_min = props.ww_ke_min
    cfg.ke_max = props.ww_ke_max
    cfg.ke_weight = props.ww_ke_weight
    cfg.spawn_rate = props.ww_spawn_rate
    cfg.life_spray = props.ww_life_spray
    cfg.life_foam = props.ww_life_foam
    cfg.life_bubble = props.ww_life_bubble
    cfg.drag_spray = props.ww_drag_spray
    cfg.drag_foam = props.ww_drag_foam
    cfg.buoyancy_bubble = props.ww_buoyancy_bubble
    return cfg


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


# ---------------------------------------------------------------------------
# Cache de collider_triangle_count
# ---------------------------------------------------------------------------
#
# collider_triangle_count() est appelee par `ui.py` A CHAQUE REDESSIN du
# panneau — plusieurs fois par seconde des que la souris bouge dans la barre
# laterale, meme quand rien n'a change dans la scene. Sans cache, chaque
# redessin refait un `to_mesh()` / `calc_loop_triangles()` / produit
# matriciel PAR COLLIDER (voir `sampling.evaluated_world_mesh`),
# potentiellement plusieurs fois par seconde sur un maillage dense.
#
# Signal d'invalidation retenu : un compteur global incremente par un
# handler `depsgraph_update_post`, qui ne se declenche QUE quand la scene
# change reellement (edition de maillage, modificateur, transformation,
# changement de frame, ajout/suppression d'objet...), jamais sur un simple
# redessin de panneau sans interaction sur la scene — c'est precisement la
# distinction qui manquait. On sur-invalide deliberement (un
# `depsgraph_update_post` sans rapport avec un collider invalide quand meme
# le cache) plutot que d'inspecter finement quel objet a change : le but
# n'est pas un cache parfait, seulement d'arreter de re-extraire le maillage
# a chaque frame de redessin de l'UI, ce qui est deja obtenu avec ce signal
# grossier.
_triangle_count_generation = 0
_triangle_count_cache = {"generation": None, "scene_name": None, "count": 0}


@bpy.app.handlers.persistent
def _bq_bump_triangle_count_generation(scene, depsgraph):
    global _triangle_count_generation
    _triangle_count_generation += 1


def _remove_triangle_count_handler():
    # Filtre par nom de fonction plutot que par identite d'objet : un
    # rechargement du module change l'identite de la fonction sans changer
    # son nom (voir meme discipline dans `display.py`).
    for fn in list(bpy.app.handlers.depsgraph_update_post):
        if fn.__name__ == _bq_bump_triangle_count_generation.__name__:
            bpy.app.handlers.depsgraph_update_post.remove(fn)


def collider_triangle_count(scene):
    """Nombre total de triangles, tous colliders confondus (maillage EVALUE,
    modificateurs appliques) — purement indicatif pour l'UI (`ui.py`), qui
    n'affiche un avertissement qu'au-dela d'un seuil eleve (le champ de
    distance du coeur, calcule par grille de buckets + propagation de
    signe, ne croit PAS lineairement avec ce nombre — voir
    `ui._COLLIDER_TRIANGLE_WARNING_THRESHOLD`).

    Mis en cache entre deux redessins du panneau (voir le commentaire de
    section ci-dessus) : ne re-extrait les maillages que si le depsgraph a
    reellement change depuis le dernier appel, ou si la scene interrogee a
    change de nom. Un compte legerement perime (une frame de retard sur une
    edition qui n'a pas encore declenche `depsgraph_update_post`, cas rare
    en pratique) est prefere a une re-extraction systematique.

    Meme discipline defensive que `_mesh_block_count` : `extension.sampling`
    peut etre absent ou en panne, ce panneau ne doit jamais planter pour
    autant. Une exception sur un collider individuel (maillage degenere,
    par exemple) ne fait pas echouer le compte des autres.
    """
    cache = _triangle_count_cache
    if (
        cache["generation"] == _triangle_count_generation
        and cache["scene_name"] == scene.name
    ):
        return cache["count"]

    try:
        from .sampling import evaluated_world_mesh
    except ImportError:
        return 0

    total = 0
    depsgraph = None
    for obj in scene.objects:
        if obj.bourrasque.role != "COLLIDER":
            continue
        try:
            if depsgraph is None:
                depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(depsgraph)
            _verts, tris = evaluated_world_mesh(obj_eval)
            total += tris.shape[0]
        except Exception:
            continue

    cache["generation"] = _triangle_count_generation
    cache["scene_name"] = scene.name
    cache["count"] = total
    return total


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
    BqMaterialProps,
    BqObjectProps,
    BqSceneProps,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    Object.bourrasque = PointerProperty(type=BqObjectProps)
    Scene.bourrasque = PointerProperty(type=BqSceneProps)

    _remove_triangle_count_handler()
    bpy.app.handlers.depsgraph_update_post.append(_bq_bump_triangle_count_generation)


def unregister():
    _remove_triangle_count_handler()
    del Scene.bourrasque
    del Object.bourrasque
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
