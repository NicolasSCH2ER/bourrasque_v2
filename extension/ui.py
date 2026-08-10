"""ui.py — panneaux et UIList de l'extension Bourrasque.

C'est la surface que voit l'artiste, dans la barre laterale du viewport 3D
(categorie « Bourrasque »). Ce module ne fait QUE de la presentation : il lit
`scene.bourrasque` / `obj.bourrasque` (voir props.py) et invoque les
operateurs de ops.py (bq.add_domain, bq.add_emitter, bq.add_collider,
bq.remove_element, bq.material_add, bq.material_remove,
bq.material_duplicate, bq.migrate_materials, bq.bake, bq.cancel_bake,
bq.free_cache, bq.bake_mesh, bq.bake_all, bq.free_mesh_cache,
bq.bake_whitewater, bq.free_whitewater_cache, bq.setup_fluid_display,
bq.setup_whitewater_display, bq.setup_whitewater_display_volume,
bq.clear_rigid_keys). Aucun appel a la DLL, aucune logique de simulation
ici : voir lib.py et ops.py.

Regle absolue (voir chaque `draw()` ci-dessous) : un `draw()` ne modifie
JAMAIS de donnees Blender. Il detecte un etat et affiche des boutons ; ce
sont les OPERATEURS qui ecrivent. Ecrire depuis un `draw()` casse l'undo et
peut faire planter Blender (redessin en cascade).

Arborescence des panneaux (categorie « Bourrasque », tous de premier niveau
sauf mention contraire — voir `classes` en bas de fichier pour l'ordre
d'enregistrement, qui determine l'ordre d'affichage) :

    BQ_PT_dashboard   « Tableau de bord »   vue d'ensemble + actions globales
    BQ_PT_domain      « Domaine »
    BQ_PT_elements    « Éléments »          liste + ajout/retrait
      BQ_PT_emitter   « Émetteur »          sous-panneau, poll role==EMITTER
      BQ_PT_collider  « Collider »          sous-panneau, poll role==COLLIDER
    BQ_PT_materials   « Matériaux »         bibliotheque de materiaux
    BQ_PT_simulation  « Simulation »
    BQ_PT_output      « Sortie »            conteneur
      BQ_PT_bake      « Particules »
      BQ_PT_mesh      « Maillage »
      BQ_PT_whitewater « Whitewater »
      BQ_PT_display   « Affichage »
"""

import math
import os

import bpy
import mathutils
from bpy.types import Panel, UIList

from . import cache, lib, meshcache, ops, whitewatercache
from .props import (
    collider_triangle_count,
    domain_resolution,
    domain_transform,
    emitter_overflow,
    estimate_particle_count,
    iter_elements,
    material_usage,
    used_material_slot_count,
    mesh_cache_path,
    mesh_effective_radii,
    mesh_layout,
    mesh_particle_spacing,
    mesh_resolution_state,
    mesh_vram_estimate_bytes,
    mesh_vram_warning_threshold_bytes,
    whitewater_cache_path,
)

# Le champ de distance du coeur est calcule par grille de buckets + propa-
# gation de signe (pas par test brut triangle x cellule) : mesure sur le
# reference du coeur, 6,6 ms pour 5000 triangles, 41 ms pour 50 000 (a
# comparer a ~50,8 ms pour un `step` complet du solveur a cette meme
# resolution) — le cout croit BEAUCOUP moins vite que lineairement avec le
# nombre de triangles, contrairement a une premiere estimation. Le seuil
# ci-dessous est calibre pour ne s'allumer que quand le champ de distance
# devient comparable au cout d'un step (donc perceptible sur le temps total
# de bake), pas des la premiere dizaine de milliers de triangles.
_COLLIDER_TRIANGLE_WARNING_THRESHOLD = 100000

# Au-dela de ce nombre de corps rigides dynamiques, le tableau de bord bascule
# d'une liste nom-par-nom a une synthese (compte flotte/coule) : le tableau
# de bord existe pour donner l'etat de la simulation D'UN COUP D'OEIL (voir
# docstring du module), une liste qui deborde de l'ecran va a l'encontre de
# ce but. Pas de mesure de cout de rendu ici (contrairement au seuil
# triangles ci-dessus) : c'est un choix de densite visuelle, pas de
# performance -- comparer `density` a 1000 est gratuit quel que soit le
# nombre de corps.
_DASHBOARD_RIGID_BODY_LIST_THRESHOLD = 6

__all__ = ("classes", "register", "unregister")


# ---------------------------------------------------------------------------
# Aides internes (presentation uniquement)
# ---------------------------------------------------------------------------


def _domain_extents(obj):
    """Etendues (dx, dy, dz) de la bounding box monde de `obj`.

    Duplique volontairement le calcul de `props.domain_transform` (qui
    renvoie desormais un triplet en espace SOLVEUR, pas les etendues MONDE
    directement) : ce panneau a besoin des trois etendues monde separement
    pour l'affichage de la taille du domaine tel que place par l'artiste.
    """
    mat = obj.matrix_world
    corners = [mat @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))


def _thousands(n):
    """Formate un entier avec un espace comme separateur de milliers."""
    return f"{n:,}".replace(",", " ")


def _format_bytes(n):
    """Formate un nombre d'octets en unite lisible (Ko/Mo/Go), pour
    l'empreinte memoire estimee du maillage et la taille du cache `.bqm`."""
    value = float(n)
    for unit in ("octets", "Ko", "Mo", "Go"):
        if value < 1024.0 or unit == "Go":
            if unit == "octets":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} Go"


# Libelles francais des champs de `meshcache.MeshProductionParams`, pour
# nommer EXPLICITEMENT le parametre qui rend le cache `.bqm` caduc (voir
# `meshcache.diff_params`, docs/plan-milestone-7.md, decision D1) plutot que
# de se contenter d'un « cache invalide » generique.
_MESH_PARAM_LABELS = {
    "mesh_res": "résolution du maillage",
    "cell_size": "taille de cellule",
    "influence_radius": "rayon d'influence",
    "particle_radius": "rayon de particule",
    "collider_offset": "décalage collider",
    "smoothing_iters": "itérations de lissage",
    "min_component_tris": "seuil de triangles minimum par composante",
    "src_frames": "nombre de frames du cache de particules (.bqd)",
    "src_n_max": "nombre de particules du cache de particules (.bqd)",
}


def _read_bqd_source_counts(cache_dir, scene_name):
    """Lit `(frames, n_max)` du `.bqd` source, LECTURE LEGERE (header + table
    d'index seulement) — helper partage par `_current_mesh_params` et
    `_current_whitewater_params`. Renvoie `(-1, -1)` si le fichier est absent
    ou illisible : des sentinelles qui forcent `diff_params` a signaler un
    ecart plutot que de supposer une source inchangee.
    """
    bqd_path, _mat_path = cache.cache_paths(cache_dir, scene_name)
    if not os.path.isfile(bqd_path):
        return (-1, -1)
    try:
        reader = cache.CacheReader(bqd_path)
        try:
            return (reader.frame_count, reader.n_particles)
        finally:
            reader.close()
    except (OSError, ValueError):
        return (-1, -1)


def _current_mesh_params(scene, res, cell_size):
    """Construit les `MeshProductionParams` COURANTS (reglages de `scene` +
    resolution/cellule effectives) pour les comparer, via
    `meshcache.diff_params`, aux parametres stockes dans le `.bqm` existant
    — voir `BQ_PT_mesh.draw`.
    """
    props = scene.bourrasque
    cache_dir = bpy.path.abspath(props.cache_dir)
    src_frames, src_n_max = _read_bqd_source_counts(cache_dir, scene.name)

    # Comparaison au format ABSOLU stocke dans le `.bqm` (voir
    # `meshcache.MeshProductionParams`) : les reglages artiste sont des
    # FACTEURS (ergonomie M7.1), `mesh_effective_radii` les convertit en
    # valeurs absolues courantes, source de verite unique — voir props.py.
    influence_radius, particle_radius, collider_offset = mesh_effective_radii(scene)

    return meshcache.MeshProductionParams(
        mesh_res=tuple(res),
        cell_size=cell_size,
        influence_radius=influence_radius,
        particle_radius=particle_radius,
        collider_offset=collider_offset,
        smoothing_iters=props.mesh_smoothing_iters,
        min_component_tris=props.mesh_min_component_tris,
        src_frames=src_frames,
        src_n_max=src_n_max,
    )


# Libelles francais des champs de `whitewatercache.WhitewaterProductionParams`,
# meme role que `_MESH_PARAM_LABELS` (voir docs/plan-milestone-8.md, D7 —
# meme raisonnement que D1 de M7 : nommer le parametre qui rend le cache
# `.bqw` caduc plutot que d'afficher un « cache invalide » generique).
_WHITEWATER_PARAM_LABELS = {
    "influence_radius": "rayon d'influence",
    "spawn_rate": "taux de génération",
    "ta_min": "air piégé — seuil bas",
    "ta_max": "air piégé — seuil haut",
    "ta_weight": "air piégé — poids",
    "wc_min": "crête de vague — seuil bas",
    "wc_max": "crête de vague — seuil haut",
    "wc_weight": "crête de vague — poids",
    "ke_min": "énergie cinétique — seuil bas",
    "ke_max": "énergie cinétique — seuil haut",
    "ke_weight": "énergie cinétique — poids",
    "life_spray": "durée de vie — embruns",
    "life_foam": "durée de vie — écume",
    "life_bubble": "durée de vie — bulles",
    "drag_spray": "traînée — embruns",
    "drag_foam": "traînée — écume/bulles",
    "buoyancy_bubble": "flottabilité — bulles",
    "src_frames": "nombre de frames du cache de particules (.bqd)",
    "src_n_max": "nombre de particules du cache de particules (.bqd)",
}


def _current_whitewater_params(scene):
    """Construit les `WhitewaterProductionParams` COURANTS (reglages de
    `scene` + rayon d'influence effectif du maillage, cf. D2 du plan M8)
    pour les comparer, via `whitewatercache.diff_params`, aux parametres
    stockes dans le `.bqw` existant — meme motif que `_current_mesh_params`
    pour le `.bqm`.

    Renvoie `None` si aucun domaine n'est defini (le rayon d'influence
    effectif, `mesh_effective_radii(scene)`, n'a alors pas de sens).
    """
    radii = mesh_effective_radii(scene)
    if radii is None:
        return None
    influence_radius, _particle_radius, _collider_offset = radii

    props = scene.bourrasque
    cache_dir = bpy.path.abspath(props.cache_dir)
    src_frames, src_n_max = _read_bqd_source_counts(cache_dir, scene.name)

    return whitewatercache.WhitewaterProductionParams(
        influence_radius=influence_radius,
        spawn_rate=props.ww_spawn_rate,
        ta_min=props.ww_ta_min,
        ta_max=props.ww_ta_max,
        ta_weight=props.ww_ta_weight,
        wc_min=props.ww_wc_min,
        wc_max=props.ww_wc_max,
        wc_weight=props.ww_wc_weight,
        ke_min=props.ww_ke_min,
        ke_max=props.ww_ke_max,
        ke_weight=props.ww_ke_weight,
        life_spray=props.ww_life_spray,
        life_foam=props.ww_life_foam,
        life_bubble=props.ww_life_bubble,
        drag_spray=props.ww_drag_spray,
        drag_foam=props.ww_drag_foam,
        buoyancy_bubble=props.ww_buoyancy_bubble,
        src_frames=src_frames,
        src_n_max=src_n_max,
    )


def _material_model_label(mat_props):
    """Libellé français du modèle physique d'un `BqMaterialProps` (« Eau »,
    « Élastique »)."""
    enum_items = mat_props.bl_rna.properties["model"].enum_items
    return enum_items[mat_props.model].name


def _is_vector_zero(vec, eps=1e-6):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]) < eps


def _has_uniform_scale(obj, rel_tol=1e-4):
    """Echelle X == Y == Z de `obj`, a `rel_tol` pres.

    Lecture gratuite de `obj.scale` (pas d'evaluation de maillage) : sert a
    detecter, SANS COUT, le cas ou un collider DYNAMIQUE aurait une echelle
    non uniforme -- la composition/decomposition en keyframes de fin de
    bake (D8/D6 du plan M17) suppose une rotation a echelle uniforme pres,
    une echelle non uniforme y introduirait un cisaillement non represente
    par une simple paire loc/rot (voir `rigidbody.decompose_loc_rot`)."""
    sx, sy, sz = obj.scale
    return math.isclose(sx, sy, rel_tol=rel_tol) and math.isclose(
        sy, sz, rel_tol=rel_tol
    )


def _is_object_animated(obj):
    """Detecte une animation de transformation ou de forme sur `obj`.

    Utilise en mode `emit_source == "MESH"` : l'echantillonnage d'interieur
    n'est fait qu'une fois, au debut du bake (voir `extension/sampling.py`,
    ecrit en parallele) — un emetteur anime donnerait un resultat faux et
    silencieux sans cet avertissement.
    """
    anim = obj.animation_data
    if anim is not None and anim.action is not None:
        return True
    data = obj.data
    shape_keys = getattr(data, "shape_keys", None)
    if shape_keys is not None:
        sk_anim = shape_keys.animation_data
        if sk_anim is not None and sk_anim.action is not None:
            return True
    return False


# ---------------------------------------------------------------------------
# UIList des elements de la simulation
#
# Choix : `template_list` sur `scene.objects`, filtre via `filter_items` pour
# ne montrer que les objets tagges (role != NONE). C'est la voie la plus
# idiomate : Blender gere nativement le rafraichissement de la liste quand
# des objets sont crees/renommes/supprimes dans la scene, sans qu'on ait a
# synchroniser une CollectionProperty derivee a la main. Le tri/filtrage
# integre (`sort_items_by_name`) prend en charge l'ordre alphabetique.
#
# Limite connue : `template_list` sur une collection d'ID (ici Object) fait
# defiler/mettre en surbrillance l'element actif via
# `active_element_index`, mais ne rend PAS cet objet actif dans la scene
# (`view_layer.objects.active`) — ce n'est pas son role. Pour realiser
# « cliquer une ligne rend l'objet actif », chaque ligne est dessinee comme
# un bouton d'operateur generique deja fourni par Blender
# (`wm.context_set_id`, qui resout un nom d'objet vers
# `view_layer.objects.active`) plutot que d'inventer un operateur
# supplementaire hors des six autorises.
# ---------------------------------------------------------------------------


class BQ_UL_elements(UIList):
    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        flags = []
        for obj in items:
            if obj.bourrasque.role != "NONE":
                flags.append(self.bitflag_filter_item)
            else:
                flags.append(0)
        order = bpy.types.UI_UL_list.sort_items_by_name(items)
        return flags, order

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        obj = item
        obj_props = obj.bourrasque
        if obj_props.role == "DOMAIN":
            role_icon = "MESH_CUBE"
        elif obj_props.role == "COLLIDER":
            role_icon = "MOD_PHYSICS"
        else:
            role_icon = "PARTICLES"

        row = layout.row(align=True)
        op = row.operator(
            "wm.context_set_id", text=obj.name, icon=role_icon, emboss=False
        )
        op.data_path = "view_layer.objects.active"
        op.value = obj.name

        if obj_props.role == "EMITTER":
            row.label(text=obj_props.material_name or "(sans matériau)")
        elif obj_props.role == "COLLIDER":
            row.label(text=f"Friction {obj_props.friction:.2f}")


# ---------------------------------------------------------------------------
# UIList de la bibliotheque de materiaux
#
# Meme motif que `BQ_UL_elements`, mais sur `scene.bourrasque.materials`
# (une CollectionProperty NOMMEE, pas les objets de la scene) : chaque ligne
# affiche la pastille de couleur, le nom EDITABLE EN PLACE (motif standard
# des UIList Blender, `layout.prop(item, "name", text="", emboss=False)`) et
# le nombre d'emetteurs qui l'utilisent (`material_usage`).
# ---------------------------------------------------------------------------


class BQ_UL_materials(UIList):
    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        mat = item
        usage, _dangling = material_usage(context.scene)
        n_users = len(usage.get(mat.name, []))

        row = layout.row(align=True)
        row.prop(mat, "viewport_color", text="")
        row.prop(mat, "name", text="", emboss=False)
        if n_users == 0:
            row.label(text="0", icon="ERROR")
        else:
            row.label(text=str(n_users))


# ---------------------------------------------------------------------------
# Tableau de bord
#
# Vue d'ensemble, lecture seule + actions globales. Compact et dense : lignes
# `split`/`row`, pas `use_property_split` (ce n'est pas un formulaire).
# Chaque helper `_draw_dashboard_*` prend `layout`/`scene`/`props` et NE
# MODIFIE JAMAIS de donnees Blender — seuls les operateurs qu'il propose
# ecrivent.
# ---------------------------------------------------------------------------


def _dashboard_row(layout, label, value):
    split = layout.split(factor=0.5)
    split.label(text=label)
    split.label(text=value)


def _draw_dashboard_domain(layout, scene, props):
    domain = props.domain_object
    box = layout.box()
    box.label(text="Domaine", icon="MESH_CUBE")

    if domain is None:
        box.label(text="Aucun domaine défini.", icon="INFO")
        box.operator("bq.add_domain", text="Créer un domaine", icon="ADD")
        return False

    extents = _domain_extents(domain)
    resolution = domain_resolution(scene)
    _dashboard_row(
        box,
        "Taille",
        f"{extents[0]:.3f} × {extents[1]:.3f} × {extents[2]:.3f} m",
    )
    if resolution is not None:
        res, dx = resolution
        n_cells = res[0] * res[1] * res[2]
        _dashboard_row(box, "dx", f"{dx:.4f} m")
        _dashboard_row(
            box, "Résolution effective", f"{res[0]} × {res[1]} × {res[2]}"
        )
        _dashboard_row(box, "Cellules", _thousands(n_cells))
    return True


def _draw_dashboard_materials(layout, scene, props):
    box = layout.box()
    box.label(text="Matériaux", icon="MATERIAL")

    usage, dangling = material_usage(scene)
    materials = props.materials
    if not materials:
        box.label(text="Bibliothèque de matériaux vide.", icon="INFO")
    else:
        for mat in materials:
            n_users = len(usage.get(mat.name, []))
            row = box.row(align=True)
            row.prop(mat, "viewport_color", text="")
            label = f"{mat.name} — {_material_model_label(mat)} — {n_users} émetteur(s)"
            if n_users == 0:
                row.label(text=label, icon="ERROR")
            else:
                row.label(text=label)

    n_used = used_material_slot_count(scene)
    if n_used > lib.BQ_MAX_MATERIALS:
        box.label(
            text=(
                f"{n_used} matériaux utilisés > maximum {lib.BQ_MAX_MATERIALS} "
                "supporté par le solveur."
            ),
            icon="ERROR",
        )

    if dangling:
        warn = box.box()
        warn.label(
            text=f"{len(dangling)} émetteur(s) sans matériau valide :",
            icon="ERROR",
        )
        for obj in dangling:
            warn.label(text=f"— {obj.name}")
        warn.operator(
            "bq.migrate_materials", text="Réparer", icon="FILE_REFRESH"
        )


def _draw_dashboard_elements(layout, scene):
    box = layout.box()
    box.label(text="Éléments", icon="OUTLINER")

    n_emitters = 0
    n_colliders = 0
    for obj in scene.objects:
        role = obj.bourrasque.role
        if role == "EMITTER":
            n_emitters += 1
        elif role == "COLLIDER":
            n_colliders += 1

    _dashboard_row(box, "Émetteurs", str(n_emitters))
    _dashboard_row(box, "Colliders", str(n_colliders))
    n_tri = collider_triangle_count(scene)
    _dashboard_row(box, "Triangles colliders", _thousands(n_tri))


def _draw_dashboard_rigid_bodies(layout, scene, props):
    """Corps rigides dynamiques (jalon M17, phase A) : combien il y en a,
    un aperçu flotte/coule par rapport à l'eau (1000 kg/m³ — repère demandé
    par l'artiste, pas une constante du solveur), les cas d'échelle non
    uniforme détectables SANS COÛT (`_has_uniform_scale`, simple lecture de
    `obj.scale`), et le bouton qui retire les clés posées par le bake.

    Contrairement au sous-panneau Collider (`BQ_PT_collider`), qui affiche
    la masse RÉELLEMENT calculée (densité × volume du maillage), ce tableau
    de bord ne recalcule PAS le volume de chaque corps à chaque redessin :
    comparer `density` à 1000 suffit à dire flotte/coule, et c'est gratuit
    quel que soit le nombre de corps dynamiques de la scène.
    """
    dynamic_objs = [
        obj
        for obj in scene.objects
        if obj.bourrasque.role == "COLLIDER" and obj.bourrasque.dynamic
    ]
    if not dynamic_objs:
        return

    box = layout.box()
    box.label(text="Corps rigides", icon="MOD_PHYSICS")
    _dashboard_row(box, "Corps dynamiques", str(len(dynamic_objs)))

    if len(dynamic_objs) <= _DASHBOARD_RIGID_BODY_LIST_THRESHOLD:
        for obj in dynamic_objs:
            density = obj.bourrasque.density
            if density < 1000.0:
                state = "flotte"
            elif density > 1000.0:
                state = "coule"
            else:
                state = "neutre"
            row = box.row(align=True)
            row.label(text=obj.name)
            row.label(text=f"{density:.0f} kg/m³ — {state}")
    else:
        n_float = sum(1 for obj in dynamic_objs if obj.bourrasque.density < 1000.0)
        n_sink = sum(1 for obj in dynamic_objs if obj.bourrasque.density > 1000.0)
        n_neutral = len(dynamic_objs) - n_float - n_sink
        _dashboard_row(box, "Flottent (< 1000 kg/m³)", str(n_float))
        _dashboard_row(box, "Coulent (> 1000 kg/m³)", str(n_sink))
        if n_neutral:
            _dashboard_row(box, "Neutres (= 1000 kg/m³)", str(n_neutral))

    non_uniform = [obj.name for obj in dynamic_objs if not _has_uniform_scale(obj)]
    if non_uniform:
        warn = box.box()
        warn.label(
            text=f"{len(non_uniform)} corps à échelle non uniforme :",
            icon="ERROR",
        )
        for name in non_uniform[:5]:
            warn.label(text=f"— {name}")
        if len(non_uniform) > 5:
            warn.label(text=f"… et {len(non_uniform) - 5} autre(s)")
        warn.label(
            text="la décomposition en keyframes de fin de bake sera "
            "approximative (cisaillement)."
        )

    busy = props.is_baking or props.is_baking_mesh or props.is_baking_whitewater
    row = box.row()
    row.enabled = not busy
    row.operator(
        "bq.clear_rigid_keys",
        text="Effacer les clés posées par le bake (corps dynamiques uniquement)",
        icon="TRASH",
    )


def _draw_dashboard_open_mesh_colliders(layout, scene):
    """Colliders STATIQUES au maillage non fermé, signalés au DERNIER bake
    (M17, phase B, tâche B4, point 3) : un plane (ou tout maillage ouvert)
    ne peut porter aucun champ de distance signée (D9) et ne participera
    donc jamais au contact solide↔solide, même s'il reste un collider
    fluide parfaitement valide — c'est précisément le piège que vit un
    artiste qui essaie d'arrêter un cube dynamique avec un plane.

    Lecture SEULE d'un cache mémorisé par `BQ_OT_bake.invoke`
    (`ops.open_mesh_collider_names`) : ne relance JAMAIS
    `sampling.check_mesh_closed` ici (lancer de rayons sur un BVH, bien trop
    coûteux pour un `draw()` appelé à chaque redessin). N'affiche donc rien
    tant qu'aucun bake n'a encore été lancé sur cette scène.
    """
    names = ops.open_mesh_collider_names(scene)
    if not names:
        return

    box = layout.box()
    box.label(text="Colliders au maillage non fermé", icon="ERROR")
    for name in names[:5]:
        box.label(text=f"— {name}")
    if len(names) > 5:
        box.label(text=f"… et {len(names) - 5} autre(s)")
    box.label(
        text="Ne participent pas au contact solide↔solide (fluide inchangé)."
    )
    box.label(
        text="Utilisez une boîte fermée, même très aplatie, pour arrêter "
        "un solide dynamique."
    )


def _draw_dashboard_particle_estimate(layout, scene, props):
    box = layout.box()
    box.label(text="Particules estimées", icon="PARTICLES")
    count = estimate_particle_count(scene)
    _dashboard_row(box, "Estimation", _thousands(count))
    _dashboard_row(box, "Maximum", _thousands(props.max_particles))
    if count > props.max_particles:
        box.label(
            text=f"Dépassement : {_thousands(count)} > {_thousands(props.max_particles)}",
            icon="ERROR",
        )


def _cache_state_row(box, label, exists, is_stale, frames_text, size_text):
    row = box.row(align=True)
    if not exists:
        row.label(text=label, icon="BLANK1")
        row.label(text="Aucun cache")
        return
    if is_stale:
        row.label(text=label, icon="ERROR")
    else:
        row.label(text=label, icon="CHECKMARK")
    row.label(text=f"{frames_text}, {size_text}")


def _draw_dashboard_caches(layout, scene, props, res, cell_size):
    box = layout.box()
    box.label(text="Caches", icon="FILE_CACHE")

    cache_dir = bpy.path.abspath(props.cache_dir)
    requested_frames = max(0, props.frame_end - props.frame_start + 1)

    # -- Particules (.bqd) : pas de notion de parametres caducs (le format
    # ne stocke pas les reglages de production), seulement une comparaison
    # frames bakees / plage demandee.
    bqd_path, _mat_path = cache.cache_paths(cache_dir, scene.name)
    bqd_exists = os.path.isfile(bqd_path)
    if bqd_exists:
        size_bytes = os.path.getsize(bqd_path)
        incomplete = props.baked_frames < requested_frames
        _cache_state_row(
            box,
            "Particules (.bqd)",
            True,
            incomplete,
            f"{props.baked_frames}/{requested_frames} frame(s)",
            _format_bytes(size_bytes),
        )
    else:
        _cache_state_row(box, "Particules (.bqd)", False, False, "", "")

    # -- Maillage (.bqm)
    bqm_path = mesh_cache_path(cache_dir, scene.name)
    if os.path.isfile(bqm_path):
        size_bytes = os.path.getsize(bqm_path)
        try:
            reader = meshcache.MeshCacheReader(bqm_path)
            try:
                frame_count = reader.frame_count
                stored_params = reader.params
            finally:
                reader.close()
            current_params = _current_mesh_params(scene, res, cell_size)
            stale = bool(meshcache.diff_params(stored_params, current_params))
            _cache_state_row(
                box,
                "Maillage (.bqm)",
                True,
                stale,
                f"{frame_count} frame(s)",
                _format_bytes(size_bytes),
            )
        except (OSError, ValueError) as exc:
            box.label(text=f"Maillage (.bqm) illisible : {exc}", icon="ERROR")
    else:
        _cache_state_row(box, "Maillage (.bqm)", False, False, "", "")

    # -- Whitewater (.bqw)
    bqw_path = whitewater_cache_path(cache_dir, scene.name)
    if os.path.isfile(bqw_path):
        size_bytes = os.path.getsize(bqw_path)
        try:
            reader = whitewatercache.WhitewaterCacheReader(bqw_path)
            try:
                frame_count = reader.frame_count
                stored_params = reader.params
            finally:
                reader.close()
            current_params = _current_whitewater_params(scene)
            stale = current_params is not None and bool(
                whitewatercache.diff_params(stored_params, current_params)
            )
            _cache_state_row(
                box,
                "Whitewater (.bqw)",
                True,
                stale,
                f"{frame_count} frame(s)",
                _format_bytes(size_bytes),
            )
        except (OSError, ValueError) as exc:
            box.label(text=f"Whitewater (.bqw) illisible : {exc}", icon="ERROR")
    else:
        _cache_state_row(box, "Whitewater (.bqw)", False, False, "", "")


def _draw_dashboard_actions(layout, props):
    box = layout.box()
    box.label(text="Actions", icon="PLAY")

    busy = props.is_baking or props.is_baking_mesh or props.is_baking_whitewater
    if busy:
        if props.is_baking:
            progress, text = props.bake_progress, "Particules"
        elif props.is_baking_mesh:
            progress, text = props.bake_mesh_progress, "Maillage"
        else:
            progress, text = props.bake_whitewater_progress, "Whitewater"

        if hasattr(box, "progress"):
            box.progress(factor=progress, text=f"{text} : {progress * 100:.0f} %")
        else:
            row = box.row()
            row.use_property_split = True
            row.prop(props, "bake_progress", slider=True, text=text)
        box.operator("bq.cancel_bake", text="Annuler le bake", icon="CANCEL")
        return

    row = box.row()
    row.scale_y = 1.5
    row.operator("bq.bake_all", text="Tout baker", icon="PLAY")

    row = box.row(align=True)
    row.operator("bq.free_cache", text="Vider particules", icon="TRASH")
    row.operator("bq.free_mesh_cache", text="Vider maillage", icon="TRASH")
    row.operator("bq.free_whitewater_cache", text="Vider whitewater", icon="TRASH")


class BQ_PT_dashboard(Panel):
    bl_idname = "BQ_PT_dashboard"
    bl_label = "Tableau de bord"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    # Pas de DEFAULT_CLOSED : c'est le panneau qu'on veut voir en premier,
    # ouvert, a l'ouverture de la sidebar.

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.bourrasque

        has_domain = _draw_dashboard_domain(layout, scene, props)
        if not has_domain:
            # Tout le reste du tableau de bord depend d'un domaine defini
            # (resolution, estimation de particules, chemin des caches...).
            return

        _draw_dashboard_materials(layout, scene, props)
        _draw_dashboard_elements(layout, scene)
        _draw_dashboard_rigid_bodies(layout, scene, props)
        _draw_dashboard_open_mesh_colliders(layout, scene)
        _draw_dashboard_particle_estimate(layout, scene, props)

        res_layout = mesh_layout(scene)
        if res_layout is not None:
            res, cell_size = res_layout
            _draw_dashboard_caches(layout, scene, props, res, cell_size)

        _draw_dashboard_actions(layout, props)


# ---------------------------------------------------------------------------
# Domaine
# ---------------------------------------------------------------------------


class BQ_PT_domain(Panel):
    bl_idname = "BQ_PT_domain"
    bl_label = "Domaine"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        scene = context.scene
        props = scene.bourrasque
        layout.enabled = not props.is_baking

        layout.prop(props, "domain_object")

        domain = props.domain_object
        if domain is None:
            layout.operator("bq.add_domain", text="Créer un domaine", icon="ADD")
            return

        layout.prop(props, "grid_res")

        # `domain_transform` renvoie desormais le pave du SOLVEUR (elargi
        # d'une marge de stencil sur chaque face, voir sa docstring), pas la
        # boite de l'artiste : la taille affichee ici doit rester celle de
        # la boite que l'artiste a placee (`extents`, calculee directement
        # depuis la bbox de l'objet domaine via `_domain_extents`), pour ne
        # pas laisser penser que le domaine s'est agrandi. `dx`, en
        # revanche, est bien la taille de cellule REELLEMENT utilisee par
        # le solveur (voir `domain_resolution`), coherente avec le pas du
        # reseau d'emission (`sampling.py`, `ops.py`). Depuis M5, un
        # domaine non cubique est un pave explicitement supporte : la
        # resolution EFFECTIVE (par axe) et le nombre total de cellules
        # sont affiches ici, c'est le chiffre qui gouverne la memoire et le
        # temps de calcul (voir docs/plan-milestone-5.md D4/R5).
        extents = _domain_extents(domain)

        resolution = domain_resolution(scene)
        if resolution is not None:
            res, dx = resolution
            n_cells = res[0] * res[1] * res[2]
            box = layout.box()
            box.label(
                text="Taille du domaine : "
                f"{extents[0]:.3f} × {extents[1]:.3f} × {extents[2]:.3f} m"
            )
            box.label(text=f"Taille de cellule (dx) : {dx:.4f} m")
            box.label(
                text=f"Résolution effective : {res[0]} × {res[1]} × {res[2]}"
            )
            box.label(text=f"Nombre total de cellules : {_thousands(n_cells)}")


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------


class BQ_PT_elements(Panel):
    bl_idname = "BQ_PT_elements"
    bl_label = "Éléments"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.bourrasque
        layout.enabled = not props.is_baking

        elements = iter_elements(scene)
        if not elements:
            col = layout.column(align=True)
            col.label(text="Aucun élément dans la simulation.", icon="INFO")
            col.label(text="Créez un domaine, sélectionnez un objet,")
            col.label(text="puis ajoutez-le comme émetteur ci-dessous.")
        else:
            layout.template_list(
                "BQ_UL_elements",
                "",
                scene,
                "objects",
                props,
                "active_element_index",
                rows=4,
            )

        row = layout.row(align=True)
        row.operator("bq.add_emitter", text="Émetteur", icon="ADD")
        row.operator("bq.add_collider", text="Collider", icon="MOD_PHYSICS")
        row.operator("bq.remove_element", text="", icon="X")

        n_tri = collider_triangle_count(scene)
        if n_tri > 0:
            box = layout.box()
            box.label(text=f"Triangles colliders : {_thousands(n_tri)}")
            if n_tri > _COLLIDER_TRIANGLE_WARNING_THRESHOLD:
                box.label(
                    text=(
                        "Nombre de triangles très élevé : le calcul du "
                        "champ de distance devient comparable au coût "
                        "d'un pas de simulation. Envisagez de décimer vos "
                        "colliders."
                    ),
                    icon="ERROR",
                )


# ---------------------------------------------------------------------------
# Émetteur (sous-panneau d'Éléments)
# ---------------------------------------------------------------------------


class BQ_PT_emitter(Panel):
    bl_idname = "BQ_PT_emitter"
    bl_label = "Émetteur"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_parent_id = "BQ_PT_elements"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.bourrasque.role == "EMITTER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        scene = context.scene
        layout.enabled = not scene.bourrasque.is_baking

        obj = context.active_object
        obj_props = obj.bourrasque

        layout.prop_search(
            obj_props, "material_name", scene.bourrasque, "materials", text="Matériau"
        )
        usage, dangling = material_usage(scene)
        if obj_props.material_name not in usage or obj in dangling:
            warn = layout.box()
            warn.label(
                text="Aucun matériau valide : le bake sera refusé.", icon="ERROR"
            )
            warn.operator(
                "bq.migrate_materials", text="Réparer", icon="FILE_REFRESH"
            )

        layout.prop(obj_props, "initial_velocity")

        layout.prop(obj_props, "turbulence")
        turb_row = layout.row()
        # Grise `turbulence_seed` a turbulence nulle : sans turbulence, la
        # graine n'a aucun effet (voir `_turbulent_emission`, ops.py).
        turb_row.enabled = obj_props.turbulence > 0.0
        turb_row.prop(obj_props, "turbulence_seed")

        layout.prop(obj_props, "emit_mode")
        layout.prop(obj_props, "emit_source")

        if obj_props.emit_source == "BOUNDS":
            note = layout.box()
            note.label(
                text="Seule la boîte englobante (AABB) de l'objet", icon="INFO"
            )
            note.label(text="est utilisée pour l'émission, pas sa forme.")

        # Inflow a vitesse nulle : comportement degenere mais inoffensif
        # depuis l'ensemencement volumique avec test d'occupation (voir
        # ops._emit_inflow_sites) — l'emetteur sature son volume une seule
        # fois puis ne le renouvelle plus (aucune particule ne s'en eloigne
        # pour liberer un site). Simple information, plus une erreur.
        if obj_props.emit_mode == "INFLOW" and _is_vector_zero(
            obj_props.initial_velocity
        ):
            warn = layout.box()
            warn.label(text="Émission continue à vitesse nulle :", icon="INFO")
            warn.label(text="l'émetteur se remplira sans produire d'écoulement.")

        if obj_props.emit_source == "MESH":
            # check_mesh_closed vit dans extension/sampling.py, ecrit en
            # parallele par un autre lot : import paresseux, purement
            # defensif (l'UI ne doit pas casser si le module n'existe pas
            # encore).
            try:
                from .sampling import check_mesh_closed
            except ImportError:
                check_mesh_closed = None

            if check_mesh_closed is not None:
                is_closed, message = check_mesh_closed(obj)
                if not is_closed:
                    warn = layout.box()
                    warn.label(text=message, icon="ERROR")

            # L'echantillonnage d'interieur n'est fait qu'une fois, au debut
            # du bake : un emetteur anime (transformation ou forme) donne un
            # resultat faux et silencieux. Hors perimetre de ce jalon.
            if _is_object_animated(obj):
                warn = layout.box()
                warn.label(text="Émetteur animé en mode maillage :", icon="ERROR")
                warn.label(
                    text="seule la pose de la première frame sera utilisée"
                )
                warn.label(text="pour déterminer la forme d'émission.")

            info = layout.box()
            info.label(
                text="Émission par maillage : coût de calcul au démarrage",
                icon="INFO",
            )
            info.label(text="du bake, proportionnel au volume de l'émetteur.")

        # Avertissement visible AVANT le bake : un emetteur hors domaine (ou
        # trop pres de son bord) fait planter le solveur (ecriture GPU hors
        # bornes), voir props.emitter_overflow. Le bake le refuse deja, mais
        # l'artiste doit pouvoir le voir sans avoir a lancer le bake.
        transform = domain_transform(scene)
        resolution = domain_resolution(scene)
        if transform is not None and resolution is not None:
            origin, size = transform
            _res, dx = resolution
            overflow = emitter_overflow(obj, origin, size, dx)
            if overflow is not None:
                fully_outside, offending_axes, margin = overflow
                warn = layout.box()
                if fully_outside:
                    warn.label(
                        text="Émetteur entièrement hors du domaine :",
                        icon="ERROR",
                    )
                    warn.label(text="le bake sera refusé.")
                else:
                    warn.label(
                        text="Émetteur trop proche du bord du domaine "
                        f"(axe {', '.join(offending_axes)}) :",
                        icon="ERROR",
                    )
                    warn.label(
                        text=f"marge de sécurité {margin:.4f} m — le bake "
                        "sera refusé."
                    )


# ---------------------------------------------------------------------------
# Collider (sous-panneau d'Éléments)
# ---------------------------------------------------------------------------


class BQ_PT_collider(Panel):
    bl_idname = "BQ_PT_collider"
    bl_label = "Collider"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_parent_id = "BQ_PT_elements"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.bourrasque.role == "COLLIDER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        scene = context.scene
        layout.enabled = not scene.bourrasque.is_baking

        obj = context.active_object
        obj_props = obj.bourrasque
        layout.prop(obj_props, "friction")
        layout.prop(obj_props, "restitution")
        # `friction`/`restitution` gouvernent DEUX contacts distincts (M17,
        # phase B, tache B4) : le frottement/l'elasticite du FLUIDE contre
        # cet obstacle (comme depuis M6), ET desormais le frottement de
        # Coulomb / la restitution au contact avec les AUTRES SOLIDES
        # (combinaison de paire : moyenne geometrique pour la friction,
        # maximum pour la restitution -- cf. BqRigidBody, bourrasque.h) --
        # y compris pour un collider FIXE (D12 : masse infinie, meme
        # traitement de contact qu'un corps dynamique). Precise ici en toutes
        # lettres : les libelles RNA de props.py (tooltips) ne le disent pas
        # encore pour `friction`, hors perimetre de ce lot (props.py non
        # touche).
        info = layout.box()
        info.scale_y = 0.8
        info.label(
            text="S'appliquent aussi au contact avec les autres solides",
            icon="INFO",
        )

        # Point 3 de B4 : rappel GRATUIT (dict memorise au dernier bake, cf.
        # ops.open_mesh_collider_names -- jamais un nouvel appel a
        # `check_mesh_closed` ici) si CET objet a ete signale comme collider
        # statique au maillage non ferme lors du dernier bake.
        if obj_props.role == "COLLIDER" and not obj_props.dynamic:
            if obj.name in ops.open_mesh_collider_names(scene):
                warn = layout.box()
                warn.label(
                    text="Maillage non fermé : ne participe pas au contact "
                    "solide↔solide.",
                    icon="ERROR",
                )
                warn.label(
                    text="Utilisez une boîte fermée (même très aplatie) "
                    "pour arrêter un solide dynamique."
                )

        layout.prop(obj_props, "dynamic")

        # D14 du plan M17 : le collider FIXE (le cas majoritaire, cf.
        # docstring du module) ne doit rien perdre, donc rien de plus ne se
        # dessine ci-dessous quand `dynamic` est faux -- pas de propriete de
        # corps rigide, pas d'avertissement d'echelle, pas d'information de
        # masse (`friction`/`restitution` sont deja dessines plus haut,
        # desormais valables aussi pour un collider fixe -- D12).
        if not obj_props.dynamic:
            return

        layout.prop(obj_props, "density")
        layout.prop(obj_props, "use_gravity")
        layout.prop(obj_props, "lock_location")
        layout.prop(obj_props, "lock_rotation")

        # -- Masse calculee, lecture seule (D7 du plan M17) ----------------
        #
        # C'est l'information dont l'artiste a reellement besoin pour savoir
        # si son objet va flotter ou couler (repere : l'eau est a
        # 1000 kg/m3). Reutilise l'integrale de volume DEJA ecrite pour
        # l'emission par maillage (`sampling._evaluated_world_volume_and_bbox`)
        # plutot que de la redupliquer -- import paresseux, meme motif
        # defensif que `check_mesh_closed` plus haut dans ce fichier.
        #
        # On n'appelle PAS `check_mesh_closed` ici : c'est un lancer de
        # rayons sur un BVH, beaucoup trop couteux pour un redessin appele a
        # chaque frame de survol de souris. Un maillage OUVERT donne donc
        # simplement un volume sans signification physique, affiche tel
        # quel sans faire planter le panneau -- le bake, lui, refuse
        # proprement (voir `ops._setup_collider_body`).
        try:
            from .sampling import (
                _evaluated_object_and_depsgraph,
                _evaluated_world_volume_and_bbox,
            )
        except ImportError:
            _evaluated_object_and_depsgraph = None

        if _evaluated_object_and_depsgraph is not None:
            obj_eval, _depsgraph = _evaluated_object_and_depsgraph(obj)
            volume, _bbox_volume = _evaluated_world_volume_and_bbox(obj_eval)

            box = layout.box()
            if volume <= 0.0:
                box.label(
                    text="Volume nul ou maillage ouvert : masse non calculable.",
                    icon="ERROR",
                )
            else:
                mass = obj_props.density * volume
                if obj_props.density < 1000.0:
                    state = "flotte (densité < eau, 1000 kg/m³)"
                elif obj_props.density > 1000.0:
                    state = "coule (densité > eau, 1000 kg/m³)"
                else:
                    state = "neutre (densité = eau, 1000 kg/m³)"
                box.label(text=f"Volume : {volume:.4f} m³")
                box.label(text=f"Masse calculée : {mass:.3f} kg — {state}")

        if not _has_uniform_scale(obj):
            warn = layout.box()
            warn.label(text="Échelle non uniforme :", icon="ERROR")
            warn.label(
                text="la décomposition en keyframes de fin de bake sera "
                "approximative (cisaillement)."
            )

        # `added_mass` (props.py) volontairement PAS expose ici : le
        # balayage de stabilite du jalon a montre que ce reglage est
        # structurellement casse (divise l'impulsion de flottaison sans
        # alleger la gravite appliquee a la masse reelle -- a alpha >= 0.25,
        # meme un corps de densite 20 kg/m3 traverse une colonne d'eau de
        # 0.86 m en moins d'une seconde), pas seulement couteux. Remplace
        # par une formulation implicite dans un lot separe. La propriete
        # reste declaree cote props.py, seulement retiree de l'UI.


# ---------------------------------------------------------------------------
# Materiaux (bibliotheque de la scene)
# ---------------------------------------------------------------------------


class BQ_PT_materials(Panel):
    bl_idname = "BQ_PT_materials"
    bl_label = "Matériaux"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.bourrasque
        layout.enabled = not props.is_baking

        row = layout.row()
        row.template_list(
            "BQ_UL_materials",
            "",
            props,
            "materials",
            props,
            "active_material_index",
            rows=4,
        )

        col = row.column(align=True)
        col.operator("bq.material_add", text="", icon="ADD")
        col.operator("bq.material_remove", text="", icon="REMOVE")
        col.separator()
        col.operator("bq.material_duplicate", text="", icon="DUPLICATE")

        n_used = used_material_slot_count(scene)
        if n_used > lib.BQ_MAX_MATERIALS:
            layout.label(
                text=(
                    f"{n_used} matériaux utilisés > maximum "
                    f"{lib.BQ_MAX_MATERIALS} supporté par le solveur."
                ),
                icon="ERROR",
            )

        if not (0 <= props.active_material_index < len(props.materials)):
            return

        mat = props.materials[props.active_material_index]

        editor = layout.column()
        editor.use_property_split = True
        editor.prop(mat, "preset")
        editor.prop(mat, "model")
        editor.prop(mat, "rho")

        if mat.model == "ELASTIC":
            editor.prop(mat, "young")
            editor.prop(mat, "poisson")
        elif mat.model == "WATER":
            editor.prop(mat, "bulk")
            editor.prop(mat, "gamma")
        elif mat.model == "SAND":
            editor.prop(mat, "young")
            editor.prop(mat, "poisson")
            editor.prop(mat, "friction_angle")
            # `cohesion` est VOLONTAIREMENT absente de l'UI : le coeur
            # refuse aujourd'hui tout materiau SAND avec une cohesion non
            # nulle (voir props.py, BqMaterialProps.cohesion). Exposer un
            # champ editable dont toute valeur non nulle ferait echouer le
            # bake serait un piege pour l'artiste -- meme traitement que
            # `added_mass` au jalon precedent.

        editor.prop(mat, "viewport_color")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class BQ_PT_simulation(Panel):
    bl_idname = "BQ_PT_simulation"
    bl_label = "Simulation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        scene = context.scene
        props = scene.bourrasque
        layout.enabled = not props.is_baking

        col = layout.column(align=True)
        col.prop(props, "frame_start")
        col.prop(props, "frame_end")

        layout.prop(props, "gravity")
        layout.prop(props, "cfl")
        layout.prop(props, "ppc_axis")
        layout.prop(props, "max_particles")

        count = estimate_particle_count(scene)
        info = layout.box()
        info.label(
            text=(
                "Particules estimées (émission continue incluse) : "
                f"{_thousands(count)}"
            )
        )
        if count > props.max_particles:
            info.label(
                text=(
                    f"Dépassement : {_thousands(count)} > "
                    f"max {_thousands(props.max_particles)}"
                ),
                icon="ERROR",
            )


# ---------------------------------------------------------------------------
# Sortie (conteneur)
# ---------------------------------------------------------------------------


class BQ_PT_output(Panel):
    bl_idname = "BQ_PT_output"
    bl_label = "Sortie"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        pass


# ---------------------------------------------------------------------------
# Particules (sous-panneau de Sortie)
# ---------------------------------------------------------------------------


class BQ_PT_bake(Panel):
    bl_idname = "BQ_PT_bake"
    bl_label = "Particules"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_parent_id = "BQ_PT_output"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.bourrasque

        if props.is_baking:
            if hasattr(layout, "progress"):
                layout.progress(
                    factor=props.bake_progress,
                    text=f"{props.bake_progress * 100:.0f} %",
                )
            else:
                layout.use_property_split = True
                layout.prop(props, "bake_progress", slider=True, text="Progression")
            layout.operator("bq.cancel_bake", text="Annuler le bake", icon="CANCEL")
            return

        layout.operator("bq.bake", text="Lancer le bake", icon="PLAY")

        col = layout.column()
        col.use_property_split = True
        col.prop(props, "cache_dir")

        layout.separator()
        layout.operator("bq.free_cache", text="Vider le cache", icon="TRASH")

        if props.baked_frames > 0:
            layout.label(text=f"{props.baked_frames} frame(s) en cache")
        else:
            layout.label(text="Aucun cache")


# ---------------------------------------------------------------------------
# Maillage (sous-panneau de Sortie)
# ---------------------------------------------------------------------------


class BQ_PT_mesh(Panel):
    bl_idname = "BQ_PT_mesh"
    bl_label = "Maillage"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_parent_id = "BQ_PT_output"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.bourrasque

        busy = props.is_baking or props.is_baking_mesh

        # -- Grandeurs calculees, en lecture seule (ergonomie M7.1) --------
        #
        # Affichees AVANT les reglages : c'est le lien entre le facteur que
        # l'artiste regle ci-dessous et la distance/resolution reelle que
        # cela represente. N'a de sens qu'avec un domaine defini.
        res_layout = mesh_layout(scene)
        if res_layout is None:
            layout.label(text="Aucun domaine défini.", icon="INFO")
            return

        res, cell_size = res_layout
        n_cells = res[0] * res[1] * res[2]
        spacing = mesh_particle_spacing(scene)
        resolution_state = mesh_resolution_state(scene)
        influence_radius, particle_radius, collider_offset = mesh_effective_radii(
            scene
        )

        info = layout.box()
        if spacing is not None:
            info.label(text=f"Espacement inter-particules : {spacing:.4f} m")
        if resolution_state is not None:
            _setting, is_auto, was_capped = resolution_state
            res_text = (
                f"Résolution du maillage : {res[0]} × {res[1]} × {res[2]} "
                f"({_thousands(n_cells)} cellules)"
            )
            if is_auto:
                res_text += " — automatique"
                if was_capped:
                    res_text += ", plafonnée (VRAM)"
            info.label(text=res_text)
        info.label(
            text=(
                f"Rayon d'influence effectif : {influence_radius:.4f} m — "
                f"rayon de particule : {particle_radius:.4f} m — "
                f"décalage collider : {collider_offset:.4f} m"
            )
        )

        settings = layout.column()
        settings.use_property_split = True
        settings.enabled = not busy
        settings.prop(props, "mesh_resolution")
        settings.prop(props, "mesh_influence_factor")
        settings.prop(props, "mesh_particle_factor")
        settings.prop(props, "mesh_collider_offset_factor")
        settings.prop(props, "mesh_smoothing_iters")
        settings.prop(props, "mesh_min_component_tris")

        # Empreinte memoire estimee AVANT tout bake (garde-fou VRAM, voir
        # docs/plan-milestone-7.md D3/R1) : 36 octets par cellule du champ
        # de maillage (champ Zhu-Bridson + tampons du marching cubes),
        # calcule en pur Python (`mesh_vram_estimate_bytes`), sans dependre
        # de la DLL native — l'artiste doit voir ce chiffre avant de lancer
        # un bake, pas le decouvrir par un echec.
        est_bytes = mesh_vram_estimate_bytes(scene)

        box = layout.box()
        box.label(text=f"Empreinte mémoire estimée : {_format_bytes(est_bytes)}")
        if est_bytes > mesh_vram_warning_threshold_bytes():
            box.label(
                text=(
                    "Empreinte mémoire très élevée (> "
                    f"{_format_bytes(mesh_vram_warning_threshold_bytes())}) : "
                    "risque d'échec du bake par manque de VRAM. Réduisez la "
                    "résolution du maillage."
                ),
                icon="ERROR",
            )

        layout.separator()

        if props.is_baking_mesh:
            if hasattr(layout, "progress"):
                layout.progress(
                    factor=props.bake_mesh_progress,
                    text=f"{props.bake_mesh_progress * 100:.0f} %",
                )
            else:
                prog = layout.column()
                prog.use_property_split = True
                prog.prop(
                    props, "bake_mesh_progress", slider=True, text="Progression"
                )
            layout.operator(
                "bq.cancel_bake", text="Annuler le bake de maillage", icon="CANCEL"
            )
        else:
            row = layout.row(align=True)
            row.enabled = not props.is_baking
            row.operator("bq.bake_mesh", text="Baker le maillage", icon="MOD_REMESH")
            row.operator("bq.bake_all", text="Tout baker", icon="PLAY")

        row = layout.row()
        row.enabled = not busy
        row.operator(
            "bq.free_mesh_cache", text="Vider le cache de maillage", icon="TRASH"
        )

        layout.separator()
        layout.label(
            text="Affichage : matériau triplanaire de départ (sans UV), "
            "pas un rendu final.",
            icon="INFO",
        )
        layout.operator(
            "bq.setup_fluid_display",
            text="Configurer l'affichage du maillage",
            icon="MATERIAL",
        )

        # -- Etat du cache .bqm --------------------------------------------
        cache_dir = bpy.path.abspath(props.cache_dir)
        bqm_path = mesh_cache_path(cache_dir, scene.name)

        if not os.path.isfile(bqm_path):
            layout.label(text="Aucun cache de maillage.")
            return

        size_bytes = os.path.getsize(bqm_path)
        try:
            reader = meshcache.MeshCacheReader(bqm_path)
            try:
                frame_count = reader.frame_count
                stored_params = reader.params
            finally:
                reader.close()
        except (OSError, ValueError) as exc:
            layout.label(text=f"Cache de maillage illisible : {exc}", icon="ERROR")
            return

        info = layout.box()
        info.label(
            text=f"{frame_count} frame(s) en cache, {_format_bytes(size_bytes)}"
        )

        current_params = _current_mesh_params(scene, res, cell_size)
        diffs = meshcache.diff_params(stored_params, current_params)
        if diffs:
            warn = layout.box()
            warn.label(text="Cache de maillage caduc :", icon="ERROR")
            for name in diffs:
                label = _MESH_PARAM_LABELS.get(name, name)
                warn.label(text=f"— {label} a changé")
            warn.label(text="Rebakez le maillage pour le mettre à jour.")


# ---------------------------------------------------------------------------
# Whitewater (sous-panneau de Sortie)
# ---------------------------------------------------------------------------


class BQ_PT_whitewater(Panel):
    bl_idname = "BQ_PT_whitewater"
    bl_label = "Whitewater"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_parent_id = "BQ_PT_output"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.bourrasque

        busy = props.is_baking or props.is_baking_mesh or props.is_baking_whitewater

        layout.label(
            text="Particules secondaires (écume, bulles, embruns) — "
            "indépendant du maillage, ne lit que le cache de particules.",
            icon="INFO",
        )

        settings = layout.column()
        settings.use_property_split = True
        settings.enabled = not busy
        settings.prop(props, "ww_max_particles")

        box = settings.box()
        box.label(text="Air piégé")
        box.prop(props, "ww_ta_min")
        box.prop(props, "ww_ta_max")
        box.prop(props, "ww_ta_weight")

        box = settings.box()
        box.label(text="Crête de vague")
        box.prop(props, "ww_wc_min")
        box.prop(props, "ww_wc_max")
        box.prop(props, "ww_wc_weight")

        box = settings.box()
        box.label(text="Énergie cinétique")
        box.prop(props, "ww_ke_min")
        box.prop(props, "ww_ke_max")
        box.prop(props, "ww_ke_weight")

        settings.prop(props, "ww_spawn_rate")

        box = settings.box()
        box.label(text="Durée de vie")
        box.prop(props, "ww_life_spray")
        box.prop(props, "ww_life_foam")
        box.prop(props, "ww_life_bubble")

        box = settings.box()
        box.label(text="Advection")
        box.prop(props, "ww_drag_spray")
        box.prop(props, "ww_drag_foam")
        box.prop(props, "ww_buoyancy_bubble")

        layout.separator()

        if props.is_baking_whitewater:
            if hasattr(layout, "progress"):
                layout.progress(
                    factor=props.bake_whitewater_progress,
                    text=f"{props.bake_whitewater_progress * 100:.0f} %",
                )
            else:
                prog = layout.column()
                prog.use_property_split = True
                prog.prop(
                    props, "bake_whitewater_progress", slider=True,
                    text="Progression",
                )
            layout.operator(
                "bq.cancel_bake", text="Annuler le bake de whitewater",
                icon="CANCEL",
            )
        else:
            layout.operator(
                "bq.bake_whitewater", text="Baker le whitewater", icon="PLAY"
            )

        row = layout.row()
        row.enabled = not busy
        row.operator(
            "bq.free_whitewater_cache", text="Vider le cache de whitewater",
            icon="TRASH",
        )

        layout.separator()
        layout.label(
            text="Affichage : point de départ éditable (instances + "
            "matériaux), pas un rendu final.",
            icon="INFO",
        )
        display_settings = layout.column()
        display_settings.use_property_split = True
        display_settings.prop(props, "ww_size_mult")
        row = layout.row(align=True)
        row.operator(
            "bq.setup_whitewater_display",
            text="Configurer l'affichage par particules",
            icon="GEOMETRY_NODES",
        )
        row.operator(
            "bq.setup_whitewater_display_volume",
            text="Configurer l'affichage volumétrique",
            icon="VOLUME_DATA",
        )

        if props.baked_whitewater_max_refused > 0:
            warn = layout.box()
            warn.label(
                text=(
                    "Capacité atteinte pendant le bake (jusqu'à "
                    f"{_thousands(props.baked_whitewater_max_refused)} "
                    "candidate(s) refusée(s) sur une frame) :"
                ),
                icon="ERROR",
            )
            warn.label(
                text="augmentez « Particules secondaires max » pour ne pas "
                "perdre de densité."
            )

        # -- Etat du cache .bqw --------------------------------------------
        cache_dir = bpy.path.abspath(props.cache_dir)
        bqw_path = whitewater_cache_path(cache_dir, scene.name)

        if not os.path.isfile(bqw_path):
            layout.label(text="Aucun cache de whitewater.")
            return

        size_bytes = os.path.getsize(bqw_path)
        try:
            reader = whitewatercache.WhitewaterCacheReader(bqw_path)
            try:
                frame_count = reader.frame_count
                stored_params = reader.params
            finally:
                reader.close()
        except (OSError, ValueError) as exc:
            layout.label(
                text=f"Cache de whitewater illisible : {exc}", icon="ERROR"
            )
            return

        info = layout.box()
        info.label(
            text=f"{frame_count} frame(s) en cache, {_format_bytes(size_bytes)}"
        )

        current_params = _current_whitewater_params(scene)
        if current_params is not None:
            diffs = whitewatercache.diff_params(stored_params, current_params)
            if diffs:
                warn = layout.box()
                warn.label(text="Cache de whitewater caduc :", icon="ERROR")
                for name in diffs:
                    label = _WHITEWATER_PARAM_LABELS.get(name, name)
                    warn.label(text=f"— {label} a changé")
                warn.label(text="Rebakez le whitewater pour le mettre à jour.")


# ---------------------------------------------------------------------------
# Affichage (sous-panneau de Sortie)
# ---------------------------------------------------------------------------


class BQ_PT_display(Panel):
    bl_idname = "BQ_PT_display"
    bl_label = "Affichage"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"
    bl_parent_id = "BQ_PT_output"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        layout.label(
            text="Taille d'affichage des particules : pas encore réglable",
            icon="INFO",
        )
        layout.label(
            text="Coloration par matériau : viendra avec l'affichage des",
            icon="INFO",
        )
        layout.label(text="particules (jalon ultérieur).")


# ---------------------------------------------------------------------------
# Enregistrement
# ---------------------------------------------------------------------------

classes = (
    BQ_UL_elements,
    BQ_UL_materials,
    BQ_PT_dashboard,
    BQ_PT_domain,
    BQ_PT_elements,
    BQ_PT_emitter,
    BQ_PT_collider,
    BQ_PT_materials,
    BQ_PT_simulation,
    BQ_PT_output,
    BQ_PT_bake,
    BQ_PT_mesh,
    BQ_PT_whitewater,
    BQ_PT_display,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
