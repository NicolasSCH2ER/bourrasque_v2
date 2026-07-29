"""ui.py — panneaux et UIList de l'extension Bourrasque.

C'est la surface que voit l'artiste, dans la barre laterale du viewport 3D
(categorie « Bourrasque »). Ce module ne fait QUE de la presentation : il lit
`scene.bourrasque` / `obj.bourrasque` (voir props.py) et invoque les sept
operateurs de ops.py (bq.add_domain, bq.add_emitter, bq.add_collider,
bq.remove_element, bq.bake, bq.cancel_bake, bq.free_cache). Aucun appel a la
DLL, aucune logique de simulation ici : voir lib.py et ops.py.
"""

import math

import bpy
import mathutils
from bpy.types import Panel, UIList

from .props import (
    collider_triangle_count,
    domain_resolution,
    domain_transform,
    emitter_overflow,
    estimate_particle_count,
    iter_elements,
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


def _material_label(obj_props):
    """Libellé français du preset materiau d'un emetteur (« Eau », etc.)."""
    enum_items = obj_props.bl_rna.properties["preset"].enum_items
    return enum_items[obj_props.preset].name


def _is_vector_zero(vec, eps=1e-6):
    return math.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]) < eps


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
            row.label(text=_material_label(obj_props))
        elif obj_props.role == "COLLIDER":
            row.label(text=f"Friction {obj_props.friction:.2f}")


# ---------------------------------------------------------------------------
# Panneau racine
# ---------------------------------------------------------------------------


class BQ_PT_main(Panel):
    bl_idname = "BQ_PT_main"
    bl_label = "Bourrasque"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Bourrasque"

    def draw(self, context):
        pass


# ---------------------------------------------------------------------------
# Domaine
# ---------------------------------------------------------------------------


class BQ_PT_domain(Panel):
    bl_idname = "BQ_PT_domain"
    bl_label = "Domaine"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "BQ_PT_main"

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
    bl_parent_id = "BQ_PT_main"

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
# Materiau
# ---------------------------------------------------------------------------


class BQ_PT_material(Panel):
    bl_idname = "BQ_PT_material"
    bl_label = "Matériau"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "BQ_PT_main"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.bourrasque.role == "EMITTER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        scene = context.scene
        layout.enabled = not scene.bourrasque.is_baking

        obj_props = context.active_object.bourrasque

        layout.prop(obj_props, "preset")
        layout.prop(obj_props, "model")
        layout.prop(obj_props, "rho")

        if obj_props.model == "ELASTIC":
            layout.prop(obj_props, "young")
            layout.prop(obj_props, "poisson")
        elif obj_props.model == "WATER":
            layout.prop(obj_props, "bulk")
            layout.prop(obj_props, "gamma")

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
                is_closed, message = check_mesh_closed(context.active_object)
                if not is_closed:
                    warn = layout.box()
                    warn.label(text=message, icon="ERROR")

            # L'echantillonnage d'interieur n'est fait qu'une fois, au debut
            # du bake : un emetteur anime (transformation ou forme) donne un
            # resultat faux et silencieux. Hors perimetre de ce jalon.
            if _is_object_animated(context.active_object):
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
            overflow = emitter_overflow(
                context.active_object, origin, size, dx
            )
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
# Collider
# ---------------------------------------------------------------------------


class BQ_PT_collider(Panel):
    bl_idname = "BQ_PT_collider"
    bl_label = "Collider"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "BQ_PT_main"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.bourrasque.role == "COLLIDER"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True

        scene = context.scene
        layout.enabled = not scene.bourrasque.is_baking

        obj_props = context.active_object.bourrasque
        layout.prop(obj_props, "friction")


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


class BQ_PT_simulation(Panel):
    bl_idname = "BQ_PT_simulation"
    bl_label = "Simulation"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "BQ_PT_main"

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
# Bake
# ---------------------------------------------------------------------------


class BQ_PT_bake(Panel):
    bl_idname = "BQ_PT_bake"
    bl_label = "Bake"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "BQ_PT_main"

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
# Affichage
# ---------------------------------------------------------------------------


class BQ_PT_display(Panel):
    bl_idname = "BQ_PT_display"
    bl_label = "Affichage"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_parent_id = "BQ_PT_main"

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
    BQ_PT_main,
    BQ_PT_domain,
    BQ_PT_elements,
    BQ_PT_material,
    BQ_PT_collider,
    BQ_PT_simulation,
    BQ_PT_bake,
    BQ_PT_display,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
