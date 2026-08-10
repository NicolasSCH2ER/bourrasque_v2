"""Verification des `draw()` de `extension/ui.py` pour le jalon M17, phase A
(sous-panneau Collider dynamique + section « Corps rigides » du tableau de
bord), a lancer avec :

    blender --background --factory-startup --python tools/repro/verify_rigidbody_ui.py

Pourquoi ce script vit ici et pas dans `extension/tests/` : les tests sous
`extension/tests/` tournent sous un stub `bpy` minimal (voir
`extension/tests/_bpy_stub.py` ou equivalent) qui n'a ni RNA reelle, ni
`bl_rna`, ni enum d'icones -- il ne peut structurellement pas attraper une
propriete mal orthographiee (`layout.prop(obj_props, "denisty")`) ou un nom
d'icone invalide, deux erreurs qui ne se voient AUTREMENT qu'a l'ecran une
fois l'add-on installe. C'est precisement le trou de couverture que ce script
comble, meme motif que `verify_material_ops.py` / `verify_rigidbody_bake.py`.

Comment on appelle un `draw()` de `Panel` sans fenetre ni region reelle
(`--background` n'a pas de `bpy.context.window`, donc pas de vraie
`UILayout`) : on construit un OBJET FACTICE portant seulement l'attribut
`.layout` (meme principe que `_FakeSelf` de `verify_rigidbody_bake.py`, qui
lie des methodes de production via `types.MethodType` a un objet qui n'a que
les attributs necessaires) et on appelle `cls.draw(fake_self, context)`
directement -- c'est un appel Python ordinaire sur la fonction non liee, le
mecanisme RNA d'invocation des Panels n'entre pas en jeu ici, donc aucune
fenetre n'est necessaire.

`.layout` est un `_MockLayout` : PAS un vrai `bpy.types.UILayout` (on ne peut
pas en construire un hors du systeme de dessin reel), mais il valide contre
de la VRAIE RNA Blender la ou ca compte :
  - `prop(data, name)` verifie `name in data.bl_rna.properties` -- attrape
    un nom de propriete inexistant, l'erreur la plus probable en cablant un
    nouveau panneau ;
  - `label(..., icon=...)` / `operator(..., icon=...)` verifient l'icone
    contre l'enum RNA reel de `UILayout.label` (1033 icones dans cette
    version, voir la sonde faite en amont) -- attrape une faute de frappe
    d'icone (`"MOD_PHYSIC"` au lieu de `"MOD_PHYSICS"`), invisible avec un
    stub qui accepte n'importe quelle chaine ;
  - `operator(idname, ...)` verifie que la classe d'operateur existe dans
    `bpy.types` selon la convention `GROUPE_OT_nom` deduite de `idname`
    (`bq.clear_rigid_keys` -> `BQ_OT_clear_rigid_keys`) -- attrape un idname
    d'operateur mal orthographie ou non enregistre.
Toute autre methode (`box`, `row`, `column`, `split`, `template_list`,
`prop_search`, `separator`, `progress`...) est un no-op generique qui
renvoie un nouveau `_MockLayout` pour permettre le chainage -- suffisant
pour EXERCER le code (chaque branche `if`/`for` du `draw()` s'execute
reellement) sans avoir besoin d'un vrai rendu pixel.

Limite assumee, a rapporter a l'utilisateur : ceci ne verifie ni la mise en
page ni la lisibilite reelle (alignement, largeur de colonne, texte tronque)
-- seulement l'absence d'exception et les invariants de contenu (quelles
proprietes/quels textes apparaissent) explicitement testes ci-dessous. Le
rendu visuel doit etre inspecte dans Blender interactif.
"""

import sys
import types

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

import bpy  # noqa: E402

# scipy est fourni en wheel bundle, installe par le gestionnaire d'extensions
# de Blender -- absent quand on importe l'arbre source en --factory-startup
# (meme garde que verify_material_ops.py / verify_rigidbody_bake.py).
if "scipy" not in sys.modules:
    _s = types.ModuleType("scipy")
    _sp = types.ModuleType("scipy.spatial")
    _sp.cKDTree = object
    _s.spatial = _sp
    sys.modules["scipy"] = _s
    sys.modules["scipy.spatial"] = _sp

import extension  # noqa: E402
from extension import ui  # noqa: E402

extension.register()

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# _MockLayout -- voir docstring du module pour le raisonnement complet
# ---------------------------------------------------------------------------

_VALID_ICONS = set(
    bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys()
)


class _DrawRecorder:
    """Registre PARTAGE par tous les `_MockLayout` produits pendant le
    dessin d'un panneau : accumule les erreurs de validation RNA et les
    traces de contenu (proprietes/textes/operateurs dessines) que les
    scenarios ci-dessous inspectent apres coup."""

    def __init__(self):
        self.errors = []
        self.prop_calls = []  # (type(data).__name__, propname)
        self.label_texts = []
        self.operator_idnames = []

    def check_icon(self, icon, where):
        if icon and icon != "NONE" and icon not in _VALID_ICONS:
            self.errors.append(f"icone inconnue {icon!r} ({where})")


class _MockLayout:
    def __init__(self, recorder):
        self._rec = recorder
        self.enabled = True
        self.use_property_split = False
        self.scale_y = 1.0

    # -- Conteneurs generiques : no-op, renvoient un nouveau mock ----------
    def _child(self, *_args, **_kwargs):
        return _MockLayout(self._rec)

    box = _child
    row = _child
    column = _child
    split = _child
    template_list = _child
    prop_search = _child
    separator = _child
    progress = _child

    def __getattr__(self, _name):
        # Toute autre methode UILayout jamais utilisee explicitement
        # ci-dessus (menu(), operator_menu_enum(), etc.) : no-op generique.
        return self._child

    # -- Methodes validees contre la VRAIE RNA Blender ----------------------
    def prop(self, data, propname, text=None, icon="NONE", **kwargs):
        rna_props = getattr(data.bl_rna, "properties", None)
        found = rna_props is not None and propname in rna_props
        if not found:
            self._rec.errors.append(
                f"prop introuvable : {type(data).__name__}.{propname}"
            )
        self._rec.prop_calls.append((type(data).__name__, propname))
        self._rec.check_icon(icon, f"prop {propname}")
        return self._child()

    def label(self, text="", icon="NONE", **kwargs):
        self._rec.label_texts.append(text)
        self._rec.check_icon(icon, f"label {text!r}")
        return self._child()

    def operator(self, idname, text="", icon="NONE", **kwargs):
        self._rec.operator_idnames.append(idname)
        self._rec.check_icon(icon, f"operator {idname}")
        group, _, name = idname.partition(".")
        expected_cls = f"{group.upper()}_OT_{name}"
        if not hasattr(bpy.types, expected_cls):
            self._rec.errors.append(
                f"operateur introuvable : {idname} (classe attendue "
                f"{expected_cls})"
            )
        # `BQ_UL_elements.draw_item` fait `op = row.operator(...); op.data_path = ...`
        # -- l'objet renvoye par un vrai UILayout.operator() accepte
        # l'assignation de proprietes d'OperatorProperties arbitraires.
        return types.SimpleNamespace()


class _PanelSelf:
    """Objet factice ne portant que `.layout`, comme un vrai `Panel` juste
    avant l'appel a `draw()` -- voir docstring du module."""

    def __init__(self, layout):
        self.layout = layout


def draw_panel(cls, context):
    """Appelle `cls.draw(fake_self, context)` en respectant `poll` comme le
    ferait Blender (un panneau dont le `poll` echoue n'est simplement pas
    dessine). Renvoie le `_DrawRecorder` si le panneau a ete dessine, sinon
    `None`."""
    poll = getattr(cls, "poll", None)
    if poll is not None and not poll(context):
        return None
    rec = _DrawRecorder()
    fake_self = _PanelSelf(_MockLayout(rec))
    cls.draw(fake_self, context)
    return rec


def draw_all_panels(context, label):
    """Exerce le `draw()` de TOUS les panneaux enregistres (pas seulement
    ceux touches par ce jalon) : c'est le seul moyen d'attraper une
    exception de dessin, qui ne se voit pas autrement qu'a l'ecran (cf.
    consigne de la tache). Renvoie `{nom_de_classe: _DrawRecorder}` pour les
    panneaux effectivement dessines (poll passe)."""
    results = {}
    for cls in ui.classes:
        if not (isinstance(cls, type) and issubclass(cls, bpy.types.Panel)):
            continue
        try:
            rec = draw_panel(cls, context)
        except Exception as exc:  # noqa: BLE001 -- on veut nommer le panneau fautif
            check(f"{label} : {cls.__name__}.draw() ne leve pas", False, repr(exc))
            continue
        if rec is not None:
            results[cls.__name__] = rec
            if rec.errors:
                check(
                    f"{label} : {cls.__name__}.draw() sans erreur RNA",
                    False,
                    "; ".join(rec.errors),
                )
    return results


# ---------------------------------------------------------------------------
# Harnais de scene
# ---------------------------------------------------------------------------


def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def make_domain(scene, size=1.0, grid_res=24):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = grid_res
    return domain


def add_collider(location, dims, name, dynamic, density=500.0, scale=None):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale if scale is not None else dims
    bpy.context.view_layer.update()
    obj.bourrasque.role = "COLLIDER"
    obj.bourrasque.dynamic = dynamic
    obj.bourrasque.density = density
    return obj


def add_open_mesh_collider(location, name, dynamic=True):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    bpy.context.view_layer.update()
    obj.bourrasque.role = "COLLIDER"
    obj.bourrasque.dynamic = dynamic
    obj.bourrasque.density = 500.0
    return obj


def select_only(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


# ---------------------------------------------------------------------------
# 1) Scene vide, et scene sans domaine
# ---------------------------------------------------------------------------
print("\n=== 1) Scène vide / sans domaine ===")


def test_empty_scene():
    fresh_scene()
    ctx = bpy.context
    draw_all_panels(ctx, "scène vide")


def test_no_domain_scene():
    scene = fresh_scene()
    bpy.ops.mesh.primitive_cube_add(size=0.5)
    obj = bpy.context.active_object
    obj.name = "SansDomaine"
    obj.bourrasque.role = "COLLIDER"
    select_only(obj)
    draw_all_panels(bpy.context, "sans domaine, collider sélectionné")


test_empty_scene()
test_no_domain_scene()


# ---------------------------------------------------------------------------
# 2) Collider fixe sélectionné : aucune propriété de corps rigide
# ---------------------------------------------------------------------------
print("\n=== 2) Collider fixe sélectionné ===")


def test_fixed_collider_no_rigid_body_props():
    scene = fresh_scene()
    make_domain(scene)
    fixed = add_collider((0, 0, 0), (0.3, 0.3, 0.3), "Fixe", dynamic=False)
    select_only(fixed)

    results = draw_all_panels(bpy.context, "collider fixe")
    rec = results.get("BQ_PT_collider")
    check("collider fixe : le sous-panneau Collider est bien dessiné", rec is not None)
    if rec is None:
        return

    drawn_props = {name for (_type, name) in rec.prop_calls}
    check(
        "collider fixe : `friction` visible",
        "friction" in drawn_props,
        f"props dessinées={drawn_props!r}",
    )
    check(
        "collider fixe : `dynamic` visible (interrupteur)",
        "dynamic" in drawn_props,
        f"props dessinées={drawn_props!r}",
    )
    # M17/phase B (D12, tâche B4) : `restitution` (comme `friction`) régit
    # DÉSORMAIS aussi le contact solide<->solide, où un collider FIXE est un
    # corps de masse infinie comme un autre — il n'est donc plus une "fuite"
    # qu'elle soit visible pour un collider fixe (revirement assumé par
    # rapport à la phase A, où seul le couplage fluide<->solide existait).
    check(
        "collider fixe : `restitution` visible (D12, contact solide<->solide)",
        "restitution" in drawn_props,
        f"props dessinées={drawn_props!r}",
    )
    rigid_body_only_props = {
        "density",
        "use_gravity",
        "lock_location",
        "lock_rotation",
        "added_mass",
    }
    leaked = drawn_props & rigid_body_only_props
    check(
        "collider fixe : aucune propriété EXCLUSIVEMENT dynamique dessinée (D14)",
        not leaked,
        f"fuite={leaked!r}",
    )
    check(
        "collider fixe : aucune mention de masse/volume/stabilité",
        not any(
            ("masse" in t.lower() or "volume" in t.lower() or "stabilité" in t.lower())
            for t in rec.label_texts
        ),
        f"labels={rec.label_texts!r}",
    )


test_fixed_collider_no_rigid_body_props()


# ---------------------------------------------------------------------------
# 3) Collider dynamique, maillage fermé (cas nominal)
# ---------------------------------------------------------------------------
print("\n=== 3) Collider dynamique, maillage fermé ===")


def test_dynamic_collider_closed_mesh():
    scene = fresh_scene()
    make_domain(scene)
    dyn = add_collider(
        (0, 0, 0), (0.3, 0.3, 0.3), "Dyn", dynamic=True, density=500.0
    )
    select_only(dyn)

    results = draw_all_panels(bpy.context, "collider dynamique, maillage fermé")
    rec = results.get("BQ_PT_collider")
    check("dynamique fermé : le sous-panneau Collider est dessiné", rec is not None)
    if rec is None:
        return

    drawn_props = {name for (_type, name) in rec.prop_calls}
    for expected in (
        "friction",
        "dynamic",
        "density",
        "use_gravity",
        "lock_location",
        "lock_rotation",
        "restitution",
    ):
        check(
            f"dynamique fermé : `{expected}` visible",
            expected in drawn_props,
            f"props dessinées={drawn_props!r}",
        )
    check(
        "dynamique fermé : `added_mass` volontairement PAS exposé "
        "(réglage structurellement cassé, voir consigne révisée)",
        "added_mass" not in drawn_props,
        f"props dessinées={drawn_props!r}",
    )
    check(
        "dynamique fermé : aucune mention de « stabilité »",
        not any("stabilité" in t.lower() for t in rec.label_texts),
        f"labels={rec.label_texts!r}",
    )

    check(
        "dynamique fermé : une masse calculée est affichée",
        any("masse calculée" in t.lower() for t in rec.label_texts),
        f"labels={rec.label_texts!r}",
    )
    check(
        "dynamique fermé : cube 0.3³ densité 500 -> devrait flotter",
        any("flotte" in t.lower() for t in rec.label_texts),
        f"labels={rec.label_texts!r}",
    )
    check(
        "dynamique fermé : pas d'avertissement d'échelle non uniforme",
        not any("échelle non uniforme" in t.lower() for t in rec.label_texts),
        f"labels={rec.label_texts!r}",
    )


test_dynamic_collider_closed_mesh()


# ---------------------------------------------------------------------------
# 4) Collider dynamique à échelle NON UNIFORME
# ---------------------------------------------------------------------------
print("\n=== 4) Collider dynamique, échelle non uniforme ===")


def test_dynamic_collider_non_uniform_scale():
    scene = fresh_scene()
    make_domain(scene)
    dyn = add_collider(
        (0, 0, 0), (0.3, 0.3, 0.3), "DynEtire", dynamic=True,
        density=500.0, scale=(0.3, 0.6, 0.3),
    )
    select_only(dyn)

    results = draw_all_panels(bpy.context, "collider dynamique, échelle non uniforme")
    rec = results.get("BQ_PT_collider")
    check(
        "échelle non uniforme : le sous-panneau Collider est dessiné",
        rec is not None,
    )
    if rec is None:
        return
    check(
        "échelle non uniforme : l'avertissement apparaît",
        any("échelle non uniforme" in t.lower() for t in rec.label_texts),
        f"labels={rec.label_texts!r}",
    )


test_dynamic_collider_non_uniform_scale()


# ---------------------------------------------------------------------------
# 5) Collider dynamique au maillage OUVERT : pas de plantage
# ---------------------------------------------------------------------------
print("\n=== 5) Collider dynamique, maillage ouvert ===")


def test_dynamic_collider_open_mesh_does_not_crash():
    scene = fresh_scene()
    make_domain(scene)
    open_obj = add_open_mesh_collider((0, 0, 0), "PlanOuvert", dynamic=True)
    select_only(open_obj)

    results = draw_all_panels(bpy.context, "collider dynamique, maillage ouvert")
    rec = results.get("BQ_PT_collider")
    check(
        "maillage ouvert : le sous-panneau Collider est dessiné sans exception",
        rec is not None,
    )


test_dynamic_collider_open_mesh_does_not_crash()


# ---------------------------------------------------------------------------
# 6) Tableau de bord : 0, 1, puis 3 corps dynamiques
# ---------------------------------------------------------------------------
print("\n=== 6) Tableau de bord : 0 / 1 / 3 corps dynamiques ===")


def test_dashboard_rigid_bodies(n_dynamic):
    scene = fresh_scene()
    make_domain(scene)
    names = []
    for i in range(n_dynamic):
        obj = add_collider(
            (i * 0.5, 0, 0), (0.2, 0.2, 0.2), f"Dyn{i}", dynamic=True,
            density=500.0 + i * 400.0,
        )
        names.append(obj.name)
    # Un collider fixe en plus, pour verifier qu'il n'est jamais compte.
    add_collider((0, 2, 0), (0.5, 0.5, 0.5), "Fixe", dynamic=False)

    results = draw_all_panels(bpy.context, f"tableau de bord ({n_dynamic} dynamique(s))")
    rec = results.get("BQ_PT_dashboard")
    check(
        f"dashboard {n_dynamic} corps : dessiné sans exception",
        rec is not None,
    )
    if rec is None:
        return

    if n_dynamic == 0:
        check(
            "dashboard 0 corps : pas de section « Corps rigides »",
            not any("corps rigides" in t.lower() for t in rec.label_texts),
            f"labels={rec.label_texts!r}",
        )
        return

    check(
        f"dashboard {n_dynamic} corps : section « Corps rigides » présente",
        any("corps rigides" in t.lower() for t in rec.label_texts),
        f"labels={rec.label_texts!r}",
    )
    check(
        f"dashboard {n_dynamic} corps : le compte ({n_dynamic}) apparaît",
        any(str(n_dynamic) == t for t in rec.label_texts),
        f"labels={rec.label_texts!r}",
    )
    check(
        f"dashboard {n_dynamic} corps : bq.clear_rigid_keys proposé",
        "bq.clear_rigid_keys" in rec.operator_idnames,
        f"opérateurs={rec.operator_idnames!r}",
    )
    check(
        f"dashboard {n_dynamic} corps : le collider fixe n'apparaît jamais",
        "Fixe" not in rec.label_texts,
        f"labels={rec.label_texts!r}",
    )


test_dashboard_rigid_bodies(0)
test_dashboard_rigid_bodies(1)
test_dashboard_rigid_bodies(3)


# ---------------------------------------------------------------------------
# 7) Tableau de bord : colliders au maillage non fermé (M17/phase B, B4)
# ---------------------------------------------------------------------------
print("\n=== 7) Tableau de bord : colliders au maillage non fermé (B4) ===")


def test_dashboard_open_mesh_colliders():
    from extension import ops

    scene = fresh_scene()
    make_domain(scene)
    add_collider((0, 0, 0), (0.3, 0.3, 0.3), "Fixe", dynamic=False)

    # Sans bake prealable : rien ne doit apparaitre (cache vide, D9/point 3
    # de B4 -- jamais recalcule depuis un draw()).
    results = draw_all_panels(bpy.context, "dashboard sans bake")
    rec = results.get("BQ_PT_dashboard")
    check(
        "sans bake : aucune mention de maillage non fermé",
        rec is not None
        and not any("non fermé" in t.lower() for t in rec.label_texts),
        f"labels={rec.label_texts if rec else None!r}",
    )

    # Simule un bake qui a memorise un collider statique au maillage ouvert
    # (motif reel : BQ_OT_bake.invoke ecrit ops._OPEN_MESH_COLLIDERS[scene.name]).
    ops._OPEN_MESH_COLLIDERS[scene.name] = ("OpenPlane",)
    try:
        results = draw_all_panels(bpy.context, "dashboard avec maillage ouvert mémorisé")
        rec = results.get("BQ_PT_dashboard")
        check(
            "dessiné sans exception ni erreur RNA (icônes réelles)",
            rec is not None,
        )
        check(
            "le nom du collider apparaît",
            rec is not None and any("OpenPlane" in t for t in rec.label_texts),
            f"labels={rec.label_texts if rec else None!r}",
        )
        check(
            "mention explicite « non fermé »",
            rec is not None
            and any("non fermé" in t.lower() for t in rec.label_texts),
            f"labels={rec.label_texts if rec else None!r}",
        )
    finally:
        ops._OPEN_MESH_COLLIDERS.pop(scene.name, None)


test_dashboard_open_mesh_colliders()


# ---------------------------------------------------------------------------
# 8) Sous-panneau Collider : rappel par-objet (maillage non fermé mémorisé)
# ---------------------------------------------------------------------------
print("\n=== 8) Sous-panneau Collider : rappel par-objet (B4) ===")


def test_collider_panel_open_mesh_reminder():
    from extension import ops

    scene = fresh_scene()
    make_domain(scene)
    plane = add_open_mesh_collider((0, 0, 0), "OpenPlaneObj", dynamic=False)
    select_only(plane)

    ops._OPEN_MESH_COLLIDERS[scene.name] = ("OpenPlaneObj",)
    try:
        results = draw_all_panels(bpy.context, "collider statique ouvert mémorisé")
        rec = results.get("BQ_PT_collider")
        check(
            "dessiné sans exception ni erreur RNA",
            rec is not None,
        )
        check(
            "rappel « non fermé » affiché sur l'objet lui-même",
            rec is not None
            and any("non fermé" in t.lower() for t in rec.label_texts),
            f"labels={rec.label_texts if rec else None!r}",
        )
    finally:
        ops._OPEN_MESH_COLLIDERS.pop(scene.name, None)


test_collider_panel_open_mesh_reminder()


print()
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
    sys.exit(1)
print("Toutes les vérifications M17 (UI, phase A) sont passées.")
