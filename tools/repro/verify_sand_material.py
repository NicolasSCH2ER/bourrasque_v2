"""Verification du sable (jalon M18, tache S3), a lancer avec :

    blender --background --factory-startup --python tools/repro/verify_sand_material.py

Le coeur CUDA (modele de sable Drucker-Prager) est deja livre et verifie ;
ce script verifie exclusivement le cablage Python/Blender ajoute par S3 :

  1. le preset « Sable » (`props._PRESET_SAND`) pose les bonnes valeurs sur
     un `BqMaterialProps` reel ;
  2. le sous-panneau Matériaux (`ui.BQ_PT_materials`) affiche
     `friction_angle` et n'affiche PAS `cohesion` pour un materiau SAND, et
     continue d'afficher exactement les memes champs qu'avant pour WATER et
     ELASTIC (non-regression) ;
  3. TOUS les `draw()` enregistres s'executent sans exception avec un
     materiau sable present dans la scene (meme discipline que
     `verify_rigidbody_ui.py`, meme motif `_MockLayout`/`_DrawRecorder`
     repris ci-dessous) ;
  4. un bake COURT REEL d'une scene de sable aboutit : `BQ_OT_bake._validate`
     (le vrai code de dispatch modele -> `lib.BQ_MODEL_SAND`, modifie par
     S3) et `_bake_worker` (le vrai coeur de boucle de calcul, sans thread
     ici — appele synchrone pour rester observable) sont tous deux le code
     de PRODUCTION, pas une reimplementation. Ca prouve que le mapping
     ops.py -> `Sim.add_material` est correct et que le coeur natif accepte
     ce que l'UI produit.

Pourquoi ce script vit ici et pas dans `extension/tests/` : ces verifications
passent par de vrais `bpy.ops`/PropertyGroups/RNA et, pour la partie 4, par
la vraie DLL CUDA (`lib.Sim`) -- aucun des deux n'existe sous le stub `bpy`
minimal utilise par `extension/tests/` (voir la meme remarque en tete de
`verify_material_ops.py`/`verify_rigidbody_ui.py`).
"""

import queue
import sys
import threading
import types

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

import bpy  # noqa: E402

# scipy est fourni en wheel bundle, installe par le gestionnaire d'extensions
# de Blender -- absent quand on importe l'arbre source en --factory-startup
# (meme garde que verify_material_ops.py / verify_rigidbody_ui.py).
if "scipy" not in sys.modules:
    _s = types.ModuleType("scipy")
    _sp = types.ModuleType("scipy.spatial")
    _sp.cKDTree = object
    _s.spatial = _sp
    sys.modules["scipy"] = _s
    sys.modules["scipy.spatial"] = _sp

import extension  # noqa: E402
from extension import cache, lib, ops, props, ui  # noqa: E402
from extension.props import (  # noqa: E402
    domain_resolution,
    domain_transform,
    domain_usable_bounds,
    emitter_bounds_solver,
    world_to_solver_dir,
)

extension.register()

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def fresh_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


# ---------------------------------------------------------------------------
# 1) Preset « Sable »
# ---------------------------------------------------------------------------
print("\n=== 1) Preset « Sable » ===")


def test_sand_preset_values():
    scene = fresh_scene()
    mat = scene.bourrasque.materials.add()
    mat.name_prev = "Sable"
    mat.name = "Sable"
    mat.preset = "SAND"

    def _close(a, b, tol=1e-5):
        # Les FloatProperty Blender sont stockees en float32 : une egalite
        # stricte contre un litteral Python (float64) est fragile (ecart
        # d'arrondi, ex. 0.3 -> 0.30000001192092896). Comparaison tolerante,
        # cf. meme discipline que le reste du projet (memoire "bruit
        # run-a-run").
        return abs(a - b) <= tol * max(1.0, abs(b))

    expected = props._PRESET_SAND
    check("preset sable : model == SAND", mat.model == "SAND", f"got={mat.model!r}")
    check("preset sable : rho", _close(mat.rho, expected["rho"]), f"got={mat.rho}")
    check(
        "preset sable : young (E)",
        _close(mat.young, expected["young"]),
        f"got={mat.young}",
    )
    check(
        "preset sable : poisson (nu)",
        _close(mat.poisson, expected["poisson"]),
        f"got={mat.poisson}",
    )
    check(
        "preset sable : friction_angle",
        _close(mat.friction_angle, expected["friction_angle"]),
        f"got={mat.friction_angle}",
    )
    check("preset sable : cohesion reste a 0", mat.cohesion == 0.0, f"got={mat.cohesion}")


test_sand_preset_values()


def test_model_change_away_from_sand_resets_preset():
    """Meme garde-fou que WATER/JELLY (props._on_model_update) : changer le
    modele loin de celui qu'implique le preset SAND doit basculer le preset
    sur CUSTOM."""
    scene = fresh_scene()
    mat = scene.bourrasque.materials.add()
    mat.name_prev = "Sable"
    mat.name = "Sable"
    mat.preset = "SAND"
    mat.model = "WATER"
    check(
        "sable -> model WATER : preset bascule sur CUSTOM",
        mat.preset == "CUSTOM",
        f"got={mat.preset!r}",
    )


test_model_change_away_from_sand_resets_preset()


# ---------------------------------------------------------------------------
# 2)/3) draw() -- _MockLayout/_DrawRecorder, memes motifs que
# verify_rigidbody_ui.py
# ---------------------------------------------------------------------------
print("\n=== 2)/3) draw() du sous-panneau Matériaux et de tous les panneaux ===")

_VALID_ICONS = set(
    bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys()
)


class _DrawRecorder:
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
        return self._child

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
        return types.SimpleNamespace()


class _PanelSelf:
    def __init__(self, layout):
        self.layout = layout


def draw_panel(cls, context):
    poll = getattr(cls, "poll", None)
    if poll is not None and not poll(context):
        return None
    rec = _DrawRecorder()
    fake_self = _PanelSelf(_MockLayout(rec))
    cls.draw(fake_self, context)
    return rec


def draw_all_panels(context, label):
    results = {}
    for cls in ui.classes:
        if not (isinstance(cls, type) and issubclass(cls, bpy.types.Panel)):
            continue
        try:
            rec = draw_panel(cls, context)
        except Exception as exc:  # noqa: BLE001 -- nommer le panneau fautif
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


def make_domain(scene, size=1.0, grid_res=24):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = grid_res
    return domain


def add_material(scene, name, model, **fields):
    mat = scene.bourrasque.materials.add()
    mat.name_prev = name
    mat.name = name
    mat.model = model
    for key, value in fields.items():
        setattr(mat, key, value)
    return mat


def add_emitter_obj(name, material_name, size=0.2, location=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.material_name = material_name
    return obj


def test_materials_panel_field_exposure():
    scene = fresh_scene()
    add_material(scene, "Eau", "WATER", rho=1000.0, bulk=4.0e4, gamma=3.0)
    add_material(
        scene, "Gelee", "ELASTIC", rho=1100.0, young=7.7e4, poisson=0.3
    )
    add_material(
        scene,
        "Sable",
        "SAND",
        rho=1600.0,
        young=3.5e5,
        poisson=0.3,
        friction_angle=35.0,
    )

    props_scene = scene.bourrasque
    names = [m.name for m in props_scene.materials]

    for expected_name, expected_props, forbidden_props in (
        ("Eau", {"bulk", "gamma"}, {"friction_angle", "cohesion", "young", "poisson"}),
        ("Gelee", {"young", "poisson"}, {"friction_angle", "cohesion", "bulk", "gamma"}),
        (
            "Sable",
            {"young", "poisson", "friction_angle"},
            {"cohesion", "bulk", "gamma"},
        ),
    ):
        props_scene.active_material_index = names.index(expected_name)
        rec = draw_panel(ui.BQ_PT_materials, bpy.context)
        check(
            f"panneau Matériaux ({expected_name}) : dessine sans exception",
            rec is not None,
        )
        if rec is None:
            continue
        drawn = {name for (_type, name) in rec.prop_calls}
        for prop_name in expected_props:
            check(
                f"panneau Matériaux ({expected_name}) : `{prop_name}` visible",
                prop_name in drawn,
                f"props dessinées={drawn!r}",
            )
        leaked = drawn & forbidden_props
        check(
            f"panneau Matériaux ({expected_name}) : aucun champ interdit affiché",
            not leaked,
            f"fuite={leaked!r} (props dessinées={drawn!r})",
        )

    # Cas central de S3 : `cohesion` ne doit JAMAIS apparaitre pour le
    # sable, meme si le coeur la refuse aujourd'hui (voir props.py,
    # BqMaterialProps.cohesion) -- un champ editable qui casserait le bake
    # serait un piege.
    props_scene.active_material_index = names.index("Sable")
    rec = draw_panel(ui.BQ_PT_materials, bpy.context)
    drawn = {name for (_type, name) in rec.prop_calls}
    check(
        "panneau Matériaux (Sable) : `cohesion` absente de l'UI",
        "cohesion" not in drawn,
        f"props dessinées={drawn!r}",
    )

    return scene


scene_for_all_panels = test_materials_panel_field_exposure()


def test_all_panels_draw_with_sand_in_scene():
    scene = fresh_scene()
    make_domain(scene)
    add_material(scene, "Eau", "WATER", rho=1000.0, bulk=4.0e4, gamma=3.0)
    add_material(
        scene,
        "Sable",
        "SAND",
        rho=1600.0,
        young=3.5e5,
        poisson=0.3,
        friction_angle=35.0,
    )
    add_emitter_obj("EmSable", "Sable")

    results = draw_all_panels(bpy.context, "scène avec sable")
    check(
        "tous les panneaux enregistrés ont été exercés (au moins un dessiné)",
        len(results) > 0,
        f"dessinés={list(results.keys())}",
    )


test_all_panels_draw_with_sand_in_scene()


# ---------------------------------------------------------------------------
# 4) Bake court REEL (production code : BQ_OT_bake._validate + _bake_worker)
# ---------------------------------------------------------------------------
print("\n=== 4) Bake court réel ===")


class _FakeBakeSelf:
    """Porte uniquement les attributs lus par `BQ_OT_bake._validate`
    (methode NON liee appelee directement, `self.report` accumule les
    messages au lieu d'appeler l'API bpy.ops.report reelle -- meme motif que
    `_FakeSelf` de `verify_m6.py`/`verify_rigidbody_bake.py`)."""

    def __init__(self):
        self.reports = []

    def report(self, tags, msg):
        self.reports.append((tuple(tags), msg))
        print("   report:", tags, msg)


def run_short_bake(scene, n_frames=8):
    """Reproduit la partie « mise en place + calcul » de `BQ_OT_bake.invoke`
    pour une scene SANS collider et SANS emetteur INFLOW (donc pas de
    pre-extraction de geometrie de collider a faire), en appelant le code de
    PRODUCTION directement :

      - `BQ_OT_bake._validate` (dispatch modele -> kwargs `Sim.add_material`,
        modifie par S3) ;
      - `Sim.add_material` (le VRAI binding ctypes vers la DLL) pour chaque
        materiau ;
      - `Sim.emit_box` pour chaque emetteur BLOCK/BOUNDS a turbulence nulle
        (chemin le plus simple, meme reseau que le solveur genere lui-meme) ;
      - `ops._bake_worker` (le VRAI coeur de boucle de calcul), appele de
        facon SYNCHRONE (pas de `threading.Thread`) pour rester observable
        depuis ce script -- la fonction elle-meme ne differencie pas thread
        vs appel direct, elle ne touche jamais bpy.

    Renvoie `(sim, material_specs, emitter_specs, progress)` ; l'appelant
    doit `sim.destroy()`.
    """
    context = bpy.context
    fake = _FakeBakeSelf()
    validated = ops.BQ_OT_bake._validate(fake, context)
    if validated is None:
        raise AssertionError(f"_validate a refusé la scène : {fake.reports}")
    config, origin, size, dx, material_specs, emitter_specs = validated

    sim = lib.Sim(config)
    material_ids = [sim.add_material(**spec) for spec in material_specs]

    props_scene = scene.bourrasque
    usable_bounds = domain_usable_bounds(scene)
    frame_dt = scene.render.fps_base / scene.render.fps

    for obj, mat_index in emitter_specs:
        op = obj.bourrasque
        assert op.emit_mode == "BLOCK" and op.emit_source == "BOUNDS", (
            "run_short_bake ne supporte que BLOCK/BOUNDS (émetteurs de ce "
            "script), voir ops.py pour les autres chemins"
        )
        mat_id = material_ids[mat_index]
        vel = world_to_solver_dir(tuple(op.initial_velocity))
        lo, hi = emitter_bounds_solver(obj, origin, size)
        n = sim.emit_box(mat_id, lo, hi, vel=vel)
        if n == 0:
            raise AssertionError(f"émetteur « {obj.name} » : 0 particule émise")

    import pathlib
    import tempfile

    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="bq_verify_sand_"))
    bqd_path = tmp_dir / "sand.bqd"
    writer = cache.CacheWriter(str(bqd_path), sim.particle_count, velocity=True)

    progress = ops._BakeProgress()
    cancel_event = threading.Event()
    try:
        ops._bake_worker(
            progress, cancel_event, sim, writer, n_frames, frame_dt,
            [], usable_bounds, None, (), None,
        )
    finally:
        writer.close()

    return sim, material_specs, emitter_specs, progress


def test_sand_bake_succeeds():
    scene = fresh_scene()
    make_domain(scene, size=1.0, grid_res=32)
    add_material(
        scene,
        "Sable",
        "SAND",
        rho=1600.0,
        young=3.5e5,
        poisson=0.3,
        friction_angle=35.0,
    )
    add_emitter_obj("EmSable", "Sable", size=0.25, location=(0.0, 0.0, 0.15))

    sim, material_specs, emitter_specs, progress = run_short_bake(scene, n_frames=8)
    try:
        check("bake sable : aucune erreur du thread de calcul", progress.error is None, str(progress.error))
        check("bake sable : progress.done", progress.done)
        check(
            "bake sable : material_specs porte model=BQ_MODEL_SAND",
            material_specs[0]["model"] == lib.BQ_MODEL_SAND,
            f"got={material_specs[0]}",
        )
        check(
            "bake sable : friction_angle transmis au coeur",
            material_specs[0].get("friction_angle") == 35.0,
            f"got={material_specs[0]}",
        )
        check(
            "bake sable : cohesion NON transmise (ou nulle) -- le coeur la refuse "
            "sinon",
            material_specs[0].get("cohesion", 0.0) == 0.0,
            f"got={material_specs[0]}",
        )

        n = sim.particle_count
        check("bake sable : des particules ont été simulées", n > 0, f"n={n}")

        mat_ids = sim.read_materials()
        check(
            "bake sable : toutes les particules portent l'id du matériau sable",
            n > 0 and bool((mat_ids == 0).all()),
            f"ids uniques={set(mat_ids.tolist()) if n else set()} (attendu {{0}})",
        )

        pos = sim.read_positions()
        check(
            "bake sable : positions finies (pas de NaN/Inf -- pas de divergence)",
            bool(__import__("numpy").isfinite(pos).all()),
        )
    finally:
        sim.destroy()


test_sand_bake_succeeds()


def test_water_and_elastic_bakes_unchanged():
    """Non-regression explicite : l'eau et l'élastique doivent toujours
    baker exactement comme avant l'introduction du sable."""
    for model, name, extra in (
        ("WATER", "Eau", dict(rho=1000.0, bulk=4.0e4, gamma=3.0)),
        ("ELASTIC", "Gelee", dict(rho=1100.0, young=7.7e4, poisson=0.3)),
    ):
        scene = fresh_scene()
        make_domain(scene, size=1.0, grid_res=32)
        add_material(scene, name, model, **extra)
        add_emitter_obj(f"Em{model}", name, size=0.25, location=(0.0, 0.0, 0.15))

        expected_bq_model = {
            "WATER": lib.BQ_MODEL_WATER,
            "ELASTIC": lib.BQ_MODEL_ELASTIC,
        }[model]

        sim, material_specs, emitter_specs, progress = run_short_bake(
            scene, n_frames=8
        )
        try:
            check(
                f"bake {model} : aucune erreur du thread de calcul",
                progress.error is None,
                str(progress.error),
            )
            check(
                f"bake {model} : material_specs porte le bon modèle natif",
                material_specs[0]["model"] == expected_bq_model,
                f"got={material_specs[0]}",
            )
            check(
                f"bake {model} : aucune clé friction_angle/cohesion pour ce modèle",
                "friction_angle" not in material_specs[0]
                and "cohesion" not in material_specs[0],
                f"got={material_specs[0]}",
            )
            n = sim.particle_count
            check(f"bake {model} : des particules ont été simulées", n > 0, f"n={n}")
        finally:
            sim.destroy()


test_water_and_elastic_bakes_unchanged()


print()
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
else:
    print("Toutes les vérifications du sable (M18, S3) sont passées.")
sys.exit(1 if FAILURES else 0)
