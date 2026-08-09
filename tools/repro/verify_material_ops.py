"""Non-regression des OPERATEURS de la bibliotheque de materiaux (jalon M16).

A lancer en Blender reel :
    blender --background --factory-startup --python tools/repro/verify_material_ops.py

Pourquoi ce script vit ici et pas dans `extension/tests/` : tout ce qu'il
verifie passe par `bpy.ops` et des PropertyGroups reellement enregistres. Les
tests de `extension/tests/` tournent sous un stub `bpy` minimal (pas
d'operateurs, pas de RNA) — ils couvrent le module PUR `extension/materials.py`
mais ne peuvent structurellement pas atteindre ces chemins.

C'est precisement ce trou de couverture qui a laisse passer l'ecart 1 de la
revue d'architecture M16 (perte de donnees silencieuse dans
`bq.migrate_materials`). Les quatre cas ci-dessous verrouillent les corrections
apportees en reponse a cette revue.
"""

import sys
import types

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

import bpy

# scipy est fourni en wheel bundle, installe par le gestionnaire d'extensions
# de Blender — absent quand on importe l'arbre source en --factory-startup.
# Aucun des chemins testes ici ne le touche.
if "scipy" not in sys.modules:
    _s = types.ModuleType("scipy")
    _sp = types.ModuleType("scipy.spatial")
    _sp.cKDTree = object
    _s.spatial = _sp
    sys.modules["scipy"] = _s
    sys.modules["scipy.spatial"] = _sp

from extension import display, ops, overlay, props, ui

FAILURES = []


def check(name, condition, detail=""):
    print(f"[{'OK' if condition else 'FAIL'}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def _add_material(scene, name, model, rho, a, b):
    mat = scene.bourrasque.materials.add()
    mat.model = model
    mat.rho = rho
    if model == "WATER":
        mat.bulk, mat.gamma = a, b
    else:
        mat.young, mat.poisson = a, b
    mat.name_prev = name
    mat.name = name
    return mat


def _add_emitter_obj(name, material_name):
    bpy.ops.mesh.primitive_cube_add(size=0.5)
    obj = bpy.context.active_object
    obj.name = name
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.material_name = material_name
    return obj


def test_migrate_preserves_valid_assignments():
    """Ecart 1 : `bq.migrate_materials` est expose dans l'UI sous le libelle
    « Reparer », declenche des qu'UN emetteur est pendant — ce qui arrive
    aussi dans une scene entierement post-M16, apres un `material_remove`.
    Il ne doit toucher QUE les emetteurs pendants.

    Avant correction, il reassignait TOUS les emetteurs depuis leurs champs
    DEPRECIES (qui ne sont plus ecrits depuis M16, donc identiques aux
    defauts pour tout le monde) : les trois emetteurs s'effondraient sur un
    unique materiau, detruisant deux assignations manuelles valides."""
    scene = _reset_scene()
    _add_material(scene, "Eau", "WATER", 1000.0, 4.0e4, 3.0)
    _add_material(scene, "Gelee", "ELASTIC", 1100.0, 7.7e4, 0.3)
    _add_material(scene, "SiropCustom", "WATER", 1400.0, 9.9e4, 6.0)

    a = _add_emitter_obj("EmA", "SiropCustom")
    b = _add_emitter_obj("EmB", "Gelee")
    c = _add_emitter_obj("EmC", "Eau")

    scene.bourrasque.active_material_index = list(
        scene.bourrasque.materials.keys()
    ).index("Gelee")
    bpy.ops.bq.material_remove()

    bpy.ops.bq.migrate_materials()

    check(
        "migrate : assignation valide preservee (EmA)",
        a.bourrasque.material_name == "SiropCustom",
        f"got={a.bourrasque.material_name!r}",
    )
    check(
        "migrate : assignation valide preservee (EmC)",
        c.bourrasque.material_name == "Eau",
        f"got={c.bourrasque.material_name!r}",
    )
    check(
        "migrate : l'emetteur pendant est bien repare (EmB)",
        b.bourrasque.material_name in {m.name for m in scene.bourrasque.materials},
        f"got={b.bourrasque.material_name!r}",
    )


def test_migrate_is_idempotent():
    scene = _reset_scene()
    _add_material(scene, "Eau", "WATER", 1000.0, 4.0e4, 3.0)
    _add_emitter_obj("Em", "")
    bpy.ops.bq.migrate_materials()
    n_after_first = len(scene.bourrasque.materials)
    bpy.ops.bq.migrate_materials()
    check(
        "migrate : idempotent (aucun doublon au second appel)",
        len(scene.bourrasque.materials) == n_after_first,
        f"{n_after_first} -> {len(scene.bourrasque.materials)}",
    )


def test_add_emitter_uses_active_material():
    """L'artiste qui vient de selectionner « Sable » dans la bibliotheque
    attend que son nouvel emetteur soit du sable, pas la premiere entree."""
    scene = _reset_scene()
    _add_material(scene, "Eau", "WATER", 1000.0, 4.0e4, 3.0)
    _add_material(scene, "Sable", "ELASTIC", 1600.0, 3.0e5, 0.3)
    scene.bourrasque.active_material_index = 1  # « Sable »

    bpy.ops.mesh.primitive_cube_add(size=0.5)
    bpy.context.active_object.name = "Nouveau"
    bpy.ops.bq.add_emitter()

    check(
        "add_emitter : prend le materiau ACTIF, pas library[0]",
        bpy.context.active_object.bourrasque.material_name == "Sable",
        f"got={bpy.context.active_object.bourrasque.material_name!r}",
    )


def test_add_emitter_repairs_dangling_reference():
    """Un objet qui fut emetteur, dont le role a ete retire puis le materiau
    supprime, conserve un `material_name` NON VIDE mais pendant. Le repasser
    emetteur ne doit pas le laisser avec cette reference morte."""
    scene = _reset_scene()
    _add_material(scene, "Eau", "WATER", 1000.0, 4.0e4, 3.0)

    bpy.ops.mesh.primitive_cube_add(size=0.5)
    obj = bpy.context.active_object
    obj.name = "Recycle"
    obj.bourrasque.role = "NONE"
    obj.bourrasque.material_name = "MateriauSupprime"  # pendant, non vide

    bpy.ops.bq.add_emitter()
    check(
        "add_emitter : repare une reference pendante non vide",
        obj.bourrasque.material_name == "Eau",
        f"got={obj.bourrasque.material_name!r}",
    )


def test_model_change_resets_preset():
    """Cause racine du bug de nommage attrape en M16 : `preset` et `model`
    pouvaient se contredire silencieusement."""
    scene = _reset_scene()
    mat = _add_material(scene, "Eau", "WATER", 1000.0, 4.0e4, 3.0)
    mat.preset = "WATER"
    mat.model = "ELASTIC"
    check(
        "model divergent du preset -> preset bascule sur CUSTOM",
        mat.preset == "CUSTOM",
        f"got={mat.preset!r}",
    )


def test_slot_count_matches_bake_semantics():
    """`used_material_slot_count` doit compter comme le bake (dedup par
    `material_key`), pas compter les NOMS : deux materiaux nommes
    differemment mais physiquement identiques occupent UN seul emplacement."""
    scene = _reset_scene()
    _add_material(scene, "EauA", "WATER", 1000.0, 4.0e4, 3.0)
    _add_material(scene, "EauB", "WATER", 1000.0, 4.0e4, 3.0)  # identique
    _add_material(scene, "Inutilise", "ELASTIC", 1100.0, 7.7e4, 0.3)
    _add_emitter_obj("Em1", "EauA")
    _add_emitter_obj("Em2", "EauB")

    n = props.used_material_slot_count(scene)
    check(
        "slot count : 2 noms identiques = 1 emplacement, inutilise exclu",
        n == 1,
        f"got={n} (attendu 1)",
    )


if __name__ == "__main__":
    for mod in (props, ops, ui, overlay, display):
        mod.register()

    test_migrate_preserves_valid_assignments()
    test_migrate_is_idempotent()
    test_add_emitter_uses_active_material()
    test_add_emitter_repairs_dangling_reference()
    test_model_change_resets_preset()
    test_slot_count_matches_bake_semantics()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} echec(s) : {FAILURES}")
        sys.exit(1)
    print("Tous les tests d'operateurs materiau sont passes.")
