"""test_materials.py — verification autonome de `extension/materials.py`
(bibliotheque de materiaux nommee au niveau scene, refonte UI).

Executable sans Blender : `python extension/tests/test_materials.py`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from materials import (  # noqa: E402
    collect_used_materials,
    material_key,
    plan_migration,
    unique_name,
)

FAILURES = []


def check(name, condition, detail=""):
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def _water(rho=1000.0, bulk=2.2e9, gamma=7.0, young=0.0, poisson=0.0):
    return dict(model="WATER", rho=rho, young=young, poisson=poisson, bulk=bulk, gamma=gamma)


def _elastic(rho=1000.0, young=1e5, poisson=0.3, bulk=0.0, gamma=0.0):
    return dict(model="ELASTIC", rho=rho, young=young, poisson=poisson, bulk=bulk, gamma=gamma)


def _sand(rho=1600.0, young=3.5e5, poisson=0.3, friction_angle=35.0, bulk=0.0, gamma=0.0):
    return dict(
        model="SAND",
        rho=rho,
        young=young,
        poisson=poisson,
        bulk=bulk,
        gamma=gamma,
        friction_angle=friction_angle,
    )


# -- material_key -----------------------------------------------------------


def test_material_key_water_ignores_young_poisson():
    a = _water(young=1.0, poisson=0.1)
    b = _water(young=99.0, poisson=0.9)
    check(
        "WATER: young/poisson non pertinents -> meme cle",
        material_key(a) == material_key(b),
        f"{material_key(a)} vs {material_key(b)}",
    )


def test_material_key_water_differs_by_bulk():
    a = _water(bulk=2.2e9)
    b = _water(bulk=1.0e9)
    check("WATER: bulk different -> cle differente", material_key(a) != material_key(b))


def test_material_key_elastic_vs_water_same_rho():
    a = _elastic(rho=1000.0)
    b = _water(rho=1000.0)
    check(
        "ELASTIC vs WATER de meme rho -> cles differentes",
        material_key(a) != material_key(b),
    )


def test_material_key_sand_same_params_same_key():
    a = _sand(friction_angle=35.0)
    b = _sand(friction_angle=35.0)
    check("SAND: memes parametres -> meme cle", material_key(a) == material_key(b))


def test_material_key_sand_differs_by_friction_angle():
    """L'angle de frottement DOIT entrer dans la cle de deduplication : deux
    sables qui ne different que par cet angle sont deux materiaux distincts
    pour le solveur, les confondre serait une perte silencieuse."""
    a = _sand(friction_angle=35.0)
    b = _sand(friction_angle=40.0)
    check(
        "SAND: angle de frottement different -> cle differente",
        material_key(a) != material_key(b),
        f"{material_key(a)} vs {material_key(b)}",
    )


def test_material_key_sand_vs_elastic_same_rho_young_poisson():
    a = _sand(rho=1000.0, young=1e5, poisson=0.3)
    b = _elastic(rho=1000.0, young=1e5, poisson=0.3)
    check(
        "SAND vs ELASTIC de memes rho/young/poisson -> cles differentes",
        material_key(a) != material_key(b),
    )


def test_material_key_unknown_model_raises():
    """Dispatch EXHAUSTIF : un modele non reconnu doit lever bruyamment,
    jamais retomber silencieusement sur WATER."""
    try:
        material_key(dict(model="PLASMA", rho=1.0))
    except ValueError:
        check("modele inconnu -> ValueError", True)
    else:
        check("modele inconnu -> ValueError", False, "aucune exception levee")


# -- unique_name --------------------------------------------------------------


def test_unique_name_free_unchanged():
    check("nom libre -> inchange", unique_name("Eau", []) == "Eau")


def test_unique_name_single_collision():
    check("collision simple -> .001", unique_name("Eau", ["Eau"]) == "Eau.001")


def test_unique_name_multiple_collisions():
    check(
        "collisions multiples -> .002",
        unique_name("Eau", ["Eau", "Eau.001"]) == "Eau.002",
    )


def test_unique_name_empty_falls_back_to_default():
    name = unique_name("", [])
    check("nom vide -> nom par defaut", name != "", f"name={name!r}")


def test_unique_name_empty_still_deduplicated():
    default = unique_name("", [])
    name2 = unique_name("   ", [default])
    check(
        "nom vide, defaut deja pris -> desambiguise",
        name2 == f"{default}.001",
        f"default={default!r} name2={name2!r}",
    )


# -- plan_migration -----------------------------------------------------------


def test_plan_migration_empty_list():
    materials, assignment = plan_migration([])
    check("liste vide -> ([], {})", materials == [] and assignment == {})


def test_plan_migration_identical_emitters_share_one_material():
    e1 = dict(name="Emitter1", **_water())
    e2 = dict(name="Emitter2", **_water())
    materials, assignment = plan_migration([e1, e2])
    check("2 emetteurs identiques -> 1 seul materiau", len(materials) == 1)
    check(
        "les deux emetteurs pointent sur le meme materiau",
        assignment["Emitter1"] == assignment["Emitter2"] == materials[0]["name"],
    )


def test_plan_migration_different_emitters_produce_distinct_unique_names():
    e1 = dict(name="Emitter1", **_water(bulk=2.2e9))
    e2 = dict(name="Emitter2", **_water(bulk=1.0e9))
    materials, assignment = plan_migration([e1, e2])
    check("2 emetteurs differents -> 2 materiaux", len(materials) == 2)
    names = [m["name"] for m in materials]
    check("noms de materiaux uniques", len(set(names)) == 2, f"names={names}")
    check(
        "assignation coherente avec les materiaux produits",
        assignment["Emitter1"] in names and assignment["Emitter2"] in names,
    )


def test_plan_migration_preset_labels():
    # Les deux emetteurs sont COHERENTS (preset <-> modele implique) : c'est
    # le cas nominal ou le libelle du preset doit primer. Le preset JELLY
    # implique un modele ELASTIC (cf. `_PRESET_JELLY` dans props.py) — le
    # marier a un modele WATER serait la situation incoherente couverte par
    # `test_plan_migration_preset_inconsistent_with_model` ci-dessous.
    e_water = dict(name="E1", preset="WATER", **_water())
    e_jelly = dict(name="E2", preset="JELLY", **_elastic(young=5.0e4))
    materials, assignment = plan_migration([e_water, e_jelly])
    names_by_emitter = {
        emitter_name: mat_name for emitter_name, mat_name in assignment.items()
    }
    check(
        "preset WATER -> libelle 'Eau'",
        names_by_emitter["E1"].startswith("Eau"),
        f"got={names_by_emitter['E1']!r}",
    )
    check(
        "preset JELLY -> libelle 'Gelée'",
        names_by_emitter["E2"].startswith("Gelée"),
        f"got={names_by_emitter['E2']!r}",
    )


def test_plan_migration_preset_inconsistent_with_model():
    """Regression : dans l'ancienne UI, `model` etait editable directement et
    le changer ne remettait PAS `preset` a « Personnalisé ». Une scene
    existante peut donc porter preset=WATER avec model=ELASTIC. Le libelle
    doit alors suivre le MODELE (qui fait autorite pour le solveur), pas le
    preset perime — sinon la migration baptise « Eau » un materiau elastique
    (observe en verification d'integration M16)."""
    e = dict(name="E1", preset="WATER", **_elastic())
    materials, _assignment = plan_migration([e])
    check(
        "preset WATER incoherent avec model ELASTIC -> libelle 'Élastique'",
        materials[0]["name"].startswith("Élastique"),
        f"got={materials[0]['name']!r}",
    )

    e2 = dict(name="E2", preset="JELLY", **_water())
    materials2, _a2 = plan_migration([e2])
    check(
        "preset JELLY incoherent avec model WATER -> libelle 'Eau'",
        materials2[0]["name"].startswith("Eau"),
        f"got={materials2[0]['name']!r}",
    )


def test_plan_migration_model_label_fallback():
    e = dict(name="E1", **_elastic())
    materials, assignment = plan_migration([e])
    check(
        "pas de preset reconnu -> libelle du modele ('Élastique')",
        materials[0]["name"].startswith("Élastique"),
        f"got={materials[0]['name']!r}",
    )


# -- collect_used_materials ----------------------------------------------------


def test_collect_used_materials_unused_material_excluded():
    used = dict(name="MatA", **_water())
    unused = dict(name="MatB", **_water(bulk=1.0e9))
    library = [used, unused]
    assignments = [("Emitter1", "MatA")]
    specs, indices, missing = collect_used_materials(library, assignments)
    check("materiau non utilise absent de specs", len(specs) == 1 and specs[0]["name"] == "MatA")
    check("pas d'entree manquante", missing == [])
    check("index correct", indices == [0])


def test_collect_used_materials_merges_same_key_different_names():
    mat1 = dict(name="MatA", **_water())
    mat2 = dict(name="MatB", **_water())  # meme cle, nom different
    library = [mat1, mat2]
    assignments = [("Emitter1", "MatA"), ("Emitter2", "MatB")]
    specs, indices, missing = collect_used_materials(library, assignments)
    check("2 entrees de bibliotheque identiques -> 1 seul spec", len(specs) == 1)
    check("les deux emetteurs recoivent le meme index", indices[0] == indices[1])
    check("pas de manquant", missing == [])


def test_collect_used_materials_empty_or_dangling_name_reported_missing():
    mat1 = dict(name="MatA", **_water())
    library = [mat1]
    assignments = [
        ("EmitterOK", "MatA"),
        ("EmitterEmpty", ""),
        ("EmitterDangling", "DoesNotExist"),
    ]
    specs, indices, missing = collect_used_materials(library, assignments)
    check("seul l'emetteur valide produit un spec", len(specs) == 1)
    check(
        "indices : [0, None, None]",
        indices == [0, None, None],
        f"indices={indices}",
    )
    check(
        "missing contient les deux emetteurs en defaut",
        set(missing) == {("EmitterEmpty", ""), ("EmitterDangling", "DoesNotExist")},
        f"missing={missing}",
    )
    check(
        "un emetteur en defaut n'empeche pas la resolution des autres",
        indices[0] == 0,
    )


def test_collect_used_materials_sands_differing_by_friction_angle_stay_distinct():
    """Deux sables qui ne different QUE par l'angle de frottement doivent
    produire deux specs distincts, pas un seul (piege explicitement signale
    par la spec de ce jalon : l'angle de frottement est physiquement
    pertinent, le confondre avec un autre sable serait une perte
    silencieuse)."""
    sand_35 = dict(name="Sable35", **_sand(friction_angle=35.0))
    sand_40 = dict(name="Sable40", **_sand(friction_angle=40.0))
    library = [sand_35, sand_40]
    assignments = [("EmitterA", "Sable35"), ("EmitterB", "Sable40")]
    specs, indices, missing = collect_used_materials(library, assignments)
    check(
        "deux sables d'angles differents -> 2 specs distincts",
        len(specs) == 2,
        f"len(specs)={len(specs)}",
    )
    check("pas de manquant", missing == [])
    check(
        "les deux emetteurs recoivent des index differents",
        indices[0] != indices[1],
        f"indices={indices}",
    )


def test_collect_used_materials_sands_same_friction_angle_merge():
    sand_a = dict(name="SableA", **_sand(friction_angle=35.0))
    sand_b = dict(name="SableB", **_sand(friction_angle=35.0))  # meme cle
    library = [sand_a, sand_b]
    assignments = [("EmitterA", "SableA"), ("EmitterB", "SableB")]
    specs, indices, missing = collect_used_materials(library, assignments)
    check(
        "deux sables identiques (meme angle) -> 1 seul spec",
        len(specs) == 1,
        f"len(specs)={len(specs)}",
    )
    check("les deux emetteurs recoivent le meme index", indices[0] == indices[1])


def test_collect_used_materials_empty_library_and_assignments():
    specs, indices, missing = collect_used_materials([], [])
    check(
        "bibliotheque et assignations vides -> tout vide",
        specs == [] and indices == [] and missing == [],
    )


# -- bout en bout : reproduit exactement l'ancienne deduplication par tuple ---


def test_end_to_end_matches_legacy_tuple_dedup():
    """Reconstruit a la main l'ancienne logique de `ops.py` (deduplication
    par tuple de valeurs, avant la refonte bibliotheque de materiaux) sur
    un jeu d'emetteurs, et verifie que `plan_migration` + `collect_used_
    materials` produisent EXACTEMENT la meme deduplication (meme nombre de
    materiaux distincts, meme regroupement des emetteurs par materiau)."""
    emitters_raw = [
        dict(name="Emitter1", model="ELASTIC", rho=1000.0, young=1e5, poisson=0.3, bulk=0.0, gamma=0.0),
        dict(name="Emitter2", model="WATER", rho=1000.0, young=0.0, poisson=0.0, bulk=2.2e9, gamma=7.0),
        dict(name="Emitter3", model="ELASTIC", rho=1000.0, young=1e5, poisson=0.3, bulk=0.0, gamma=0.0),  # dup d'Emitter1
        dict(name="Emitter4", model="WATER", rho=800.0, young=0.0, poisson=0.0, bulk=1.0e9, gamma=7.0),
    ]

    # -- ancienne logique (ops.py:1786-1812) --
    legacy_keys = []
    legacy_specs = []
    legacy_emitter_index = []
    for e in emitters_raw:
        # Dispatch EXHAUSTIF, meme discipline que `materials.material_key` :
        # `plan_migration` ne route jamais de SAND ici (les champs herites
        # DEPRECIES d'`BqObjectProps` n'ont jamais eu d'enum SAND), mais un
        # `else` muet resterait un piege si ca changeait un jour.
        if e["model"] == "ELASTIC":
            key = ("ELASTIC", e["rho"], e["young"], e["poisson"])
        elif e["model"] == "WATER":
            key = ("WATER", e["rho"], e["bulk"], e["gamma"])
        else:
            raise ValueError(f"modele inconnu {e['model']!r}")
        if key in legacy_keys:
            idx = legacy_keys.index(key)
        else:
            idx = len(legacy_specs)
            legacy_keys.append(key)
            legacy_specs.append(key)
        legacy_emitter_index.append(idx)

    # -- nouvelle logique : migration puis collecte --
    materials, assignment = plan_migration(emitters_raw)
    assignments = [(e["name"], assignment[e["name"]]) for e in emitters_raw]
    specs, indices, missing = collect_used_materials(materials, assignments)

    check(
        "meme nombre de materiaux distincts que l'ancienne dedup",
        len(specs) == len(legacy_specs),
        f"new={len(specs)} legacy={len(legacy_specs)}",
    )
    check("aucun manquant", missing == [])
    check(
        "meme regroupement des emetteurs par materiau (Emitter1 == Emitter3)",
        indices[0] == indices[2] and indices[0] == legacy_emitter_index[0],
    )
    check(
        "Emitter2 et Emitter4 restent distincts (bulk different)",
        indices[1] != indices[3],
    )
    check(
        "Emitter1/Emitter3 (ELASTIC) distinct d'Emitter2 (WATER)",
        indices[0] != indices[1],
    )


if __name__ == "__main__":
    test_material_key_water_ignores_young_poisson()
    test_material_key_water_differs_by_bulk()
    test_material_key_elastic_vs_water_same_rho()
    test_material_key_sand_same_params_same_key()
    test_material_key_sand_differs_by_friction_angle()
    test_material_key_sand_vs_elastic_same_rho_young_poisson()
    test_material_key_unknown_model_raises()

    test_unique_name_free_unchanged()
    test_unique_name_single_collision()
    test_unique_name_multiple_collisions()
    test_unique_name_empty_falls_back_to_default()
    test_unique_name_empty_still_deduplicated()

    test_plan_migration_empty_list()
    test_plan_migration_identical_emitters_share_one_material()
    test_plan_migration_different_emitters_produce_distinct_unique_names()
    test_plan_migration_preset_labels()
    test_plan_migration_preset_inconsistent_with_model()
    test_plan_migration_model_label_fallback()

    test_collect_used_materials_unused_material_excluded()
    test_collect_used_materials_merges_same_key_different_names()
    test_collect_used_materials_empty_or_dangling_name_reported_missing()
    test_collect_used_materials_sands_differing_by_friction_angle_stay_distinct()
    test_collect_used_materials_sands_same_friction_angle_merge()
    test_collect_used_materials_empty_library_and_assignments()

    test_end_to_end_matches_legacy_tuple_dedup()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} test(s) failed: {FAILURES}")
        sys.exit(1)
    else:
        print("Tous les tests sont passes.")
        sys.exit(0)
