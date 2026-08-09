"""materials.py — logique pure de la bibliotheque de materiaux nommee au
niveau scene (refonte UI, remplace les champs materiau recopies sur chaque
objet emetteur).

Aucune dependance a `bpy` NI a `lib` (le binding ctypes vers la DLL) : ce
module est pur, importable et testable en dehors de Blender (voir
`extension/tests/test_materials.py`), au meme titre que
`foam.py`/`whitewaterfade.py`/`whitewatervolume.py` dont il reprend la
convention. En particulier, aucune conversion vers les constantes
`lib.BQ_MODEL_*` n'a lieu ici : les dicts materiau produits par ce module
portent `model` sous forme de chaine (`"ELASTIC"`/`"WATER"`), c'est
`ops.py` qui traduira au moment de l'appel au solveur.

Deux contraintes du coeur, deja appliquees par la deduplication historique
par tuple de valeurs (`ops.py`, avant cette refonte, lignes ~1786-1812), et
que ce module reprend a l'identique :

1. Le solveur accepte au maximum `lib.BQ_MAX_MATERIALS` (8) materiaux
   enregistres simultanement — cette limite n'est PAS verifiee ici (elle
   depend de `lib`), c'est `ops.py` qui la verifie sur `specs` renvoye par
   `collect_used_materials`.
2. Seuls les materiaux EFFECTIVEMENT UTILISES par au moins un emetteur
   doivent etre enregistres au solveur : le pas de temps `dt` est calcule
   sur la vitesse du son maximale de TOUS les materiaux enregistres, donc
   enregistrer un materiau raide inutilise ralentirait toute la simulation
   (cf. `mlsmpm.cu`, calcul du dt). `collect_used_materials` existe pour
   cette raison precise : filtrer la bibliotheque de scene (qui peut
   contenir des materiaux orphelins, crees puis desassignes) a ce qui sert
   reellement au bake courant.

Un "materiau" est un dict Python plat avec les cles `name` (str), `model`
(str, `"ELASTIC"` ou `"WATER"`), `rho`, `young`, `poisson`, `bulk`, `gamma`
(floats). Les champs non pertinents pour le modele sont presents mais
ignores par `material_key` (ex. un materiau WATER porte quand meme
`young`/`poisson`, herites du PropertyGroup Blender qui a les memes champs
pour les deux modeles, mais ils ne comptent pas dans sa clé de
deduplication).
"""

__all__ = (
    "material_key",
    "unique_name",
    "plan_migration",
    "collect_used_materials",
)

# Nom par defaut attribue a un materiau demande avec un libelle vide (voir
# `unique_name`). Choisi neutre plutot que vide, pour ne jamais produire de
# nom de materiau invisible dans une liste UI.
_DEFAULT_NAME = "Matériau"

# Libelles lisibles utilises par `plan_migration` pour nommer les materiaux
# derives d'un preset ou, a defaut, d'un modele physique. Les presets
# WATER/JELLY sont ceux exposes par le PropertyGroup emetteur existant
# (`props.py`) ; tout autre preset (ou son absence) retombe sur le libelle
# du modele physique.
_PRESET_LABELS = {
    "WATER": "Eau",
    "JELLY": "Gelée",
}
# Modele constitutif que chaque preset IMPLIQUE (cf. `_PRESET_WATER` /
# `_PRESET_JELLY` dans props.py, source de verite de ces valeurs). Sert a
# n'utiliser le libelle d'un preset que s'il est COHERENT avec le modele
# reellement porte par l'emetteur -- voir `_label_for_emitter`.
_PRESET_MODEL = {
    "WATER": "WATER",
    "JELLY": "ELASTIC",
}
_MODEL_LABELS = {
    "WATER": "Eau",
    "ELASTIC": "Élastique",
}


def material_key(material):
    """Cle de deduplication d'un materiau : deux materiaux de meme cle sont
    interchangeables pour le solveur (memes parametres physiques, quel que
    soit leur `name`). MEME semantique que la deduplication historique par
    tuple de valeurs (`ops.py`, avant cette refonte) : seuls les champs
    PERTINENTS pour le modele entrent dans la cle, les autres (herites du
    PropertyGroup commun aux deux modeles) sont ignores meme s'ils different.

    - ELASTIC -> `("ELASTIC", rho, young, poisson)`
    - WATER   -> `("WATER", rho, bulk, gamma)`
    """
    if material["model"] == "ELASTIC":
        return ("ELASTIC", material["rho"], material["young"], material["poisson"])
    return ("WATER", material["rho"], material["bulk"], material["gamma"])


def unique_name(desired, existing):
    """Renvoie `desired` s'il est libre dans `existing`, sinon `desired`
    suffixe du plus petit entier libre au format Blender (`.001`, `.002`,
    ...). Un `desired` vide (ou uniquement des espaces) retombe sur
    `_DEFAULT_NAME` avant desambiguisation, pour ne jamais produire un nom
    de materiau invisible dans une liste UI.

    `existing` est parcouru une seule fois en un `set` (pas de cout
    quadratique si `unique_name` est appele en boucle sur une bibliotheque
    qui grossit — l'appelant doit alors reconstruire `existing` a chaque
    appel avec le nom nouvellement ajoute, ce module ne maintient pas
    d'etat lui-meme, coherent avec le style pur du reste du fichier).
    """
    desired = (desired or "").strip() or _DEFAULT_NAME
    taken = set(existing)

    if desired not in taken:
        return desired

    n = 1
    while True:
        candidate = f"{desired}.{n:03d}"
        if candidate not in taken:
            return candidate
        n += 1


def _label_for_emitter(emitter):
    """Libelle lisible derive d'un emetteur en cours de migration.

    Le `preset` porte l'intention utilisateur d'origine et donne un libelle
    plus parlant que le seul modele (« Gelée » plutot qu'« Élastique »), donc
    on le prefere -- mais UNIQUEMENT s'il est COHERENT avec le `model`
    reellement porte par l'emetteur.

    Ce garde-fou n'est pas theorique : dans l'ancienne UI, `model` etait un
    enum editable directement, et le modifier ne remettait PAS `preset` a
    « Personnalisé ». Une scene existante peut donc parfaitement porter
    `preset == "WATER"` et `model == "ELASTIC"` -- sans ce test, la migration
    baptisait « Eau » un materiau elastique (observe en verification
    d'integration M16). Le `model` fait autorite pour le solveur, donc le
    libelle doit s'y conformer des que les deux divergent."""
    preset = emitter.get("preset")
    model = emitter["model"]
    if preset in _PRESET_LABELS and _PRESET_MODEL.get(preset) == model:
        return _PRESET_LABELS[preset]
    return _MODEL_LABELS.get(model, model)


def plan_migration(emitters):
    """Construit une bibliotheque de materiaux depuis des emetteurs qui
    portent encore leurs reglages en propre (migration d'une scene
    existante vers la bibliotheque nommee).

    `emitters` : liste de dicts, dans l'ordre de la scene, avec au moins
    `name` (nom de l'objet emetteur), `model`, `rho`, `young`, `poisson`,
    `bulk`, `gamma`, et optionnellement `preset` (str) qui sert a NOMMER
    le materiau produit (voir `_label_for_emitter`).

    Renvoie `(materials, assignment)` :
      - `materials` : liste de dicts materiau dedupliques par
        `material_key`, DANS L'ORDRE DE PREMIERE RENCONTRE (deux emetteurs
        identiques -> un seul materiau, cree au moment du premier des
        deux), chacun avec un `name` UNIQUE (via `unique_name`).
      - `assignment` : dict `{nom_emetteur: nom_materiau}`.

    Liste vide -> `([], {})`, jamais d'exception : cas normal (scene sans
    emetteur, ou deja entierement migree en amont par l'appelant).
    """
    materials = []
    assignment = {}
    keys_seen = {}  # material_key -> nom du materiau deja cree

    for emitter in emitters:
        key = material_key(emitter)

        if key in keys_seen:
            assignment[emitter["name"]] = keys_seen[key]
            continue

        label = _label_for_emitter(emitter)
        existing_names = (m["name"] for m in materials)
        name = unique_name(label, existing_names)

        materials.append(
            {
                "name": name,
                "model": emitter["model"],
                "rho": emitter["rho"],
                "young": emitter["young"],
                "poisson": emitter["poisson"],
                "bulk": emitter["bulk"],
                "gamma": emitter["gamma"],
            }
        )
        keys_seen[key] = name
        assignment[emitter["name"]] = name

    return materials, assignment


def collect_used_materials(library, assignments):
    """Selectionne, deduplique, les materiaux a enregistrer au solveur pour
    un bake — c'est la logique que `ops._validate` consommera pour
    construire `material_specs`/`emitter_specs` a la place de la
    deduplication par tuple de valeurs qu'elle faisait jusqu'ici.

    `library` : liste ordonnee de dicts materiau (la bibliotheque de la
    scene, potentiellement plus large que ce dont le bake courant a
    besoin — ex. materiaux crees puis desassignes de tout emetteur).
    `assignments` : liste de tuples `(nom_emetteur, nom_materiau)` dans
    l'ordre des emetteurs.

    Renvoie `(specs, emitter_material_index, missing)` :
      - `specs` : les materiaux (dicts complets, tels que dans `library`)
        REELLEMENT UTILISES (au moins un emetteur les reference), dans
        l'ordre de PREMIERE UTILISATION, dedupliques par `material_key` —
        voir la contrainte 2 en tete de module, c'est la raison d'etre de
        cette fonction : un materiau non reference par aucun emetteur
        n'apparait JAMAIS dans `specs`, meme s'il est present dans
        `library`.
      - `emitter_material_index` : liste PARALLELE a `assignments` (meme
        longueur, meme ordre), donnant pour chaque emetteur l'index de son
        materiau dans `specs`. Vaut `None` a la position d'un emetteur
        dont le materiau est manquant (voir `missing`).
      - `missing` : liste de tuples `(nom_emetteur, nom_materiau)` pour
        les emetteurs dont `nom_materiau` est vide ou absent de
        `library` — l'appelant (`ops.py`) refusera le bake avec un
        message nommant l'emetteur fautif, mais cette fonction elle-meme
        ne leve jamais : un emetteur en defaut n'empeche pas les autres
        d'etre resolus normalement (`missing` collecte TOUTES les
        anomalies en un seul passage, pour un message d'erreur complet
        plutot qu'un arret au premier probleme rencontre).
    """
    by_name = {m["name"]: m for m in library}

    specs = []
    keys_to_index = {}  # material_key -> index dans specs
    emitter_material_index = []
    missing = []

    for emitter_name, material_name in assignments:
        material = by_name.get(material_name) if material_name else None

        if material is None:
            missing.append((emitter_name, material_name))
            emitter_material_index.append(None)
            continue

        key = material_key(material)
        if key not in keys_to_index:
            keys_to_index[key] = len(specs)
            specs.append(material)

        emitter_material_index.append(keys_to_index[key])

    return specs, emitter_material_index, missing
