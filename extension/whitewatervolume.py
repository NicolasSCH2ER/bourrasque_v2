"""whitewatervolume.py — calcul pur des parametres de l'affichage
volumetrique du whitewater (`BQ Whitewater Volume`/`BQ_Whitewater_Volume`,
`extension/assets/whitewater_display.blend`), a partir de la taille REELLE
des particules de la frame courante (`bq_size`, cache `.bqw`).

Aucune dependance a `bpy` : ce module est pur, importable et testable en
dehors de Blender (voir `extension/tests/test_whitewatervolume.py`), meme
convention que `whitewaterfade.py`/`foam.py`.

Contexte (verifie par rendu Cycles headless comparatif, pas suppose) : le
groupe de noeuds `BQ Whitewater Volume` et le materiau
`BQ_Whitewater_Volume` etaient jusqu'ici calibres avec deux constantes
FIXES (`Value.VoxelSize` = 0.01, `Value.DensityFactor` = 25.0), choisies sur
une scene de test a UNE echelle arbitraire. En usage reel, le rayon des
particules whitewater (`bq_size`, `= 0.3 * influence_radius` cote coeur
CUDA) varie de plusieurs ordres de grandeur selon la scene (mesure :
0.007 m avec `influence_radius=0.0234`) : un voxel plus GROS que la
particule (resolution insuffisante) et surtout une densite bien trop faible
pour qu'une epaisseur optique de quelques mm absorbe la lumiere rendaient
le volume quasi invisible. Ce module derive les deux valeurs de `bq_size`
mesure sur la frame courante, meme politique que `ww_size_mult`
(`display.py`) : calcul en Python, injection directe dans le graphe
Blender, aucune logique dans les noeuds au-dela de ce qui est necessaire.
"""

__all__ = ("compute_whitewater_volume_display_params",)

# Facteur voxel : le voxel doit rester nettement plus petit que le rayon
# TYPIQUE d'une particule pour que `Points to Volume` echantillonne
# correctement chaque particule (plusieurs voxels par sphere), sans pour
# autant produire une resolution inutilement fine (cout memoire/temps de
# rendu). 0.3 place le voxel a moins d'un tiers du rayon moyen des
# particules de la frame.
_VOXEL_SIZE_FACTOR = 0.3

# Constante de densite derivee EMPIRIQUEMENT (pas arbitraire) de la mesure
# de la session de verification : `density_factor=400` sur une scene de
# test a `bq_size=0.007` donnait un resultat visuellement net (bleu fonce),
# contre une tache a peine perceptible a `density_factor=25`. `400*0.007 ~=
# 2.8` : on fixe ce produit constant plutot que le facteur lui-meme, pour
# que la densite optique absorbee reste comparable quelle que soit
# l'echelle de `bq_size` observee sur une autre scene.
_DENSITY_FACTOR_CONSTANT = 2.8

# Plancher de `bq_size_mean` pour eviter une division par une valeur
# nulle/quasi nulle (cache degenere, toutes les particules a taille ~0).
_MIN_BQ_SIZE = 1e-6


def compute_whitewater_volume_display_params(bq_size_mean):
    """Renvoie `(voxel_size, density_factor)` a partir de `bq_size_mean`
    (moyenne, sur la frame courante, du tableau `bq_size` LU DEPUIS LE CACHE
    `.bqw`, AVANT le multiplicateur cosmetique `ww_size_mult` — voir
    docstring de `display.refresh_whitewater`).

    `voxel_size = bq_size_mean * _VOXEL_SIZE_FACTOR` : voir la constante
    pour la justification.

    `density_factor = _DENSITY_FACTOR_CONSTANT / max(bq_size_mean,
    _MIN_BQ_SIZE)` : voir la constante pour la provenance empirique. La
    division est protegee contre une valeur nulle/quasi nulle de
    `bq_size_mean`.
    """
    safe_size = max(float(bq_size_mean), _MIN_BQ_SIZE)
    voxel_size = safe_size * _VOXEL_SIZE_FACTOR
    density_factor = _DENSITY_FACTOR_CONSTANT / safe_size
    return voxel_size, density_factor
