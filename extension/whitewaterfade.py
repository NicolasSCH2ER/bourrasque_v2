"""whitewaterfade.py — calcul pur du facteur de fondu par age
(`bq_fade`), applique a l'echelle des particules whitewater affichees.

Aucune dependance a `bpy` : ce module est pur, importable et testable en
dehors de Blender (voir `extension/tests/test_whitewaterfade.py`), au meme
titre que `foam.py`/`transform.py`/`meshcache.py` dont il reprend la
convention.

Contexte : sans fondu, une particule secondaire (spray/foam/bubble)
apparait et disparait d'un coup entre deux frames (l'"effet pop" signale
par l'utilisateur), puisque `refresh_whitewater` (display.py) ne fait que
recopier les positions/tailles actives d'une frame a l'autre sans aucune
transition. Ce module calcule, a partir de l'age de chaque particule
(`bq_age`, deja ecrit sur le cache `.bqw`) et de sa duree de vie par type
(deja stockee dans `WhitewaterProductionParams`), un facteur `[0, 1]`
multiplie a l'echelle affichee (voir le noeud `Math.SizeFade` du groupe
Geometry Nodes `BQ Whitewater Display`, `extension/assets/
whitewater_display.blend`) : la particule grandit depuis 0 a la naissance
et retrecit vers 0 avant sa mort, au lieu d'un pop.
"""

import numpy as np

__all__ = ("compute_age_fade",)


def _smoothstep(t):
    """Interpolation lisse standard `3t^2 - 2t^3` sur `t` DEJA clampe a
    `[0, 1]` par l'appelant. Pas de cassure de derivee en `t=0`/`t=1`
    (contrairement a une rampe lineaire), ce qui evite un "coude" visible
    en debut/fin de fondu."""
    return t * t * (3.0 - 2.0 * t)


def compute_age_fade(
    age,
    type_,
    life_spray,
    life_foam,
    life_bubble,
    fade_in_frac=0.15,
    fade_out_frac=0.25,
):
    """Renvoie un facteur `[0, 1]` par particule (ndarray float32, meme
    forme que `age`) : monte en douceur de 0 a 1 durant les premiers
    `fade_in_frac` de la duree de vie, reste a 1, redescend en douceur vers
    0 durant les derniers `fade_out_frac`. Interpolation smoothstep
    (`3t^2 - 2t^3`), pas lineaire, pour ne laisser aucune cassure visible
    en debut/fin de fondu.

    `age`/`type_` : ndarray `(n,)`, memes tableaux que ceux lus depuis le
    cache `.bqw` (`WhitewaterCacheReader.read_frame`). `type_` vaut 0
    (spray), 1 (foam) ou 2 (bubble) — `life_spray`/`life_foam`/
    `life_bubble` sont assignes par particule en consequence (vectorise
    numpy, pas de boucle Python par particule).

    `t = age / life`, clampe a `[0, 1]`. `life <= 0` (duree de vie nulle
    ou negative, cas degenere/mal configure) : fade = 1 partout pour cette
    particule, pas de division par zero.

    Si `fade_in_frac + fade_out_frac >= 1` (duree de vie tres courte par
    rapport aux fractions demandees), les deux rampes se chevauchent : la
    portion pleine (`t` entre la fin de la montee et le debut de la
    descente) est vide, et les deux smoothstep sont normalises sur leurs
    plages respectives sans jamais diviser par une plage nulle ni produire
    de valeur negative (voir details d'implementation ci-dessous).
    """
    age = np.asarray(age, dtype=np.float64)
    type_ = np.asarray(type_)

    life_by_type = np.array(
        [float(life_spray), float(life_foam), float(life_bubble)], dtype=np.float64
    )
    # `type_` est cense valoir 0/1/2 (voir docstring), mais on se protege
    # d'une valeur hors plage (cache corrompu) par un clip plutot qu'une
    # exception — meme discipline "jamais lever" que `display.py`.
    type_idx = np.clip(type_.astype(np.int64), 0, 2)
    life = life_by_type[type_idx]

    fade = np.ones_like(age, dtype=np.float64)

    valid = life > 0.0
    if not np.any(valid):
        return fade.astype(np.float32)

    t = np.zeros_like(age)
    t[valid] = np.clip(age[valid] / life[valid], 0.0, 1.0)

    fade_in = float(fade_in_frac)
    fade_out = float(fade_out_frac)

    # Point de bascule entre rampe montante et descendante. Cas normal
    # (`fade_in + fade_out < 1`) : montee sur `[0, fade_in]`, plein sur
    # `[fade_in, 1-fade_out]`, descente sur `[1-fade_out, 1]` — le "point de
    # bascule" pertinent pour le calcul ci-dessous est alors la fin de la
    # rampe montante et le debut de la rampe descendante, traites separement.
    #
    # Cas degenere (`fade_in + fade_out >= 1`, duree de vie tres courte par
    # rapport aux fractions demandees) : les deux rampes se chevauchent, il
    # n'existe plus de plage pleine. On ramene alors les DEUX rampes a un
    # seul point de bascule `split = fade_in / (fade_in + fade_out)` (jamais
    # de division par zero : ce cas n'est atteint que si `fade_in+fade_out
    # >= 1 > 0`), chaque rampe etant renormalisee sur sa propre demi-plage
    # `[0, split]` / `[split, 1]` — un pic continu (jamais de valeur
    # negative ni de discontinuite), plutot qu'une plage "pleine" de
    # longueur negative.
    overlap = (fade_in + fade_out) >= 1.0

    if overlap:
        split = fade_in / (fade_in + fade_out)
        rise_end = split
        fall_start = split
    else:
        rise_end = fade_in
        fall_start = 1.0 - fade_out

    result = np.ones_like(t)

    if rise_end > 0.0:
        rising = t < rise_end
        result = np.where(rising, _smoothstep(np.clip(t / rise_end, 0.0, 1.0)), result)

    if fall_start < 1.0:
        falling = t >= fall_start
        tf = np.clip((t - fall_start) / (1.0 - fall_start), 0.0, 1.0)
        result = np.where(falling, 1.0 - _smoothstep(tf), result)

    fade[valid] = result[valid]
    return fade.astype(np.float32)
