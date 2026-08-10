"""ops.py — operateurs bpy de l'extension Bourrasque.

Cote Python/Blender de la frontiere : ce module cable les objets de scene
(domaine, emetteurs) vers `lib.Sim` et le cache `.bqd`/`.mat`, mais ne parle
jamais directement a ctypes — c'est le role exclusif de `lib.py`.

Le coeur de ce module est `BQ_OT_bake`, un operateur modal : le bake peut
durer plusieurs minutes et doit rester annulable (ESC ou bouton Annuler)
sans jamais fuir de memoire GPU ni laisser un timer actif. Tous les chemins
de sortie de la boucle modale (fin normale, annulation, exception) convergent
vers `BQ_OT_bake._cleanup`, seul endroit qui ferme le `CacheWriter`, detruit
la simulation et retire le timer.

Le CALCUL de chaque bake (particules ou maillage, `BQ_OT_bake_mesh`) tourne
dans un `threading.Thread(daemon=True)` (docs/plan-milestone-7.md, decision
D10) : le tick modal ne fait plus qu'un sondage periodique d'un objet
`_BakeProgress` partage (compteur de frames faites, drapeau de fin, message
d'erreur eventuel, file de messages a reporter), jamais de calcul. C'est ce
qui laisse Blender manipulable (viewport, ESC) pendant tout le bake, y
compris une frame de maillage qui peut durer plusieurs secondes.

REGLE NON NEGOCIABLE : AUCUN appel a `bpy` (lecture de scene, ecriture de
propriete, `self.report`, `evaluated_depsgraph_get`...) ne doit se produire
dans un thread de calcul — Blender n'est pas thread-safe, un tel appel hors
du thread principal fait tomber le processus sans message exploitable. Le
thread ne touche que : la `Sim`/le `Mesher` (appels ctypes, qui relachent le
GIL — c'est ce qui rend le thread utile), l'ecriture de fichier cache, et
l'objet `_BakeProgress` (compteurs/drapeaux simples, plus une `queue.Queue`
pour les messages a reporter). Tout ce qui vient de la scene — y compris la
geometrie EVALUEE des colliders animes, qui depend du depsgraph a chaque
frame — est donc collecte AVANT le lancement du thread, sur le thread
principal (voir `BQ_OT_bake.invoke`, pre-extraction dans
`self._collider_frames`) : le thread ne consomme plus que des tableaux
numpy deja extraits, jamais un objet bpy.
"""

import math
import os
import queue
import threading

import bpy
import numpy as np

from . import cache, lib, materials, meshcache, whitewatercache
from .props import (
    _PRESET_WATER,
    domain_resolution,
    domain_transform,
    domain_usable_bounds,
    emitter_bounds_solver,
    emitter_overflow,
    estimate_particle_count,
    material_usage,
    mesh_cache_path,
    mesh_effective_radii,
    mesh_layout,
    whitewater_cache_path,
    whitewater_config_from_scene,
    world_to_solver_dir,
)
from .transform import (
    solver_to_world,
    world_to_solver,
    world_to_solver_array,
    world_to_solver_dir_array,
)

__all__ = ("classes", "register", "unregister")


# ---------------------------------------------------------------------------
# Contact solide<->solide (jalon M17, phase B, B4) — constantes de cablage
# ---------------------------------------------------------------------------
#
# Plafond du nombre d'echantillons de surface par corps (D10 du plan) : une
# densite ciblee de `dx` (voir `rigidbody.surface_samples`) sur un maillage
# tres detaille pourrait sinon produire des dizaines de milliers de points,
# couteux a tester par sous-pas pour chaque paire de corps en recouvrement.
# Large devant les colliders typiques d'une scene (quelques milliers de
# triangles) : n'entre en jeu que pour un maillage exceptionnellement dense.
_BODY_SURFACE_SAMPLE_CAP = 20_000

# Noms des colliders STATIQUES au maillage NON FERME du DERNIER bake lance
# sur chaque scene (cle = `scene.name`) — memorise par `BQ_OT_bake.invoke`
# au moment ou `sampling.check_mesh_closed` est de toute facon deja appele
# (point 3 de B4), jamais recalcule depuis un `draw()` (lancer de rayons sur
# un BVH, bien trop couteux pour un redessin de panneau). Lu par
# `open_mesh_collider_names` (ui.py, tableau de bord) : information
# "gratuite" tant qu'aucun nouveau bake n'a ete lance depuis. Cache
# process-local (jamais persiste dans le .blend) : une scene qui n'a encore
# jamais ete bakee n'a simplement aucune entree ici.
_OPEN_MESH_COLLIDERS = {}


def open_mesh_collider_names(scene):
    """Noms des colliders STATIQUES au maillage non ferme, tels que memorises
    par le DERNIER bake de `scene` (voir `_OPEN_MESH_COLLIDERS`) — tuple
    vide si `scene` n'a jamais ete bakee depuis le chargement de
    l'extension, ou si tous ses colliders statiques etaient fermes."""
    return _OPEN_MESH_COLLIDERS.get(scene.name, ())


# ---------------------------------------------------------------------------
# Geometrie de l'inflow — fonctions pures, testables sans bpy
# ---------------------------------------------------------------------------
#
# Un emetteur INFLOW ensemence son VOLUME COMPLET — pas une tranche collee
# a une "face aval" (approche abandonnee : fausse par construction pour une
# forme non prismatique, ex. une sphere dont la face aval est un pole, donc
# un disque de rayon quasi nul) — et maintient ce volume SATURE pour toute
# la duree du bake : c'est l'approche
# des implementations de reference (FLIP Fluids exige un emetteur "avec du
# volume" et contraint la vitesse du fluide a l'interieur de l'inflow).
#
# Le NUAGE COMPLET de sites d'emission (`sites`, espace solveur) est calcule
# UNE FOIS a la mise en place du bake (`BQ_OT_bake.invoke`) — exactement ce
# que produit `sampling.sample_mesh_interior` en mode MESH, ou le reseau
# complet de la boite en mode BOUNDS (`_lattice_points_in_box`) — et reste
# fixe pour toute la duree du bake, il ne bouge jamais.
#
# A chaque frame, avant le `step` (voir `BQ_OT_bake._emit_inflow_sites`) :
# on determine quels sites sont deja occupes par une particule existante
# (filtre par bbox de l'emetteur puis indexation au site le plus proche), et
# on emet une particule a chaque site NON occupe, a la vitesse initiale de
# l'emetteur. Le volume reste ainsi sature en permanence : le fluide en sort
# a `v`, les sites se liberent exactement au rythme voulu, le debit
# s'auto-regule sans accumulateur de distance.
#
# Ce nombre de sites libres n'est PAS a lui seul un plafond fiable des que
# le jitter positionnel ou la turbulence spatialement coherente sont actifs
# : un bruit qui deplace des blocs entiers de particules dans la meme
# direction cree a la fois des sites vides (reemis, on GAGNE) et des sites
# doublement occupes (rien n'est retire, on ne PERD rien), d'ou une
# sur-emission structurelle si on emet a chaque site libre sans plafond.
# L'emission est donc plafonnee par CONSERVATION DU NOMBRE : au plus
# `deficit = max(0, n_sites - n_occupants)` particules par emetteur et par
# frame, les sites libres excedentaires etant tires au hasard (voir
# `BQ_OT_bake._emit_inflow_sites`) — turbulence et debit restent ainsi deux
# reglages orthogonaux pour l'artiste.


def _lattice_points_in_box(lo, hi, spacing):
    """Points du reseau GLOBAL du solveur (ancres a l'origine du domaine,
    positions `spacing/2 + k*spacing` sur chaque axe, k >= 0) contenus dans
    `[lo, hi]` (espace solveur), en `(n, 3)` float32.

    Reprend en miniature la logique de
    `sampling._lattice_candidates_solver` / `_lattice_k_range` (meme
    convention de reseau global, voir la docstring de `sampling.py` pour la
    justification) plutot que d'importer ces symboles prives d'un autre
    module.
    """
    if spacing <= 0.0:
        return np.empty((0, 3), dtype=np.float32)

    ranges = []
    for axis in range(3):
        axis_lo, axis_hi = lo[axis], hi[axis]
        if axis_hi <= axis_lo:
            return np.empty((0, 3), dtype=np.float32)
        k_min = max(0, math.ceil((axis_lo - spacing / 2.0) / spacing))
        k_max = math.floor((axis_hi - spacing / 2.0) / spacing)
        if k_max < k_min:
            return np.empty((0, 3), dtype=np.float32)
        ranges.append((int(k_min), int(k_max)))

    axes = []
    for k_min, k_max in ranges:
        ks = np.arange(k_min, k_max + 1, dtype=np.float64)
        axes.append(spacing / 2.0 + ks * spacing)

    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    return pts.astype(np.float32)


def _box_lattice_points(lo, hi, spacing):
    """Reproduit, cote Python, EXACTEMENT le reseau que `bq_emit_box`
    genere cote solveur (`core/src/mlsmpm.cu`) :

        for (float x = lo[0] + spacing/2; x < hi[0]; x += spacing)
            for (float y = lo[1] + spacing/2; y < hi[1]; y += spacing)
                for (float z = lo[2] + spacing/2; z < hi[2]; z += spacing)
                    ...

    Ce reseau est ANCRE SUR L'EMETTEUR (`lo`), contrairement au reseau
    GLOBAL de `_lattice_points_in_box` (ancre a l'origine du domaine, utilise
    pour l'inflow) : les deux ne coincident pas, NE PAS les confondre. Cette
    fonction n'est necessaire que pour appliquer la turbulence a un emetteur
    BLOCK/BOUNDS, cas ou c'est normalement le solveur qui genere le reseau
    (`bq_emit_box`). L'arithmetique est faite en float32, comme cote C
    (`float x`, pas `double`), pour reproduire fidelement le compte de
    particules issu de la meme accumulation en simple precision — un compte
    different casserait silencieusement `props.estimate_particle_count`.
    """
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    step = np.float32(spacing)
    half = np.float32(step / np.float32(2.0))

    axes = []
    for a in range(3):
        vals = []
        x = np.float32(lo[a] + half)
        while x < hi[a]:
            vals.append(x)
            x = np.float32(x + step)
        axes.append(np.array(vals, dtype=np.float32))

    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)
    return pts.astype(np.float32)


def _turbulence_rng(seed, frame_index, emitter_index):
    """Generateur numpy DEDIE, jamais l'etat global de numpy (determinisme
    du bake, voir docstring de `BQ_OT_bake`). La sequence `[seed,
    frame_index, emitter_index]` isole trois sources de variation : la
    graine reglee par l'artiste sur CET emetteur, la frame courante (deux
    frames consecutives ne recoivent pas le meme bruit), et l'indice de
    l'emetteur (deux emetteurs de meme graine ne recoivent pas le meme
    bruit).

    Ne sert plus qu'au JITTER POSITIONNEL (voir `_turbulent_emission') : le
    bruit de VITESSE est desormais porte par `_CurlNoise`, qui a sa propre
    seed independante par tranche temporelle (voir sa docstring)."""
    return np.random.default_rng([int(seed), int(frame_index), int(emitter_index)])


def _points_bbox(points):
    """Bbox `(lo, hi)` de `points` `(n, 3)`, ou l'origine degeneree `(0,0,0)`
    a `(0,0,0)` si `points` est vide (evite un `.min()`/`.max()` sur tableau
    vide, qui leve `ValueError`) : un `_CurlNoise` construit sur cette bbox
    degeneree reste valide (voir `_CurlNoise.__init__`, la marge/le clamp de
    taille de grille absorbe le cas), il ne sera simplement jamais
    echantillonne (aucun point a emettre)."""
    if points.shape[0] == 0:
        zero = np.zeros(3, dtype=np.float64)
        return zero, zero
    return points.min(axis=0), points.max(axis=0)


def _curl_from_potential(psi, cell):
    """Rotationnel de `psi` (`(nx, ny, nz, 3)`, grille reguliere de pas
    `cell`) par differences finies centrees.

    `np.roll` (plutot qu'un padding explicite) enveloppe les bords de la
    grille : le rotationnel y est donc physiquement arbitraire. C'est sans
    consequence, car `_CurlNoise` entoure toujours la bbox utile d'une
    marge de 2 cellules grossieres — seuls des noeuds JAMAIS echantillonnes
    touchent ce bord enveloppe.
    """
    def d(a, axis):
        return (np.roll(a, -1, axis=axis) - np.roll(a, 1, axis=axis)) / (2.0 * cell)

    curl = np.empty_like(psi)
    curl[..., 0] = d(psi[..., 2], 1) - d(psi[..., 1], 2)
    curl[..., 1] = d(psi[..., 0], 2) - d(psi[..., 2], 0)
    curl[..., 2] = d(psi[..., 1], 0) - d(psi[..., 0], 1)
    return curl


def _sample_trilinear(field, points, lo, cell):
    """Interpolation trilineaire de `field` (grille reguliere `(nx, ny, nz,
    k)`, origine `lo`, pas `cell`) aux positions `points` `(n, 3)`, toutes
    deux en `float64`.

    Les indices de cellule sont CLIPPES a l'interieur de la grille : un
    point legerement hors de la bbox couverte (jitter positionnel, ou marge
    de securite du domaine) ne provoque donc jamais d'acces hors tableau,
    seulement une extrapolation par la cellule de bord la plus proche.
    """
    g = (points - lo) / cell
    nx, ny, nz = field.shape[0] - 1, field.shape[1] - 1, field.shape[2] - 1
    i0 = np.floor(g).astype(np.int64)
    i0[:, 0] = np.clip(i0[:, 0], 0, max(nx - 1, 0))
    i0[:, 1] = np.clip(i0[:, 1], 0, max(ny - 1, 0))
    i0[:, 2] = np.clip(i0[:, 2], 0, max(nz - 1, 0))
    frac = np.clip(g - i0, 0.0, 1.0)

    out = np.zeros((points.shape[0], field.shape[-1]), dtype=np.float64)
    for ox in (0, 1):
        wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
        for oy in (0, 1):
            wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
            for oz in (0, 1):
                wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                w = wx * wy * wz
                out += w[:, None] * field[i0[:, 0] + ox, i0[:, 1] + oy, i0[:, 2] + oz]
    return out


class _CurlNoise:
    """Champ de turbulence de VITESSE, spatialement coherent, pour
    l'emission de particules (voir `_turbulent_emission`).

    Remplace un bruit blanc par particule (erreur de conception mesuree :
    le transfert particules->grille du solveur MLS-MPM moyenne le bruit sur
    le stencil de grille, ce qui detruit ~92% d'un bruit blanc des la
    premiere frame). Ce champ est construit pour SURVIVRE a ce filtrage :

    1. Un potentiel vectoriel `psi` aleatoire gaussien est tire sur une
       grille grossiere reguliere de pas `L_noise = 4 * dx` (`dx` = pas de
       grille FIN du solveur), couvrant la bbox de l'emetteur avec une
       marge de 2 cellules grossieres de chaque cote (pour que
       l'interpolation trilineaire reste valide en bord d'emetteur, y
       compris apres jitter positionnel).
    2. La vitesse est le rotationnel de `psi` (differences finies
       centrees, `_curl_from_potential`), donc a DIVERGENCE NULLE : elle
       n'injecte pas de compression parasite, que l'equation d'etat de
       Tait (materiau eau) paierait en onde de pression.
    3. Le champ DERIVE LENTEMENT dans le temps plutot que d'etre fixe (ce
       qui tordrait le jet de facon figee) ou retire independamment a
       chaque frame (ce qui ramenerait un bruit blanc EN TEMPS, filtre de
       la meme facon par le solveur) : une 4e dimension temporelle est
       ajoutee, de pas `T_noise = L_noise / v_ref` (temps de retournement
       naturel d'un tourbillon de taille `L_noise`), et le champ
       echantillonne est interpole LINEAIREMENT entre les deux tranches
       temporelles ("slabs") encadrant l'instant courant.

    Les slabs sont generees PARESSEUSEMENT, chacune depuis un generateur
    numpy DEDIE seede par son propre indice (`[seed, emitter_index,
    slab_index]`, jamais l'etat global de numpy) : la duree du bake n'est
    donc pas bornee (aucune preallocation de memoire), et le resultat reste
    reproductible bit pour bit. Les slabs deja tirees sont mises en cache
    (`self._slabs`) pour ne pas etre regenerees a chaque frame.

    L'amplitude est normalisee : chaque slab de vitesse est divisee par son
    ECART-TYPE CALCULE SUR LA GRILLE GROSSIERE COMPLETE (jamais sur
    l'echantillon de particules courant, qui rendrait l'amplitude
    dependante du nombre de particules et bruitee quand elles sont peu
    nombreuses) — `sample()` renvoie donc un champ d'ecart-type ~1,
    a multiplier par `sigma_v` cote appelant (`_turbulent_emission`).
    """

    __slots__ = (
        "seed", "emitter_index", "lo", "cell", "shape", "v_ref", "t_noise",
        "_slabs",
    )

    def __init__(self, seed, emitter_index, lo, hi, dx, vel, spacing, frame_dt):
        cell = 4.0 * float(dx)
        margin = 2.0 * cell
        lo_arr = np.asarray(lo, dtype=np.float64) - margin
        hi_arr = np.asarray(hi, dtype=np.float64) + margin
        span = np.maximum(hi_arr - lo_arr, cell)
        n_cells = np.maximum(1, np.ceil(span / cell).astype(np.int64))

        self.seed = int(seed)
        self.emitter_index = int(emitter_index)
        self.lo = lo_arr
        self.cell = cell
        self.shape = (
            int(n_cells[0]) + 1, int(n_cells[1]) + 1, int(n_cells[2]) + 1,
        )

        vel_arr = np.asarray(vel, dtype=np.float64)
        speed = float(np.linalg.norm(vel_arr))
        self.v_ref = max(speed, spacing / frame_dt)
        self.t_noise = cell / self.v_ref

        self._slabs = {}

    def _slab_velocity(self, slab_index):
        cached = self._slabs.get(slab_index)
        if cached is not None:
            return cached
        rng = np.random.default_rng(
            [self.seed, self.emitter_index, int(slab_index)]
        )
        psi = rng.normal(0.0, 1.0, size=self.shape + (3,))
        curl = _curl_from_potential(psi, self.cell)
        std = float(np.std(curl))
        if std > 1e-12:
            curl /= std
        self._slabs[slab_index] = curl
        return curl

    def sample(self, points, t):
        """Vitesse de turbulence `(n, 3)` float64, ecart-type ~1 par
        construction, aux positions `points` `(n, 3)` (espace solveur,
        float64) et a l'instant `t` (secondes, origine arbitraire mais
        commune a tout le bake — voir `BQ_OT_bake` pour la convention
        utilisee : `frame_index * frame_dt`)."""
        if points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float64)

        slab_f = float(t) / self.t_noise
        slab0 = int(math.floor(slab_f))
        frac = slab_f - slab0

        field0 = self._slab_velocity(slab0)
        v0 = _sample_trilinear(field0, points, self.lo, self.cell)
        if frac <= 0.0:
            return v0
        field1 = self._slab_velocity(slab0 + 1)
        v1 = _sample_trilinear(field1, points, self.lo, self.cell)
        return (1.0 - frac) * v0 + frac * v1


def _turbulent_emission(points, vel, turbulence, spacing, frame_dt, rng, dx=0.0, t=0.0, noise=None):
    """Perturbe des positions/vitesses d'emission pour casser la regularite
    parfaite d'un reseau d'emission (jet "trop lisse").

    `points` : `(n, 3)` positions non perturbees (espace solveur).
    `vel` : vitesse uniforme `(3,)` avant perturbation (espace solveur).
    `turbulence` : reglage artiste, fraction de la vitesse caracteristique.
    `spacing` : pas du reseau d'emission (espace solveur).
    `frame_dt` : duree de la frame courante (secondes).
    `rng` : `numpy.random.Generator` DEDIE a cet appel (voir docstring de
    `BQ_OT_bake` sur la construction du generateur, determinisme requis) —
    ne pilote plus que le JITTER POSITIONNEL, le bruit de vitesse etant
    porte par `noise` (voir plus bas).
    `dx` : pas de grille FIN du solveur, uniquement utilise (turbulence >
    0) pour verifier la coherence de `noise` avec la resolution courante.
    `t` : instant courant (secondes), pour l'interpolation temporelle du
    champ de turbulence (voir `_CurlNoise.sample`).
    `noise` : `_CurlNoise` PORTEUR du champ de turbulence pour cet
    emetteur — construit par l'appelant (persiste via `_InflowState` pour
    un emetteur INFLOW, construit a la volee pour un emetteur BLOCK, voir
    `BQ_OT_bake.invoke`). Requis (non `None`) des que `turbulence > 0`.

    Renvoie `(positions_perturbees, vitesses)`, toutes deux `(n, 3)`
    float32. Si `turbulence <= 0`, renvoie les positions INCHANGEES et la
    vitesse repetee sur `n` lignes (passe-plat exact, aucune consommation du
    generateur aleatoire, `noise` jamais touche).

    Regle (contraintes de conception, pas des choix libres) :

        v_ref   = max(|vel|, spacing / frame_dt)   (= noise.v_ref)
        sigma_v = turbulence * v_ref
        amp_pos = turbulence * 0.4 * spacing

        velocities = vel + sigma_v * noise.sample(points, t)
        positions  = points + jitter uniforme U(-amp_pos, amp_pos) par axe

    `v_ref = max(|vel|, spacing/frame_dt)` : la turbulence est une FRACTION
    d'une vitesse caracteristique. Prendre `|vel|` seul rendrait le reglage
    inoperant sur un emetteur a vitesse nulle ; `spacing/frame_dt` (la
    vitesse a laquelle une particule franchit un site en une frame) est
    l'echelle de vitesse naturelle du solveur et sert de plancher. C'est
    aussi ce `v_ref` qui fixe le pas temporel de derive de `noise` (voir
    `_CurlNoise`), les deux etant calcules une seule fois, a la
    construction de `noise`, pour rester coherents entre eux.

    `0.4 * spacing` sur le jitter positionnel est une BORNE DURE, a ne
    jamais augmenter. L'emission continue (INFLOW) identifie les sites
    occupes en arrondissant chaque position au site de reseau le plus
    proche (`_site_indices`, ce fichier). Un deplacement de plus de
    `0.5 * spacing` ferait basculer une particule dans la case du site
    voisin et casserait le test d'occupation, donc le debit. `0.4` laisse
    la marge. Le jitter positionnel n'est PAS filtre par le transfert
    particules->grille du solveur (il agit sur les positions, pas sur un
    champ transporte) : il garde sa regle d'origine, un bruit uniforme
    INDEPENDANT par particule et par frame, sans besoin de coherence
    spatiale.

    Le bruit de VITESSE, lui, EST filtre par ce transfert (P2G) : un bruit
    blanc y est detruit statistiquement (mesure : ~92% des la premiere
    frame), d'ou le champ spatialement coherent `_CurlNoise` — voir sa
    docstring pour le detail de la conception et sa justification.
    """
    n = points.shape[0] if hasattr(points, "shape") else len(points)
    vel_arr = np.asarray(vel, dtype=np.float64)

    if turbulence <= 0.0:
        pos_out = np.asarray(points, dtype=np.float32)
        vel_out = np.broadcast_to(vel_arr.astype(np.float32), (n, 3)).copy()
        return pos_out, vel_out

    assert noise is not None, (
        "_turbulent_emission: noise (_CurlNoise) requis des que "
        "turbulence > 0"
    )
    assert abs(noise.cell - 4.0 * dx) <= 1e-6 * max(1.0, abs(4.0 * dx)), (
        "_turbulent_emission: le champ de turbulence fourni ne correspond "
        "pas au pas de grille fin courant (incoherence dx <-> _CurlNoise, "
        "bug d'appelant)"
    )

    v_ref = noise.v_ref
    sigma_v = turbulence * v_ref
    amp_pos = turbulence * 0.4 * spacing

    points_f64 = np.asarray(points, dtype=np.float64)
    noise_vel = noise.sample(points_f64, t)
    velocities = vel_arr + sigma_v * noise_vel
    positions = points_f64 + rng.uniform(-amp_pos, amp_pos, size=(n, 3))

    return positions.astype(np.float32), velocities.astype(np.float32)


def _site_indices(points, spacing):
    """Indices `(n, 3)` int64 du reseau global correspondant a `points`
    (espace solveur), arrondis au site le plus proche : `round((p -
    spacing/2) / spacing)` — l'inverse de la formule de placement des
    sites dans `_lattice_points_in_box` (`spacing/2 + k*spacing`)."""
    return np.rint((points - spacing / 2.0) / spacing).astype(np.int64)


def _void_rows(int_arr):
    """Vue `(n,)` de type `void` sur un tableau entier `(n, k)` C-contigu,
    pour comparer des LIGNES entieres via `np.isin` (test d'appartenance
    vectorise, sans boucle Python) plutot que composante par composante."""
    arr = np.ascontiguousarray(int_arr)
    return arr.view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))).ravel()


class _InflowState:
    """Etat maintenu par `BQ_OT_bake` pour un emetteur `INFLOW`, pour toute
    la duree du bake. `sites` (le nuage complet de sites d'emission,
    espace solveur) est calcule UNE FOIS et reste FIXE (voir docstring de
    module) : aucun autre etat mutable n'est necessaire, la saturation du
    volume se lit directement dans les positions courantes de la
    simulation a chaque frame (voir `BQ_OT_bake._emit_inflow_sites`).

    `noise` (un `_CurlNoise`, voir sa docstring) porte le champ de
    turbulence de VITESSE pour cet emetteur — construit UNE FOIS a
    `BQ_OT_bake.invoke` (comme `sites`) et REUTILISE (avec son cache de
    slabs temporelles) a chaque frame : c'est ce qui permet au champ de
    deriver de facon COHERENTE dans le temps plutot que d'etre retire
    independamment chaque frame (voir `_CurlNoise`, paragraphe sur la
    derive temporelle)."""

    __slots__ = (
        "name",
        "mat_id",
        "vel",
        "spacing",
        "dx",
        "sites",
        "turbulence",
        "turbulence_seed",
        "emitter_index",
        "noise",
    )

    def __init__(
        self,
        name,
        mat_id,
        vel,
        spacing,
        sites,
        dx=0.0,
        turbulence=0.0,
        turbulence_seed=0,
        emitter_index=0,
        noise=None,
    ):
        self.name = name
        self.mat_id = mat_id
        self.vel = vel
        self.spacing = spacing
        self.dx = dx
        self.sites = sites
        self.turbulence = turbulence
        self.turbulence_seed = turbulence_seed
        self.emitter_index = emitter_index
        self.noise = noise


# ---------------------------------------------------------------------------
# Colliders animes
# ---------------------------------------------------------------------------
#
# Contrairement aux emetteurs, un collider est reevalue a CHAQUE frame (voir
# `BQ_OT_bake._advance_scene_frame` / `_update_colliders`) : sa geometrie et
# sa vitesse par sommet dependent de la pose courante du depsgraph, qui
# n'existe que parce que `_advance_frame` fait desormais avancer la frame
# Blender avant chaque `step` (le point structurant de ce jalon).
#
# Vitesse par sommet : `(position_monde_courante - position_monde_precedente)
# / frame_dt`, calculee en espace MONDE puis convertie en espace solveur par
# `world_to_solver_dir_array` (SANS translation — voir sa docstring, le piege
# documente du jalon). Nulle a la premiere frame d'un collider (pas de
# position precedente).
#
# Topologie changeante (remesh, modificateur variable) : si le nombre de
# sommets change d'une frame a l'autre, l'appariement sommet a sommet est
# faux (l'indice N ne designe plus le "meme" sommet). Ce cas donne une
# vitesse NULLE au collider pour cette frame (plutot qu'un vecteur delirant
# calcule entre deux sommets sans rapport), et n'est signale a l'artiste
# qu'UNE SEULE fois par collider (`warned_topology`), pas a chaque frame.


class _ColliderState:
    """Etat maintenu par `BQ_OT_bake` pour un collider, pour toute la duree
    du bake.

    Un collider FIXE (`dynamic` faux — cas majoritaire : bassin, sol,
    obstacle anime a la main) conserve EXACTEMENT le comportement historique
    cote FLUIDE : position MONDE de ses sommets a la frame precedente (pour
    la difference finie de vitesse), avertissement de topologie changeante
    au plus une fois, aucune propriete massique calculee, aucun refus de
    bake sur maillage ouvert, jamais de keyframe posee (voir
    `_collect_collider_frame`, qui l'ignore purement et simplement dans son
    propre traitement pour la geometrie transmise au FLUIDE).

    Depuis M17/phase B (D9, D12), TOUT collider — dynamique OU fixe — porte
    en plus un repere de CORPS (origine, `com0_solver`/`com0_world`/`m0`) et
    son maillage de repos (`rest_tris`/`rest_offsets`, espace SOLVEUR,
    evalue UNE SEULE FOIS a la premiere frame du bake par
    `_setup_collider_body`) : c'est ce qui permet au SOLIDE de heurter un
    sol fixe, pas seulement d'etre pousse par le fluide. Un collider
    DYNAMIQUE porte en plus ses proprietes massiques, necessaires a la fois
    a la declaration du corps rigide aupres du solveur
    (`Sim.set_collider_bodies`) et a la reconstruction de ses triangles a
    chaque frame depuis l'etat rigide courant (voir
    `_dynamic_collider_geometry`), ainsi qu'a la composition des keyframes
    en fin de bake (`BQ_OT_bake._post_keyframes`). Un collider FIXE, lui,
    n'a pas de masse (D12 : masse infinie, `mass=0`/`inv_inertia=0` cote
    `BqRigidBody`) mais A BESOIN de son repere de corps pour que le SDF
    local construit par `_setup_collider_body` (D9) soit interroge au bon
    endroit.

    `mesh_closed`/`closed_message` (D9/point 3 de B4) : resultat de
    `sampling.check_mesh_closed`, calcule UNE FOIS au bake par
    `_setup_collider_body`, jamais depuis un `draw()` (trop couteux — lancer
    de rayons sur un BVH). `body_sdf_built` est vrai si (et seulement si) le
    SDF local et les echantillons de surface ont ete construits pour ce
    corps (`Sim.build_body_sdf`/`Sim.set_body_samples`, appeles par
    `BQ_OT_bake.invoke`) : faux pour un collider DYNAMIQUE au maillage
    ouvert (le bake refuse deja ce cas, voir plus bas), et faux pour un
    collider FIXE au maillage ouvert (D9 : un plan ne delimite aucun
    interieur, il ne peut pas porter de champ de distance signee — il reste
    un collider FLUIDE valide, mais ne participe jamais au contact solide,
    voir l'avertissement pose par `BQ_OT_bake.invoke`).

    `body_index` est l'indice de ce collider dans le tableau `BqRigidBody`
    transmis a `Sim.set_collider_bodies` — TOUS les colliders y figurent,
    dynamiques ou non (voir docs/plan-milestone-17.md, section fichiers :
    "le solveur attend un corps par collider pour que l'attribution
    tri_body soit coherente") — assigne par `BQ_OT_bake.invoke` dans l'ordre
    de `self._collider_states`.
    """

    __slots__ = (
        "obj",
        "friction",
        "prev_verts_world",
        "warned_topology",
        "dynamic",
        "body_index",
        "density",
        "use_gravity",
        "added_mass",
        "lock_lin",
        "lock_ang",
        "restitution",
        "rest_tris",
        "rest_offsets",
        "com0_solver",
        "com0_world",
        "m0",
        "mass",
        "inv_inertia",
        "mesh_closed",
        "closed_message",
        "body_sdf_built",
        "prev_pose_loc_world",
        "prev_pose_rot_world",
    )

    def __init__(self, obj, friction):
        self.obj = obj
        self.friction = friction
        self.prev_verts_world = None
        self.warned_topology = False
        # Pose MONDE precedente (repere de corps, pas maillage) d'un
        # collider CINEMATIQUE au SDF construit -- voir
        # `_kinematic_pose_frame`. `None` tant qu'aucune frame precedente
        # n'a ete traitee (vitesse nulle a la premiere frame, meme
        # convention que `prev_verts_world`).
        self.prev_pose_loc_world = None
        self.prev_pose_rot_world = None

        op = obj.bourrasque
        self.dynamic = bool(op.dynamic)
        self.body_index = 0
        self.density = float(op.density)
        self.use_gravity = bool(op.use_gravity)
        self.added_mass = float(op.added_mass)
        self.lock_lin = tuple(int(bool(b)) for b in op.lock_location)
        self.lock_ang = tuple(int(bool(b)) for b in op.lock_rotation)
        self.restitution = float(op.restitution)

        # Rempli par `_setup_collider_body`, pour TOUS les colliders (D9,
        # D12) : repere de corps, maillage de repos (espace solveur). Les
        # proprietes massiques (mass/inv_inertia) ne sont significatives que
        # pour un collider DYNAMIQUE (D12 : un collider fixe garde mass=0 /
        # inv_inertia=None, traduit en masse infinie cote `BqRigidBody`).
        self.rest_tris = None
        self.rest_offsets = None
        self.com0_solver = None
        self.com0_world = None
        self.m0 = None
        self.mass = 0.0
        self.inv_inertia = None
        self.mesh_closed = None
        self.closed_message = ""
        self.body_sdf_built = False


def _setup_collider_body(state, depsgraph, origin, size):
    """Capture le repere de CORPS et le maillage de repos d'UN collider,
    dynamique OU fixe, a la premiere frame du bake (D6/D9 du plan M17) —
    generalise `_setup_dynamic_collider` (phase A) a TOUS les colliders
    (phase B, D12 : « les colliders statiques sont des corps de masse
    infinie... aucun cas particulier »).

    Collider DYNAMIQUE (`state.dynamic` vrai, comportement D6/D7 inchange) :
    maillage ferme EXIGE — leve `ValueError` (message DEJA FORME, nommant
    l'objet) si le maillage est ouvert ou de volume degenere, traduit par
    l'appelant en refus de bake. Proprietes massiques calculees
    (`rigidbody.mass_properties`), origine du repere de corps = centre de
    masse.

    Collider FIXE (`state.dynamic` faux) : AUCUN refus de bake sur maillage
    ouvert (D14 : le collider fixe ne perd rien — un sol ouvert continue de
    passer, comme avant ce jalon). Aucune propriete massique. Origine du
    repere de corps = centre de l'AABB du maillage (D9 : « le centroide ou
    le centre de son AABB suffit », il ne bouge pas — ou, pour un
    cinematique anime, seule sa pose de REPOS a la premiere frame compte
    ici).

    Dans les deux cas, `state.mesh_closed` est memorise (D9/point 3 de
    B4) : un maillage OUVERT ne peut pas porter de champ de distance
    signee coherent (le SDF signe suppose un volume ferme), donc
    `state.body_sdf_built` reste faux et l'appelant (`BQ_OT_bake.invoke`)
    ne construit NI SDF NI echantillons de surface pour ce corps — il
    reste un collider FLUIDE valide (voir `_collect_collider_frame`, qui
    l'ignore purement et simplement), mais ne participera jamais au
    contact solide<->solide. Un collider dynamique au maillage ouvert ne
    peut de toute facon jamais atteindre ce point (refus ci-dessus) :
    `body_sdf_built` n'est donc faux pour un dynamique que si son maillage
    est degenere autrement (aucun triangle).

    Renvoie un message d'avertissement (str) si l'echelle de l'objet
    DYNAMIQUE n'est pas uniforme (la decomposition en keyframes sera
    approximative, D8), sinon None — sans objet pour un collider fixe, qui
    ne recoit jamais de keyframe. Mute `state` en place.
    """
    from . import sampling

    obj = state.obj

    closed, message = sampling.check_mesh_closed(obj)
    state.mesh_closed = closed
    state.closed_message = message

    if state.dynamic and not closed:
        raise ValueError(
            f"« {obj.name} » ne peut pas être un collider dynamique : {message}"
        )

    obj_eval = obj.evaluated_get(depsgraph)
    verts_world, tris = sampling.evaluated_world_mesh(obj_eval)

    if verts_world.shape[0] == 0 or tris.shape[0] == 0:
        # Collider sans triangle exploitable (maillage vide) : repere de
        # corps degenere a l'origine de l'objet, aucun SDF constructible —
        # il ne delimite de toute facon rien. `state.dynamic` ne peut pas
        # etre vrai ici (aurait deja leve ValueError plus haut via
        # `check_mesh_closed`, un maillage vide n'etant jamais ferme).
        loc_world = np.array(obj.matrix_world.translation, dtype=np.float64)
        state.com0_world = loc_world
        state.com0_solver = np.array(
            world_to_solver(tuple(loc_world), origin, size), dtype=np.float64
        )
        state.m0 = np.array(obj.matrix_world, dtype=np.float64)
        state.rest_tris = np.empty((0, 3), dtype=np.int64)
        state.rest_offsets = np.empty((0, 3), dtype=np.float64)
        state.body_sdf_built = False
        return None

    verts_solver = world_to_solver_array(verts_world, origin, size)

    if state.dynamic:
        from . import rigidbody

        try:
            _volume, mass, com0_solver, inertia_solver = rigidbody.mass_properties(
                verts_solver, tris, state.density
            )
        except ValueError as exc:
            raise ValueError(f"« {obj.name} » : {exc}") from exc
        state.mass = mass
        state.inv_inertia = np.linalg.inv(inertia_solver)
    else:
        # D9 : « le centroide ou le centre de son AABB suffit » — pas
        # d'integrale de volume ici, un collider fixe peut etre ouvert.
        lo = verts_solver.min(axis=0)
        hi = verts_solver.max(axis=0)
        com0_solver = (lo + hi) / 2.0

    state.rest_tris = tris
    state.rest_offsets = verts_solver - com0_solver
    state.com0_solver = com0_solver
    state.com0_world = np.array(
        solver_to_world(tuple(com0_solver), origin, size), dtype=np.float64
    )
    state.m0 = np.array(obj.matrix_world, dtype=np.float64)
    # SDF/echantillons construits par l'appelant UNIQUEMENT si le maillage
    # est ferme (voir docstring) — jamais pour un collider (fixe ou,
    # structurellement, dynamique) au maillage ouvert.
    state.body_sdf_built = closed

    if not state.dynamic:
        return None

    loc, rot, scale = obj.matrix_world.decompose()
    scale_ref = max(abs(scale.x), abs(scale.y), abs(scale.z), 1.0)
    if (
        abs(scale.x - scale.y) > 1e-4 * scale_ref
        or abs(scale.y - scale.z) > 1e-4 * scale_ref
        or abs(scale.x - scale.z) > 1e-4 * scale_ref
    ):
        return (
            f"« {obj.name} » a une échelle non uniforme : la décomposition "
            "de sa transformation rigide en position/rotation pour les "
            "keyframes sera approximative."
        )
    return None


def _build_rigid_bodies(collider_states):
    """Construit le tableau `lib.BqRigidBody` transmis a
    `Sim.set_collider_bodies` — TOUS les colliders y figurent (dynamiques ET
    fixes), dans l'ordre de `collider_states` (= `body_index`), pour que
    l'attribution `tri_body` reste coherente (voir docs/plan-milestone-17.md).

    Un collider FIXE (`dynamic` faux, D12) recoit un corps de masse
    INFINIE (`mass=0`, `inv_inertia=0`, jamais integre — `dynamic=0` fait
    retomber `k_grid_update`/`k_body_predict`/`k_advance_bodies` sur le
    chemin cinematique actuel, voir `bourrasque.h`), mais `x` porte
    desormais le VRAI repere de corps (`state.com0_solver`, calcule par
    `_setup_collider_body` pour TOUS les colliders depuis M17/phase B — ce
    n'etait pas le cas en phase A, ou un collider fixe recevait `x=(0,0,0)`
    puisqu'aucun SDF local n'existait encore pour lui) : le SDF local
    construit par `Sim.build_body_sdf` (D9) est exprime en repere de corps,
    une requete de contact `p_local = R^T(p - x)` serait fausse si `x` ne
    correspondait pas a l'origine reellement utilisee a la construction.
    """
    bodies = []
    for state in collider_states:
        if state.com0_solver is not None:
            x = tuple(float(v) for v in state.com0_solver)
        else:
            # Compatibilite avec un appelant qui n'a pas encore appele
            # `_setup_collider_body` sur ce corps (ex. harnais de test de
            # phase A qui n'exerce que les colliders dynamiques) — repli
            # neutre sur l'origine du domaine, comme le comportement
            # historique de phase A.
            x = (0.0, 0.0, 0.0)
        if state.dynamic:
            inv_inertia = tuple(float(v) for v in state.inv_inertia.reshape(-1))
            mass = float(state.mass)
        else:
            inv_inertia = (0.0,) * 9
            mass = 0.0
        bodies.append(
            lib.BqRigidBody(
                dynamic=1 if state.dynamic else 0,
                mass=mass,
                inv_inertia=inv_inertia,
                x=x,
                q=(1.0, 0.0, 0.0, 0.0),
                v=(0.0, 0.0, 0.0),
                w=(0.0, 0.0, 0.0),
                use_gravity=1 if state.use_gravity else 0,
                added_mass=float(state.added_mass),
                lock_lin=state.lock_lin,
                lock_ang=state.lock_ang,
                restitution=float(state.restitution),
                # Contact corps<->corps (D11) : coefficient de Coulomb du
                # corps, meme champ `friction` que l'artiste regle pour le
                # contact FLUIDE<->solide (obj.bourrasque.friction) — voir
                # BQ_PT_collider (ui.py) pour la clarification de portee.
                friction=float(state.friction),
            )
        )
    return bodies


# Rotation FIXE reliant les axes SOLVEUR aux axes MONDE — partie lineaire de
# `world_to_solver`/`world_to_solver_dir` (voir transform.py) :
# (dx, dy, dz) -> (dx, dz, -dy). Orthogonale, determinant +1 (rotation pure,
# pas une reflexion).
#
# Necessaire pour convertir l'ORIENTATION d'un corps rigide en keyframe
# monde : le couple qu'accumule le solveur (D1 du plan) vient de positions
# de GRILLE, donc de vecteurs SOLVEUR (x_noeud - com) — l'integration de
# l'etat rigide (x, q, v, w) qu'il en deduit est donc necessairement tenue
# en axes SOLVEUR d'un bout a l'autre (c'est aussi pourquoi
# `_setup_collider_body` calcule les proprietes massiques sur des
# sommets deja convertis en espace solveur — meme coherence d'axes que
# `inv_inertia`). La position se convertit simplement par
# `solver_to_world` (translation + cette meme rotation) ; l'orientation
# exige une CONJUGAISON par cette rotation fixe (changement de repere d'une
# matrice de rotation), pas une simple substitution — une conjugaison par
# l'identite laisse l'identite inchangee, ce qui NE distingue PAS cette
# conversion d'un cablage naif a l'etat de repos (voir l'invariant de
# `rigidbody.compose_body_transform`) : verifiee independamment sur une
# rotation non triviale par le script de validation Blender reel du jalon.
_R_SOLVER_FROM_WORLD = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)


def _solver_quat_to_world(q_solver):
    """Convertit un quaternion d'orientation de corps rigide — integre par
    le solveur en axes SOLVEUR — vers son equivalent en axes MONDE, par
    conjugaison avec la rotation fixe `_R_SOLVER_FROM_WORLD` (voir sa
    docstring)."""
    from . import rigidbody

    r_solver = rigidbody.quat_to_matrix(q_solver)
    r_world = _R_SOLVER_FROM_WORLD.T @ r_solver @ _R_SOLVER_FROM_WORLD
    return rigidbody.quat_from_matrix(r_world)


def _world_matrix_to_solver_quat(r_world):
    """Inverse de `_solver_quat_to_world` : convertit une matrice de
    rotation MONDE en quaternion d'orientation SOLVEUR, par conjugaison
    avec la meme rotation fixe `_R_SOLVER_FROM_WORLD` (voir sa docstring).
    Prend directement une MATRICE (pas un quaternion) car l'appelant
    (`_kinematic_pose_frame`) en a deja une sous la main — evite un aller-
    retour quaternion inutile."""
    from . import rigidbody

    r_solver = _R_SOLVER_FROM_WORLD @ r_world @ _R_SOLVER_FROM_WORLD.T
    return rigidbody.quat_from_matrix(r_solver)


def _kinematic_pose_frame(collider_states, origin, size, frame_dt):
    """Pose/vitesse SOLVEUR courante de chaque collider CINEMATIQUE
    (`dynamic` faux) dont le SDF de contact a ete construit
    (`state.body_sdf_built`), pour la frame COURANTE de la scene — suppose
    `scene.frame_set` deja appele par l'appelant (meme convention que
    `_collect_collider_frame`, dont c'est le pendant pour le contact
    SOLIDE plutot que pour le FLUIDE).

    Necessaire depuis M17/phase B (D9/D12, tache B4, point 2) : un
    collider `dynamic == 0` n'est JAMAIS avance par le solveur (voir
    `bourrasque.h`, `bq_set_body_pose`) — sans cet appel explicite, son SDF
    de contact resterait fige a la pose de la premiere frame du bake alors
    que le fluide, lui, continue de voir sa geometrie animee reelle.

    Position : transformation rigide de `obj.matrix_world` relativement a
    la pose de reference `state.m0` (capturee par `_setup_collider_body` a
    la premiere frame du bake, D6) appliquee a `state.com0_world` (origine
    du repere de corps, MONDE) — PAS une simple lecture de
    `obj.matrix_world.translation`, qui donnerait l'origine de l'OBJET, pas
    celle du repere de corps (com0/centroide, potentiellement decale de
    l'origine de l'objet).

    Orientation : extraite de la meme transformation relative
    (`rigidbody.decompose_loc_rot`), convertie en axes SOLVEUR par
    `_world_matrix_to_solver_quat` (meme conjugaison que pour les
    keyframes de sortie, D8 — voir `_R_SOLVER_FROM_WORLD`).

    Vitesse lineaire ET angulaire : DIFFERENCE FINIE entre deux appels
    successifs (meme motif que la vitesse par sommet de
    `_collect_collider_frame`, meme `frame_dt`) — le solveur de contact
    (B3b, D11) travaille sur la vitesse RELATIVE au contact : sans elle, un
    mur qui avance ne transfererait rien au premier contact, penetrerait,
    puis serait repousse par la seule correction de position (split
    impulse) — un contact mou plutot qu'un entrainement franc. Vitesse
    angulaire deduite de l'ECART DE ROTATION entre deux frames (formule
    standard axe*angle depuis la partie antisymetrique de `R(f) . R(f-1)^T`
    — equivalente a l'ecart de quaternion, evite un aller-retour matrice ->
    quaternion -> matrice puisque `_kinematic_pose_frame` a deja les
    matrices sous la main). Nulle a la premiere frame d'un collider (pas de
    pose precedente, `state.prev_pose_loc_world is None`) ou si l'angle de
    rotation entre deux frames est negligeable (evite une division par un
    sinus quasi nul).

    Limite ASSUMEE et distincte de celle-ci : un collider dont le MAILLAGE
    se deforme (armature, shape keys, modificateur deformant) n'est pas
    rigide — sa transformation `matrix_world` seule ne capture pas la
    deformation, son SDF de contact resterait celui de la premiere frame
    meme si cette fonction met a jour x/q/v/w de l'objet dans son ensemble.
    Non detectable a bas cout (il faudrait re-echantillonner le maillage
    chaque frame, exactement ce que ce jalon evite pour le contact), donc
    non signale ici — a documenter a l'artiste hors de ce lot si le besoin
    se presente.

    Mute `state.prev_pose_loc_world`/`state.prev_pose_rot_world` en place
    (etat persistant entre appels, une frame apres l'autre — meme motif que
    `state.prev_verts_world`).

    Renvoie une liste de `(body_index, x_solver, q_solver, v_solver,
    w_solver)`, un tuple par collider concerne — directement au format
    attendu par `Sim.set_body_pose` (voir `lib.py`).
    """
    from . import rigidbody

    out = []
    for state in collider_states:
        if state.dynamic or not state.body_sdf_built:
            continue

        m_now = np.array(state.obj.matrix_world, dtype=np.float64)
        delta = m_now @ np.linalg.inv(state.m0)

        _loc, quat_world, _uniform_scale_ok = rigidbody.decompose_loc_rot(delta)
        r_world = rigidbody.quat_to_matrix(quat_world)
        loc_world = (delta @ np.append(state.com0_world, 1.0))[:3]

        prev_loc = state.prev_pose_loc_world
        prev_rot = state.prev_pose_rot_world
        if prev_loc is None:
            v_world = np.zeros(3, dtype=np.float64)
            w_world = np.zeros(3, dtype=np.float64)
        else:
            v_world = (loc_world - prev_loc) / frame_dt

            r_rel = r_world @ prev_rot.T
            cos_angle = float(np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0))
            angle = math.acos(cos_angle)
            if angle > 1e-7:
                axis = np.array(
                    [
                        r_rel[2, 1] - r_rel[1, 2],
                        r_rel[0, 2] - r_rel[2, 0],
                        r_rel[1, 0] - r_rel[0, 1],
                    ],
                    dtype=np.float64,
                ) / (2.0 * math.sin(angle))
                w_world = axis * angle / frame_dt
            else:
                w_world = np.zeros(3, dtype=np.float64)

        state.prev_pose_loc_world = loc_world
        state.prev_pose_rot_world = r_world

        x_solver = world_to_solver(tuple(loc_world), origin, size)
        q_solver = tuple(float(c) for c in _world_matrix_to_solver_quat(r_world))
        v_solver = world_to_solver_dir(tuple(v_world))
        w_solver = world_to_solver_dir(tuple(w_world))

        out.append((state.body_index, x_solver, q_solver, v_solver, w_solver))

    return out


def _dynamic_collider_geometry(dynamic_states, body_state):
    """Triangles/vitesses/frictions/tri_body des colliders DYNAMIQUES,
    reconstruits depuis l'etat rigide COURANT lu du solveur (`body_state`,
    ndarray `(n_bodies, 13)`, meme mise en forme que
    `Sim.read_collider_bodies` : `x[3], q[4], v[3], w[3]` par corps,
    entierement en espace SOLVEUR) — fonction PURE (aucun bpy), callable
    depuis le thread de calcul du bake (voir `_bake_worker`).

    Pour chaque sommet de repos (espace solveur, capture UNE FOIS a la
    premiere frame du bake, voir `_setup_collider_body`/D6) : position
    courante `x_corps + R(q) @ decalage`, vitesse `v + w x (position -
    x_corps)` — champ de vitesse RIGIDE, PAS une difference finie (D4 du
    plan : c'est ce qui stabilise le couplage, le corps qui accelere est
    moins pousse des le sous-pas suivant).

    `dynamic_states` : sous-ensemble de `collider_states` dont `dynamic` est
    vrai. Renvoie `(tri_all, vel_all, fric_all, tri_body_all)`, memes
    conventions que `_collect_collider_frame`.
    """
    from . import rigidbody

    tri_chunks = []
    vel_chunks = []
    fric_chunks = []
    body_chunks = []

    for state in dynamic_states:
        n_tri = state.rest_tris.shape[0]
        if n_tri == 0:
            continue

        row = body_state[state.body_index]
        x = row[0:3]
        q = row[3:7]
        v = row[7:10]
        w = row[10:13]

        R = rigidbody.quat_to_matrix(q)
        verts_now = x[np.newaxis, :] + state.rest_offsets @ R.T
        rel = verts_now - x[np.newaxis, :]
        vel_now = v[np.newaxis, :] + np.cross(np.broadcast_to(w, rel.shape), rel)

        tri_chunks.append(verts_now[state.rest_tris].astype(np.float32))
        vel_chunks.append(vel_now[state.rest_tris].astype(np.float32))
        fric_chunks.append(np.full(n_tri, state.friction, dtype=np.float32))
        body_chunks.append(np.full(n_tri, state.body_index, dtype=np.int32))

    if tri_chunks:
        tri_all = np.concatenate(tri_chunks, axis=0)
        vel_all = np.concatenate(vel_chunks, axis=0)
        fric_all = np.concatenate(fric_chunks, axis=0)
        body_all = np.concatenate(body_chunks, axis=0)
    else:
        tri_all = np.empty((0, 3, 3), dtype=np.float32)
        vel_all = np.empty((0, 3, 3), dtype=np.float32)
        fric_all = np.empty((0,), dtype=np.float32)
        body_all = np.empty((0,), dtype=np.int32)

    return tri_all, vel_all, fric_all, body_all


def _combine_collider_geometry(static_frame, dynamic_geo):
    """Concatene la geometrie STATIQUE (pre-extraite, `_collect_collider_frame`)
    et la geometrie DYNAMIQUE (recalculee depuis l'etat rigide,
    `_dynamic_collider_geometry`) d'une meme frame, en un seul quadruplet
    pret pour `Sim.set_colliders(..., tri_body=...)`. Court-circuite la
    concatenation si l'un des deux cotes est vide (cas courant : un bake
    sans collider dynamique, ou sans collider fixe)."""
    tri_s, vel_s, fric_s, body_s = static_frame
    tri_d, vel_d, fric_d, body_d = dynamic_geo
    if tri_s.shape[0] == 0:
        return tri_d, vel_d, fric_d, body_d
    if tri_d.shape[0] == 0:
        return tri_s, vel_s, fric_s, body_s
    return (
        np.concatenate([tri_s, tri_d], axis=0),
        np.concatenate([vel_s, vel_d], axis=0),
        np.concatenate([fric_s, fric_d], axis=0),
        np.concatenate([body_s, body_d], axis=0),
    )


def _tag_redraw(context):
    """Force le redessin de toutes les zones UI (barre de progression)."""
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


# ---------------------------------------------------------------------------
# BQ_OT_add_domain
# ---------------------------------------------------------------------------


class BQ_OT_add_domain(bpy.types.Operator):
    """Cree un cube domaine a l'origine du curseur 3D."""

    bl_idname = "bq.add_domain"
    bl_label = "Créer un domaine"
    bl_description = "Crée un cube de simulation à l'origine du curseur 3D"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        existing = scene.bourrasque.domain_object
        if existing is None:
            # Un domaine peut exister sans etre reference par la scene (ex.
            # scene rechargee) : on le retrouve via le role avant d'en creer
            # un nouveau.
            for obj in scene.objects:
                if obj.bourrasque.role == "DOMAIN":
                    existing = obj
                    break

        if existing is not None:
            for obj in scene.objects:
                obj.select_set(False)
            existing.select_set(True)
            context.view_layer.objects.active = existing
            scene.bourrasque.domain_object = existing
            self.report({"INFO"}, "Un domaine existe déjà : sélectionné.")
            return {"FINISHED"}

        bpy.ops.mesh.primitive_cube_add(
            size=1.0, location=context.scene.cursor.location
        )
        domain = context.active_object
        domain.name = "Bourrasque_Domain"
        domain.display_type = "WIRE"
        domain.show_in_front = True
        domain.bourrasque.role = "DOMAIN"
        scene.bourrasque.domain_object = domain
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_add_emitter
# ---------------------------------------------------------------------------


class BQ_OT_add_emitter(bpy.types.Operator):
    """Passe l'objet actif en emetteur."""

    bl_idname = "bq.add_emitter"
    bl_label = "Ajouter l'objet actif"
    bl_description = "Fait de l'objet actif un émetteur de particules"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            return False
        if obj == context.scene.bourrasque.domain_object:
            return False
        return True

    def execute(self, context):
        obj = context.active_object
        if obj.bourrasque.role == "EMITTER":
            self.report({"INFO"}, f"« {obj.name} » est déjà un émetteur.")
            return {"FINISHED"}
        obj.bourrasque.role = "EMITTER"

        # Un emetteur sans materiau VALIDE est une reference pendante des sa
        # creation : une scene neuve ne doit jamais en presenter (voir
        # `material_usage`/`ui.py`).
        #
        # Le test porte sur la VALIDITE, pas sur le simple fait que le champ
        # soit non vide : un objet qui a deja ete emetteur, dont le role a ete
        # retire puis le materiau supprime, conserve un `material_name` non
        # vide mais PENDANT — le repasser emetteur le laisserait avec cette
        # reference morte.
        library = context.scene.bourrasque.materials
        if obj.bourrasque.material_name not in {mat.name for mat in library}:
            if len(library) == 0:
                # Bibliotheque vide : on cree une entree « Eau » par defaut
                # (memes valeurs que le preset WATER historique,
                # `_PRESET_WATER`).
                mat = library.add()
                mat.model = _PRESET_WATER["model"]
                mat.rho = _PRESET_WATER["rho"]
                mat.bulk = _PRESET_WATER["bulk"]
                mat.gamma = _PRESET_WATER["gamma"]
                mat.preset = "WATER"
                name = materials.unique_name("Eau", [])
                mat.name_prev = name
                mat.name = name
                obj.bourrasque.material_name = name
            else:
                # Bibliotheque non vide : on prend le materiau ACTIF de la
                # liste, pas `library[0]` — l'artiste qui vient de selectionner
                # « Sable » dans le panneau Materiaux attend que son nouvel
                # emetteur soit du sable, pas la premiere entree de la liste.
                index = context.scene.bourrasque.active_material_index
                if not (0 <= index < len(library)):
                    index = 0
                obj.bourrasque.material_name = library[index].name

        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_add_collider
# ---------------------------------------------------------------------------


class BQ_OT_add_collider(bpy.types.Operator):
    """Passe l'objet actif en collider."""

    bl_idname = "bq.add_collider"
    bl_label = "Ajouter l'objet actif comme collider"
    bl_description = "Fait de l'objet actif un obstacle avec lequel le fluide interagit"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != "MESH":
            return False
        if obj == context.scene.bourrasque.domain_object:
            return False
        return True

    def execute(self, context):
        obj = context.active_object
        if obj.bourrasque.role == "COLLIDER":
            self.report({"INFO"}, f"« {obj.name} » est déjà un collider.")
            return {"FINISHED"}
        obj.bourrasque.role = "COLLIDER"
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_remove_element
# ---------------------------------------------------------------------------


class BQ_OT_remove_element(bpy.types.Operator):
    """Retire le role de l'element selectionne dans la UIList."""

    bl_idname = "bq.remove_element"
    bl_label = "Retirer"
    bl_description = "Retire cet objet de la simulation (domaine ou émetteur)"
    bl_options = {"REGISTER", "UNDO"}

    # `active_element_index` indexe `scene.objects`, PAS la liste filtree de
    # `iter_elements` (voir la convention documentee sur la propriete dans
    # props.py) : le `template_list` de ui.py est construit sur
    # `scene.objects`, et `filter_items` ne fait que masquer/reordonner
    # l'affichage sans changer ce que l'index actif adresse.

    @classmethod
    def poll(cls, context):
        objects = context.scene.objects
        index = context.scene.bourrasque.active_element_index
        if not (0 <= index < len(objects)):
            return False
        return objects[index].bourrasque.role != "NONE"

    def execute(self, context):
        scene = context.scene
        objects = scene.objects
        index = scene.bourrasque.active_element_index
        if not (0 <= index < len(objects)):
            self.report({"ERROR"}, "Aucun élément sélectionné.")
            return {"CANCELLED"}

        obj = objects[index]
        if obj.bourrasque.role == "NONE":
            self.report({"ERROR"}, "Aucun élément sélectionné.")
            return {"CANCELLED"}

        was_domain = obj.bourrasque.role == "DOMAIN"
        obj.bourrasque.role = "NONE"
        if was_domain and scene.bourrasque.domain_object == obj:
            scene.bourrasque.domain_object = None
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_material_add / _remove / _duplicate, BQ_OT_migrate_materials
# ---------------------------------------------------------------------------


class BQ_OT_material_add(bpy.types.Operator):
    """Ajoute un materiau a la bibliotheque de materiaux de la scene."""

    bl_idname = "bq.material_add"
    bl_label = "Ajouter un matériau"
    bl_description = "Ajoute un nouveau matériau à la bibliothèque de la scène"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene
        library = scene.bourrasque.materials

        existing = [m.name for m in library]
        name = materials.unique_name("Eau", existing)

        mat = library.add()
        # `name_prev` initialise a la MEME valeur que `name` avant meme la
        # premiere affectation : sans ca, le callback `update` de `name`
        # (props.py::_on_material_name_update) verrait un `name_prev` vide
        # comme un "vrai" ancien nom lors du tout premier renommage venu de
        # l'utilisateur et tenterait (a tort) de propager depuis une valeur
        # perimee.
        mat.name_prev = name
        mat.name = name

        scene.bourrasque.active_material_index = len(library) - 1
        return {"FINISHED"}


class BQ_OT_material_remove(bpy.types.Operator):
    """Supprime le materiau actif de la bibliotheque de la scene."""

    bl_idname = "bq.material_remove"
    bl_label = "Supprimer le matériau"
    bl_description = (
        "Supprime le matériau actif ; les émetteurs qui le référencent "
        "perdent leur assignation"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.bourrasque.materials) > 0

    def execute(self, context):
        scene = context.scene
        library = scene.bourrasque.materials
        index = scene.bourrasque.active_material_index

        if not (0 <= index < len(library)):
            self.report({"ERROR"}, "Aucun matériau sélectionné.")
            return {"CANCELLED"}

        mat = library[index]

        # AVANT la suppression : vide `material_name` sur tous les
        # emetteurs qui referencent ce materiau, pour ne jamais laisser de
        # reference pendante silencieuse.
        usage, _dangling = material_usage(scene)
        affected = usage.get(mat.name, [])
        for obj in affected:
            obj.bourrasque.material_name = ""

        library.remove(index)
        scene.bourrasque.active_material_index = max(
            0, min(index, len(library) - 1)
        )

        if affected:
            self.report(
                {"INFO"},
                f"Matériau supprimé ; {len(affected)} émetteur(s) ont perdu "
                "leur matériau assigné.",
            )
        else:
            self.report({"INFO"}, "Matériau supprimé.")
        return {"FINISHED"}


class BQ_OT_material_duplicate(bpy.types.Operator):
    """Duplique le materiau actif de la bibliotheque de la scene."""

    bl_idname = "bq.material_duplicate"
    bl_label = "Dupliquer le matériau"
    bl_description = "Duplique le matériau actif (mêmes réglages, nom distinct)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        index = scene.bourrasque.active_material_index
        return 0 <= index < len(scene.bourrasque.materials)

    def execute(self, context):
        scene = context.scene
        library = scene.bourrasque.materials
        index = scene.bourrasque.active_material_index
        source = library[index]

        existing = [m.name for m in library]
        name = materials.unique_name(source.name, existing)

        dup = library.add()
        # `preset` est affecte EN PREMIER : son callback `update`
        # (props.py::_on_preset_update) reecrit model/rho/young/poisson
        # d'apres le preset choisi si celui-ci n'est pas CUSTOM. Affecter
        # ensuite explicitement TOUS les champs physiques depuis `source`
        # garantit que la copie finale est exacte, quel que soit l'effet de
        # bord de ce callback.
        dup.preset = source.preset
        dup.model = source.model
        dup.rho = source.rho
        dup.young = source.young
        dup.poisson = source.poisson
        dup.bulk = source.bulk
        dup.gamma = source.gamma
        dup.friction_angle = source.friction_angle
        dup.cohesion = source.cohesion
        dup.viewport_color = source.viewport_color[:]

        dup.name_prev = name
        dup.name = name

        scene.bourrasque.active_material_index = len(library) - 1
        return {"FINISHED"}


class BQ_OT_migrate_materials(bpy.types.Operator):
    """Construit/complete la bibliotheque de materiaux de la scene depuis
    les reglages materiau herites (DEPRECIES, voir props.py) portes par
    chaque emetteur, puis reassigne `material_name` en consequence.

    Idempotent : un materiau de la bibliotheque dont la `material_key` (voir
    materials.py) correspond deja a ce qu'un emetteur porte est REUTILISE,
    jamais duplique — relancer cet operateur sur une scene deja migree ne
    cree rien de nouveau et ne change aucune assignation.
    """

    bl_idname = "bq.migrate_materials"
    bl_label = "Migrer les matériaux"
    bl_description = (
        "Construit la bibliothèque de matériaux de la scène à partir des "
        "réglages hérités de chaque émetteur (opération idempotente)"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        scene = context.scene

        # N'agit QUE sur les emetteurs dont la reference est vide ou pendante.
        #
        # Ce filtre n'est pas une optimisation, c'est une garde contre une
        # PERTE DE DONNEES (ecart 1 de la revue d'architecture M16, reproduit
        # et mesure avant correction). L'UI expose cet operateur sous le
        # libelle « Reparer », declenche des qu'UN SEUL emetteur est pendant —
        # ce qui arrive aussi dans une scene entierement post-M16, par exemple
        # apres un `bq.material_remove`. Sans ce filtre, l'operateur
        # reassignait TOUS les emetteurs depuis leurs champs DEPRECIES, qui ne
        # sont plus ecrits depuis M16 et valent donc les defauts identiques
        # pour tout le monde : mesure sur une scene de 3 emetteurs assignes a
        # la main (SiropCustom / Gelee / Eau), la suppression d'un seul
        # materiau suivie d'un clic sur « Reparer » les effondrait tous les
        # trois sur un unique « Eau.001 », detruisant deux assignations
        # valides et creant un doublon. Un emetteur deja correctement assigne
        # n'a par definition rien a migrer.
        library_names = {mat.name for mat in scene.bourrasque.materials}
        emitter_objs = [
            obj
            for obj in scene.objects
            if obj.bourrasque.role == "EMITTER"
            and obj.bourrasque.material_name not in library_names
        ]
        if not emitter_objs:
            self.report(
                {"INFO"},
                "Tous les émetteurs ont déjà un matériau valide : rien à migrer.",
            )
            return {"FINISHED"}

        emitter_dicts = [
            {
                "name": obj.name,
                "model": obj.bourrasque.model,
                "rho": obj.bourrasque.rho,
                "young": obj.bourrasque.young,
                "poisson": obj.bourrasque.poisson,
                "bulk": obj.bourrasque.bulk,
                "gamma": obj.bourrasque.gamma,
                "preset": obj.bourrasque.preset,
            }
            for obj in emitter_objs
        ]

        planned_materials, assignment = materials.plan_migration(emitter_dicts)

        library = scene.bourrasque.materials
        # `friction_angle` est INDISPENSABLE ici : `material_key` le lit pour le
        # modele SAND. Sans lui, ce dictionnaire levait KeyError des que la
        # bibliotheque contenait un materiau sable -- il n'etait meme pas
        # necessaire qu'un emetteur l'utilise. L'operateur est expose dans l'UI
        # sous le libelle « Reparer », declenche des qu'un emetteur est pendant :
        # l'artiste cliquait et recevait un traceback.
        #
        # Troisieme occurrence de la meme famille (apres `used_material_slot_count`
        # et `BQ_OT_material_duplicate`) : tout endroit qui reconstruit a la main
        # un dict de materiau doit porter TOUS les champs lus par `material_key`.
        by_key = {
            materials.material_key(
                {
                    "model": mat.model,
                    "rho": mat.rho,
                    "young": mat.young,
                    "poisson": mat.poisson,
                    "bulk": mat.bulk,
                    "gamma": mat.gamma,
                    "friction_angle": mat.friction_angle,
                }
            ): mat.name
            for mat in library
        }

        # Nom planifie par `plan_migration` -> nom REEL dans la
        # bibliotheque (reutilise s'il existe deja, sinon nouvellement cree
        # ci-dessous).
        resolved_names = {}
        created = 0
        for planned in planned_materials:
            key = materials.material_key(planned)
            existing_name = by_key.get(key)
            if existing_name is not None:
                resolved_names[planned["name"]] = existing_name
                continue

            existing = [m.name for m in library]
            name = materials.unique_name(planned["name"], existing)

            mat = library.add()
            mat.model = planned["model"]
            mat.rho = planned["rho"]
            mat.young = planned["young"]
            mat.poisson = planned["poisson"]
            mat.bulk = planned["bulk"]
            mat.gamma = planned["gamma"]
            mat.preset = "CUSTOM"
            mat.name_prev = name
            mat.name = name

            by_key[key] = name
            resolved_names[planned["name"]] = name
            created += 1

        assigned = 0
        for obj in emitter_objs:
            planned_name = assignment.get(obj.name)
            if planned_name is None:
                continue
            obj.bourrasque.material_name = resolved_names.get(
                planned_name, planned_name
            )
            assigned += 1

        self.report(
            {"INFO"},
            f"{created} matériau(x) créé(s), {assigned} émetteur(s) assigné(s).",
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_cancel_bake
# ---------------------------------------------------------------------------


class BQ_OT_cancel_bake(bpy.types.Operator):
    """Positionne le drapeau d'annulation observe par le modal de bake.

    Sert aussi bien au bake de particules qu'au bake de maillage (voir
    `BQ_OT_bake_mesh`) : les deux passes du bake modulaire (voir
    docs/plan-milestone-7.md, D8) partagent ce meme bouton « Annuler »,
    plutot que d'en dupliquer un par passe. `BQ_OT_bake_all` (qui enchaine
    les deux) n'a besoin d'aucune logique d'annulation propre : ESC ou ce
    bouton atteint directement l'operateur modal de la passe en cours, qui
    gere deja sa propre annulation."""

    bl_idname = "bq.cancel_bake"
    bl_label = "Annuler"
    bl_description = "Annule le bake en cours"

    @classmethod
    def poll(cls, context):
        props = context.scene.bourrasque
        return props.is_baking or props.is_baking_mesh or props.is_baking_whitewater

    def execute(self, context):
        props = context.scene.bourrasque
        if props.is_baking:
            BQ_OT_bake.cancel_requested = True
        if props.is_baking_mesh:
            BQ_OT_bake_mesh.cancel_requested = True
        if props.is_baking_whitewater:
            BQ_OT_bake_whitewater.cancel_requested = True
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_free_cache
# ---------------------------------------------------------------------------


class BQ_OT_free_cache(bpy.types.Operator):
    """Supprime les fichiers de cache de la scene et remet baked_frames a 0."""

    bl_idname = "bq.free_cache"
    bl_label = "Vider le cache"
    bl_description = "Supprime le cache de simulation sur le disque"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return not context.scene.bourrasque.is_baking

    def execute(self, context):
        scene = context.scene
        cache_dir = bpy.path.abspath(scene.bourrasque.cache_dir)
        bqd_path, mat_path = cache.cache_paths(cache_dir, scene.name)

        # Referme le CacheReader eventuellement garde ouvert par un scrub de
        # la timeline AVANT de supprimer le fichier : sur Windows, un
        # descripteur de fichier encore ouvert fait echouer os.remove
        # (PermissionError, WinError 32) -- cf. display.close_particle_reader.
        from . import display

        display.close_particle_reader()

        removed_any = False
        for path in (bqd_path, mat_path):
            if os.path.isfile(path):
                os.remove(path)
                removed_any = True

        scene.bourrasque.baked_frames = 0

        # Vide la geometrie affichee : sans ca, la derniere frame bakee
        # resterait affichee dans le viewport alors que le cache qui la
        # sous-tend vient d'etre supprime.
        display.clear_particle_object()

        if removed_any:
            self.report({"INFO"}, "Cache vidé.")
        else:
            self.report({"INFO"}, "Aucun cache à vider.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_free_mesh_cache
# ---------------------------------------------------------------------------


class BQ_OT_free_mesh_cache(bpy.types.Operator):
    """Supprime le cache de maillage (.bqm) de la scene, sur le modele exact
    de `BQ_OT_free_cache` (voir sa docstring) — mais ne touche PAS au cache
    de particules (.bqd/.mat) : les deux caches sont independants (bake
    modulaire, voir docs/plan-milestone-7.md, D1/D8)."""

    bl_idname = "bq.free_mesh_cache"
    bl_label = "Vider le cache de maillage"
    bl_description = "Supprime le cache de maillage (.bqm) sur le disque"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return not context.scene.bourrasque.is_baking_mesh

    def execute(self, context):
        scene = context.scene
        cache_dir = bpy.path.abspath(scene.bourrasque.cache_dir)
        bqm_path = mesh_cache_path(cache_dir, scene.name)

        # Referme le CacheReader de maillage eventuellement garde ouvert
        # AVANT de supprimer le fichier -- meme raison que
        # BQ_OT_free_cache (voir sa docstring / display.close_mesh_reader).
        from . import display

        display.close_mesh_reader()

        removed = False
        if os.path.isfile(bqm_path):
            os.remove(bqm_path)
            removed = True

        scene.bourrasque.baked_mesh_frames = 0

        # Vide la geometrie de maillage affichee : sans ca, la derniere
        # frame bakee resterait affichee alors que le cache qui la
        # sous-tend vient d'etre supprime (meme discipline que
        # BQ_OT_free_cache/display.clear_particle_object).
        display.clear_mesh_object()

        if removed:
            self.report({"INFO"}, "Cache de maillage vidé.")
        else:
            self.report({"INFO"}, "Aucun cache de maillage à vider.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_free_whitewater_cache
# ---------------------------------------------------------------------------


class BQ_OT_free_whitewater_cache(bpy.types.Operator):
    """Supprime le cache whitewater (.bqw) de la scene, sur le modele exact
    de `BQ_OT_free_mesh_cache` (voir sa docstring) — mais ne touche PAS aux
    caches de particules (.bqd/.mat) ni de maillage (.bqm) : les trois
    caches sont independants (bake modulaire, voir
    docs/plan-milestone-8.md, D8)."""

    bl_idname = "bq.free_whitewater_cache"
    bl_label = "Vider le cache de whitewater"
    bl_description = "Supprime le cache de whitewater (.bqw) sur le disque"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return not context.scene.bourrasque.is_baking_whitewater

    def execute(self, context):
        scene = context.scene
        cache_dir = bpy.path.abspath(scene.bourrasque.cache_dir)
        bqw_path = whitewater_cache_path(cache_dir, scene.name)

        # Referme le CacheReader de whitewater eventuellement garde ouvert
        # AVANT de supprimer le fichier -- meme raison que BQ_OT_free_cache
        # (voir sa docstring / display.close_whitewater_reader). Sans cet
        # appel, vider le cache echoue avec PermissionError (WinError 32) des
        # qu'on a scrubbe la timeline au moins une fois depuis le bake.
        from . import display

        display.close_whitewater_reader()

        removed = False
        if os.path.isfile(bqw_path):
            os.remove(bqw_path)
            removed = True

        scene.bourrasque.baked_whitewater_frames = 0
        scene.bourrasque.baked_whitewater_max_refused = 0

        # Vide la geometrie whitewater affichee : sans ca, la derniere
        # frame bakee resterait affichee alors que le cache qui la
        # sous-tend vient d'etre supprime (meme discipline que
        # BQ_OT_free_cache/BQ_OT_free_mesh_cache).
        display.clear_whitewater_object()

        if removed:
            self.report({"INFO"}, "Cache de whitewater vidé.")
        else:
            self.report({"INFO"}, "Aucun cache de whitewater à vider.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Asset d'affichage whitewater (Geometry Nodes + materiaux) — jalon 8bis
# ---------------------------------------------------------------------------
#
# Decision produit : l'add-on ne genere JAMAIS de Geometry Nodes par code
# Python a l'execution (fragile, l'API des noeuds change entre versions de
# Blender). Il fournit a la place un ASSET tout fait, construit une fois
# hors-ligne (voir le script de fabrication, non embarque dans l'extension)
# et livre dans `assets/whitewater_display.blend` : un groupe de noeuds
# `BQ Whitewater Display` et trois materiaux (`BQ_Spray`, `BQ_Foam`,
# `BQ_Bubble`). Cet operateur se contente de l'APPENDRE (jamais de le lier :
# l'artiste doit pouvoir modifier librement le resultat sans dependre du
# fichier source) sur `Bourrasque_Whitewater`, sans jamais construire de
# noeuds lui-meme. C'est un point de depart editable, pas un rendu final.

_WHITEWATER_ASSET_NODE_GROUP = "BQ Whitewater Display"
_WHITEWATER_ASSET_FILENAME = "whitewater_display.blend"


def _whitewater_asset_path():
    """Chemin du .blend d'asset, relatif au fichier de l'extension — meme
    motif que `lib._DLL_PATH` pour `bourrasque.dll`."""
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(ext_dir, "assets", _WHITEWATER_ASSET_FILENAME)


class BourrasqueAssetError(Exception):
    """Erreur locale a la resolution de l'asset whitewater — jamais
    remontee telle quelle a `self.report` (voir discipline `str(exc)`)."""


def _whitewater_display_modifier(obj):
    """Renvoie le modificateur Geometry Nodes de `obj` qui utilise deja le
    groupe de noeuds `BQ Whitewater Display`, ou `None` s'il n'y en a
    aucun — c'est ce test qui rend `BQ_OT_setup_whitewater_display`
    idempotent (jamais de modificateur en double)."""
    for mod in obj.modifiers:
        if mod.type != "NODES":
            continue
        node_group = mod.node_group
        if node_group is not None and node_group.name == _WHITEWATER_ASSET_NODE_GROUP:
            return mod
    return None


class BQ_OT_setup_whitewater_display(bpy.types.Operator):
    """Ajoute a `Bourrasque_Whitewater` le modificateur Geometry Nodes de
    depart (asset livre avec l'extension, voir docstring ci-dessus).
    Idempotent : ne fait rien de plus si le modificateur est deja present."""

    bl_idname = "bq.setup_whitewater_display"
    bl_label = "Configurer l'affichage du whitewater"
    bl_description = (
        "Ajoute un modificateur Geometry Nodes de depart (instances + "
        "materiaux editables) sur l'objet whitewater"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from . import display

        scene = context.scene
        obj = display.ensure_whitewater_object(scene)

        existing = _whitewater_display_modifier(obj)
        if existing is not None:
            self.report(
                {"INFO"}, "Affichage du whitewater déjà configuré."
            )
            return {"FINISHED"}

        asset_path = _whitewater_asset_path()
        if not os.path.isfile(asset_path):
            self.report(
                {"ERROR"}, f"Asset d'affichage introuvable : {asset_path}"
            )
            return {"CANCELLED"}

        try:
            with bpy.data.libraries.load(asset_path, link=False) as (
                data_from,
                data_to,
            ):
                if _WHITEWATER_ASSET_NODE_GROUP not in data_from.node_groups:
                    raise BourrasqueAssetError(
                        f"Groupe de noeuds « {_WHITEWATER_ASSET_NODE_GROUP} » "
                        f"absent de {asset_path}"
                    )
                data_to.node_groups = [_WHITEWATER_ASSET_NODE_GROUP]
                data_to.materials = list(data_from.materials)
        except (OSError, BourrasqueAssetError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        node_group = bpy.data.node_groups.get(_WHITEWATER_ASSET_NODE_GROUP)
        if node_group is None:
            self.report(
                {"ERROR"},
                f"Échec de l'ajout du groupe de noeuds « "
                f"{_WHITEWATER_ASSET_NODE_GROUP} »",
            )
            return {"CANCELLED"}

        mod = obj.modifiers.new(name="BQ Whitewater Display", type="NODES")
        mod.node_group = node_group

        self.report({"INFO"}, "Affichage du whitewater configuré.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Asset d'affichage volumetrique du whitewater — alternative a l'affichage
# par instances ci-dessus
#
# Ihmsen et al. 2012 ("Unified Spray, Foam and Bubbles for Particle-Based
# Fluids") ne rend JAMAIS de geometrie de particule individuelle : le nuage
# de particules diffuses est traite comme un VOLUME implicite (densite
# constante dans un rayon autour de chaque particule, 0 ailleurs), rendu par
# ray-marching avec ABSORPTION de radiance uniquement — le papier "neglige
# les effets de diffusion". Cette alternative reproduit ce principe avec des
# noeuds NATIFS Blender plutot qu'un rendu volumetrique custom :
# `Points to Volume` (converti le nuage en volume fog, rayon par point) +
# `Volume Absorption` (absorption pure, sans diffusion — contrairement a
# `Principled Volume`).
#
# Meme politique produit que les deux assets precedents (voir leurs
# docstrings de section) : jamais de construction de noeuds par code Python
# a l'execution, un asset tout fait (`assets/whitewater_display.blend`, un
# SECOND groupe de noeuds `BQ Whitewater Volume` a cote de `BQ Whitewater
# Display` — meme fichier, cible le meme objet `Bourrasque_Whitewater`,
# plutot qu'un fichier separe) est APPENDU (jamais lie) par cet operateur.
#
# Les deux affichages COEXISTENT (decision produit) : le rendu volumetrique
# peut etre beaucoup plus couteux en rendu (Cycles marche a travers un
# volume dense) que l'affichage par instances, et certains artistes voudront
# garder ce dernier pour la performance viewport ou un style different. Cet
# operateur n'enleve jamais le modificateur de l'autre affichage — un objet
# peut porter les deux modificateurs Geometry Nodes simultanement (l'artiste
# choisit lequel activer/desactiver dans la pile de modificateurs).

_WHITEWATER_VOLUME_NODE_GROUP = "BQ Whitewater Volume"


def _whitewater_volume_modifier(obj):
    """Renvoie le modificateur Geometry Nodes de `obj` qui utilise deja le
    groupe de noeuds `BQ Whitewater Volume`, ou `None` s'il n'y en a aucun —
    meme role que `_whitewater_display_modifier` pour l'affichage par
    instances (idempotence de `BQ_OT_setup_whitewater_display_volume`)."""
    for mod in obj.modifiers:
        if mod.type != "NODES":
            continue
        node_group = mod.node_group
        if node_group is not None and node_group.name == _WHITEWATER_VOLUME_NODE_GROUP:
            return mod
    return None


class BQ_OT_setup_whitewater_display_volume(bpy.types.Operator):
    """Ajoute a `Bourrasque_Whitewater` le modificateur Geometry Nodes
    volumetrique de depart (asset livre avec l'extension, voir docstring
    ci-dessus). Idempotent : ne fait rien de plus si le modificateur est
    deja present. Coexiste avec `BQ_OT_setup_whitewater_display` (affichage
    par instances) — les deux peuvent etre configures sur le meme objet."""

    bl_idname = "bq.setup_whitewater_display_volume"
    bl_label = "Configurer l'affichage volumétrique du whitewater"
    bl_description = (
        "Ajoute un modificateur Geometry Nodes volumetrique (Points to "
        "Volume + Volume Absorption, sans diffusion — Ihmsen et al. 2012) "
        "sur l'objet whitewater"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from . import display

        scene = context.scene
        obj = display.ensure_whitewater_object(scene)

        existing = _whitewater_volume_modifier(obj)
        if existing is not None:
            self.report(
                {"INFO"}, "Affichage volumétrique du whitewater déjà configuré."
            )
            return {"FINISHED"}

        asset_path = _whitewater_asset_path()
        if not os.path.isfile(asset_path):
            self.report(
                {"ERROR"}, f"Asset d'affichage introuvable : {asset_path}"
            )
            return {"CANCELLED"}

        try:
            with bpy.data.libraries.load(asset_path, link=False) as (
                data_from,
                data_to,
            ):
                if _WHITEWATER_VOLUME_NODE_GROUP not in data_from.node_groups:
                    raise BourrasqueAssetError(
                        f"Groupe de noeuds « {_WHITEWATER_VOLUME_NODE_GROUP} » "
                        f"absent de {asset_path}"
                    )
                data_to.node_groups = [_WHITEWATER_VOLUME_NODE_GROUP]
                data_to.materials = list(data_from.materials)
        except (OSError, BourrasqueAssetError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        node_group = bpy.data.node_groups.get(_WHITEWATER_VOLUME_NODE_GROUP)
        if node_group is None:
            self.report(
                {"ERROR"},
                f"Échec de l'ajout du groupe de noeuds « "
                f"{_WHITEWATER_VOLUME_NODE_GROUP} »",
            )
            return {"CANCELLED"}

        mod = obj.modifiers.new(name="BQ Whitewater Volume", type="NODES")
        mod.node_group = node_group

        self.report({"INFO"}, "Affichage volumétrique du whitewater configuré.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Asset d'affichage du maillage fluide (materiau triplanaire) — jalon 14 T3
# ---------------------------------------------------------------------------
#
# Meme politique produit que l'asset whitewater ci-dessus (voir sa docstring
# de section) : l'add-on ne genere JAMAIS de noeuds de shader par code
# Python a l'execution. Il fournit a la place un ASSET tout fait, construit
# une fois hors-ligne (voir le script de fabrication, non embarque dans
# l'extension) et livre dans `assets/fluid_display.blend` : un materiau
# `BQ_Fluid` a mapping TRIPLANAIRE (coordonnees Object du noeud Texture
# Coordinate — pas Generated, qui est recalcule depuis la bounding box du
# mesh a chaque frame et ferait "nager" la texture puisque cette bounding
# box change de forme/taille a chaque pas de simulation, cf. docs/plan-
# milestone-14.md D2). Necessaire car `Bourrasque_Mesh` est entierement
# reconstruit (marching cubes) a chaque frame : aucun UV stable n'a de sens
# ici. Cet operateur se contente d'APPENDRE le materiau (jamais de le lier)
# sur `Bourrasque_Mesh`, sans jamais construire de noeuds lui-meme. C'est un
# point de depart editable (texture Checker de demonstration), pas un rendu
# final.

_FLUID_ASSET_MATERIAL = "BQ_Fluid"
_FLUID_ASSET_FILENAME = "fluid_display.blend"


def _fluid_asset_path():
    """Chemin du .blend d'asset, relatif au fichier de l'extension — meme
    motif que `_whitewater_asset_path` / `lib._DLL_PATH`."""
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(ext_dir, "assets", _FLUID_ASSET_FILENAME)


def _fluid_display_material(obj):
    """Renvoie le materiau `BQ_Fluid` deja assigne a un slot de `obj`, ou
    `None` s'il n'y en a aucun — c'est ce test qui rend
    `BQ_OT_setup_fluid_display` idempotent (jamais de slot en double)."""
    for mat in obj.data.materials:
        if mat is not None and mat.name == _FLUID_ASSET_MATERIAL:
            return mat
    return None


class BQ_OT_setup_fluid_display(bpy.types.Operator):
    """Assigne a `Bourrasque_Mesh` le materiau triplanaire de depart (asset
    livre avec l'extension, voir docstring ci-dessus). Idempotent : ne fait
    rien de plus si le materiau est deja assigne."""

    bl_idname = "bq.setup_fluid_display"
    bl_label = "Configurer l'affichage du maillage"
    bl_description = (
        "Assigne un materiau triplanaire de depart (sans UV, editable) sur "
        "l'objet maillage fluide"
    )
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from . import display

        scene = context.scene
        obj = display.ensure_mesh_object(scene)

        existing = _fluid_display_material(obj)
        if existing is not None:
            self.report({"INFO"}, "Affichage du maillage déjà configuré.")
            return {"FINISHED"}

        asset_path = _fluid_asset_path()
        if not os.path.isfile(asset_path):
            self.report(
                {"ERROR"}, f"Asset d'affichage introuvable : {asset_path}"
            )
            return {"CANCELLED"}

        try:
            with bpy.data.libraries.load(asset_path, link=False) as (
                data_from,
                data_to,
            ):
                if _FLUID_ASSET_MATERIAL not in data_from.materials:
                    raise BourrasqueAssetError(
                        f"Materiau « {_FLUID_ASSET_MATERIAL} » "
                        f"absent de {asset_path}"
                    )
                data_to.materials = [_FLUID_ASSET_MATERIAL]
        except (OSError, BourrasqueAssetError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        material = bpy.data.materials.get(_FLUID_ASSET_MATERIAL)
        if material is None:
            self.report(
                {"ERROR"},
                f"Échec de l'ajout du materiau « {_FLUID_ASSET_MATERIAL} »",
            )
            return {"CANCELLED"}

        obj.data.materials.append(material)

        self.report({"INFO"}, "Affichage du maillage configuré.")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Bake en thread (D10) — objet de progression partage, et fonctions PURES
# (aucun bpy) reutilisees par le thread de calcul de BQ_OT_bake ET par les
# methodes d'instance historiques (compatibilite des scripts de
# tools/repro/*.py, qui les invoquent via `types.MethodType` sur un faux
# operateur — voir leurs docstrings). Les methodes d'instance
# (`BQ_OT_bake._emit_inflow_sites`, `BQ_OT_bake._update_colliders`) restent
# donc en place, mais deleguent desormais a ces fonctions pures plutot que
# de dupliquer leur logique.
# ---------------------------------------------------------------------------


class _BakeProgress:
    """Etat partage entre un thread de calcul de bake et le tick modal qui
    le sonde (docs/plan-milestone-7.md, D10).

    Seul le THREAD DE CALCUL ecrit `frame_index`/`done`/`error` et empile
    dans `reports` ; seul le TICK MODAL (thread principal) les lit et
    draine `reports` pour transformer chaque message en un vrai
    `self.report()` bpy — jamais l'inverse. Ces lectures/ecritures
    d'attributs simples (int, bool, str) sont atomiques sous le GIL,
    aucun verrou n'est necessaire pour elles ; `reports` est une
    `queue.Queue`, deja thread-safe par construction.
    """

    __slots__ = ("frame_index", "done", "error", "reports", "max_refused")

    def __init__(self):
        self.frame_index = 0
        self.done = False
        self.error = None
        self.reports = queue.Queue()
        # Uniquement rempli par `_whitewater_bake_worker` (voir sa
        # docstring) : maximum de `Whitewater.last_refused()` observe sur
        # toute la duree du bake whitewater. Inerte (reste a 0) pour les
        # bakes de particules/maillage, qui ne l'ecrivent jamais.
        self.max_refused = 0


def _emit_inflow_sites_impl(sim, inflow_states, usable_bounds, frame_index,
                             frame_dt, saturated, report):
    """Emet, pour chaque emetteur INFLOW, les sites de son nuage complet
    (`state.sites`, FIXE, voir `_InflowState`) qui ne sont pas deja occupes
    par une particule existante — ensemencement volumique avec test
    d'occupation (voir docstring de module).

    Fonction PURE (aucun bpy) : `sim` n'est sollicite que par ses appels
    ctypes (`read_positions`/`emit_points_vel`), et `report` est un simple
    callable `report(level, message)` — `self.report` bpy quand appele
    depuis `BQ_OT_bake._emit_inflow_sites` (thread principal), ou un
    depot dans une `queue.Queue` quand appele depuis le thread de calcul
    (`_bake_worker`), jamais un appel bpy direct. C'est ce qui rend cette
    fonction appelable indifferemment des deux contextes.

    Pour chaque emetteur : filtre les particules courantes (lues UNE fois
    pour tous les emetteurs, avant le `step` de cette frame) a celles
    tombant dans la bbox du nuage de sites (filtre vectorise numpy, c'est
    ce qui rend l'operation peu couteuse), convertit ces positions
    retenues en indices de site du reseau (arrondi au site le plus
    proche), et emet un point a chaque site du nuage dont l'indice
    n'apparait pas parmi ces occupants — SOUS RESERVE du plafond de
    conservation du nombre ci-dessous. Le volume de l'emetteur reste ainsi
    sature en permanence : le fluide en sort a la vitesse `v` (constante,
    celle de l'emetteur — les sites sont au niveau de l'emetteur, ou la
    vitesse du fluide est `v`), les sites se liberent au rythme voulu, le
    debit s'auto-regule sans accumulateur.

    Plafond par CONSERVATION DU NOMBRE : le nombre de sites libres n'est
    PAS a lui seul un plafond fiable des lors que le jitter positionnel ou
    le bruit de turbulence sont actifs. Un bruit spatialement coherent
    (`_CurlNoise`) deplace des BLOCS entiers de particules dans la meme
    direction, ce qui cree simultanement des sites vides (le bloc s'en est
    eloigne — reemis, on GAGNE des particules) et des sites occupes par
    deux particules ou plus (deux particules arrondissent au meme site —
    rien n'est retire, on ne PERD rien). Le bilan naif est donc
    structurellement positif : sans correction, le debit derive vers le
    haut avec la turbulence, alors que les deux reglages doivent rester
    orthogonaux pour l'artiste. On calcule donc `deficit = max(0, n_sites
    - n_occupants)` (le nombre de particules manquantes pour saturer
    exactement le nuage) et on n'emet jamais plus de `deficit` particules :
    si les sites libres sont plus nombreux que `deficit`, `deficit` d'entre
    eux sont tires AU HASARD (jamais un prefixe, qui biaiserait
    spatialement l'emission vers un coin de l'emetteur — les sites sont
    ranges dans un ordre de reseau), via un generateur numpy DEDIE seede
    par la meme convention que `_turbulence_rng`. A turbulence nulle,
    aucune paire de particules n'arrondit au meme site : `deficit` egale
    alors exactement le nombre de sites libres et le plafond ne mord
    jamais (comportement inchange).

    Un site peut tomber hors de la zone utile du domaine (pres d'une
    paroi) : les points hors `usable_bounds` sont filtres avant l'appel,
    comme pour un emetteur BLOCK.

    Ne fait rien si la simulation est deja saturee (`saturated`). Sur le
    premier depassement de `max_particles`, rapporte l'avertissement UNE
    SEULE fois ; les emetteurs suivants (meme frame ou frames suivantes)
    ne tentent plus d'emission.

    Renvoie `(total, saturated)` : le nombre total de particules
    effectivement emises pour cette frame, et le drapeau de saturation mis
    a jour (a repasser en entree de l'appel suivant).
    """
    if not inflow_states or saturated:
        return 0, saturated

    usable = usable_bounds
    # Lues UNE fois pour tous les emetteurs de cette frame : c'est l'etat
    # de la simulation AVANT l'emission de cette frame (les sites qui
    # viennent d'etre liberes par le mouvement des particules depuis la
    # frame precedente).
    positions = sim.read_positions()

    total = 0
    for state in inflow_states:
        if saturated:
            break

        spacing = state.spacing
        sites = state.sites
        if sites.shape[0] == 0:
            continue

        lo = sites.min(axis=0) - spacing / 2.0
        hi = sites.max(axis=0) + spacing / 2.0

        if positions.shape[0] > 0:
            in_bbox = np.all((positions >= lo) & (positions <= hi), axis=1)
            occupants = positions[in_bbox]
        else:
            occupants = positions

        site_idx = _site_indices(sites, spacing)
        if occupants.shape[0] > 0:
            occ_idx = _site_indices(occupants, spacing)
            free_mask = ~np.isin(_void_rows(site_idx), _void_rows(occ_idx))
        else:
            free_mask = np.ones(sites.shape[0], dtype=bool)

        n_sites = sites.shape[0]
        n_occupants = occupants.shape[0]
        deficit = max(0, n_sites - n_occupants)

        free_idx = np.nonzero(free_mask)[0]
        if free_idx.shape[0] > deficit:
            select_rng = _turbulence_rng(
                state.turbulence_seed, frame_index, state.emitter_index
            )
            free_idx = select_rng.choice(free_idx, size=deficit, replace=False)

        pts = sites[free_idx]
        if pts.shape[0] == 0:
            continue

        rng = _turbulence_rng(
            state.turbulence_seed, frame_index, state.emitter_index
        )
        t = frame_index * frame_dt
        pts, vels = _turbulent_emission(
            pts, state.vel, state.turbulence, spacing, frame_dt,
            rng, state.dx, t, state.noise,
        )

        if usable is not None:
            ulo = np.asarray(usable[0], dtype=np.float64)
            uhi = np.asarray(usable[1], dtype=np.float64)
            tol = 1e-6 * state.dx
            mask = np.all((pts >= ulo - tol) & (pts <= uhi + tol), axis=1)
            pts = pts[mask]
            vels = vels[mask]
        if pts.shape[0] == 0:
            continue

        try:
            total += sim.emit_points_vel(state.mat_id, pts, vels)
        except lib.BourrasqueError as exc:
            msg = str(exc)
            if "capacite depassee" in msg:
                saturated = True
                report(
                    {"WARNING"},
                    "Capacité maximale de particules atteinte : "
                    "l'émission continue (inflow) est arrêtée, la "
                    "simulation se poursuit jusqu'à la fin du bake avec "
                    "les particules déjà émises.",
                )
            else:
                report(
                    {"WARNING"},
                    f"« {state.name} » : émission refusée par le "
                    f"solveur pour cette frame ({msg}). L'inflow "
                    "continue aux frames suivantes.",
                )
    return total, saturated


def _collect_collider_frame(collider_states, depsgraph, origin, size,
                             frame_dt, report):
    """Triangles/vitesses/frictions concatenes de tous les colliders de
    `collider_states`, EVALUES au `depsgraph` fourni — fonction PURE (aucun
    bpy au-dela de `depsgraph`/`state.obj.evaluated_get`, deja resolus par
    l'appelant) reutilisee par :

    - `BQ_OT_bake._update_colliders` (thread principal, une frame a la
      fois, pendant la boucle modale historique/les scripts de
      tools/repro/*.py) ;
    - `BQ_OT_bake.invoke` (thread principal aussi, mais en PRE-EXTRACTION :
      TOUTES les frames du bake sont calculees ici, avant de lancer le
      thread de calcul — voir docs/plan-milestone-7.md D10 et la docstring
      de module. Les colliders animes exigent le depsgraph bpy a chaque
      frame, qui ne peut donc jamais etre lu depuis le thread de calcul).

    Positions : converties monde -> solveur via `world_to_solver_array`
    (AVEC translation). Vitesse par sommet : difference finie MONDE
    convertie via `world_to_solver_dir_array` (SANS translation) — ne
    jamais confondre les deux, voir docstring de module sur les colliders
    animes. Nulle a la premiere frame d'un collider, ou si son nombre de
    sommets a change depuis la frame precedente (topologie non appariable,
    voir `_ColliderState`). Mute `state.prev_verts_world`/
    `state.warned_topology` EN PLACE (etat persistant entre deux appels
    successifs, une frame apres l'autre).

    Un collider DYNAMIQUE (`state.dynamic` vrai) est totalement IGNORE ici
    (`continue` immediat, aucun controle de maillage/topologie, aucune
    lecture de `evaluated_world_mesh`) : sa geometrie est reconstruite
    ailleurs depuis l'etat rigide courant, jamais reevaluee depuis la scene
    (D6 du plan — voir `_dynamic_collider_geometry`). C'est ce qui garantit
    qu'un collider FIXE (le cas majoritaire) ne perd RIEN de son
    comportement historique, y compris pour un maillage ouvert ou de volume
    nul : cette fonction ne lui applique jamais `sampling.check_mesh_closed`
    ni de calcul de proprietes massiques.

    `report` : callable `report(level, message)`, jamais un appel bpy
    direct fait par cette fonction — voir `_emit_inflow_sites_impl` pour la
    meme convention.

    Renvoie `(tri_all, vel_all, fric_all, tri_body_all)`, quatre ndarray
    (potentiellement vides) prets pour `Sim.set_colliders`. `tri_body_all`
    (int32) porte `state.body_index` par triangle — assigne par l'appelant
    (`BQ_OT_bake.invoke`) AVANT le premier appel a cette fonction.
    """
    from . import sampling

    tri_chunks = []
    vel_chunks = []
    fric_chunks = []
    body_chunks = []

    for state in collider_states:
        if state.dynamic:
            # Geometrie recalculee depuis l'etat rigide courant (voir
            # _dynamic_collider_geometry), jamais reevaluee ici — D6.
            continue

        obj_eval = state.obj.evaluated_get(depsgraph)
        verts_world, tris = sampling.evaluated_world_mesh(obj_eval)
        n_tri = tris.shape[0]
        if n_tri == 0:
            state.prev_verts_world = verts_world
            continue

        prev = state.prev_verts_world
        topology_ok = prev is not None and prev.shape[0] == verts_world.shape[0]
        if prev is not None and not topology_ok and not state.warned_topology:
            report(
                {"WARNING"},
                f"« {state.obj.name} » : le nombre de sommets a changé "
                "d'une frame à l'autre (remesh, modificateur variable) "
                "— vitesse nulle pour ce collider tant que sa "
                "topologie n'est pas stable.",
            )
            state.warned_topology = True

        if topology_ok:
            vel_world = (verts_world - prev) / frame_dt
        else:
            vel_world = np.zeros_like(verts_world)

        verts_solver = world_to_solver_array(verts_world, origin, size)
        vel_solver = world_to_solver_dir_array(vel_world)

        tri_chunks.append(verts_solver[tris].astype(np.float32))
        vel_chunks.append(vel_solver[tris].astype(np.float32))
        fric_chunks.append(np.full(n_tri, state.friction, dtype=np.float32))
        body_chunks.append(np.full(n_tri, state.body_index, dtype=np.int32))

        state.prev_verts_world = verts_world

    if tri_chunks:
        tri_all = np.concatenate(tri_chunks, axis=0)
        vel_all = np.concatenate(vel_chunks, axis=0)
        fric_all = np.concatenate(fric_chunks, axis=0)
        body_all = np.concatenate(body_chunks, axis=0)
    else:
        tri_all = np.empty((0, 3, 3), dtype=np.float32)
        vel_all = np.empty((0, 3, 3), dtype=np.float32)
        fric_all = np.empty((0,), dtype=np.float32)
        body_all = np.empty((0,), dtype=np.int32)

    return tri_all, vel_all, fric_all, body_all


def _static_collider_triangles(collider_objs, depsgraph, origin, size):
    """Triangles concatenes de `collider_objs`, EVALUES au `depsgraph`
    fourni, en espace solveur — variante STATIQUE (pas de vitesse, pas
    d'etat persistant entre appels) de `_collect_collider_frame`, reservee
    au rognage du mailleur (`BQ_OT_bake_mesh`, docs/plan-milestone-7.md
    D5) : le champ de distance des colliders y est construit UNE FOIS,
    a l'etat courant de la scene, jamais reevalue par frame. Reutilise les
    memes briques (`sampling.evaluated_world_mesh`, `world_to_solver_array`)
    que `_collect_collider_frame` plutot que de refaire l'extraction de
    geometrie.

    Renvoie `tri_all`, un ndarray `(n_tri, 3, 3)` float32 (potentiellement
    vide).
    """
    from . import sampling

    tri_chunks = []
    for obj in collider_objs:
        obj_eval = obj.evaluated_get(depsgraph)
        verts_world, tris = sampling.evaluated_world_mesh(obj_eval)
        if tris.shape[0] == 0:
            continue
        verts_solver = world_to_solver_array(verts_world, origin, size)
        tri_chunks.append(verts_solver[tris].astype(np.float32))

    if tri_chunks:
        return np.concatenate(tri_chunks, axis=0)
    return np.empty((0, 3, 3), dtype=np.float32)


def _bake_worker(progress, cancel_event, sim, writer, frame_count, frame_dt,
                  inflow_states, usable_bounds, collider_frames,
                  dynamic_collider_states, body_track, kinematic_pose_frames):
    """Boucle de calcul du bake de particules — executee dans un
    `threading.Thread(daemon=True)` (voir `BQ_OT_bake.invoke`,
    docs/plan-milestone-7.md D10).

    NE TOUCHE JAMAIS bpy (voir la garde en tete de module) : uniquement
    `sim` (appels ctypes, qui relachent le GIL — c'est ce qui rend ce
    thread utile), `writer` (ecriture de fichier), et `progress`/
    `cancel_event` (objets Python simples). Les colliders FIXES animes ont
    deja ete PRE-EXTRAITS sur le thread principal avant l'appel a cette
    fonction (`collider_frames`, une liste de `(tri, vel, fric, tri_body)`
    par frame — voir `_collect_collider_frame` — ou `None` si aucun
    collider n'est configure pour ce bake) : c'est l'unique moyen de leur
    faire traverser la frontiere thread, leur extraction necessitant le
    depsgraph bpy (non thread-safe). Les colliders DYNAMIQUES, eux, sont
    reconstruits ICI a chaque frame, en PUR numpy (`_dynamic_collider_geometry`),
    depuis leur maillage de repos (deja capture sur le thread principal,
    voir `_setup_collider_body`) et l'etat rigide COURANT relu du solveur
    (`sim.read_collider_bodies`, un appel ctypes — jamais bpy) : c'est ce qui
    permet a un corps dynamique de traverser la frontiere thread sans jamais
    toucher la scene.

    `kinematic_pose_frames` (M17/phase B, tache B4, point 2) : meme
    principe de PRE-EXTRACTION que `collider_frames`, une liste de listes
    de `(body_index, x, q, v, w)` par frame — voir `_kinematic_pose_frame`
    — ou `None` si aucun collider n'est configure. Applique via
    `Sim.set_body_pose` AVANT `sim.set_colliders`/`sim.step` de chaque
    frame : c'est ce qui fait qu'un mur ou une pale animes entrainent
    reellement un corps dynamique en contact (D9/D12), pas seulement le
    fluide (chemin `collider_frames`, inchange). `Sim.set_body_pose` ne
    touche NI l'etat des corps dynamiques NI le sommeil/warm-starting du
    contact des autres corps (voir `bq_set_body_pose`, bourrasque.h) :
    contrairement a `set_collider_bodies`, l'appeler chaque frame est le
    comportement VOULU, pas une erreur.

    `body_track` (liste mutee EN PLACE) accumule une copie de l'etat de
    TOUS les corps apres CHAQUE frame effectivement calculee — y compris en
    cas d'annulation en cours de route, ce qui permet a l'appelant de poser
    des keyframes sur les frames deja bakees plutot que de jeter le travail
    (voir `BQ_OT_bake._post_keyframes`).

    Toute exception est capturee et deposee dans `progress.error` (une
    chaine, jamais l'exception elle-meme — un traceback ou un objet
    d'exception ne doit pas etre lu depuis le thread principal sans
    precaution) ; `progress.done` est mis a vrai dans tous les cas
    (`finally`), y compris une annulation ou une exception, pour que le
    tick modal cesse d'attendre.
    """
    try:
        saturated = False
        pos_buffer = None
        vel_buffer = None
        body_state = sim.read_collider_bodies() if collider_frames is not None else None
        for frame_index in range(frame_count):
            if cancel_event.is_set():
                return
            if kinematic_pose_frames is not None:
                for body_index, x, q, v, w in kinematic_pose_frames[frame_index]:
                    sim.set_body_pose(body_index, x, q, v=v, w=w)
            if collider_frames is not None:
                static_frame = collider_frames[frame_index]
                if dynamic_collider_states:
                    dynamic_geo = _dynamic_collider_geometry(
                        dynamic_collider_states, body_state
                    )
                    tri_all, vel_all, fric_all, body_all = _combine_collider_geometry(
                        static_frame, dynamic_geo
                    )
                else:
                    tri_all, vel_all, fric_all, body_all = static_frame
                sim.set_colliders(tri_all, vel_all, fric_all, tri_body=body_all)
            _total, saturated = _emit_inflow_sites_impl(
                sim, inflow_states, usable_bounds, frame_index, frame_dt,
                saturated, lambda level, msg: progress.reports.put((level, msg)),
            )
            sim.step(frame_dt)
            if collider_frames is not None:
                body_state = sim.read_collider_bodies()
                body_track.append(body_state.copy())
            pos_buffer = sim.read_positions(out=pos_buffer)
            vel_buffer = sim.read_velocities(out=vel_buffer)
            writer.append_frame(pos_buffer, vel_buffer)
            progress.frame_index = frame_index + 1
    except Exception as exc:  # noqa: BLE001 — remonte au tick modal, jamais bpy ici
        progress.error = str(exc)
    finally:
        progress.done = True


# ---------------------------------------------------------------------------
# BQ_OT_bake — operateur modal
# ---------------------------------------------------------------------------


class BQ_OT_bake(bpy.types.Operator):
    """Bake la simulation : cree la Sim native, emet les particules de
    chaque emetteur, puis avance frame par frame en ecrivant le cache.

    Modal : une frame simulee par evenement TIMER, pour ne jamais bloquer
    l'interface le temps d'un bake qui peut durer plusieurs minutes.
    """

    bl_idname = "bq.bake"
    bl_label = "Baker"
    bl_description = "Lance le bake de la simulation"
    bl_options = {"REGISTER"}

    # Drapeau d'annulation observe par le modal ; positionne par
    # BQ_OT_cancel_bake. Attribut de classe : un seul bake modal actif a la
    # fois par definition (bpy ne permet pas deux invocations modales
    # concurrentes du meme operateur).
    cancel_requested = False

    # Reference vers l'instance modale en cours, pour permettre a
    # unregister() de nettoyer un bake reste actif (extension desactivee
    # pendant un bake). None quand aucun bake n'est en cours.
    _active_instance = None

    _timer = None
    _sim = None
    _writer = None
    _pos_buffer = None
    _frame_index = 0
    _frame_count = 0
    _frame_dt = 1.0 / 24.0
    # Etat des emetteurs INFLOW (liste de _InflowState) et drapeau de
    # saturation (max_particles atteint) : voir _emit_inflow_sites.
    _inflow_states = ()
    _saturated = False
    # Bornes de la zone utile du domaine (espace solveur), memorisees a
    # invoke() pour rester lisibles depuis _emit_inflow_sites sans dependre
    # de context/scene (methode appelable hors cycle modal, cf.
    # _advance_frame).
    _usable_bounds = None
    # (origin, size) du pave SOLVEUR, memorises a invoke() : necessaires a
    # chaque frame pour convertir la geometrie des colliders animes
    # (_update_colliders), pas seulement a la mise en place du bake.
    _domain_transform = None
    # Colliders de la scene (liste de _ColliderState), collectes une fois a
    # invoke() ; chacun est reevalue a chaque frame (_update_colliders).
    _collider_states = ()
    # Sous-ensemble de _collider_states dont `dynamic` est vrai (jalon M17,
    # phase A) — memorise separement pour ne pas reparcourir/refiltrer
    # _collider_states a chaque frame du bake. `()` si aucun collider
    # dynamique (bake inchange, voir _update_colliders/_bake_worker).
    _dynamic_collider_states = ()
    # Etat de TOUS les corps rigides (dynamiques et fixes), une entree par
    # frame effectivement bakee (ndarray (n_bodies, 13), voir
    # Sim.read_collider_bodies) — accumule par _bake_worker, consomme par
    # _post_keyframes. `None` tant qu'aucun collider n'est configure.
    _body_track = None
    # Pose/vitesse SOLVEUR des colliders CINEMATIQUES au SDF construit,
    # PRE-EXTRAITE frame par frame (liste de listes de `(body_index, x, q,
    # v, w)`, voir `_kinematic_pose_frame`) — consommee par `_bake_worker`
    # via `Sim.set_body_pose` (M17/phase B, B4, point 2). `None` tant
    # qu'aucun collider n'est configure.
    _kinematic_pose_frames = None
    # Frame Blender courante au moment ou bq.bake a ete invoque : restauree
    # dans _cleanup, sur TOUS les chemins de sortie (fin normale,
    # annulation, exception) — voir _advance_scene_frame, qui fait avancer
    # scene.frame_current a chaque frame simulee.
    _start_frame = None
    # Scene visee par ce bake, memorisee dans invoke() : voir sa docstring.
    # Ne jamais lire context.scene apres invoke() dans cet operateur.
    _scene = None

    # -- D10 : le calcul tourne dans un thread, le modal ne fait que sonder
    # -- (voir la garde en tete de module et BQ_OT_bake.invoke/modal) -----
    # Geometrie des colliders animes, PRE-EXTRAITE frame par frame sur le
    # thread principal avant le lancement du thread de calcul (liste de
    # `(tri, vel, fric)`, longueur `_frame_count`), ou None si aucun
    # collider n'est configure pour ce bake — voir _collect_collider_frame.
    _collider_frames = None
    # Objet de progression partage (`_BakeProgress`) : ecrit par le thread
    # de calcul, lu par le tick modal.
    _progress = None
    # Signale au thread de calcul qu'il doit s'arreter entre deux frames
    # (ESC ou bouton Annuler) — jamais une interruption forcee.
    _cancel_event = None
    # Le thread de calcul lui-meme.
    _thread = None

    @classmethod
    def poll(cls, context):
        return not context.scene.bourrasque.is_baking

    # -- validation avant lancement -----------------------------------

    def _validate(self, context):
        """Renvoie (config, materials, emitter_specs) si tout est valide,
        None sinon (avec self.report deja appele)."""
        scene = context.scene
        props = scene.bourrasque

        transform = domain_transform(scene)
        if transform is None:
            self.report({"ERROR"}, "Aucun domaine défini.")
            return None
        origin, size = transform
        resolution = domain_resolution(scene)
        # `resolution` ne peut etre None que si `transform` l'est aussi (meme
        # garde dans `props._domain_layout`) : deja exclu ci-dessus.
        res, dx = resolution

        # Depuis M5, un domaine non cubique est un pave EXPLICITEMENT
        # supporte (voir docs/plan-milestone-5.md) : plus d'avertissement
        # « le domaine n'est pas un cube », `domain_transform` calcule
        # desormais une resolution par axe plutot que d'approximer par un
        # cube englobant.
        if min(size) <= 0:
            self.report(
                {"ERROR"},
                "Le domaine est dégénéré (taille nulle ou négative) : "
                "vérifiez la bounding box de l'objet domaine.",
            )
            return None

        if props.frame_end < props.frame_start:
            self.report(
                {"ERROR"},
                f"« Frame de fin » ({props.frame_end}) est antérieure à "
                f"« Frame de début » ({props.frame_start}).",
            )
            return None

        emitters = [
            obj for obj in scene.objects if obj.bourrasque.role == "EMITTER"
        ]
        if not emitters:
            self.report(
                {"ERROR"},
                "Aucun émetteur défini. Ajoutez au moins un objet émetteur "
                "avant de lancer le bake.",
            )
            return None

        # Un emetteur dont la boite deborde du domaine produit des
        # coordonnees solveur hors [0, size] : k_p2g ecrit alors hors du
        # buffer GPU, ce qui tue le contexte CUDA de tout le processus
        # Blender. Voir `props.emitter_overflow` pour le detail de la marge
        # de securite.
        for obj in emitters:
            overflow = emitter_overflow(obj, origin, size, dx)
            if overflow is None:
                continue
            fully_outside, offending_axes, margin = overflow
            if fully_outside:
                self.report(
                    {"ERROR"},
                    f"L'émetteur « {obj.name} » est entièrement hors du "
                    "domaine (ou trop près de son bord, marge de "
                    f"{margin:.4f} m). Bake refusé.",
                )
            else:
                self.report(
                    {"ERROR"},
                    f"L'émetteur « {obj.name} » déborde du domaine sur "
                    f"l'axe {', '.join(offending_axes)} (marge de "
                    f"sécurité {margin:.4f} m). Ajustez sa position/"
                    "taille avant de baker.",
                )
            return None

        estimated = estimate_particle_count(scene)
        if estimated > props.max_particles:
            self.report(
                {"ERROR"},
                f"Nombre de particules estimé ({estimated}) dépasse la "
                f"capacité configurée ({props.max_particles}). Augmentez "
                "« Particules max » ou réduisez la résolution/les "
                "émetteurs.",
            )
            return None

        try:
            config = lib.default_config()
        except lib.BourrasqueError as exc:
            self.report({"ERROR"}, str(exc))
            return None

        config.grid_res[:] = res
        config.cell_size = dx
        config.gravity_y = props.gravity
        config.cfl = props.cfl
        config.ppc_axis = props.ppc_axis
        config.max_particles = props.max_particles

        # Seuls les materiaux EFFECTIVEMENT UTILISES par au moins un
        # emetteur sont enregistres, sinon le pas de temps (calcule sur le
        # max de TOUS les materiaux enregistres, cf. mlsmpm.cu:319) serait
        # penalise par des materiaux inutilises mais plus raides — voir
        # `materials.collect_used_materials`, qui porte desormais cette
        # regle (avant cette refonte, la deduplication se faisait ici meme,
        # par tuple de valeurs lues sur chaque emetteur).
        library = [
            {
                "name": mat.name,
                "model": mat.model,
                "rho": mat.rho,
                "young": mat.young,
                "poisson": mat.poisson,
                "bulk": mat.bulk,
                "gamma": mat.gamma,
                "friction_angle": mat.friction_angle,
                "cohesion": mat.cohesion,
            }
            for mat in props.materials
        ]
        assignments = [
            (obj.name, obj.bourrasque.material_name) for obj in emitters
        ]
        specs, emitter_material_index, missing = materials.collect_used_materials(
            library, assignments
        )

        if missing:
            names = "; ".join(
                f"« {emitter_name} »" for emitter_name, _material_name in missing
            )
            self.report(
                {"ERROR"},
                "Émetteur(s) sans matériau assigné (référence manquante ou "
                f"vide) : {names}. Assignez-leur un matériau de la "
                "bibliothèque avant de baker.",
            )
            return None

        if len(specs) > lib.BQ_MAX_MATERIALS:
            self.report(
                {"ERROR"},
                f"{len(specs)} matériaux distincts sont utilisés par les "
                f"émetteurs, mais le solveur n'en accepte que "
                f"{lib.BQ_MAX_MATERIALS} au maximum. Assignez le même "
                "matériau de la bibliothèque à plusieurs émetteurs plutôt "
                "que d'en utiliser un distinct par émetteur.",
            )
            return None

        material_specs = []
        for material in specs:
            model = material["model"]
            # Dispatch EXHAUSTIF et explicite : un modele non reconnu leve
            # plutot que de retomber silencieusement sur l'eau (piege connu
            # de ce jalon -- voir materials.material_key pour la meme
            # discipline).
            if model == "ELASTIC":
                kwargs = dict(
                    model=lib.BQ_MODEL_ELASTIC,
                    rho=material["rho"],
                    E=material["young"],
                    nu=material["poisson"],
                )
            elif model == "WATER":
                kwargs = dict(
                    model=lib.BQ_MODEL_WATER,
                    rho=material["rho"],
                    bulk=material["bulk"],
                    gamma=material["gamma"],
                )
            elif model == "SAND":
                kwargs = dict(
                    model=lib.BQ_MODEL_SAND,
                    rho=material["rho"],
                    E=material["young"],
                    nu=material["poisson"],
                    friction_angle=material["friction_angle"],
                )
            else:
                raise ValueError(
                    f"BQ_OT_bake: modele de materiau inconnu {model!r} pour "
                    f"« {material['name']} »"
                )
            material_specs.append(kwargs)

        emitter_specs = [
            (obj, mat_index)
            for obj, mat_index in zip(emitters, emitter_material_index)
        ]

        return (config, origin, size, dx, material_specs, emitter_specs)

    # -- invoke : validation + mise en place ---------------------------

    def invoke(self, context, event):
        # Memorise la scene visee par CE bake : `modal` renvoie
        # PASS_THROUGH pour les evenements non-TIMER, donc l'utilisateur
        # peut changer `context.scene` (changement de scene active) pendant
        # que le bake tourne. Tout le reste de l'operateur doit lire
        # `self._scene`, jamais `context.scene`.
        self._scene = context.scene
        scene = self._scene
        props = scene.bourrasque

        # Memorisee AVANT tout `frame_set` (voir _advance_scene_frame) :
        # c'est la valeur a restaurer dans _cleanup, sur tous les chemins de
        # sortie (fin normale, annulation, exception).
        self._start_frame = scene.frame_current

        validated = self._validate(context)
        if validated is None:
            return {"CANCELLED"}
        config, origin, size, dx, material_specs, emitter_specs = validated

        try:
            self._sim = lib.Sim(config)
        except lib.BourrasqueError as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        # A partir d'ici, la Sim (memoire GPU) existe : TOUT chemin de
        # sortie doit passer par self._cleanup(context), y compris les
        # echecs de mise en place du timer/modal ci-dessous.
        try:
            material_ids = []
            for spec in material_specs:
                material_ids.append(self._sim.add_material(**spec))

            # Importe paresseusement : `sampling` n'est necessaire que pour
            # les emetteurs en mode Maillage (BLOCK ou INFLOW), et ce
            # module fait du lancer de rayons (bmesh/BVHTree) qu'on ne veut
            # pas payer/importer pour un bake sans emetteur MESH.
            from . import sampling

            spacing = dx / props.ppc_axis

            # fps/fps_base ensemble donnent la frequence de rendu reelle
            # (ex. 23.976 = 24/1.001) ; fps seul l'ignore et decale le dt
            # de 0,1 %. Calcule ICI (avant la boucle d'emission ci-dessous)
            # car `_turbulent_emission` (mode BLOCK) en a besoin des la mise
            # en place, pas seulement pendant le modal.
            self._frame_dt = scene.render.fps_base / scene.render.fps

            total_emitted = 0
            self._inflow_states = []
            self._saturated = False
            self._usable_bounds = domain_usable_bounds(scene)
            self._domain_transform = (origin, size)

            # Colliders : collectes une fois ici (objet + friction), chacun
            # est reevalue a CHAQUE frame du bake (voir
            # _advance_scene_frame / _update_colliders), contrairement aux
            # emetteurs dont la geometrie n'est echantillonnee qu'une fois.
            self._collider_states = [
                _ColliderState(obj, obj.bourrasque.friction)
                for obj in scene.objects
                if obj.bourrasque.role == "COLLIDER"
            ]
            # Indice de corps STABLE (jalon M17, phase A) : l'ordre de
            # collecte ci-dessus EST l'ordre transmis a
            # `Sim.set_collider_bodies`, TOUS les colliders y figurant
            # (dynamiques et fixes, voir _build_rigid_bodies) pour que
            # l'attribution `tri_body` reste coherente.
            for body_index, state in enumerate(self._collider_states):
                state.body_index = body_index
            self._dynamic_collider_states = [
                state for state in self._collider_states if state.dynamic
            ]

            for emitter_index, (obj, mat_index) in enumerate(emitter_specs):
                op = obj.bourrasque
                mat_id = material_ids[mat_index]
                vel = world_to_solver_dir(tuple(op.initial_velocity))

                mesh_points = None
                if op.emit_source == "MESH":
                    closed, message = sampling.check_mesh_closed(obj)
                    if not closed:
                        self.report(
                            {"ERROR"},
                            f"« {obj.name} » ne peut pas être utilisé comme "
                            f"émetteur en mode Maillage : {message}",
                        )
                        self._cleanup(context)
                        return {"CANCELLED"}

                    mesh_points = sampling.sample_mesh_interior(
                        obj, origin, size, dx, props.ppc_axis
                    )
                    # Un maillage FERME (verifie ci-dessus) mais plus petit
                    # que le pas du reseau du solveur ne fait plus echouer
                    # le bake : voir le repli sur une particule centrale
                    # ci-dessous (mode BLOCK) — un maillage ouvert reste,
                    # lui, toujours refuse par `check_mesh_closed` plus
                    # haut, la parite des rayons y etant ambigue plutot que
                    # simplement grossiere.

                if op.emit_mode == "BLOCK":
                    if op.emit_source == "MESH":
                        # Points deja explicites : il suffit de les
                        # perturber (passe-plat exact a turbulence == 0,
                        # voir `_turbulent_emission`) et d'emettre avec une
                        # vitesse par particule. `noise` est construit A LA
                        # VOLEE (emission unique, pas de bake a etaler dans
                        # le temps : `t = 0.0` suffit), contrairement au cas
                        # INFLOW ou il est persiste via `_InflowState`.
                        rng = _turbulence_rng(op.turbulence_seed, 0, emitter_index)
                        bbox_lo, bbox_hi = _points_bbox(mesh_points)
                        noise = _CurlNoise(
                            op.turbulence_seed, emitter_index, bbox_lo, bbox_hi,
                            dx, vel, spacing, self._frame_dt,
                        )
                        pts, vels = _turbulent_emission(
                            mesh_points, vel, op.turbulence, spacing,
                            self._frame_dt, rng, dx, 0.0, noise,
                        )
                        n = self._sim.emit_points_vel(mat_id, pts, vels)
                    elif op.turbulence <= 0.0:
                        # Chemin STRICTEMENT inchange a turbulence nulle :
                        # c'est le solveur (bq_emit_box) qui genere le
                        # reseau, aucun passage par le chemin Python.
                        lo, hi = emitter_bounds_solver(obj, origin, size)
                        n = self._sim.emit_box(mat_id, lo, hi, vel=vel)
                    else:
                        # `bq_emit_box` genere le reseau cote solveur ; pour
                        # y appliquer la turbulence il faut le reproduire
                        # cote Python (`_box_lattice_points`, PAS
                        # `_lattice_points_in_box` — reseau global, ancrage
                        # different, voir sa docstring) puis emettre avec
                        # une vitesse par particule.
                        lo, hi = emitter_bounds_solver(obj, origin, size)
                        lattice = _box_lattice_points(lo, hi, spacing)
                        rng = _turbulence_rng(op.turbulence_seed, 0, emitter_index)
                        noise = _CurlNoise(
                            op.turbulence_seed, emitter_index, lo, hi,
                            dx, vel, spacing, self._frame_dt,
                        )
                        pts, vels = _turbulent_emission(
                            lattice, vel, op.turbulence, spacing,
                            self._frame_dt, rng, dx, 0.0, noise,
                        )
                        n = self._sim.emit_points_vel(mat_id, pts, vels)

                    if n == 0:
                        # Emetteur plus petit que le pas du reseau
                        # (`spacing`) : aucun point du reseau global ne
                        # tombe dedans. Plutot que de refuser le bake,
                        # on emet UNE particule au centre de l'emetteur
                        # (centre de sa bbox solveur, qui coincide avec le
                        # centre du volume pour une forme convexe centree,
                        # et en est une approximation raisonnable sinon) —
                        # a condition que ce centre tombe dans la zone
                        # utile du domaine (deja garanti par la validation
                        # `emitter_overflow` de `_validate`, revérifié ici
                        # par prudence).
                        lo, hi = emitter_bounds_solver(obj, origin, size)
                        center = tuple((lo[a] + hi[a]) / 2.0 for a in range(3))
                        usable = domain_usable_bounds(self._scene)
                        inside_usable = usable is not None and all(
                            usable[0][a] <= center[a] <= usable[1][a]
                            for a in range(3)
                        )
                        if inside_usable:
                            center_arr = np.array([center], dtype=np.float32)
                            n = self._sim.emit_points(mat_id, center_arr, vel=vel)
                            self.report(
                                {"WARNING"},
                                f"« {obj.name} » est plus petit que le pas "
                                f"du réseau du solveur ({spacing:.5f} m en "
                                "espace solveur) : une seule particule a "
                                "été émise, au centre de l'émetteur. "
                                "Augmentez « Résolution de grille » ou "
                                "« Particules/cellule/axe » pour un "
                                "échantillonnage plus fin.",
                            )
                    total_emitted += n
                    continue

                # INFLOW : n'emet rien ici. Le nuage COMPLET de sites est
                # memorise ; la premiere frame du modal (`_emit_inflow_sites`)
                # le trouve entierement libre (aucune particule encore
                # emise) et sature donc le volume d'un coup — c'est le
                # "remplissage initial" attendu, obtenu naturellement par la
                # regle generale plutot que par un cas particulier.
                if op.emit_source == "MESH":
                    sites = mesh_points
                else:
                    lo, hi = emitter_bounds_solver(obj, origin, size)
                    sites = _lattice_points_in_box(lo, hi, spacing)

                if sites.shape[0] == 0:
                    self.report(
                        {"ERROR"},
                        f"« {obj.name} » (émission continue) : aucun site "
                        "d'émission (émetteur trop fin devant le pas du "
                        f"réseau du solveur, {spacing:.5f} m). Agrandissez "
                        "l'émetteur ou augmentez la résolution.",
                    )
                    self._cleanup(context)
                    return {"CANCELLED"}

                # `noise` est construit UNE FOIS ici (comme `sites`) et
                # persiste dans l'etat pour toute la duree du bake : c'est
                # ce qui permet au champ de deriver de facon COHERENTE
                # d'une frame a l'autre (voir `_CurlNoise`, `_InflowState`).
                bbox_lo, bbox_hi = sites.min(axis=0), sites.max(axis=0)
                noise = _CurlNoise(
                    op.turbulence_seed, emitter_index, bbox_lo, bbox_hi,
                    dx, vel, spacing, self._frame_dt,
                )

                self._inflow_states.append(
                    _InflowState(
                        obj.name, mat_id, vel, spacing, sites,
                        dx=dx,
                        turbulence=op.turbulence,
                        turbulence_seed=op.turbulence_seed,
                        emitter_index=emitter_index,
                        noise=noise,
                    )
                )

            if total_emitted == 0 and not self._inflow_states:
                self.report(
                    {"ERROR"},
                    "Aucune particule émise : vérifiez que les émetteurs "
                    "sont bien à l'intérieur du domaine et que leurs "
                    "boîtes ne sont pas dégénérées (dimension nulle).",
                )
                self._cleanup(context)
                return {"CANCELLED"}

            cache_dir = bpy.path.abspath(props.cache_dir)
            bqd_path, _ = cache.cache_paths(cache_dir, scene.name)
            cache.ensure_cache_dir(cache_dir)
            # velocity=True : le bake whitewater (independant du mesh, cf.
            # docs/plan-milestone-8.md D2) a besoin du canal vitesse du .bqd
            # pour classer/advecter les particules secondaires. L'ecrire
            # systematiquement evite un pre-requis invisible que l'artiste
            # decouvrirait au moment d'un bake_all, bien apres avoir deja
            # attendu le bake de particules.
            self._writer = cache.CacheWriter(
                bqd_path, self._sim.particle_count, velocity=True
            )
            # Le sidecar .mat est ecrit a la FIN du bake (voir _cleanup) :
            # les ids materiaux des particules ajoutees en cours de route
            # par un emetteur INFLOW ne sont connus qu'une fois la
            # derniere frame ecrite (voir cache.py, doc du sidecar .mat).

            self._frame_count = props.frame_end - props.frame_start + 1

            # D10 : PRE-EXTRACTION de la geometrie des colliders FIXES
            # animes, frame par frame, ICI sur le thread PRINCIPAL — le
            # thread de calcul lance plus bas (voir _bake_worker) ne doit
            # plus jamais toucher bpy/le depsgraph (voir la garde en tete de
            # module). Reutilise _advance_scene_frame (avance
            # self._frame_index puis self._scene, evalue self._depsgraph)
            # et _collect_collider_frame (extraction pure, qui IGNORE
            # desormais les colliders dynamiques — voir sa docstring)
            # exactement comme le faisait l'ancienne boucle modale frame par
            # frame.
            self._collider_frames = None
            self._body_track = None
            self._kinematic_pose_frames = None
            if self._collider_states:
                origin, size = self._domain_transform

                # M17/phase B (D9, D12) : repere de corps + maillage de
                # repos de TOUS les colliders (dynamiques ET fixes),
                # captures UNE SEULE FOIS, a la PREMIERE frame du bake —
                # donc AVANT la boucle de pre-extraction ci-dessous, sur la
                # meme frame qu'elle (idx == 0). C'est le traitement
                # UNIFORME de D12 (« aucun cas particulier dans le solveur
                # de contact ») qui fait qu'une caisse flottante se pose sur
                # le fond d'un bassin FIXE : sans repere de corps pour le
                # collider fixe, son SDF local n'aurait rien a quoi
                # s'ancrer. Un collider FIXE ne subit toujours AUCUN refus
                # de bake sur maillage ouvert/degenere (D14) — seul un
                # DYNAMIQUE au maillage ouvert fait toujours echouer le
                # bake (voir `_setup_collider_body`).
                self._frame_index = 0
                self._advance_scene_frame()
                warnings = []
                open_static_names = []
                for state in self._collider_states:
                    try:
                        warning = _setup_collider_body(
                            state, self._depsgraph, origin, size
                        )
                    except ValueError as exc:
                        self.report({"ERROR"}, str(exc))
                        self._cleanup(context)
                        return {"CANCELLED"}
                    if warning is not None:
                        warnings.append(warning)
                    if not state.dynamic and not state.mesh_closed:
                        open_static_names.append(state.obj.name)
                for warning in warnings:
                    self.report({"WARNING"}, warning)

                # Point 3 de B4 : un maillage OUVERT (ex. un plane) ne peut
                # porter aucun champ de distance signee coherent (D9) — il
                # reste un collider FLUIDE parfaitement valide (inchange),
                # mais ne participera JAMAIS au contact solide<->solide.
                # C'est precisement le piege que vit l'artiste qui essaie
                # d'arreter un cube dynamique avec un plane : plutot que de
                # le laisser le decouvrir en silence, on le nomme et on dit
                # quoi faire. Memorise (jamais recalcule depuis un `draw()`,
                # trop couteux) pour le tableau de bord — voir
                # `open_mesh_collider_names`, consomme par ui.py.
                _OPEN_MESH_COLLIDERS[self._scene.name] = tuple(open_static_names)
                if open_static_names:
                    joined = ", ".join(f"« {n} »" for n in open_static_names)
                    self.report(
                        {"WARNING"},
                        f"Maillage non fermé, contact solide↔solide "
                        f"désactivé pour : {joined}. Ces colliders restent "
                        "valides pour le fluide, mais ne peuvent pas "
                        "arrêter un solide dynamique : utilisez une boîte "
                        "fermée (même très aplatie) si vous voulez qu'ils "
                        "en arrêtent un.",
                    )

                # Un corps par collider (dynamique ET fixe), meme ordre que
                # self._collider_states == body_index : voir
                # _build_rigid_bodies. Appele UNE SEULE FOIS (voir
                # bq_set_collider_bodies, bourrasque.h) — jamais par frame :
                # rappeler cette fonction ecraserait l'etat interne (v, w,
                # sommeil, warm starting du contact) integre par bq_step.
                bodies = _build_rigid_bodies(self._collider_states)
                self._sim.set_collider_bodies(bodies)

                # D9/D10 : SDF local + echantillons de surface, UNIQUEMENT
                # pour les corps dont le maillage est ferme
                # (`state.body_sdf_built`, pose par `_setup_collider_body`)
                # — TOUS les colliders, dynamiques ET fixes (D12). C'est ce
                # qui fait qu'un cube dynamique heurte et se pose sur un sol
                # FIXE : sans cet appel pour les colliders fixes (l'etat de
                # la phase A), `k_gen_contacts` n'avait tout simplement
                # aucun SDF a interroger pour eux (« pas de SDF construit —
                # body_sdf le traite comme pas de collider », mlsmpm.cu).
                # N'exige PAS que `set_collider_bodies` ait deja ete appele
                # (les deux pipelines sont independants, voir
                # `bq_build_body_sdf`) — l'ordre choisi ici (bodies d'abord)
                # est une simple lisibilite.
                from . import rigidbody

                for state in self._collider_states:
                    if not state.body_sdf_built:
                        continue
                    tri_local = state.rest_offsets[state.rest_tris].astype(
                        np.float32
                    )
                    self._sim.build_body_sdf(
                        state.body_index, tri_local, dx, max_res=128
                    )
                    samples = rigidbody.surface_samples(
                        state.rest_offsets, state.rest_tris, dx,
                        _BODY_SURFACE_SAMPLE_CAP,
                    )
                    self._sim.set_body_samples(state.body_index, samples)

                self._body_track = []

                # Limite ASSUMEE et TOUJOURS VRAIE (contrairement au fige-
                # ment de la pose de contact, resolu ci-dessous par
                # `Sim.set_body_pose`) : un collider dont le MAILLAGE se
                # DEFORME (armature, shape keys, modificateur deformant)
                # n'est pas rigide. `_kinematic_pose_frame` met a jour
                # x/q/v/w de l'OBJET dans son ensemble, mais son SDF de
                # contact — echantillonne UNE SEULE FOIS a la premiere
                # frame (D9) — reste celui de sa forme au repos : le fluide,
                # lui, continue de voir la geometrie deformee reelle
                # (`_collect_collider_frame` reevalue le maillage chaque
                # frame). Non detectable a bas cout (il faudrait re-
                # echantillonner le maillage chaque frame, exactement ce
                # que ce jalon evite pour le contact) : a signaler a
                # l'artiste hors de ce lot si le besoin s'en fait sentir.
                frames = []
                kinematic_pose_frames = []
                for idx in range(self._frame_count):
                    self._frame_index = idx
                    self._advance_scene_frame()
                    frames.append(
                        _collect_collider_frame(
                            self._collider_states, self._depsgraph, origin,
                            size, self._frame_dt, self.report,
                        )
                    )
                    # Point 2 de B4 : pose/vitesse SOLVEUR de chaque
                    # collider CINEMATIQUE au SDF construit, pour que son
                    # contact SOLIDE suive sa pose animee (D9/D12) — pas
                    # seulement le fluide. Applique par `_bake_worker`
                    # (`Sim.set_body_pose`), pas ici : le thread de calcul
                    # n'est pas encore lance, mais l'ordre par frame doit
                    # etre respecte au moment du `bq_step` correspondant.
                    kinematic_pose_frames.append(
                        _kinematic_pose_frame(
                            self._collider_states, origin, size, self._frame_dt
                        )
                    )
                self._collider_frames = frames
                self._kinematic_pose_frames = kinematic_pose_frames

                # La pre-extraction vient de parcourir tout l'intervalle du
                # bake : remet la scene a la frame de depart plutot que de
                # la laisser sur la derniere frame pre-extraite pendant
                # toute la duree du calcul en arriere-plan (_cleanup la
                # restaurera de toute facon a la fin, mais autant eviter cet
                # etat transitoire trompeur affiche entre-temps).
                scene.frame_set(self._start_frame)

            self._frame_index = 0
            self._pos_buffer = None

            BQ_OT_bake.cancel_requested = False
            props.is_baking = True
            props.bake_progress = 0.0
            props.baked_frames = 0

            # Le calcul (emission d'inflow, step, lecture des positions,
            # ecriture cache) part dans un thread daemon : voir la garde en
            # tete de module et docs/plan-milestone-7.md D10. Le tick modal
            # ne fait plus que sonder `self._progress` (voir `modal`).
            self._progress = _BakeProgress()
            self._cancel_event = threading.Event()
            self._thread = threading.Thread(
                target=_bake_worker,
                args=(
                    self._progress, self._cancel_event, self._sim,
                    self._writer, self._frame_count, self._frame_dt,
                    self._inflow_states, self._usable_bounds,
                    self._collider_frames, self._dynamic_collider_states,
                    self._body_track, self._kinematic_pose_frames,
                ),
                daemon=True,
            )

            wm = context.window_manager
            # Intervalle de sondage, PAS le pas de calcul (qui tourne dans
            # le thread, a sa propre vitesse) : 50 ms est trois a quatre
            # fois sous le seuil de reactivite ESC exige (V10, < 200 ms),
            # sans imposer au thread principal un sondage inutilement
            # frequent pendant tout le bake.
            self._timer = wm.event_timer_add(0.05, window=context.window)
            if not wm.modal_handler_add(self):
                raise RuntimeError(
                    "impossible d'installer le gestionnaire modal "
                    "(wm.modal_handler_add a renvoye faux)"
                )
            self._thread.start()
        except lib.BourrasqueError as exc:
            self._cleanup(context)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        except Exception as exc:
            self._cleanup(context)
            self.report({"ERROR"}, f"Erreur lors de la préparation du bake : {exc}")
            return {"CANCELLED"}

        BQ_OT_bake._active_instance = self
        return {"RUNNING_MODAL"}

    # -- modal ----------------------------------------------------------

    def modal(self, context, event):
        scene = self._scene
        props = scene.bourrasque

        # ESC ou le bouton Annuler ne font plus que POSITIONNER le drapeau
        # que le thread de calcul consulte entre deux frames (jamais une
        # interruption forcee, voir docs/plan-milestone-7.md D10) : le
        # nettoyage effectif attend que le thread ait reellement fini
        # (`progress.done`, plus bas) — c'est ce qui garde le viewport
        # manipulable et ESC reactif (< 200 ms, V10) sans jamais detruire
        # `self._sim`/`self._writer` sous les pieds du thread qui les
        # utilise encore.
        if event.type == "ESC" or BQ_OT_bake.cancel_requested:
            self._cancel_event.set()

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        progress = self._progress

        # Seul CE tick modal (thread principal) peut appeler self.report —
        # les messages accumules par le thread de calcul (saturation,
        # emission refusee...) transitent par `progress.reports`, jamais
        # par un appel bpy direct depuis le thread (voir la garde en tete
        # de module).
        while True:
            try:
                level, msg = progress.reports.get_nowait()
            except queue.Empty:
                break
            self.report(level, msg)

        self._frame_index = progress.frame_index
        props.baked_frames = self._frame_index
        props.bake_progress = self._frame_index / max(1, self._frame_count)
        _tag_redraw(context)

        if not progress.done:
            return {"RUNNING_MODAL"}

        # Le thread a fini (normalement, par annulation, ou par exception) :
        # `progress.done` n'est mis a vrai qu'apres son dernier geste (voir
        # `_bake_worker`), rejoindre est donc immediat.
        self._thread.join()

        if progress.error is not None:
            self._cleanup(context)
            self.report({"ERROR"}, f"Erreur pendant le bake : {progress.error}")
            return {"CANCELLED"}

        if self._cancel_event.is_set():
            # Annulation : les keyframes des corps dynamiques sont posees
            # sur les frames EFFECTIVEMENT calculees avant de nettoyer (D8
            # du plan — le travail deja fait n'est jamais jete). Voir
            # _post_keyframes, qui lit self._body_track/self._dynamic_
            # collider_states, tous deux reinitialises par _cleanup().
            self._post_keyframes(context)
            self._cleanup(context)
            self.report({"INFO"}, "Bake annulé.")
            return {"CANCELLED"}

        self._finish(context)
        return {"FINISHED"}

    # -- avance d'une frame — factorise pour rester appelable directement,
    # -- sans timer ni evenement bpy (voir script de validation du jalon) -

    def _emit_inflow_sites(self):
        """Delegue a `_emit_inflow_sites_impl` (fonction PURE, aucun bpy —
        voir sa docstring pour le detail de la regle d'emission), en lui
        passant `self.report` comme callable de reporting. Conservee comme
        methode d'instance UNIQUEMENT pour la compatibilite des scripts de
        `tools/repro/*.py`, qui la lient a un faux operateur via
        `types.MethodType` et lisent/ecrivent `self._saturated` — le thread
        de calcul de `BQ_OT_bake` (voir `_bake_worker`) appelle
        `_emit_inflow_sites_impl` DIRECTEMENT, jamais cette methode (qui
        appellerait `self.report`, un acces bpy interdit hors du thread
        principal — voir la garde en tete de module).

        Renvoie le nombre total de particules effectivement emises pour
        cette frame (utile aux scripts de validation) ; met a jour
        `self._saturated` en place.
        """
        total, self._saturated = _emit_inflow_sites_impl(
            self._sim, self._inflow_states, self._usable_bounds,
            self._frame_index, self._frame_dt, self._saturated, self.report,
        )
        return total

    def _advance_scene_frame(self):
        """Avance `self._scene` a la frame Blender correspondant a
        `self._frame_index`, et reevalue le depsgraph UNE FOIS pour cette
        frame (`self._depsgraph`) — c'est le seul moyen d'obtenir une
        geometrie de collider animee (voir docstring de module) : la scene
        n'etait auparavant JAMAIS avancee pendant un bake.

        Le cout de reevaluation du depsgraph est assume, mais ne doit pas
        etre paye plusieurs fois par frame : `_update_colliders` reutilise
        `self._depsgraph` pour TOUS les colliders de cette frame plutot que
        d'appeler `evaluated_depsgraph_get()` par collider.

        Hors perimetre de ce jalon : ceci rend de fait les emetteurs
        animables (leur `matrix_world`/forme suivrait desormais l'anim), en
        particulier l'ensemencement volumique de l'inflow, dont le nuage de
        sites (`_InflowState.sites`) reste calcule UNE FOIS dans `invoke()`
        (avant tout `frame_set`) et n'est PAS recalcule ici.

        `bpy.context.evaluated_depsgraph_get()` resout le depsgraph via
        `context.scene`/`context.view_layer`, PAS via `self._scene`
        directement : sans precaution, si l'utilisateur change la scene
        active de la fenetre pendant le bake (le modal ne consomme que les
        evenements TIMER, tout le reste passe en PASS_THROUGH — voir
        `invoke`), le depsgraph obtenu serait celui de la MAUVAISE scene.
        `temp_override` force la resolution sur `self._scene` quel que soit
        l'etat de `context.window.scene` a cet instant, en mode UI comme en
        `--background` (ou `context.window` n'existe pas, mais
        `evaluated_depsgraph_get` n'a besoin que de `scene`/`view_layer`).
        """
        target = self._scene.bourrasque.frame_start + self._frame_index
        with bpy.context.temp_override(
            scene=self._scene, view_layer=self._scene.view_layers[0]
        ):
            self._scene.frame_set(target)
            self._depsgraph = bpy.context.evaluated_depsgraph_get()

    def _update_colliders(self):
        """Reevalue tous les colliders a la frame courante (geometrie via
        `self._depsgraph`, deja mis a jour par `_advance_scene_frame`) et
        transmet leurs triangles au coeur pour cette frame, en UN SEUL appel
        (`Sim.set_colliders`) concatenant tous les colliders.

        Positions : converties monde -> solveur via `world_to_solver_array`
        (AVEC translation, ce sont des points). Vitesse par sommet :
        `(position_monde_courante - position_monde_precedente) /
        frame_dt`, calculee en espace MONDE puis convertie via
        `world_to_solver_dir_array` (SANS translation, ce sont des
        directions) — ne jamais confondre les deux, voir docstring de
        module. Nulle a la premiere frame d'un collider, ou si son nombre
        de sommets a change depuis la frame precedente (topologie non
        appariable, voir `_ColliderState`).

        Ne fait rien si aucun collider n'est configure pour ce bake (la
        simulation cote coeur n'a jamais eu de collider a effacer).

        Delegue a `_collect_collider_frame` (fonction PURE, aucun bpy au-
        dela de `self._depsgraph` deja resolu par `_advance_scene_frame`)
        pour l'extraction geometrique des colliders FIXES, la combine
        (`_combine_collider_geometry`) a la geometrie des colliders
        DYNAMIQUES reconstruite depuis l'etat rigide COURANT
        (`self._sim.read_collider_bodies`, `_dynamic_collider_geometry`),
        puis applique le resultat a `self._sim` — ces derniers gestes SONT
        les seuls propres a cette methode d'instance. Conservee pour la
        compatibilite des scripts de `tools/repro/*.py` (voir
        `_emit_inflow_sites` pour la meme discipline) : le thread de calcul
        de `BQ_OT_bake` (voir `_bake_worker`) n'appelle jamais cette
        methode, il consomme directement `self._collider_frames`,
        PRE-EXTRAIT par `_collect_collider_frame` sur le thread principal
        avant son lancement (voir `BQ_OT_bake.invoke` et la garde en tete de
        module), et reconstruit sa propre part dynamique en pur numpy.

        `getattr(self, "_dynamic_collider_states", ())` : les scripts de
        `tools/repro/*.py` anterieurs a ce jalon lient cette methode a un
        faux operateur minimal qui ne definit pas cet attribut — absent, on
        le traite comme "aucun collider dynamique" (comportement inchange).

        Applique aussi (M17/phase B, B4, point 2) la pose/vitesse SOLVEUR
        des colliders CINEMATIQUES au SDF construit, via `Sim.set_body_pose`
        — le pendant, pour le contact SOLIDE, de la geometrie transmise ici
        au fluide. `_bake_worker` (chemin de production, thread de calcul)
        fait l'equivalent depuis `self._kinematic_pose_frames`, PRE-EXTRAIT
        sur le thread principal ; cette methode, elle, tourne DEJA sur le
        thread principal (compatibilite `tools/repro/*.py`), donc appelle
        `_kinematic_pose_frame` directement, sans pre-extraction.
        """
        if not self._collider_states:
            return

        origin, size = self._domain_transform
        static_frame = _collect_collider_frame(
            self._collider_states, self._depsgraph, origin, size,
            self._frame_dt, self.report,
        )
        dynamic_states = getattr(self, "_dynamic_collider_states", ())
        if dynamic_states:
            body_state = self._sim.read_collider_bodies()
            dynamic_geo = _dynamic_collider_geometry(dynamic_states, body_state)
            tri_all, vel_all, fric_all, body_all = _combine_collider_geometry(
                static_frame, dynamic_geo
            )
        else:
            tri_all, vel_all, fric_all, body_all = static_frame
        self._sim.set_colliders(tri_all, vel_all, fric_all, tri_body=body_all)

        for body_index, x, q, v, w in _kinematic_pose_frame(
            self._collider_states, origin, size, self._frame_dt
        ):
            self._sim.set_body_pose(body_index, x, q, v=v, w=w)

    def _advance_frame(self):
        """Avance la simulation d'UNE frame : avance la frame Blender (pour
        une geometrie de collider animee), met a jour les colliders, emet
        les sites d'inflow libres, fait avancer le solveur, lit les
        positions et les ajoute au cache.

        Ne depend d'aucun etat modal (timer, evenement bpy) : appelable
        directement depuis un script de validation hors du cycle modal de
        Blender (l'operateur modal ne s'execute pas en `--background`).
        """
        self._advance_scene_frame()
        self._update_colliders()
        self._emit_inflow_sites()
        self._sim.step(self._frame_dt)
        self._pos_buffer = self._sim.read_positions(out=self._pos_buffer)
        self._writer.append_frame(self._pos_buffer)

    # -- keyframes des corps rigides dynamiques (D8 du plan) -------------

    def _post_keyframes(self, context):
        """Pose les keyframes `location`/`rotation_quaternion` des colliders
        DYNAMIQUES sur les frames EFFECTIVEMENT bakees (D8 du plan) —
        appelee AVANT `_cleanup` (qui reinitialise `self._collider_states`/
        `self._dynamic_collider_states`/`self._body_track`), aussi bien en
        fin normale (`_finish`) qu'en annulation (`modal`) : le travail deja
        calcule n'est jamais jete.

        Un collider FIXE (`dynamic` faux) n'est JAMAIS keyframe ici — seuls
        `self._dynamic_collider_states` sont parcourus.

        `self._body_track[frame_index]` est l'etat de TOUS les corps, en
        espace SOLVEUR, apres le pas de la frame `frame_index` (voir
        `_bake_worker`). La position se convertit en monde par
        `solver_to_world` (point). L'orientation se convertit par
        `_solver_quat_to_world` (conjugaison par la rotation fixe
        solveur<->monde, voir sa docstring) AVANT d'etre composee avec
        `com0_world`/`m0` (deja en espace monde, captures par
        `_setup_collider_body`) via `rigidbody.compose_body_transform`.
        """
        if not self._dynamic_collider_states or not self._body_track:
            return

        from . import rigidbody

        origin, size = self._domain_transform
        frame_start = self._scene.bourrasque.frame_start
        n_done = len(self._body_track)

        for state in self._dynamic_collider_states:
            obj = state.obj
            obj.rotation_mode = "QUATERNION"
            for frame_index in range(n_done):
                row = self._body_track[frame_index][state.body_index]
                x_solver = row[0:3]
                q_solver = row[3:7]

                x_world = np.array(
                    solver_to_world(tuple(x_solver), origin, size),
                    dtype=np.float64,
                )
                q_world = _solver_quat_to_world(q_solver)

                m = rigidbody.compose_body_transform(
                    x_world, q_world, state.com0_world, state.m0
                )
                loc, quat, _uniform_scale_ok = rigidbody.decompose_loc_rot(m)

                blender_frame = frame_start + frame_index
                obj.location = tuple(loc)
                obj.rotation_quaternion = tuple(quat)
                obj.keyframe_insert(data_path="location", frame=blender_frame)
                obj.keyframe_insert(
                    data_path="rotation_quaternion", frame=blender_frame
                )

    # -- fin normale : ferme proprement puis rafraichit l'affichage ------

    def _finish(self, context):
        scene = self._scene
        n_particles = self._sim.particle_count
        mat_array = self._sim.read_materials()
        self._post_keyframes(context)
        self._cleanup(context)

        from . import display

        obj = display.ensure_particle_object(context, n_particles)
        display.write_material_attribute(obj, mat_array)
        display.refresh(scene)

    # -- nettoyage : SEUL point qui ferme le writer, detruit la sim et ----
    # -- retire le timer, sur tous les chemins (fin, ESC, exception) -----

    def _cleanup(self, context):
        scene = self._scene
        props = scene.bourrasque

        # D10 : arrete et rejoint le thread de calcul AVANT de toucher a
        # `self._sim`/`self._writer` — sur TOUT chemin qui invoque
        # `_cleanup` pendant qu'il tourne encore (ex. `unregister()`
        # appele pendant un bake, extension desactivee en cours de route),
        # jamais detruire la sim GPU ou fermer le fichier sous les pieds
        # du thread qui les utilise. `ident is not None` signifie que
        # `start()` a bien ete appele (un thread jamais demarre ne peut
        # pas etre `join()`) ; un timeout genereux mais fini evite de
        # bloquer indefiniment ce filet de securite si le thread reste
        # coince sur un appel CUDA. `getattr(..., None)` (plutot que
        # `self._thread` direct) : cette methode reste liee par
        # `types.MethodType` a un faux operateur minimal dans
        # `tools/repro/verify_m6.py`, qui ne definit pas ces attributs D10
        # (il n'exerce que le chemin synchrone historique) — absents, on
        # les traite comme "pas de thread a nettoyer".
        thread = getattr(self, "_thread", None)
        if thread is not None:
            cancel_event = getattr(self, "_cancel_event", None)
            if cancel_event is not None:
                cancel_event.set()
            if thread.ident is not None:
                thread.join(timeout=10.0)
            self._thread = None
        self._progress = None
        self._cancel_event = None
        self._collider_frames = None
        self._kinematic_pose_frames = None

        if self._writer is not None and self._sim is not None:
            # Le sidecar .mat est ecrit ICI, au tout dernier moment ou la
            # sim GPU est encore vivante et sur TOUS les chemins de sortie
            # (fin normale, ESC, exception) : les ids materiaux des
            # particules ajoutees en cours de route par un emetteur INFLOW
            # ne sont connus qu'a la derniere frame effectivement ecrite
            # (voir cache.py, doc du sidecar .mat).
            try:
                self._writer.write_materials(self._sim.read_materials())
            except (lib.BourrasqueError, ValueError):
                # Aucune frame ecrite (annulation avant le premier step) ou
                # sim dans un etat incoherent (cleanup declenche par
                # l'exception meme qu'on est en train de gerer) : le
                # sidecar est omis plutot que de masquer l'erreur d'origine.
                pass

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        if self._sim is not None:
            self._sim.destroy()
            self._sim = None

        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None

        self._inflow_states = ()
        self._saturated = False
        self._collider_states = ()
        self._dynamic_collider_states = ()
        self._body_track = None

        # Restaure la frame Blender d'origine, sur TOUS les chemins de
        # sortie (fin normale, ESC, exception) : `_advance_scene_frame` a
        # deplace `scene.frame_current` a chaque frame simulee, la scene ne
        # doit pas rester sur la derniere frame du bake.
        if self._start_frame is not None:
            scene.frame_set(self._start_frame)
            self._start_frame = None

        props.is_baking = False
        BQ_OT_bake.cancel_requested = False
        BQ_OT_bake._active_instance = None
        _tag_redraw(context)


# ---------------------------------------------------------------------------
# BQ_OT_clear_rigid_keys
# ---------------------------------------------------------------------------


class BQ_OT_clear_rigid_keys(bpy.types.Operator):
    """Retire les keyframes `location`/`rotation_quaternion` posees par un
    bake precedent (voir `BQ_OT_bake._post_keyframes`, D8 du plan) sur les
    colliders DYNAMIQUES de la scene.

    Ne touche JAMAIS un collider FIXE (`dynamic` faux), meme s'il porte sa
    propre animation posee a la main par l'artiste (obstacle anime, cas
    majoritaire) — c'est le piege du jalon : cet operateur cible les corps
    DYNAMIQUES, pas tous les colliders. Ne supprime que les F-curves
    `location`/`rotation_quaternion` de l'action courante, jamais
    `animation_data_clear()` en bloc : un collider dynamique ne devrait
    porter aucune autre animation (D6 — « dynamique » est un interrupteur
    exclusif avec l'anim), mais ne pas presumer d'un `custom property
    driver` ou autre F-curve qu'un artiste y aurait tout de meme ajoute.
    """

    bl_idname = "bq.clear_rigid_keys"
    bl_label = "Effacer les clés des corps rigides"
    bl_description = (
        "Retire les clés location/rotation posées par le bake sur les "
        "colliders dynamiques (n'affecte jamais un collider fixe)"
    )
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return any(
            obj.bourrasque.role == "COLLIDER" and obj.bourrasque.dynamic
            for obj in context.scene.objects
        )

    @staticmethod
    def _fcurve_containers(anim):
        """Renvoie la liste des conteneurs de F-curves (chacun offrant
        `.find(data_path, index=...)` et `.remove(fcurve)`) portant
        potentiellement les clés posees par `_post_keyframes` sur `anim`.

        Deux representations coexistent selon la version de Blender :
        `action.fcurves` directement (action "legacy", < 4.4) ou un
        `ActionChannelbag` par `(layer, strip)` (action "layered",
        Blender 4.4+/5.x — meme motif que `add_animated_fixed_collider`
        dans `tools/repro/verify_rigidbody_bake.py`). Renvoie une liste vide
        si `anim`/son action est absente.
        """
        action = anim.action if anim is not None else None
        if action is None:
            return []
        try:
            return [action.fcurves]
        except AttributeError:
            pass
        containers = []
        for layer in action.layers:
            for strip in layer.strips:
                channelbag = strip.channelbag(anim.action_slot)
                if channelbag is not None:
                    containers.append(channelbag.fcurves)
        return containers

    def execute(self, context):
        cleared = 0
        for obj in context.scene.objects:
            if obj.bourrasque.role != "COLLIDER" or not obj.bourrasque.dynamic:
                continue

            removed_here = False
            for container in self._fcurve_containers(obj.animation_data):
                for data_path, n_components in (
                    ("location", 3),
                    ("rotation_quaternion", 4),
                ):
                    for index in range(n_components):
                        fcurve = container.find(data_path, index=index)
                        if fcurve is not None:
                            container.remove(fcurve)
                            removed_here = True
            if removed_here:
                cleared += 1

        self.report(
            {"INFO"},
            f"Clés retirées sur {cleared} collider(s) dynamique(s).",
        )
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# BQ_OT_bake_mesh — operateur modal, deuxieme passe du bake modulaire
# ---------------------------------------------------------------------------
#
# Lit le `.bqd` (deja bake par BQ_OT_bake) frame par frame, maille chacune
# via `lib.Mesher`, ecrit le `.bqm` a cote — voir docs/plan-milestone-7.md,
# D1/D7/D8. Independant de tout `BqSim` (le mailleur ne consomme que des
# positions, cf. `lib.Mesher`) : c'est ce qui rend les deux passes du bake
# reellement modulaires, l'une relancable sans l'autre (V6 du plan).
#
# Meme squelette modal que BQ_OT_bake (timer, drapeau d'annulation de
# classe, `_cleanup` unique convergeant TOUS les chemins de sortie), en
# nettement plus simple : pas d'emission, pas de colliders animes, pas
# d'avance de la frame Blender (le mailleur ne depend d'aucun etat vivant de
# la scene, seulement du `.bqd` deja sur disque).


def _mesh_bake_worker(progress, cancel_event, reader, mesher, writer, frame_count):
    """Boucle de calcul du bake de maillage — executee dans un
    `threading.Thread(daemon=True)` (voir `BQ_OT_bake_mesh.invoke`,
    docs/plan-milestone-7.md D10).

    NE TOUCHE JAMAIS bpy (voir la garde en tete de module) : `reader`
    (lecture du `.bqd`, fichier binaire pur, voir `cache.py`), `mesher`
    (appels ctypes) et `writer` (ecriture du `.bqm`, fichier binaire pur,
    voir `meshcache.py`) n'en dependent d'aucune facon. Contrairement au
    bake de particules (`_bake_worker`), AUCUNE pre-extraction n'est
    requise avant de lancer ce thread : le rognage collider, s'il y en a
    un, est un champ STATIQUE deja construit et fourni a `mesher` avant le
    lancement du thread (voir `BQ_OT_bake_mesh.invoke`, docs/plan-
    milestone-7.md D5) — le mailleur ne depend plus d'aucun etat vivant de
    la scene une fois ce thread demarre.

    Meme discipline que `_bake_worker` pour `progress`/`cancel_event`/la
    capture d'exception (voir sa docstring).
    """
    try:
        for frame_index in range(frame_count):
            if cancel_event.is_set():
                return
            positions = reader.read_frame(frame_index)
            mesher.run(positions)
            verts, tris = mesher.read()
            writer.append_frame(verts, tris)
            progress.frame_index = frame_index + 1
    except Exception as exc:  # noqa: BLE001 — remonte au tick modal, jamais bpy ici
        progress.error = str(exc)
    finally:
        progress.done = True


class BQ_OT_bake_mesh(bpy.types.Operator):
    """Bake le maillage de surface a partir du cache de particules `.bqd`
    deja bake, frame par frame, dans un `.bqm` a cote."""

    bl_idname = "bq.bake_mesh"
    bl_label = "Baker le maillage"
    bl_description = "Reconstruit la surface pour chaque frame déjà bakée du cache de particules"
    bl_options = {"REGISTER"}

    cancel_requested = False
    _active_instance = None

    _timer = None
    _reader = None
    _writer = None
    _mesher = None
    _frame_index = 0
    _frame_count = 0
    _scene = None

    # -- D10 : le calcul tourne dans un thread, le modal ne fait que sonder
    # -- (voir la garde en tete de module) ---------------------------------
    _progress = None
    _cancel_event = None
    _thread = None

    @classmethod
    def poll(cls, context):
        props = context.scene.bourrasque
        return not props.is_baking and not props.is_baking_mesh

    def invoke(self, context, event):
        self._scene = context.scene
        scene = self._scene
        props = scene.bourrasque

        cache_dir = bpy.path.abspath(props.cache_dir)
        bqd_path, _mat_path = cache.cache_paths(cache_dir, scene.name)

        if not os.path.isfile(bqd_path):
            self.report(
                {"ERROR"},
                "Aucun cache de particules (.bqd) trouvé : lancez d'abord "
                "le bake de particules (« Lancer le bake ») avant de baker "
                "le maillage.",
            )
            return {"CANCELLED"}

        try:
            self._reader = cache.CacheReader(bqd_path)
        except (OSError, ValueError) as exc:
            self.report(
                {"ERROR"}, f"Impossible de lire le cache de particules : {exc}"
            )
            return {"CANCELLED"}

        if self._reader.frame_count == 0:
            self._reader.close()
            self._reader = None
            self.report(
                {"ERROR"},
                "Le cache de particules (.bqd) est vide (aucune frame "
                "bakée) : rien à mailler.",
            )
            return {"CANCELLED"}

        layout = mesh_layout(scene)
        if layout is None:
            self._reader.close()
            self._reader = None
            self.report({"ERROR"}, "Aucun domaine défini.")
            return {"CANCELLED"}
        res, cell_size = layout
        if min(res) <= 0 or cell_size <= 0:
            self._reader.close()
            self._reader = None
            self.report(
                {"ERROR"},
                "Le domaine est dégénéré (taille nulle ou négative) : "
                "vérifiez la bounding box de l'objet domaine.",
            )
            return {"CANCELLED"}

        # Facteurs -> valeurs absolues (ergonomie M7.1) : les reglages
        # `mesh_*_factor` de `scene.bourrasque` sont des multiples de
        # l'espacement inter-particules, pas des longueurs — c'est
        # `mesh_effective_radii` qui fait cette conversion, source de
        # verite unique (voir props.py). `layout` n'etant pas `None` ici,
        # le domaine est defini et cet appel ne peut pas renvoyer `None`.
        radii = mesh_effective_radii(scene)
        influence_radius, particle_radius, collider_offset = radii

        try:
            cfg = lib.default_mesher_config()
        except lib.BourrasqueError as exc:
            self._reader.close()
            self._reader = None
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        cfg.grid_res[:] = res
        cfg.cell_size = cell_size
        cfg.influence_radius = influence_radius
        cfg.particle_radius = particle_radius
        cfg.collider_offset = collider_offset
        cfg.smoothing_iters = props.mesh_smoothing_iters
        cfg.min_component_tris = props.mesh_min_component_tris
        # Canal vitesse : reserve cote coeur (bq_mesher_read l'ignore encore,
        # voir bourrasque.h) — aucun reglage artiste ne peut donc l'activer
        # pour l'instant, cf. docs/plan-milestone-7.md D9. Le .bqm est donc
        # toujours ecrit sans ce canal.
        cfg.channels = 0

        try:
            self._mesher = lib.Mesher(cfg)
        except lib.BourrasqueError as exc:
            self._reader.close()
            self._reader = None
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        # Rognage contre les colliders (docs/plan-milestone-7.md, D5) : le
        # champ de distance signee des colliders est construit dES l'appel
        # a bq_set_colliders (les kernels de champ tournent DANS cette
        # fonction, aucun bq_step necessaire, cf. core/src/mlsmpm.cu) — une
        # Sim LEGERE (max_particles=1, rien alloue par particule) suffit
        # donc a le produire : meme grid_res/cell_size que le domaine (pas
        # la resolution du champ de maillage, independante, cf.
        # mesh_layout), triangles de colliders de la scene reutilises tels
        # quels via `_static_collider_triangles` (memes briques que le bake
        # de particules), champ relu par `Sim.read_sdf` (echantillonne AUX
        # NOEUDS) et transmis SANS conversion au mailleur. S'il n'y a aucun
        # collider dans la scene, aucune Sim n'est creee et
        # `set_collider_sdf` n'est jamais appelee.
        collider_objs = [
            obj for obj in scene.objects if obj.bourrasque.role == "COLLIDER"
        ]
        if collider_objs:
            try:
                d_res, d_dx = domain_resolution(scene)
                d_origin, d_size = domain_transform(scene)

                collider_cfg = lib.default_config()
                collider_cfg.grid_res[:] = d_res
                collider_cfg.cell_size = d_dx
                collider_cfg.max_particles = 1

                depsgraph = context.evaluated_depsgraph_get()
                tri_all = _static_collider_triangles(
                    collider_objs, depsgraph, d_origin, d_size
                )
                n_tri = tri_all.shape[0]
                vel_zero = np.zeros_like(tri_all)
                fric_zero = np.zeros((n_tri,), dtype=np.float32)

                with lib.Sim(collider_cfg) as collider_sim:
                    # `bq_set_colliders` refuse de s'executer tant qu'aucun
                    # materiau n'est enregistre (garde de mlsmpm.cu : le champ
                    # de contact depend de la vitesse du son du materiau le
                    # plus raide). Cette Sim ne fait JAMAIS de pas de temps —
                    # elle n'existe que pour produire le champ de distance des
                    # colliders — donc le materiau n'a aucune influence sur le
                    # resultat ; il satisfait seulement la precondition.
                    collider_sim.add_material(
                        lib.BQ_MODEL_WATER, 1000.0, bulk=4.0e4, gamma=3.0
                    )
                    collider_sim.set_colliders(tri_all, vel_zero, fric_zero)
                    collider_sdf = collider_sim.read_sdf()

                self._mesher.set_collider_sdf(collider_sdf, d_res, d_dx)
            except lib.BourrasqueError as exc:
                self._mesher.destroy()
                self._mesher = None
                self._reader.close()
                self._reader = None
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

        bqm_path = mesh_cache_path(cache_dir, scene.name)
        cache.ensure_cache_dir(cache_dir)

        params = meshcache.MeshProductionParams(
            mesh_res=tuple(res),
            cell_size=cell_size,
            influence_radius=influence_radius,
            particle_radius=particle_radius,
            collider_offset=collider_offset,
            smoothing_iters=props.mesh_smoothing_iters,
            min_component_tris=props.mesh_min_component_tris,
            src_frames=self._reader.frame_count,
            src_n_max=self._reader.n_particles,
        )

        try:
            self._writer = meshcache.MeshCacheWriter(bqm_path, params, velocity=False)
        except OSError as exc:
            self._mesher.destroy()
            self._mesher = None
            self._reader.close()
            self._reader = None
            self.report({"ERROR"}, f"Impossible d'écrire le cache de maillage : {exc}")
            return {"CANCELLED"}

        self._frame_count = self._reader.frame_count
        self._frame_index = 0

        BQ_OT_bake_mesh.cancel_requested = False
        props.is_baking_mesh = True
        props.bake_mesh_progress = 0.0
        props.baked_mesh_frames = 0

        # Le calcul (lecture d'une frame .bqd, reconstruction, ecriture
        # .bqm) part dans un thread daemon : voir la garde en tete de
        # module et docs/plan-milestone-7.md D10. Contrairement au bake de
        # particules, aucune pre-extraction n'est necessaire ici : le
        # mailleur ne depend d'aucun etat vivant de la scene (le rognage
        # collider, s'il y en a un, est un champ STATIQUE deja construit et
        # fourni a `self._mesher` plus haut, avant ce point) — `reader`,
        # `mesher`, `writer` ne touchent jamais bpy.
        self._progress = _BakeProgress()
        self._cancel_event = threading.Event()
        self._thread = threading.Thread(
            target=_mesh_bake_worker,
            args=(
                self._progress, self._cancel_event, self._reader,
                self._mesher, self._writer, self._frame_count,
            ),
            daemon=True,
        )

        wm = context.window_manager
        # Meme intervalle de sondage que BQ_OT_bake (voir sa docstring) :
        # sous le seuil de reactivite ESC (V10, < 200 ms) sans sonder plus
        # souvent que necessaire.
        self._timer = wm.event_timer_add(0.05, window=context.window)
        if not wm.modal_handler_add(self):
            self._cleanup(context)
            self.report(
                {"ERROR"},
                "impossible d'installer le gestionnaire modal du bake de maillage",
            )
            return {"CANCELLED"}
        self._thread.start()

        BQ_OT_bake_mesh._active_instance = self
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        props = self._scene.bourrasque

        # Meme discipline que BQ_OT_bake.modal : ESC/Annuler ne font que
        # positionner un drapeau consulte par le thread entre deux frames,
        # jamais une interruption forcee (docs/plan-milestone-7.md D10).
        if event.type == "ESC" or BQ_OT_bake_mesh.cancel_requested:
            self._cancel_event.set()

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        progress = self._progress
        self._frame_index = progress.frame_index
        props.baked_mesh_frames = self._frame_index
        props.bake_mesh_progress = self._frame_index / max(1, self._frame_count)
        _tag_redraw(context)

        if not progress.done:
            return {"RUNNING_MODAL"}

        self._thread.join()

        if progress.error is not None:
            self._cleanup(context)
            self.report(
                {"ERROR"}, f"Erreur pendant le bake de maillage : {progress.error}"
            )
            return {"CANCELLED"}

        if self._cancel_event.is_set():
            self._cleanup(context)
            self.report({"INFO"}, "Bake de maillage annulé.")
            return {"CANCELLED"}

        self._finish(context)
        return {"FINISHED"}

    def _finish(self, context):
        scene = self._scene
        self._cleanup(context)

        from . import display

        display.refresh_mesh(scene)

    def _cleanup(self, context):
        scene = self._scene
        props = scene.bourrasque

        # Meme discipline que BQ_OT_bake._cleanup : arrete et rejoint le
        # thread AVANT de toucher a reader/mesher/writer (voir sa
        # docstring pour le detail du raisonnement).
        if self._thread is not None:
            if self._cancel_event is not None:
                self._cancel_event.set()
            if self._thread.ident is not None:
                self._thread.join(timeout=10.0)
            self._thread = None
        self._progress = None
        self._cancel_event = None

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        if self._mesher is not None:
            self._mesher.destroy()
            self._mesher = None

        if self._reader is not None:
            self._reader.close()
            self._reader = None

        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None

        props.is_baking_mesh = False
        BQ_OT_bake_mesh.cancel_requested = False
        BQ_OT_bake_mesh._active_instance = None
        _tag_redraw(context)


# ---------------------------------------------------------------------------
# BQ_OT_bake_whitewater — operateur modal, troisieme passe du bake modulaire
# ---------------------------------------------------------------------------
#
# Lit le `.bqd` (deja bake par BQ_OT_bake) frame par frame et avance un
# `lib.Whitewater` a ETAT PERSISTANT (contrairement au mailleur, PAS une
# reconstruction sans memoire, voir docs/plan-milestone-8.md D1/D5), ecrit
# le `.bqw` a cote. INDEPENDANT du `.bqm` : ne lit ni n'ecrit jamais le
# maillage (D8 du plan) — `bq.bake_whitewater` ne depend que du `.bqd`.
#
# Meme squelette modal que BQ_OT_bake_mesh (timer, drapeau d'annulation de
# classe, `_cleanup` unique convergeant TOUS les chemins de sortie), avec en
# plus le respect du CONTRAT DE SEQUENTIALITE (D5) : les frames sont
# consommees dans l'ordre CROISSANT, sans saut, par la boucle
# `range(frame_count)` de `_whitewater_bake_worker` — l'API native ne le
# verifie pas elle-meme (cf. bourrasque.h, risque 3 du plan).


def _whitewater_bake_worker(progress, cancel_event, reader, ww, writer,
                             frame_count, frame_dt):
    """Boucle de calcul du bake whitewater — executee dans un
    `threading.Thread(daemon=True)` (voir `BQ_OT_bake_whitewater.invoke`,
    docs/plan-milestone-8.md D8).

    NE TOUCHE JAMAIS bpy (voir la garde en tete de module) : `reader`
    (lecture du `.bqd`, fichier binaire pur, voir `cache.py`), `ww`
    (`lib.Whitewater`, appels ctypes) et `writer` (ecriture du `.bqw`,
    fichier binaire pur, voir `whitewatercache.py`) n'en dependent d'aucune
    facon.

    Contrat de sequentialite STRICT (docs/plan-milestone-8.md, D5) :
    `ww.step` doit etre appele une fois par frame, dans l'ordre CROISSANT
    des frames du `.bqd`, sans saut ni retour arriere — c'est cette boucle
    `range(frame_count)` qui garantit le contrat, l'API native elle-meme ne
    le verifie pas (cf. bourrasque.h et le risque 3 du plan).

    Meme discipline que `_mesh_bake_worker`/`_bake_worker` pour
    `progress`/`cancel_event`/la capture d'exception (voir leurs
    docstrings). `progress.max_refused` accumule le MAXIMUM de
    `ww.last_refused()` observe sur toute la duree du bake (cf. plan,
    risque 4 : le compte de candidates refusees doit etre remonte dans le
    rapport de bake, pas seulement loggue) : ecrit ICI, lu par le tick
    modal seulement une fois le thread termine, jamais lu depuis le thread
    lui-meme.
    """
    try:
        pos_buffer = None
        vel_buffer = None
        for frame_index in range(frame_count):
            if cancel_event.is_set():
                return
            pos_buffer = reader.read_frame(frame_index, out=pos_buffer)
            vel_buffer = reader.read_velocity(frame_index, out=vel_buffer)
            ww.step(pos_buffer, vel_buffer, frame_dt)
            refused = ww.last_refused()
            if refused > progress.max_refused:
                progress.max_refused = refused
            pos, type_, size, age, vel = ww.read()
            writer.append_frame(pos, type_, size, age, velocity=vel)
            progress.frame_index = frame_index + 1
    except Exception as exc:  # noqa: BLE001 — remonte au tick modal, jamais bpy ici
        progress.error = str(exc)
    finally:
        progress.done = True


class BQ_OT_bake_whitewater(bpy.types.Operator):
    """Bake les particules secondaires (whitewater) a partir du cache de
    particules `.bqd` deja bake, frame par frame, dans un `.bqw` a cote.

    Independant du `.bqm` : ne lit jamais le maillage (voir docstring de
    section)."""

    bl_idname = "bq.bake_whitewater"
    bl_label = "Baker le whitewater"
    bl_description = (
        "Simule les particules secondaires (écume, bulles, embruns) à "
        "partir du cache de particules déjà baké"
    )
    bl_options = {"REGISTER"}

    cancel_requested = False
    _active_instance = None

    _timer = None
    _reader = None
    _writer = None
    _ww = None
    _frame_index = 0
    _frame_count = 0
    _scene = None

    # -- D10 : le calcul tourne dans un thread, le modal ne fait que sonder
    _progress = None
    _cancel_event = None
    _thread = None

    @classmethod
    def poll(cls, context):
        props = context.scene.bourrasque
        return (
            not props.is_baking
            and not props.is_baking_mesh
            and not props.is_baking_whitewater
        )

    def invoke(self, context, event):
        self._scene = context.scene
        scene = self._scene
        props = scene.bourrasque

        cache_dir = bpy.path.abspath(props.cache_dir)
        bqd_path, _mat_path = cache.cache_paths(cache_dir, scene.name)

        if not os.path.isfile(bqd_path):
            self.report(
                {"ERROR"},
                "Aucun cache de particules (.bqd) trouvé : lancez d'abord "
                "le bake de particules (« Lancer le bake ») avant de baker "
                "le whitewater.",
            )
            return {"CANCELLED"}

        try:
            self._reader = cache.CacheReader(bqd_path)
        except (OSError, ValueError) as exc:
            self.report(
                {"ERROR"}, f"Impossible de lire le cache de particules : {exc}"
            )
            return {"CANCELLED"}

        if self._reader.frame_count == 0:
            self._reader.close()
            self._reader = None
            self.report(
                {"ERROR"},
                "Le cache de particules (.bqd) est vide (aucune frame "
                "bakée) : rien à simuler.",
            )
            return {"CANCELLED"}

        # Le canal vitesse du .bqd est requis (D5 de docs/plan-milestone-8.md
        # : bq_whitewater_step attend une vitesse par particule fluide) —
        # verifie AVANT de commencer le bake, avec un message actionnable,
        # plutot que d'echouer sur la premiere frame.
        if not self._reader.has_velocity:
            self._reader.close()
            self._reader = None
            self.report(
                {"ERROR"},
                "Le bake whitewater nécessite le canal vitesse du .bqd : "
                "rebakez les particules avec la vitesse activée.",
            )
            return {"CANCELLED"}

        cfg = whitewater_config_from_scene(scene)
        if cfg is None:
            self._reader.close()
            self._reader = None
            self.report({"ERROR"}, "Aucun domaine défini.")
            return {"CANCELLED"}

        # Bornes de domaine (BqWhitewaterConfig.grid_res/cell_size, meme
        # convention que BqConfig) : SANS elles, les deux champs restent a
        # zero (ctypes les initialise a zero par defaut) et toute particule
        # serait immediatement consideree hors domaine des le premier
        # sous-pas — voir bourrasque.h, section whitewater.
        d_res, d_dx = domain_resolution(scene)
        cfg.grid_res[:] = d_res
        cfg.cell_size = d_dx

        try:
            self._ww = lib.Whitewater(cfg)
        except lib.BourrasqueError as exc:
            self._reader.close()
            self._reader = None
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        # Rognage contre les colliders (meme limitation assumee que le bake
        # de maillage, voir BQ_OT_bake_mesh.invoke ci-dessus : collider
        # STATIQUE, calcule UNE FOIS avant la boucle sur les frames, pas les
        # colliders animes). S'il n'y a aucun collider dans la scene, aucune
        # Sim n'est creee et `set_collider_sdf` n'est jamais appelee.
        collider_objs = [
            obj for obj in scene.objects if obj.bourrasque.role == "COLLIDER"
        ]
        if collider_objs:
            try:
                collider_cfg = lib.default_config()
                collider_cfg.grid_res[:] = d_res
                collider_cfg.cell_size = d_dx
                collider_cfg.max_particles = 1

                depsgraph = context.evaluated_depsgraph_get()
                d_origin, d_size = domain_transform(scene)
                tri_all = _static_collider_triangles(
                    collider_objs, depsgraph, d_origin, d_size
                )
                n_tri = tri_all.shape[0]
                vel_zero = np.zeros_like(tri_all)
                fric_zero = np.zeros((n_tri,), dtype=np.float32)

                with lib.Sim(collider_cfg) as collider_sim:
                    # `bq_set_colliders` refuse de s'executer tant qu'aucun
                    # materiau n'est enregistre (garde de mlsmpm.cu). Cette
                    # Sim ne fait JAMAIS de pas de temps — elle n'existe que
                    # pour produire le champ de distance des colliders — donc
                    # le materiau n'a aucune influence sur le resultat ; il
                    # satisfait seulement la precondition.
                    collider_sim.add_material(
                        lib.BQ_MODEL_WATER, 1000.0, bulk=4.0e4, gamma=3.0
                    )
                    collider_sim.set_colliders(tri_all, vel_zero, fric_zero)
                    collider_sdf = collider_sim.read_sdf()
                    collider_cnrm = collider_sim.read_cnrm()

                self._ww.set_collider_sdf(collider_sdf, d_res, d_dx)
                self._ww.set_collider_cnrm(collider_cnrm, d_res, d_dx)
            except lib.BourrasqueError as exc:
                self._ww.close()
                self._ww = None
                self._reader.close()
                self._reader = None
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

        bqw_path = whitewater_cache_path(cache_dir, scene.name)
        cache.ensure_cache_dir(cache_dir)

        # Params ecrits en en-tete pour l'invalidation du cache (voir
        # whitewatercache.py, D7 du plan) : miroir exact de `cfg` (deja
        # construite depuis les reglages courants de la scene, voir
        # `whitewater_config_from_scene`) plus `src_frames`/`src_n_max` du
        # `.bqd` source.
        params = whitewatercache.WhitewaterProductionParams(
            influence_radius=cfg.influence_radius,
            spawn_rate=cfg.spawn_rate,
            ta_min=cfg.ta_min,
            ta_max=cfg.ta_max,
            ta_weight=cfg.ta_weight,
            wc_min=cfg.wc_min,
            wc_max=cfg.wc_max,
            wc_weight=cfg.wc_weight,
            ke_min=cfg.ke_min,
            ke_max=cfg.ke_max,
            ke_weight=cfg.ke_weight,
            life_spray=cfg.life_spray,
            life_foam=cfg.life_foam,
            life_bubble=cfg.life_bubble,
            drag_spray=cfg.drag_spray,
            drag_foam=cfg.drag_foam,
            buoyancy_bubble=cfg.buoyancy_bubble,
            src_frames=self._reader.frame_count,
            src_n_max=self._reader.n_particles,
        )

        try:
            # Canal vitesse TOUJOURS active (motion blur Cycles) : la
            # vitesse par particule secondaire est desormais calculee cote
            # GPU (bq_whitewater_read, ABI 10) et coute peu d'espace disque
            # supplementaire (un tiers du volume deja occupe par les
            # positions) — pas de reglage UI pour ne pas ajouter une option
            # de plus a gerer, contrairement au choix "reserve" fait pour le
            # canal vitesse du .bqm (D9 de M7), qui restait non calcule cote
            # coeur au moment de cette decision.
            self._writer = whitewatercache.WhitewaterCacheWriter(
                bqw_path, params, velocity=True
            )
        except OSError as exc:
            self._ww.close()
            self._ww = None
            self._reader.close()
            self._reader = None
            self.report(
                {"ERROR"}, f"Impossible d'écrire le cache de whitewater : {exc}"
            )
            return {"CANCELLED"}

        self._frame_count = self._reader.frame_count
        self._frame_index = 0

        # fps/fps_base ensemble donnent la frequence de rendu reelle (voir
        # BQ_OT_bake.invoke pour la meme justification).
        frame_dt = scene.render.fps_base / scene.render.fps

        BQ_OT_bake_whitewater.cancel_requested = False
        props.is_baking_whitewater = True
        props.bake_whitewater_progress = 0.0
        props.baked_whitewater_frames = 0

        # Le calcul (lecture d'une frame .bqd, avance du whitewater,
        # ecriture .bqw) part dans un thread daemon : voir la garde en tete
        # de module et docs/plan-milestone-8.md D8. `reader`, `ww`, `writer`
        # ne touchent jamais bpy.
        self._progress = _BakeProgress()
        self._cancel_event = threading.Event()
        self._thread = threading.Thread(
            target=_whitewater_bake_worker,
            args=(
                self._progress, self._cancel_event, self._reader,
                self._ww, self._writer, self._frame_count, frame_dt,
            ),
            daemon=True,
        )

        wm = context.window_manager
        # Meme intervalle de sondage que BQ_OT_bake/BQ_OT_bake_mesh (voir
        # leurs docstrings) : sous le seuil de reactivite ESC (V8, < 200 ms)
        # sans sonder plus souvent que necessaire.
        self._timer = wm.event_timer_add(0.05, window=context.window)
        if not wm.modal_handler_add(self):
            self._cleanup(context)
            self.report(
                {"ERROR"},
                "impossible d'installer le gestionnaire modal du bake de whitewater",
            )
            return {"CANCELLED"}
        self._thread.start()

        BQ_OT_bake_whitewater._active_instance = self
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        props = self._scene.bourrasque

        # Meme discipline que BQ_OT_bake_mesh.modal : ESC/Annuler ne font
        # que positionner un drapeau consulte par le thread entre deux
        # frames, jamais une interruption forcee (docs/plan-milestone-7.md
        # D10, reprise ici).
        if event.type == "ESC" or BQ_OT_bake_whitewater.cancel_requested:
            self._cancel_event.set()

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        progress = self._progress
        self._frame_index = progress.frame_index
        props.baked_whitewater_frames = self._frame_index
        props.bake_whitewater_progress = self._frame_index / max(1, self._frame_count)
        _tag_redraw(context)

        if not progress.done:
            return {"RUNNING_MODAL"}

        self._thread.join()

        if progress.error is not None:
            self._cleanup(context)
            self.report(
                {"ERROR"}, f"Erreur pendant le bake de whitewater : {progress.error}"
            )
            return {"CANCELLED"}

        if self._cancel_event.is_set():
            self._cleanup(context)
            self.report({"INFO"}, "Bake de whitewater annulé.")
            return {"CANCELLED"}

        self._finish(context)
        return {"FINISHED"}

    def _finish(self, context):
        scene = self._scene
        # Capture AVANT _cleanup (qui remet self._progress a None) : voir
        # docstring de `_whitewater_bake_worker` sur `progress.max_refused`
        # (cf. plan, risque 4).
        max_refused = self._progress.max_refused if self._progress is not None else 0
        self._cleanup(context)
        scene.bourrasque.baked_whitewater_max_refused = max_refused

        from . import display

        display.refresh_whitewater(scene)

    def _cleanup(self, context):
        scene = self._scene
        props = scene.bourrasque

        # Meme discipline que BQ_OT_bake_mesh._cleanup : arrete et rejoint
        # le thread AVANT de toucher a reader/ww/writer (voir sa docstring
        # pour le detail du raisonnement).
        if self._thread is not None:
            if self._cancel_event is not None:
                self._cancel_event.set()
            if self._thread.ident is not None:
                self._thread.join(timeout=10.0)
            self._thread = None
        self._progress = None
        self._cancel_event = None

        if self._writer is not None:
            self._writer.close()
            self._writer = None

        if self._ww is not None:
            self._ww.close()
            self._ww = None

        if self._reader is not None:
            self._reader.close()
            self._reader = None

        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None

        props.is_baking_whitewater = False
        BQ_OT_bake_whitewater.cancel_requested = False
        BQ_OT_bake_whitewater._active_instance = None
        _tag_redraw(context)


# ---------------------------------------------------------------------------
# BQ_OT_bake_all — enchaine bq.bake puis bq.bake_mesh, sans dupliquer
# ---------------------------------------------------------------------------
#
# N'implemente AUCUNE logique de simulation, de maillage ou de whitewater
# propre : ce n'est qu'un chef d'orchestre modal qui invoque `bq.bake` puis,
# une fois qu'il a reellement termine (toutes les frames demandees, pas une
# annulation ou une erreur), `bq.bake_mesh`, puis `bq.bake_whitewater` —
# chacun restant un operateur modal complet, avec son propre timer et sa
# propre gestion d'annulation (ESC atteint directement l'operateur modal le
# plus recemment enregistre, donc la passe en cours, sans code
# supplementaire ici). Voir docs/plan-milestone-7.md, D8, et
# docs/plan-milestone-8.md, D8 (le whitewater est independant du mesh :
# `bq.bake_whitewater` seul ne lit que le `.bqd`, mais `bq.bake_all` enchaine
# quand meme les trois passes dans l'ordre).


class BQ_OT_bake_all(bpy.types.Operator):
    """Lance le bake des particules, puis, une fois terminé, celui du
    maillage, puis celui du whitewater — sans dupliquer la logique des
    trois passes (voir docstring de section)."""

    bl_idname = "bq.bake_all"
    bl_label = "Tout baker"
    bl_description = (
        "Lance le bake des particules puis, à sa fin, le bake du maillage "
        "et du whitewater"
    )
    bl_options = {"REGISTER"}

    _phase = "PARTICLES"
    _timer = None
    _scene = None
    _particle_target = 0

    @classmethod
    def poll(cls, context):
        props = context.scene.bourrasque
        return (
            not props.is_baking
            and not props.is_baking_mesh
            and not props.is_baking_whitewater
        )

    def invoke(self, context, event):
        self._scene = context.scene
        scene = self._scene
        props = scene.bourrasque

        # Cible de succes de la passe particules : meme calcul que
        # `BQ_OT_bake.invoke` (voir `_frame_count` la-bas), pour distinguer
        # une passe terminee normalement d'une annulation/erreur (ou
        # `baked_frames` reste en-deca) une fois `is_baking` retombe a faux.
        self._particle_target = max(0, props.frame_end - props.frame_start + 1)

        result = bpy.ops.bq.bake("INVOKE_DEFAULT")
        if "RUNNING_MODAL" not in result:
            # bq.bake a refuse des l'invocation (validation) : deja rapporte
            # par son propre self.report, rien de plus a ajouter.
            return {"CANCELLED"}

        self._phase = "PARTICLES"
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        props = self._scene.bourrasque

        if self._phase == "PARTICLES":
            if props.is_baking:
                return {"RUNNING_MODAL"}
            # bq.bake vient de rendre la main (fin normale, annulation ou
            # erreur, toutes convergees par BQ_OT_bake._cleanup).
            if props.baked_frames < self._particle_target:
                self._cleanup(context)
                self.report(
                    {"WARNING"},
                    "Bake de particules interrompu : le bake de maillage "
                    "n'a pas été lancé.",
                )
                return {"CANCELLED"}

            result = bpy.ops.bq.bake_mesh("INVOKE_DEFAULT")
            if "RUNNING_MODAL" not in result:
                self._cleanup(context)
                self.report(
                    {"WARNING"},
                    "Bake de particules terminé, mais le bake de maillage "
                    "n'a pas pu démarrer (voir le message précédent).",
                )
                return {"CANCELLED"}

            self._phase = "MESH"
            return {"RUNNING_MODAL"}

        if self._phase == "MESH":
            if props.is_baking_mesh:
                return {"RUNNING_MODAL"}
            # bq.bake_mesh vient de rendre la main (fin normale, annulation
            # ou erreur, toutes convergees par BQ_OT_bake_mesh._cleanup) :
            # lance la troisieme passe quel que soit l'issue du maillage —
            # le whitewater est INDEPENDANT du mesh (docs/plan-milestone-8.md,
            # D8) et ne lit que le `.bqd`, deja disponible a ce stade.

            result = bpy.ops.bq.bake_whitewater("INVOKE_DEFAULT")
            if "RUNNING_MODAL" not in result:
                self._cleanup(context)
                self.report(
                    {"WARNING"},
                    "Bake de maillage terminé, mais le bake de whitewater "
                    "n'a pas pu démarrer (voir le message précédent).",
                )
                return {"CANCELLED"}

            self._phase = "WHITEWATER"
            return {"RUNNING_MODAL"}

        # phase WHITEWATER
        if props.is_baking_whitewater:
            return {"RUNNING_MODAL"}

        self._cleanup(context)
        return {"FINISHED"}

    def _cleanup(self, context):
        wm = context.window_manager
        if self._timer is not None:
            wm.event_timer_remove(self._timer)
            self._timer = None


# ---------------------------------------------------------------------------
# Enregistrement
# ---------------------------------------------------------------------------

classes = (
    BQ_OT_add_domain,
    BQ_OT_add_emitter,
    BQ_OT_add_collider,
    BQ_OT_remove_element,
    BQ_OT_material_add,
    BQ_OT_material_remove,
    BQ_OT_material_duplicate,
    BQ_OT_migrate_materials,
    BQ_OT_bake,
    BQ_OT_cancel_bake,
    BQ_OT_clear_rigid_keys,
    BQ_OT_free_cache,
    BQ_OT_free_mesh_cache,
    BQ_OT_bake_mesh,
    BQ_OT_free_whitewater_cache,
    BQ_OT_bake_whitewater,
    BQ_OT_bake_all,
    BQ_OT_setup_whitewater_display,
    BQ_OT_setup_whitewater_display_volume,
    BQ_OT_setup_fluid_display,
)


# Filet de securite : un crash pendant un bake precedent (ou un .blend
# sauvegarde alors qu'un bake etait en cours) peut laisser `is_baking = True`
# (ou `is_baking_mesh = True`) fige sur une scene, sans instance modale pour
# la reinitialiser (BQ_OT_bake._active_instance / BQ_OT_bake_mesh._active_
# instance ne survivent pas a un redemarrage de Blender). Sans ce reset, le
# bouton Baker (ou Baker le maillage) resterait grise indefiniment, sans
# recours possible depuis l'UI.
@bpy.app.handlers.persistent
def _bq_reset_baking_flags(*_args):
    for scene in bpy.data.scenes:
        try:
            scene.bourrasque.is_baking = False
            scene.bourrasque.is_baking_mesh = False
            scene.bourrasque.is_baking_whitewater = False
        except AttributeError:
            pass  # props.py pas encore enregistre


def _deferred_reset():
    """Execute le reset une fois les restrictions sur bpy.data levees."""
    _bq_reset_baking_flags()
    return None  # None => timer ponctuel, non replanifie


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # bpy.data est un `_RestrictData` pendant register() : y acceder leve
    # AttributeError. Le reset est donc differe par un timer ponctuel, qui
    # s'execute une fois l'enregistrement termine.
    bpy.app.timers.register(_deferred_reset, first_interval=0.0)

    # Et rejoue a chaque ouverture de .blend, pour le cas d'un fichier
    # sauvegarde alors qu'un bake etait en cours. Filtrage par nom : un
    # rechargement de module change l'identite de la fonction.
    names = {h.__name__ for h in bpy.app.handlers.load_post}
    if _bq_reset_baking_flags.__name__ not in names:
        bpy.app.handlers.load_post.append(_bq_reset_baking_flags)


def unregister():
    for h in list(bpy.app.handlers.load_post):
        if h.__name__ == _bq_reset_baking_flags.__name__:
            bpy.app.handlers.load_post.remove(h)

    if bpy.app.timers.is_registered(_deferred_reset):
        bpy.app.timers.unregister(_deferred_reset)

    # Filet de securite : si l'extension est desactivee pendant un bake,
    # nettoie l'instance modale active (writer, sim GPU, timer) plutot que
    # de laisser une fuite ou un timer pendre sur une fenetre dont
    # l'operateur va disparaitre.
    active = BQ_OT_bake._active_instance
    if active is not None:
        try:
            active._cleanup(bpy.context)
        except Exception:
            pass
        BQ_OT_bake._active_instance = None

    active_mesh = BQ_OT_bake_mesh._active_instance
    if active_mesh is not None:
        try:
            active_mesh._cleanup(bpy.context)
        except Exception:
            pass
        BQ_OT_bake_mesh._active_instance = None

    active_whitewater = BQ_OT_bake_whitewater._active_instance
    if active_whitewater is not None:
        try:
            active_whitewater._cleanup(bpy.context)
        except Exception:
            pass
        BQ_OT_bake_whitewater._active_instance = None

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
