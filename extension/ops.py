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
"""

import math
import os

import bpy
import numpy as np

from . import cache, lib
from .props import (
    domain_resolution,
    domain_transform,
    domain_usable_bounds,
    emitter_bounds_solver,
    emitter_overflow,
    estimate_particle_count,
    world_to_solver_dir,
)

__all__ = ("classes", "register", "unregister")


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
# BQ_OT_cancel_bake
# ---------------------------------------------------------------------------


class BQ_OT_cancel_bake(bpy.types.Operator):
    """Positionne le drapeau d'annulation observe par le modal de bake."""

    bl_idname = "bq.cancel_bake"
    bl_label = "Annuler"
    bl_description = "Annule le bake en cours"

    @classmethod
    def poll(cls, context):
        return context.scene.bourrasque.is_baking

    def execute(self, context):
        BQ_OT_bake.cancel_requested = True
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

        removed_any = False
        for path in (bqd_path, mat_path):
            if os.path.isfile(path):
                os.remove(path)
                removed_any = True

        scene.bourrasque.baked_frames = 0

        # Vide la geometrie affichee : sans ca, la derniere frame bakee
        # resterait affichee dans le viewport alors que le cache qui la
        # sous-tend vient d'etre supprime.
        from . import display

        display.clear_particle_object()

        if removed_any:
            self.report({"INFO"}, "Cache vidé.")
        else:
            self.report({"INFO"}, "Aucun cache à vider.")
        return {"FINISHED"}


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
    # Scene visee par ce bake, memorisee dans invoke() : voir sa docstring.
    # Ne jamais lire context.scene apres invoke() dans cet operateur.
    _scene = None

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

        # Deduplique les materiaux : seuls ceux effectivement portes par un
        # emetteur sont enregistres, sinon le pas de temps (calcule sur le
        # max de TOUS les materiaux enregistres, cf. mlsmpm.cu:319) serait
        # penalise par des materiaux inutilises mais plus raides.
        material_keys = []  # liste de tuples -> index dans material_specs
        material_specs = []  # liste de dict pour lib.Sim.add_material
        emitter_specs = []  # (obj, material_index)

        for obj in emitters:
            op = obj.bourrasque
            if op.model == "ELASTIC":
                model = lib.BQ_MODEL_ELASTIC
                key = ("ELASTIC", op.rho, op.young, op.poisson)
                kwargs = dict(model=model, rho=op.rho, E=op.young, nu=op.poisson)
            else:
                model = lib.BQ_MODEL_WATER
                key = ("WATER", op.rho, op.bulk, op.gamma)
                kwargs = dict(model=model, rho=op.rho, bulk=op.bulk, gamma=op.gamma)

            if key in material_keys:
                mat_index = material_keys.index(key)
            else:
                mat_index = len(material_specs)
                material_keys.append(key)
                material_specs.append(kwargs)

            emitter_specs.append((obj, mat_index))

        if len(material_specs) > lib.BQ_MAX_MATERIALS:
            self.report(
                {"ERROR"},
                f"{len(material_specs)} matériaux distincts sont utilisés "
                f"par les émetteurs, mais le solveur n'en accepte que "
                f"{lib.BQ_MAX_MATERIALS} au maximum. Réutilisez les mêmes "
                "réglages matériau sur plusieurs émetteurs (plutôt que "
                "« Personnalisé » avec des valeurs légèrement différentes).",
            )
            return None

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
            self._writer = cache.CacheWriter(bqd_path, self._sim.particle_count)
            # Le sidecar .mat est ecrit a la FIN du bake (voir _cleanup) :
            # les ids materiaux des particules ajoutees en cours de route
            # par un emetteur INFLOW ne sont connus qu'une fois la
            # derniere frame ecrite (voir cache.py, doc du sidecar .mat).

            self._frame_count = props.frame_end - props.frame_start + 1
            self._frame_index = 0
            self._pos_buffer = None

            BQ_OT_bake.cancel_requested = False
            props.is_baking = True
            props.bake_progress = 0.0
            props.baked_frames = 0

            wm = context.window_manager
            self._timer = wm.event_timer_add(1e-6, window=context.window)
            if not wm.modal_handler_add(self):
                raise RuntimeError(
                    "impossible d'installer le gestionnaire modal "
                    "(wm.modal_handler_add a renvoye faux)"
                )
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

        if event.type == "ESC" or BQ_OT_bake.cancel_requested:
            self._cleanup(context)
            self.report({"INFO"}, "Bake annulé.")
            return {"CANCELLED"}

        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        try:
            self._advance_frame()
        except Exception as exc:
            self._cleanup(context)
            self.report({"ERROR"}, f"Erreur pendant le bake : {exc}")
            return {"CANCELLED"}

        self._frame_index += 1
        props.baked_frames = self._frame_index
        props.bake_progress = self._frame_index / max(1, self._frame_count)
        _tag_redraw(context)

        if self._frame_index >= self._frame_count:
            self._finish(context)
            return {"FINISHED"}

        return {"RUNNING_MODAL"}

    # -- avance d'une frame — factorise pour rester appelable directement,
    # -- sans timer ni evenement bpy (voir script de validation du jalon) -

    def _emit_inflow_sites(self):
        """Emet, pour chaque emetteur INFLOW, les sites de son nuage complet
        (`state.sites`, FIXE, voir `_InflowState`) qui ne sont pas deja
        occupes par une particule existante — ensemencement volumique avec
        test d'occupation (voir docstring de module).

        Pour chaque emetteur : filtre les particules courantes (lues UNE
        fois pour tous les emetteurs, avant le `step` de cette frame) a
        celles tombant dans la bbox du nuage de sites (filtre vectorise
        numpy, c'est ce qui rend l'operation peu couteuse), convertit ces
        positions retenues en indices de site du reseau (arrondi au site le
        plus proche), et emet un point a chaque site du nuage dont l'indice
        n'apparait pas parmi ces occupants — SOUS RESERVE du plafond de
        conservation du nombre ci-dessous. Le volume de l'emetteur reste
        ainsi sature en permanence : le fluide en sort a la vitesse `v`
        (constante, celle de l'emetteur — les sites sont au niveau de
        l'emetteur, ou la vitesse du fluide est `v`), les sites se liberent
        au rythme voulu, le debit s'auto-regule sans accumulateur.

        Plafond par CONSERVATION DU NOMBRE : le nombre de sites libres
        n'est PAS a lui seul un plafond fiable des lors que le jitter
        positionnel ou le bruit de turbulence sont actifs. Un bruit
        spatialement coherent (`_CurlNoise`) deplace des BLOCS entiers de
        particules dans la meme direction, ce qui cree simultanement des
        sites vides (le bloc s'en est eloigne — reemis, on GAGNE des
        particules) et des sites occupes par deux particules ou plus (deux
        particules arrondissent au meme site — rien n'est retire, on ne
        PERD rien). Le bilan naif est donc structurellement positif : sans
        correction, le debit derive vers le haut avec la turbulence, alors
        que les deux reglages doivent rester orthogonaux pour l'artiste.
        La methode calcule donc `deficit = max(0, n_sites - n_occupants)`
        (le nombre de particules manquantes pour saturer exactement le
        nuage) et n'emet jamais plus de `deficit` particules : si les
        sites libres sont plus nombreux que `deficit`, `deficit` d'entre
        eux sont tires AU HASARD (jamais un prefixe, qui biaiserait
        spatialement l'emission vers un coin de l'emetteur — les sites
        sont ranges dans un ordre de reseau), via un generateur numpy
        DEDIE seede par la meme convention que `_turbulence_rng`. A
        turbulence nulle, aucune paire de particules n'arrondit au meme
        site : `deficit` egale alors exactement le nombre de sites libres
        et le plafond ne mord jamais (comportement inchange).

        Un site peut tomber hors de la zone utile du domaine (pres d'une
        paroi) : les points hors `self._usable_bounds` sont filtres avant
        l'appel, comme pour un emetteur BLOCK.

        Ne fait rien si la simulation est deja saturee (`self._saturated`).
        Sur le premier depassement de `max_particles`, positionne
        `self._saturated` et rapporte l'avertissement UNE SEULE fois ; les
        emetteurs suivants (meme frame ou frames suivantes) ne tentent plus
        d'emission.

        Renvoie le nombre total de particules effectivement emises pour
        cette frame (utile aux scripts de validation).
        """
        if not self._inflow_states or self._saturated:
            return 0

        usable = self._usable_bounds
        # Lues UNE fois pour tous les emetteurs de cette frame : c'est
        # l'etat de la simulation AVANT l'emission de cette frame (les
        # sites qui viennent d'etre libere par le mouvement des particules
        # depuis la frame precedente).
        positions = self._sim.read_positions()

        total = 0
        for state in self._inflow_states:
            if self._saturated:
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

            # Plafond par CONSERVATION DU NOMBRE (voir docstring de la
            # methode) : sans lui, un bruit spatialement coherent deplace
            # des blocs entiers de particules dans la meme direction, ce
            # qui cree simultanement des sites vides (reemis, on GAGNE) et
            # des sites doublement occupes (rien n'est retire, on ne PERD
            # rien) — bilan structurellement positif, donc sur-emission.
            # `deficit` est le nombre de particules manquantes pour
            # saturer exactement le nuage ; on n'emet jamais plus.
            n_sites = sites.shape[0]
            n_occupants = occupants.shape[0]
            deficit = max(0, n_sites - n_occupants)

            free_idx = np.nonzero(free_mask)[0]
            if free_idx.shape[0] > deficit:
                # Plus de sites libres que de deficit (turbulence > 0,
                # sites doublement occupes ailleurs dans le nuage) : on en
                # tire `deficit` AU HASARD parmi les sites libres, jamais
                # un prefixe — les sites sont ranges dans un ordre de
                # reseau (balayage x,y,z), en garder les premiers
                # biaiserait spatialement l'emission vers un coin de
                # l'emetteur. Generateur DEDIE, meme convention de graine
                # que `_turbulence_rng` (seed emetteur + frame), jamais
                # l'etat global de numpy (determinisme du bake).
                select_rng = _turbulence_rng(
                    state.turbulence_seed, self._frame_index, state.emitter_index
                )
                free_idx = select_rng.choice(free_idx, size=deficit, replace=False)

            pts = sites[free_idx]
            if pts.shape[0] == 0:
                # Emetteur entierement sature (ou deficit nul) : rien a
                # emettre.
                continue

            # Turbulence appliquee AVANT le filtrage par la zone utile : une
            # particule jittee peut sortir de `usable` (voir docstring de
            # `_turbulent_emission`), le filtrage doit donc porter sur les
            # positions DEJA perturbees, pas sur les sites bruts.
            rng = _turbulence_rng(
                state.turbulence_seed, self._frame_index, state.emitter_index
            )
            # `t` : instant courant du bake, commun a toute la duree du
            # bake (pas seulement a cet emetteur) — c'est ce qui fait
            # deriver `state.noise` de facon coherente d'une frame a
            # l'autre (voir `_CurlNoise`).
            t = self._frame_index * self._frame_dt
            pts, vels = _turbulent_emission(
                pts, state.vel, state.turbulence, spacing, self._frame_dt,
                rng, state.dx, t, state.noise,
            )

            if usable is not None:
                ulo, uhi = usable
                # Tolerance qui absorbe la divergence float64 (ce filtre,
                # cote Python) / float32 (la borne recalculee cote coeur a
                # partir de `config.cell_size`, tronque a float32) sur la
                # meme valeur de dx : ~1e-8 relatif entre les deux chemins
                # de calcul. Sans marge, un point que Python juge tout
                # juste dans la zone utile peut etre rejete par le coeur
                # comme hors domaine (voir bq_last_error, `emit_particles`
                # dans mlsmpm.cu). 1e-6 * dx est trois ordres de grandeur
                # au-dessus de cet ecart et trois ordres en dessous de
                # toute grandeur physique du domaine : marge negligeable,
                # mais qui absorbe l'ecart de precision.
                tol = 1e-6 * state.dx
                mask = np.all((pts >= ulo - tol) & (pts <= uhi + tol), axis=1)
                pts = pts[mask]
                vels = vels[mask]
            if pts.shape[0] == 0:
                # Emetteur entierement hors de la zone utile (proche d'une
                # paroi) : rien a emettre.
                continue

            try:
                total += self._sim.emit_points_vel(state.mat_id, pts, vels)
            except lib.BourrasqueError as exc:
                # Le coeur remonte deux causes distinctes sur ce meme appel
                # (voir emit_particles dans core/src/mlsmpm.cu) : la
                # capacite depassee (message "capacite depassee ...") et le
                # rejet d'un point hors domaine (message "point %d hors
                # domaine ..."). Seule la premiere est une saturation reelle
                # qui justifie d'arreter l'inflow pour le reste du bake ;
                # attribuer la seconde a la premiere donnerait un
                # diagnostic faux a l'artiste.
                msg = str(exc)
                if "capacite depassee" in msg:
                    self._saturated = True
                    self.report(
                        {"WARNING"},
                        "Capacité maximale de particules atteinte : "
                        "l'émission continue (inflow) est arrêtée, la "
                        "simulation se poursuit jusqu'à la fin du bake avec "
                        "les particules déjà émises.",
                    )
                else:
                    self.report(
                        {"WARNING"},
                        f"« {state.name} » : émission refusée par le "
                        f"solveur pour cette frame ({msg}). L'inflow "
                        "continue aux frames suivantes.",
                    )
        return total

    def _advance_frame(self):
        """Avance la simulation d'UNE frame : emet les sites d'inflow
        libres, fait avancer le solveur, lit les positions et les ajoute au
        cache.

        Ne depend d'aucun etat modal (timer, evenement bpy) : appelable
        directement depuis un script de validation hors du cycle modal de
        Blender (l'operateur modal ne s'execute pas en `--background`).
        """
        self._emit_inflow_sites()
        self._sim.step(self._frame_dt)
        self._pos_buffer = self._sim.read_positions(out=self._pos_buffer)
        self._writer.append_frame(self._pos_buffer)

    # -- fin normale : ferme proprement puis rafraichit l'affichage ------

    def _finish(self, context):
        scene = self._scene
        n_particles = self._sim.particle_count
        mat_array = self._sim.read_materials()
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

        props.is_baking = False
        BQ_OT_bake.cancel_requested = False
        BQ_OT_bake._active_instance = None
        _tag_redraw(context)


# ---------------------------------------------------------------------------
# Enregistrement
# ---------------------------------------------------------------------------

classes = (
    BQ_OT_add_domain,
    BQ_OT_add_emitter,
    BQ_OT_remove_element,
    BQ_OT_bake,
    BQ_OT_cancel_bake,
    BQ_OT_free_cache,
)


# Filet de securite : un crash pendant un bake precedent (ou un .blend
# sauvegarde alors qu'un bake etait en cours) peut laisser `is_baking = True`
# fige sur une scene, sans instance modale pour la reinitialiser
# (BQ_OT_bake._active_instance ne survit pas a un redemarrage de Blender).
# Sans ce reset, le bouton Baker resterait grise indefiniment, sans recours
# possible depuis l'UI.
@bpy.app.handlers.persistent
def _bq_reset_baking_flags(*_args):
    for scene in bpy.data.scenes:
        try:
            scene.bourrasque.is_baking = False
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

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
