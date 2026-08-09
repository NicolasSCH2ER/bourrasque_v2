"""D7 / V6 -- cohérence dJ/dt vs J*div(v) (diagnostic, PAS un gate).

Contexte (cf. docs/plan-milestone-12.md, constat 1 et D7) : ce solveur est
MLS-MPM + EOS de Tait, volontairement faiblement compressible. Il n'a pas de
projection de pression qui impose div(v) = 0 comme un solveur incompressible
(FLIP). La compression locale est exactement ce que `J` encode, et l'EOS la
corrige progressivement -- il n'existe donc pas de "bonne" valeur attendue a
priori pour l'écart entre les deux façons suivantes de représenter le
changement de volume :

  1. `dJ/dt` par différence finie : deux lectures de `bq_read_J` à un pas de
     temps d'écart, par particule.
  2. Une estimation de `J * div(v)` via un opérateur de divergence SPH
     standard sur le voisinage de chaque particule.

Ce test mesure et rapporte la corrélation et l'écart typique entre ces deux
séries. Il ne contient AUCUNE assertion de seuil sur ces valeurs -- la seule
assertion est une sanité minimale (corrélation finie, pas NaN), qui détecte
une erreur de configuration du test (voisinages vides, séries constantes),
pas un résultat physique "mauvais".

Choix méthodologiques documentés ici (demandés explicitement par la spec) :

- Noyau SPH : cubique spline standard (Monaghan 1992), support compact
  2h, normalisation 3D sigma = 1/(pi h^3). Choix standard, sans
  particularité -- la précision de l'estimateur n'est pas critique ici
  puisque c'est un diagnostic, pas un gate (cf. plan, risque 2).
- Rayon de recherche de voisins : 2.5 * spacing (spacing =
  cfg.cell_size / cfg.ppc_axis, l'espacement de la grille de semis de
  particules). On choisit h = rayon_recherche / 2, de sorte que le
  support du noyau (2h) coïncide exactement avec le rayon de requête
  cKDTree -- aucun voisin à contribution non nulle n'est manqué, et on
  n'interroge pas plus loin que nécessaire.
- Correspondance de particule entre les deux lectures de J : l'indexation
  n'est PAS stable d'un bq_step à l'autre dès que le reseeding (actif à
  chaque frame, cf. reseed()/k_reseed_compact_survivors dans mlsmpm.cu)
  déclenche une mort quelque part dans le nuage -- les survivants sont
  compactés (tassés vers le bas), donc l'indice i à t0 peut correspondre à
  un indice différent à t1 même pour une particule qui a simplement
  survécu sans interagir. On ne suppose donc jamais que l'indice i à t0
  est le même objet physique que l'indice i à t1 : on retrouve la
  particule par correspondance spatiale (position prédite x0 + v0*dt vs
  positions réelles à t1, plus proche voisin, avec tolérance). Les
  échantillons sans correspondance proche (mort probable pendant ce pas)
  sont exclus et comptés.
"""
import ctypes

import numpy as np
import pytest
from scipy.spatial import cKDTree

from _bindings import (
    add_water,
    create_sim,
    emit_box,
    load_dll,
    read_J,
    read_positions,
    read_velocities,
)

# ---------------------------------------------------------------------------
# Scénario : bassin + bloc lâché (éclaboussure standard de cette session),
# grid_res=128, ppc_axis=3, matériau WATER standard. Domaine cubique 1 m^3
# (cell_size = 1/128) pour des coordonnées de scène lisibles.
# ---------------------------------------------------------------------------
GRID_RES = 128
CELL_SIZE = 1.0 / 128.0
PPC_AXIS = 3
DT = 1.0 / 24.0
N_STEPS_TO_ACTIVITY = 18  # dans la fenêtre 15-20 demandée : bloc déjà tombé
                          # et en pleine éclaboussure (impact attendu vers
                          # frame ~6, cf. calcul de chute libre ci-dessous).

SAMPLE_SIZE = 3000
RNG_SEED = 0
SEARCH_RADIUS_FACTOR = 2.5  # "quelques spacing" -- cf. docstring module.


def _basin_and_block(spacing):
    # Bassin large et peu profond, centré au fond du domaine.
    basin_lo = (0.30, 0.05, 0.30)
    basin_hi = (0.70, 0.065, 0.70)
    # Bloc lâché depuis au-dessus du bassin (vitesse initiale nulle --
    # "lâché", pas projeté), assez haut pour un impact franc avant la
    # frame ~15-20 mais pas au repos avant (cf. N_STEPS_TO_ACTIVITY).
    block_lo = (0.45, 0.40, 0.45)
    block_hi = (0.55, 0.50, 0.55)
    return basin_lo, basin_hi, block_lo, block_hi


# ---------------------------------------------------------------------------
# Noyau SPH cubique spline (Monaghan 1992), 3D, support compact 2h.
# ---------------------------------------------------------------------------
def _cubic_spline_dWdr(r, h):
    """dW/dr pour le noyau cubique spline standard, r et h scalaires ou
    tableaux numpy alignés. Nul au-delà du support q = r/h >= 2."""
    sigma3 = 1.0 / (np.pi * h ** 3)
    q = r / h
    dWdq = np.zeros_like(q)
    m1 = (q >= 0) & (q < 1)
    m2 = (q >= 1) & (q < 2)
    dWdq[m1] = sigma3 * (-3.0 * q[m1] + 2.25 * q[m1] ** 2)
    dWdq[m2] = sigma3 * (-0.75 * (2.0 - q[m2]) ** 2)
    return dWdq / h


def _sph_divergence(i_local, x_i, v_i, tree, pos, vel, V_current, h, search_radius):
    """div(v)_i ~= sum_j V_j * (v_j - v_i) . grad_W(x_i - x_j, h), formule de
    la spec (D7) -- pas de facteur 1/rho_i devant la somme, volume courant
    V_j = p_vol * J_j déjà calculé côté appelant."""
    neighbor_idx = tree.query_ball_point(x_i, r=search_radius)
    if i_local in neighbor_idx:
        neighbor_idx = [j for j in neighbor_idx if j != i_local]
    if not neighbor_idx:
        return 0.0
    neighbor_idx = np.asarray(neighbor_idx, dtype=np.int64)
    r_vec = x_i - pos[neighbor_idx]          # (k, 3), x_i - x_j
    r = np.linalg.norm(r_vec, axis=1)
    valid = r > 1e-12
    if not np.any(valid):
        return 0.0
    neighbor_idx = neighbor_idx[valid]
    r_vec = r_vec[valid]
    r = r[valid]
    dWdr = _cubic_spline_dWdr(r, h)
    grad_w = (dWdr / r)[:, None] * r_vec       # (k, 3)
    dv = vel[neighbor_idx] - v_i                # (k, 3)
    contrib = V_current[neighbor_idx] * np.sum(dv * grad_w, axis=1)
    return float(np.sum(contrib))


def test_divergence_J_consistency_diagnostic():
    d = load_dll()
    sim, cfg = create_sim(d, grid_res=GRID_RES, cell_size=CELL_SIZE, ppc_axis=PPC_AXIS)
    try:
        mat_id, rho = add_water(d, sim)
        spacing = cfg.cell_size / cfg.ppc_axis
        p_vol = spacing ** 3

        basin_lo, basin_hi, block_lo, block_hi = _basin_and_block(spacing)
        n_basin = emit_box(d, sim, mat_id, basin_lo, basin_hi)
        n_block = emit_box(d, sim, mat_id, block_lo, block_hi)
        print(f"\n[scene] bassin={n_basin} particules, bloc={n_block} particules, "
              f"spacing={spacing:.6f}")

        # bq_step retourne le nombre de sous-pas effectues (>= 0) en cas de
        # succes, negatif en cas d'erreur (cf. core/src/mlsmpm.cu bq_step) --
        # PAS un code d'erreur 0/non-0.
        for step in range(N_STEPS_TO_ACTIVITY):
            rc = d.bq_step(sim, ctypes.c_float(DT))
            assert rc >= 0, f"bq_step a echoue au pas {step}: {d.bq_last_error().decode()}"

        n0 = d.bq_particle_count(sim)
        pos0 = read_positions(d, sim, n0)
        vel0 = read_velocities(d, sim, n0)
        J0 = read_J(d, sim, n0)

        rc = d.bq_step(sim, ctypes.c_float(DT))
        assert rc >= 0, f"bq_step (pas de mesure) a echoue: {d.bq_last_error().decode()}"

        n1 = d.bq_particle_count(sim)
        pos1 = read_positions(d, sim, n1)
        J1 = read_J(d, sim, n1)

        print(f"[particules] n0={n0} n1={n1} "
              f"(delta={n1 - n0:+d}, reseeding actif -- indices non supposes stables)")

        # ------------------------------------------------------------
        # Sous-échantillon pour le calcul (recherche de voisins coûteuse
        # en Python/scipy sur des centaines de milliers de particules).
        # ------------------------------------------------------------
        rng = np.random.default_rng(RNG_SEED)
        sample_size = min(SAMPLE_SIZE, n0)
        sample_idx = rng.choice(n0, size=sample_size, replace=False)

        # ------------------------------------------------------------
        # Correspondance spatiale t0 -> t1 (indices non fiables, cf.
        # docstring module) : position prédite par advection simple vs
        # plus proche voisin réel à t1. Tolerance calibree empiriquement
        # sur ce scenario (pas devinee a
        # priori, meme discipline que le reste de cette session, cf.
        # politique "verification par bruit run-a-run") : avec la
        # prediction balistique x0 + v0*dt, la distance au plus proche
        # voisin reel a t1 vaut ~0.6*spacing en mediane et ~1*spacing au
        # 90e percentile meme en pleine activite (impact/eclaboussure) --
        # 2*spacing capture ~98.5% des echantillons sur ce scenario tout
        # en restant strictement sous le rayon de recherche SPH
        # (2.5*spacing), donc un match reste bien une correspondance de
        # particule, pas un voisin quelconque du meme voisinage.
        match_tol = 2.0 * spacing
        tree1 = cKDTree(pos1)
        predicted = pos0[sample_idx] + vel0[sample_idx] * DT
        match_dist, match_idx1 = tree1.query(predicted, k=1)
        matched = match_dist < match_tol
        n_unmatched = int(np.sum(~matched))
        print(f"[correspondance t0->t1] {n_unmatched}/{sample_size} echantillons sans "
              f"correspondance proche (< {match_tol:.6f} m) -- exclus (mort probable "
              f"pendant le pas de mesure).")

        used_idx0 = sample_idx[matched]
        used_idx1 = match_idx1[matched]

        dJdt = (J1[used_idx1] - J0[used_idx0]) / DT

        # ------------------------------------------------------------
        # Estimation SPH de div(v) au temps t0, sur le voisinage complet
        # (arbre construit sur TOUTES les particules à t0, pas seulement
        # l'échantillon, pour ne manquer aucun voisin réel).
        # ------------------------------------------------------------
        tree0 = cKDTree(pos0)
        search_radius = SEARCH_RADIUS_FACTOR * spacing
        h = search_radius / 2.0
        V_current = p_vol * J0.astype(np.float64)

        div_v = np.empty(used_idx0.shape[0], dtype=np.float64)
        for k, i_local in enumerate(used_idx0):
            div_v[k] = _sph_divergence(
                int(i_local), pos0[i_local], vel0[i_local],
                tree0, pos0, vel0, V_current, h, search_radius)

        J_div_v = J0[used_idx0].astype(np.float64) * div_v

        # ------------------------------------------------------------
        # Rapport (diagnostic informatif, PAS un critere de reussite).
        # ------------------------------------------------------------
        n_used = dJdt.shape[0]
        abs_diff = np.abs(dJdt - J_div_v)
        median_abs_diff = float(np.median(abs_diff))
        mean_abs_diff = float(np.mean(abs_diff))
        if np.std(dJdt) > 0 and np.std(J_div_v) > 0:
            corr = float(np.corrcoef(dJdt, J_div_v)[0, 1])
        else:
            corr = float("nan")

        print(f"\n=== D7/V6 -- cohérence dJ/dt vs J*div(v) (DIAGNOSTIC, pas un gate) ===")
        print(f"echantillons utilises      : {n_used} / {sample_size}")
        print(f"noyau SPH                  : cubique spline (Monaghan), h={h:.6f}, "
              f"rayon recherche={search_radius:.6f} (={SEARCH_RADIUS_FACTOR}*spacing)")
        print(f"dJ/dt        : mean={np.mean(dJdt):.6g} std={np.std(dJdt):.6g} "
              f"min={np.min(dJdt):.6g} max={np.max(dJdt):.6g}")
        print(f"J*div(v)     : mean={np.mean(J_div_v):.6g} std={np.std(J_div_v):.6g} "
              f"min={np.min(J_div_v):.6g} max={np.max(J_div_v):.6g}")
        print(f"correlation (numpy.corrcoef)     : {corr:.6f}")
        print(f"ecart absolu median |dJ/dt - J*div(v)| : {median_abs_diff:.6g}")
        print(f"ecart absolu moyen  |dJ/dt - J*div(v)| : {mean_abs_diff:.6g}")
        print(
            "Lecture : ceci est une mesure de coherence interne entre deux "
            "representations du changement de volume, PAS un critere "
            "pass/fail. Ce solveur (MLS-MPM + EOS de Tait) est volontairement "
            "faiblement compressible -- il n'existe pas de 'bonne' valeur "
            "attendue a priori pour la correlation ou l'ecart. cf. plan "
            "milestone 12, D7 et constat 1."
        )

        # Seule assertion : sanite minimale, pas une mesure physique. Une
        # correlation NaN signalerait une erreur de configuration du test
        # (voisinages tous vides ou series constantes), pas un resultat
        # physique "mauvais" -- cf. spec.
        assert np.isfinite(corr), (
            "correlation non finie (NaN/inf) -- signe d'un probleme de "
            "configuration du test (voisinages SPH vides ou series "
            "constantes), pas d'un resultat physique.")
    finally:
        d.bq_destroy(sim)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
