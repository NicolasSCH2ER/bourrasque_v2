"""Test de l'invariant volume (plan M12, D6 / critere V5).

Le seul des 5 invariants du jalon a porter un vrai risque MPM documente dans
la litterature : perte (ou gain) de volume numerique cumulee sur un run long,
via la derive de J (le determinant du gradient de deformation, integre par
particule). Volume total = somme(V0 * J_i), V0 = p_vol constant par materiau
(cf. _bindings.total_volume).

Pas de seuil d'assertion fixe a priori (plan D6, explicite : "pas de seuil
fixe a priori -- mesure et rapport, comme le reste"). Une seule assertion de
garde-fou GROSSIER est presente en fin de test (ordre de grandeur), distincte
de la mesure de derive elle-meme qui est seulement rapportee, jamais assertee
finement.
"""
import numpy as np

from _bindings import (
    add_water,
    create_sim,
    emit_box,
    load_dll,
    read_J,
    total_volume,
)

N_STEPS = 200
MEASURE_EVERY = 20
FRAME_DT = 1.0 / 24.0


def test_derive_volume_run_long():
    d = load_dll()
    # Config par defaut (aucun override) -- meme scenario que
    # test_masse_stable_au_repos de test_mass.py : nappe au repos couvrant le
    # fond du domaine, reseeding quasi-inerte a cette resolution (V1).
    sim, cfg = create_sim(d)
    mat_id, rho = add_water(d, sim)

    dx = cfg.cell_size
    lo_v = 3.0 * dx
    hi_v = cfg.grid_res[0] * dx - 3.0 * dx
    lo = (lo_v, lo_v, lo_v)
    hi = (hi_v, lo_v + 0.25, hi_v)
    n0 = emit_box(d, sim, mat_id, lo, hi)
    assert n0 > 0

    p_vol = (cfg.cell_size / cfg.ppc_axis) ** 3

    n_cur = d.bq_particle_count(sim)
    J = read_J(d, sim, n_cur)
    measurements = [(0, total_volume(J, p_vol), n_cur)]

    for step_i in range(1, N_STEPS + 1):
        substeps = d.bq_step(sim, FRAME_DT)
        assert substeps >= 0, f"bq_step a echoue: {d.bq_last_error().decode()}"
        if step_i % MEASURE_EVERY == 0:
            n_cur = d.bq_particle_count(sim)
            J = read_J(d, sim, n_cur)
            measurements.append((step_i, total_volume(J, p_vol), n_cur))

    d.bq_destroy(sim)

    steps = [m[0] for m in measurements]
    vols = np.array([m[1] for m in measurements], dtype=np.float64)
    n_particles = [m[2] for m in measurements]

    v0 = float(vols[0])
    v_final = float(vols[-1])
    v_min = float(vols.min())
    v_max = float(vols.max())
    rel_drift = (vols - v0) / v0

    print(f"\n[volume] p_vol={p_vol:.6e}  n_particules initial={n_particles[0]}")
    print("[volume] serie temporelle (pas, volume, derive relative, n_particules) :")
    for s, v, r, npart in zip(steps, vols, rel_drift, n_particles):
        print(f"    pas={s:4d}  V={v:.6e}  derive={r * 100:+.3f}%  n_particules={npart}")

    # Lecture qualitative de la FORME de la derive (monotone / oscillante /
    # stabilisee), demandee par le plan D6 en plus de l'amplitude -- comptage
    # des changements de signe de la variation d'une mesure a l'autre, purement
    # rapporte, jamais asserte.
    diffs = np.diff(vols)
    diff_signs = np.sign(diffs)
    diff_signs = diff_signs[diff_signs != 0]
    sign_changes = int(np.sum(np.diff(diff_signs) != 0)) if diff_signs.size > 1 else 0
    if diff_signs.size == 0:
        shape_desc = "volume constant (aucune variation mesuree)"
    elif sign_changes == 0:
        shape_desc = "monotone (croissante ou decroissante sans inversion)"
    elif sign_changes <= 2:
        shape_desc = "quasi-monotone avec une inflexion (ex: transitoire puis stabilisation)"
    else:
        shape_desc = "oscillante / non-monotone"
    print(f"[volume] V(min)={v_min:.6e}  V(max)={v_max:.6e}")
    print(f"[volume] changements de signe de la variation entre mesures : {sign_changes}")
    print(f"[volume] lecture qualitative de la forme : {shape_desc}")
    print(f"[volume] derive relative finale (pas {N_STEPS} vs pas 0) : {rel_drift[-1] * 100:+.3f}%")

    # Garde-fou de non-regression GROSSIER, VOLONTAIREMENT distinct de la
    # mesure de derive ci-dessus (qui n'a pas de seuil, cf. plan D6) : le
    # volume final doit rester dans un ordre de grandeur raisonnable du
    # volume initial -- ne pas s'effondrer a near-zero (fuite de volume
    # catastrophique) et ne pas exploser x10 (instabilite numerique). Bornes
    # deliberement tres larges (x0.1 / x10), choisies pour attraper une
    # regression GROSSIERE (effondrement/explosion), pas pour calibrer la
    # physique fine de la derive elle-meme.
    assert 0.1 * v0 < v_final < 10.0 * v0, (
        f"volume final hors de l'ordre de grandeur du volume initial : "
        f"V0={v0:.6e} V_final={v_final:.6e} ratio={v_final / v0:.3f}"
    )
