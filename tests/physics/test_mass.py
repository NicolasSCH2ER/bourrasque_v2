"""Tests de l'invariant masse (plan M12, D3 / criteres V1, V2).

masse_totale(materiau) = bq_particle_count(materiau) * p_mass(materiau) -- la
masse par particule est un scalaire constant du materiau (rho * p_vol), jamais
un champ par particule (cf. plan M12, constat 3). La seule facon de faire
varier la masse totale d'un materiau est de faire varier son nombre de
particules, ce qui arrive uniquement via le reseeding (naissances/morts,
M10). Les deux tests ci-dessous couvrent les deux regimes : reseeding
quasi-inerte (nappe au repos) et reseeding actif (eclaboussure).
"""
from _bindings import (
    add_water,
    create_sim,
    emit_box,
    load_dll,
    particle_mass,
)


def test_masse_stable_au_repos():
    """V1 -- nappe au repos, config par defaut, reseeding quasi-inerte
    (aucun seuil de naissance/mort ne se declenche sur une surface plane
    stable). Ecart de masse attendu < 1%, aligne sur la mesure M10 deja
    faite sur ce meme scenario (+0.4% observe sur 48 pas)."""
    d = load_dll()
    sim, cfg = create_sim(d)
    mat_id, rho = add_water(d, sim)
    p_mass = particle_mass(cfg, rho)

    dx = cfg.cell_size
    lo_v = 3 * dx
    hi_v = cfg.grid_res[0] * dx - 3 * dx
    lo = (lo_v, lo_v, lo_v)
    hi = (hi_v, lo_v + 0.25, hi_v)
    emit_box(d, sim, mat_id, lo, hi)

    n_before = d.bq_particle_count(sim)
    mass_before = n_before * p_mass

    for _ in range(48):
        # bq_step retourne le nombre de sous-pas executes (positif), pas un
        # code d'erreur -- seule une valeur negative signale un echec.
        assert d.bq_step(sim, 1.0 / 24.0) >= 0

    n_after = d.bq_particle_count(sim)
    mass_after = n_after * p_mass

    rel_err = abs(mass_after - mass_before) / mass_before
    print(
        f"\n[test_masse_stable_au_repos] n_before={n_before} n_after={n_after} "
        f"mass_before={mass_before:.6f} mass_after={mass_after:.6f} "
        f"rel_err={rel_err * 100:.4f}%"
    )

    d.bq_destroy(sim)

    assert rel_err < 0.01, (
        f"ecart de masse {rel_err * 100:.4f}% >= 1% sur une scene au repos "
        f"(reseeding cense etre quasi-inerte, cf. plan M12 D3)"
    )


def test_masse_sous_reseeding_rapportee():
    """V2 -- scene d'eclaboussure (bassin + bloc lache en chute dedans),
    reseeding actif. Le plan M12 (D3, constat 3) est explicite : la masse
    totale N'EST PAS conservee par construction des que le reseeding cree ou
    detruit des particules sans fusion (fusion ecartee en M10). Ce test
    RAPPORTE l'ecart avec son signe -- il n'y a volontairement AUCUNE
    assertion de seuil ici : ce n'est pas un oubli, c'est le critere V2 du
    plan ("pas de seuil pass/fail arbitraire -- objectif de ce test est
    d'exposer le comportement reel")."""
    d = load_dll()
    # max_particles releve : le bassin plein-fond a res=128/ppc_axis=3
    # depasse a lui seul le plafond par defaut (2_000_000, ~2.7M mesure).
    sim, cfg = create_sim(d, grid_res=128, ppc_axis=3, max_particles=4_000_000)
    mat_id, rho = add_water(d, sim)
    p_mass = particle_mass(cfg, rho)

    dx = cfg.cell_size
    lo_v = 3 * dx
    hi_v = cfg.grid_res[0] * dx - 3 * dx
    center = (lo_v + hi_v) / 2.0

    # Bassin : couvre le fond du domaine, hauteur 0.10 en y.
    pool_lo = (lo_v, lo_v, lo_v)
    pool_hi = (hi_v, lo_v + 0.10, hi_v)
    emit_box(d, sim, mat_id, pool_lo, pool_hi)

    # Bloc 0.20x0.20x0.20 centre en x/z, au-dessus du bassin (bassin plafonne
    # a lo_v + 0.10 ~= 0.147 ; le bloc part de y=1.0, tres au-dessus, pour
    # laisser le temps a un impact franc puis a l'eclaboussure de se
    # developper sur les pas restants avant le pas 22).
    block_lo = (center - 0.10, 1.0, center - 0.10)
    block_hi = (center + 0.10, 1.2, center + 0.10)
    emit_box(d, sim, mat_id, block_lo, block_hi, vel=(0.0, -4.5, 0.0))

    n_before = d.bq_particle_count(sim)
    mass_before = n_before * p_mass

    for _ in range(22):
        assert d.bq_step(sim, 1.0 / 24.0) >= 0

    n_after = d.bq_particle_count(sim)
    mass_after = n_after * p_mass

    delta = mass_after - mass_before
    rel_delta = delta / mass_before
    print(
        f"\n[test_masse_sous_reseeding_rapportee] n_before={n_before} "
        f"n_after={n_after} (delta_n={n_after - n_before:+d}) "
        f"mass_before={mass_before:.6f} mass_after={mass_after:.6f} "
        f"delta={delta:+.6f} rel_delta={rel_delta * 100:+.4f}% "
        f"({'naissances nettes' if delta > 0 else 'morts nettes' if delta < 0 else 'stable'})"
    )

    d.bq_destroy(sim)

    # Pas d'assertion de seuil ici (voir docstring) : la masse n'est pas
    # conservee sous reseeding actif par construction, ce test mesure et
    # rapporte l'ecart, il ne le fait jamais echouer.
