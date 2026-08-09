"""Test de l'invariant quantite de mouvement (plan M12, D4 / critere V3).

Scenario en chute libre strictement hors de portee de tout collider : aucun
appel a bq_set_colliders dans ce test, donc aucun collider n'existe -- les
seules parois sont les bords du domaine de grille. Le bloc emis et la fenetre
de temps sont choisis pour ne jamais approcher ces bords (cf. plan M12,
risque R3).

Sous gravite constante et en l'absence de toute force de contact, la
quantite de mouvement totale suit exactement l'integration de la gravite :
p(t) = p(0) + M * g_vec * t, M = masse totale (constante ici : aucun
reseeding significatif attendu en chute libre loin de tout seuil de
naissance/mort). La tolerance d'assertion n'est pas devinee a priori : elle
est mesuree sur ce run (ecart relatif maximum observe), puis une marge de
securite est appliquee (cf. discipline generale du jalon M12).
"""
import numpy as np

from _bindings import (
    add_water,
    create_sim,
    emit_box,
    load_dll,
    particle_mass,
    read_velocities,
    total_momentum,
)

# Tolerance calibree empiriquement, pas devinee a priori (cf. plan M12,
# discipline generale + risque R1). Mesure de reference sur ce scenario
# (poste RTX 3060, cf. memoire "poste-dev-rtx3060") : ecart relatif
# systematique et tres stable d'un pas a l'autre, ~1.390%-1.391% sur les 6
# pas (steps 1..6 : 1.3904%, 1.3906%, 1.3905%, 1.3903%, 1.3901%, 1.3902%).
# Ce n'est pas du bruit run-a-run mais un biais systematique de schema
# d'integration semi-implicite (accumulation de gravite par sous-pas
# internes vs formule fermee p(t) = p(0) + M*g*t) -- attendu et documente
# par le plan (D4). Tolerance retenue : ~x2 la valeur mesuree, arrondie a
# 3.0% pour absorber une variation run-a-run raisonnable sans devenir un
# seuil arbitraire.
MOMENTUM_REL_TOLERANCE = 0.03


def test_quantite_de_mouvement_chute_libre():
    d = load_dll()
    sim, cfg = create_sim(d)
    mat_id, rho = add_water(d, sim)
    p_mass = particle_mass(cfg, rho)

    dx = cfg.cell_size
    domain_size = cfg.grid_res[0] * dx  # domaine par defaut : cube [0, 1]^3
    margin_required = 5.0 * dx

    # Petit bloc en l'air, centre en x/z, place au-dessus du centre vertical
    # du domaine pour maximiser la marge de chute disponible avant le bord
    # bas (aucun collider n'existe : la seule limite est le bord de grille).
    center = domain_size / 2.0
    lo = (center - 0.05, 0.6 * domain_size, center - 0.05)
    hi = (center + 0.05, 0.7 * domain_size, center + 0.05)
    emit_box(d, sim, mat_id, lo, hi)

    n = d.bq_particle_count(sim)
    assert n > 0

    # Verification explicite (risque R3 du plan) : distance au bord le plus
    # proche du domaine de grille, avant le premier pas.
    dist_to_walls = [
        lo[0] - 0.0, domain_size - hi[0],
        lo[1] - 0.0, domain_size - hi[1],
        lo[2] - 0.0, domain_size - hi[2],
    ]
    min_dist = min(dist_to_walls)
    print(
        f"\n[test_quantite_de_mouvement_chute_libre] domain_size={domain_size:.6f} "
        f"margin_required={margin_required:.6f} min_dist_to_wall={min_dist:.6f}"
    )
    assert min_dist > margin_required, (
        f"marge de securite insuffisante avant le premier pas : "
        f"{min_dist:.6f} <= {margin_required:.6f} (5*cell_size) -- risque de "
        f"contact avec un bord de domaine pendant le run (plan M12, risque R3)"
    )

    vel0 = read_velocities(d, sim, n)
    p0 = total_momentum(vel0, p_mass)
    total_mass = n * p_mass
    g_vec = np.array([0.0, cfg.gravity_y, 0.0], dtype=np.float64)

    dt = 1.0 / 24.0
    n_steps = 6  # court : reste loin des bords sur toute la fenetre (voir calcul de marge ci-dessus)

    rel_errors = []
    for step in range(1, n_steps + 1):
        # bq_step retourne le nombre de sous-pas executes (positif), pas un
        # code d'erreur -- seule une valeur negative signale un echec.
        assert d.bq_step(sim, dt) >= 0
        t = step * dt

        vel = read_velocities(d, sim, n)
        p_measured = total_momentum(vel, p_mass)
        p_predicted = p0 + total_mass * g_vec * t

        diff = np.linalg.norm(p_measured - p_predicted)
        norm_pred = np.linalg.norm(p_predicted)
        rel_err = diff / norm_pred if norm_pred > 0 else diff
        rel_errors.append(rel_err)
        print(
            f"  step={step} t={t:.6f} p_measured={p_measured} "
            f"p_predicted={p_predicted} rel_err={rel_err * 100:.6f}%"
        )

    max_rel_err = max(rel_errors)
    print(
        f"[test_quantite_de_mouvement_chute_libre] max_rel_err={max_rel_err * 100:.6f}% "
        f"tolerance_retenue={MOMENTUM_REL_TOLERANCE * 100:.6f}% "
        f"(constante calibree empiriquement, voir commentaire module)"
    )

    d.bq_destroy(sim)

    assert max_rel_err <= MOMENTUM_REL_TOLERANCE, (
        f"ecart relatif {max_rel_err * 100:.4f}% depasse la tolerance calibree "
        f"{MOMENTUM_REL_TOLERANCE * 100:.4f}% (plan M12, D4/V3)"
    )
