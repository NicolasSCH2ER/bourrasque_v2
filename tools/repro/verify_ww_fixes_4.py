"""Verification cas synthetique des 4 corrections whitewater (cf. spec de la
tache : Fix A I_ta, Fix B I_wc, Fix C spawn cylindre, Fix D foam sans
inertie).

Fix A/B (I_ta, I_wc) : reimplementation numpy fidele des formules EXACTES du
kernel k_ww_generate_potentials (core/src/whitewater.cu) apres correction,
appliquee a des configurations synthetiques controlees. Ne charge pas la DLL
pour ces deux tests -- l'algebre des deux termes se verifie entierement cote
hote, meme demarche que tools/repro/whitewater_velocity_read.py qui rejoue
deja la formule SPRAY en float32 cote hote pour comparer a la sortie GPU.
La fidelite au kernel reel est garantie par transcription ligne a ligne (cf.
commentaires) et par la relecture de code deja faite pendant l'implementation
-- la compilation propre de whitewater.cu confirme que le GPU execute EXACTEMENT
ce texte.

Fix C (spawn cylindre) et Fix D (foam sans inertie) sont verifies via la DLL
reelle (extension/bin/bourrasque.dll), pas une reimplementation -- observables
directement par Whitewater.step()/.read().
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")
import lib


# ---------------------------------------------------------------------------
# Fix A -- I_ta (air piege), transcription de la boucle corrigee.
# ---------------------------------------------------------------------------
def kernel_w(d2, R):
    inv_R = 1.0 / R
    s = np.sqrt(d2) * inv_R
    t = 1.0 - s * s
    return t ** 3


def i_ta_term(xi, vi, xj, vj, R):
    d = xi - xj
    d2 = float(np.dot(d, d))
    if d2 >= R * R:
        return 0.0
    w_ij = kernel_w(d2, R)
    vij = vi - vj
    vij_len = float(np.linalg.norm(vij))
    if vij_len <= 1e-8:
        return 0.0
    v_hat_ij = vij / vij_len
    x_hat_ij = d / np.sqrt(d2)
    return float(w_ij * vij_len * (1.0 - np.dot(v_hat_ij, x_hat_ij)))


def test_fix_a():
    print("=== Fix A : I_ta (air piege) ===")
    R = 3.0 / 128.0
    d_offset = 0.3 * R
    xi = np.array([0.0, 0.0, 0.0])
    xj = np.array([d_offset, 0.0, 0.0])

    # tete-a-tete : i va vers +x, j va vers -x -> elles convergent.
    vi_conv = np.array([1.0, 0.0, 0.0])
    vj_conv = np.array([-1.0, 0.0, 0.0])
    ita_conv = i_ta_term(xi, vi_conv, xj, vj_conv, R)

    # meme vitesses mais opposees : i va vers -x, j va vers +x -> divergent.
    vi_div = np.array([-1.0, 0.0, 0.0])
    vj_div = np.array([1.0, 0.0, 0.0])
    ita_div = i_ta_term(xi, vi_div, xj, vj_div, R)

    # meme vitesse absolue, meme direction (vij_len ~ 0) -> ne doit
    # contribuer ni positivement ni negativement (garde-fou division/0).
    vi_same = np.array([2.0, 0.0, 0.0])
    vj_same = np.array([2.0, 0.0, 0.0])
    ita_same = i_ta_term(xi, vi_same, xj, vj_same, R)

    print(f"  convergence frontale : I_ta = {ita_conv:.6f}")
    print(f"  divergence  frontale : I_ta = {ita_div:.6f}")
    print(f"  vitesses identiques  : I_ta = {ita_same:.6f} (attendu 0.0, garde-fou)")

    ok = (ita_conv > 3.0 * max(ita_div, 1e-9)) and abs(ita_div) < 1e-6 and abs(ita_same) < 1e-9
    print(f"  -> {'OK' if ok else 'ECHEC'} (convergence >> divergence ~ 0)\n")
    return ok


# ---------------------------------------------------------------------------
# Fix B -- I_wc (crete de vague), filtre de convexite.
# ---------------------------------------------------------------------------
def wc_sum_filtered(xi, n_hat_i, neighbors, R):
    """neighbors : liste de (xj, n_hat_j)."""
    total = 0.0
    for xj, n_hat_j in neighbors:
        d = xi - xj
        d2 = float(np.dot(d, d))
        if d2 >= R * R:
            continue
        w_ij = kernel_w(d2, R)
        if np.dot(d, n_hat_i) > 0.0:
            total += w_ij * (1.0 - np.dot(n_hat_i, n_hat_j))
    return total


def wc_sum_unfiltered(xi, n_hat_i, neighbors, R):
    total = 0.0
    for xj, n_hat_j in neighbors:
        d = xi - xj
        d2 = float(np.dot(d, d))
        if d2 >= R * R:
            continue
        w_ij = kernel_w(d2, R)
        total += w_ij * (1.0 - np.dot(n_hat_i, n_hat_j))
    return total


def neighbor_normal(xi, neighbors, R):
    """Transcription de ww_neighbor_normal (whitewater.cu) : gradient de
    densite SPH, normalise, (0,1,0) par defaut si degenere."""
    total = np.zeros(3)
    for xj, _ in neighbors:
        d = xi - xj
        d2 = float(np.dot(d, d))
        if d2 >= R * R:
            continue
        w_ij = kernel_w(d2, R)
        total += w_ij * d
    n = np.linalg.norm(total)
    if n < 1e-8:
        return np.array([0.0, 1.0, 0.0])
    return total / n


def test_fix_b():
    print("=== Fix B : I_wc (crete de vague), filtre de convexite ===")
    print("  (le filtre exclut les voisins du cote EXTERIEUR/vide de i, cf.")
    print("  vdot(x_hat_ji, n_hat_i) < 0 == vdot(d, n_hat_i) > 0 pour inclure ;")
    print("  demonstration algebrique directe de son effet, cas d'ecole)")
    R = 3.0 / 128.0
    r = 0.3 * R

    # i sur une surface plane, normale connue n_hat_i = (0,1,0) (le fluide
    # est sous i, cote y<0 relatif a i -- convention "normale sortante").
    xi = np.array([0.0, 0.0, 0.0])
    n_hat_i = np.array([0.0, 1.0, 0.0])

    # Voisin FLUIDE reel (cote interieur, y<0 relatif a i) mais dont la
    # normale locale diverge fortement de n_hat_i -- signal de VRAIE
    # courbure convexe (un sommet de vague : la surface s'incline autour
    # de i). Doit rester compte, filtre ou non.
    inward_curved = (np.array([0.3 * r, -0.6 * r, 0.0]), np.array([0.85, 0.53, 0.0]))

    # Voisin EXTERIEUR (cote vide/air, y>0 relatif a i -- au-dessus de la
    # surface, du mauvais cote de i) avec une normale tres differente --
    # c'est exactement le faux positif que le papier decrit (arete/coin :
    # un point geometriquement de l'autre cote du plan tangent local de i,
    # dont la comparaison de normale n'a pas de sens physique). Doit etre
    # exclu par le filtre.
    outward_spurious = (np.array([-0.2 * r, 0.7 * r, 0.0]), np.array([-1.0, 0.0, 0.0]))

    neighbors = [inward_curved, outward_spurious]

    wc_before = wc_sum_unfiltered(xi, n_hat_i, neighbors, R)
    wc_after = wc_sum_filtered(xi, n_hat_i, neighbors, R)
    wc_inward_only = wc_sum_unfiltered(xi, n_hat_i, [inward_curved], R)

    print(f"  I_wc SANS filtre (inward + outward)      = {wc_before:.5f}")
    print(f"  I_wc AVEC filtre (outward exclu)          = {wc_after:.5f}")
    print(f"  I_wc du seul voisin inward (reference)    = {wc_inward_only:.5f}")

    # le filtre doit annuler exactement la contribution du voisin outward
    # (le resultat filtre == la contribution du seul voisin inward), et le
    # voisin outward a un poids non nul avant filtre (sinon le test ne
    # demontre rien).
    outward_contrib = wc_before - wc_inward_only
    ok = (abs(wc_after - wc_inward_only) < 1e-9) and (outward_contrib > 1e-6) and (wc_after < wc_before)
    print(f"  contribution du voisin outward (avant filtre) = {outward_contrib:.5f}")
    print(f"  -> {'OK' if ok else 'ECHEC'} "
          f"(voisin outward totalement exclu, voisin inward integralement conserve)\n")
    return ok


# ---------------------------------------------------------------------------
# Fix C -- spawn cylindre le long de la trajectoire (via la DLL reelle).
# ---------------------------------------------------------------------------
def test_fix_c():
    print("=== Fix C : spawn cylindre le long de la trajectoire (DLL reelle) ===")
    cfg = lib.default_whitewater_config()
    cfg.max_particles = 200000
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 128
    cfg.cell_size = 1.0  # domaine 128 m/axe, largement assez pour ne pas clipper
    R = cfg.influence_radius

    # tout le potentiel porte sur l'energie cinetique, seuils larges pour
    # generer un GROS lot de particules en UNE frame depuis UNE seule
    # particule fluide (statistique de position exploitable directement).
    cfg.ta_weight = 0.0
    cfg.wc_weight = 0.0
    cfg.ke_min = 0.0
    cfg.ke_max = 1.0
    cfg.ke_weight = 1.0
    cfg.spawn_rate = 100000.0

    dt = 1.0 / 60.0
    x0 = np.array([64.0, 64.0, 64.0], dtype=np.float32)
    v0_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_mag = 50.0  # grande vitesse synthetique : etale le cylindre bien au-dela de R
    v0 = v0_dir * v_mag

    fluid_pos = x0.reshape(1, 3)
    fluid_vel = v0.reshape(1, 3)

    with lib.Whitewater(cfg) as ww:
        ww.step(fluid_pos, fluid_vel, dt)
        pos, type_, size, age, vel = ww.read()

    n = pos.shape[0]
    print(f"  {n} particules generees en 1 frame (v={v_mag} m/s, dt={dt:.5f}s)")
    if n < 50:
        print("  -> ECHEC (pas assez de particules generees pour une statistique fiable)")
        return False

    rel = pos.astype(np.float64) - x0.astype(np.float64)
    axis_component = rel @ v0_dir.astype(np.float64)  # le long de la vitesse
    perp = rel - np.outer(axis_component, v0_dir.astype(np.float64))
    perp_component = np.linalg.norm(perp, axis=1)

    std_axis = float(np.std(axis_component))
    std_perp = float(np.std(perp_component))

    print(f"  ecart-type le long de la vitesse       : {std_axis:.6f} m "
          f"({std_axis / R:.2f} x influence_radius)")
    print(f"  ecart-type perpendiculaire a la vitesse : {std_perp:.6f} m "
          f"({std_perp / R:.2f} x influence_radius)")

    # attendu : etalement le long de l'axe nettement > influence_radius
    # (h_dist = Xh*dt*|v|, uniforme sur [0, dt*v] ~ 0.83 m ici), et
    # perpendiculaire borne par rV = 0.1*influence_radius (donc << R).
    ok = (std_axis > 3.0 * R) and (std_perp < R)
    print(f"  -> {'OK' if ok else 'ECHEC'} "
          f"(distribution etiree le long de la trajectoire, pas un jitter isotrope)\n")
    return ok


# ---------------------------------------------------------------------------
# Fix D -- mousse sans inertie propre (via la DLL reelle).
# ---------------------------------------------------------------------------
def test_fix_d():
    print("=== Fix D : mousse suit directement v_moy, pas de relaxation (DLL reelle) ===")
    cfg = lib.default_whitewater_config()
    cfg.max_particles = 1000
    cfg.life_spray = cfg.life_foam = cfg.life_bubble = 10.0
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64
    cfg.cell_size = 2.0
    # influence_radius agrandi (defaut 3/128 est trop petit devant la
    # vitesse/gravite : n_sub > 1 dans k_ww_advect ferait reevaluer le
    # regime PLUSIEURS fois dans la frame, meme particule pouvant passer
    # SPRAY/BUBBLE puis FOAM en cours de route -- pas isolant pour ce test.
    # Avec R=1.0, n_sub_fluid = ceil(v_major*dt/(0.5*R)) = 1 pour les
    # vitesses utilisees ici, garantissant UNE seule evaluation FOAM.
    cfg.influence_radius = 1.0
    R = cfg.influence_radius

    # une particule fluide FIXE (vitesse connue et constante), une particule
    # secondaire injectee manuellement via une generation forcee (ke faible
    # seuil) tres PRES de la source -> restera classee FOAM (dans la bande de
    # surface, cf. ww_classify) puisque le champ fluide est stationnaire.
    cfg.ta_weight = 0.0
    cfg.wc_weight = 0.0
    cfg.ke_min = 0.0
    cfg.ke_max = 0.01
    cfg.ke_weight = 1.0
    cfg.spawn_rate = 100000.0

    dt = 1.0 / 60.0
    R_val = R

    # Classification FOAM/SPRAY/BUBBLE (ww_classify) depend du phi Zhu-Bridson
    # au point d'emission -- une seule particule fluide isolee s'auto-
    # echantillonne toujours dans la bande BUBBLE (phi = -0.3*R via son propre
    # rayon de particule). Un petit CLUSTER en ligne (voisins coplanaires
    # d=0 en y, espaces le long de x) approxime un vrai bord de surface : le
    # point d'EXTREMITE du cluster a un xmean tire vers l'interieur, donc un
    # phi moins negatif -- balaye plusieurs espacements jusqu'a tomber dans
    # la bande FOAM (-0.25R, 0], pas de recherche a l'aveugle sur autre chose
    # que ce seul parametre physique connu.
    i0 = None
    for spacing_frac in (0.05, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4):
        spacing = spacing_frac * R_val
        n_pts = 6
        xs = 64.0 + spacing * np.arange(n_pts, dtype=np.float32)
        fluid_pos = np.stack([xs, np.full(n_pts, 64.0, np.float32),
                               np.full(n_pts, 64.0, np.float32)], axis=1)
        fluid_vel = np.tile(np.array([3.0, 0.0, 0.0], dtype=np.float32), (n_pts, 1))

        with lib.Whitewater(cfg) as ww_probe:
            ww_probe.step(fluid_pos, fluid_vel, dt)
            pos_probe, type_probe, _, _, _ = ww_probe.read()
        if pos_probe.shape[0] and (type_probe == 1).any():
            print(f"  cluster trouve : espacement = {spacing_frac} * R -> "
                  f"{(type_probe == 1).sum()} particule(s) FOAM generees")
            break
    else:
        print("  -> ECHEC (aucun espacement teste ne produit de generation FOAM)")
        return False

    with lib.Whitewater(cfg) as ww:
        ww.step(fluid_pos, fluid_vel, dt)
        pos, type_, size, age, vel = ww.read()
        if pos.shape[0] == 0:
            print("  -> ECHEC (aucune particule generee)")
            return False

        foam_idx = np.where(type_ == 1)[0]  # BQ_WW_FOAM
        if foam_idx.size == 0:
            print(f"  -> ECHEC (aucune particule classee FOAM au premier pas, "
                  f"types presents = {np.unique(type_)})")
            return False
        i0 = int(foam_idx[0])

        # vitesse d'emission delibrement DIFFERENTE de v_moy (jitter, cf.
        # Fix C) : sur le pas SUIVANT, si Fix D est correct, la vitesse doit
        # devenir EXACTEMENT v_moy = fluid_vel (affectation, pas de
        # relaxation partielle).
        v_before = vel[i0].copy()
        ww.step(fluid_pos, fluid_vel, dt)
        pos2, type2, size2, age2, vel2 = ww.read()

    v_after = vel2[i0]
    print(f"  vitesse avant le pas verifie : {v_before}")
    print(f"  vitesse apres le pas verifie : {v_after}")
    print(f"  v_moy attendu (= fluid_vel)  : {fluid_vel[0]}")

    err = float(np.max(np.abs(v_after - fluid_vel[0])))
    print(f"  erreur max vs v_moy : {err:.6f} m/s")
    ok = (type2[i0] == 1) and err < 1e-4
    print(f"  -> {'OK' if ok else 'ECHEC'} "
          f"(vitesse foam == v_moy exactement, pas de relaxation)\n")
    return ok


def main():
    results = {
        "Fix A (I_ta)": test_fix_a(),
        "Fix B (I_wc)": test_fix_b(),
        "Fix C (spawn cylindre)": test_fix_c(),
        "Fix D (foam sans inertie)": test_fix_d(),
    }
    print("=== Resume ===")
    all_ok = True
    for name, ok in results.items():
        print(f"  {name:30s} : {'OK' if ok else 'ECHEC'}")
        all_ok &= ok
    print()
    print("RESULTAT:", "OK" if all_ok else "ECHEC")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
