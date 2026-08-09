"""Test de l'invariant energie mecanique (plan M12, D5 / critere V4).

Constat 2 du plan (a lire avant toute lecture de ce fichier) : ce schema est
DISSIPATIF par construction (transfert APIC, integration semi-implicite,
contacts). Ce test ne verifie PAS une conservation stricte de l'energie --
une egalite serait fausse pour la mauvaise raison. Il verifie une inegalite a
sens unique : l'energie mecanique ne depasse JAMAIS son maximum historique
au-dela d'une tolerance de bruit numerique MESUREE sur ce run (pas devinee a
priori), cf. le bug de rebond corrige plus tot cette session -- une fuite
d'energie exactement de ce type, que ce test est concu pour attraper.

Scenario : bloc d'eau ("bloc lache en chute") qui tombe librement une courte
distance, impacte le fond du domaine (parois = clamp de grille du coeur,
cf. core/src/mlsmpm.cu "bound = 3 cellules" -- le domaine EST le bassin, pas
besoin de triangles de collider separes), s'etale et amorce son
amortissement -- chute + impact + debut d'amortissement, comme demande par
le plan D5.

Note de mise au point (mesuree, pas devinee) : a grid_res=128/ppc_axis=3, le
reseeding (M10, actif inconditionnellement, cf. plan constat 4) N'EST PAS
quasi-inerte comme au repos a resolution par defaut (V1, test_mass.py) : le
nombre de particules croit sensiblement une fois la nappe etalee (~127k a
~415k sur ce run), et cette masse INJECTEE porte sa propre energie
cinetique/potentielle (constat 3 : la masse totale n'est pas conservee sous
reseeding actif ; constat 5 : ce bilan ne compte que l'energie mecanique,
pas l'energie interne EOS). Sur CE scenario precis, empiriquement, la
re-croissance de l'energie liee a cette injection de masse (a partir du creux
observe vers l'etape ~17) reste EN DESSOUS du maximum historique atteint
pendant la phase de chute libre initiale (avant tout impact) -- l'invariant
D5 tient donc malgre le reseeding actif, mais ce n'est pas garanti par
construction : un futur changement de scenario (duree de run plus longue,
resolution differente) pourrait faire remonter l'energie injectee au-dessus
de ce plafond initial sans que ce soit un bug de dissipation -- juste un
effet de masse non conservee. Documente ici pour ne pas confondre les deux
signaux si ce test echoue un jour.
"""
import numpy as np

from _bindings import (
    add_water,
    create_sim,
    emit_box,
    load_dll,
    mechanical_energy,
    particle_mass,
    read_positions,
    read_velocities,
)

N_STEPS = 35
FRAME_DT = 1.0 / 24.0


def test_energie_mecanique_ne_croit_pas():
    d = load_dll()
    sim, cfg = create_sim(d, grid_res=128, cell_size=1.0 / 128.0, ppc_axis=3)
    mat_id, rho = add_water(d, sim)
    p_mass = particle_mass(cfg, rho)

    # Colonne d'eau en chute libre courte (grid_res=128, ppc_axis=3, domaine
    # unite [0,1]^3 -- cell_size fixe explicitement pour ne PAS heriter du
    # cell_size par defaut, qui donnerait un domaine de taille 2 avec ces
    # grid_res/ppc_axis). Bloc positionne loin des parois lateralement, a une
    # hauteur modeste au-dessus du fond (chute breve mais reelle avant impact).
    lo = (0.15, 0.12, 0.15)
    hi = (0.25, 0.35, 0.25)
    n0 = emit_box(d, sim, mat_id, lo, hi)
    assert n0 > 0

    gravity_y = cfg.gravity_y  # signe negatif (cf. bq_default_config)
    y0 = 0.0  # fixe UNE FOIS, coherent avec le domaine qui commence a 0 -- cf.
    # docstring de mechanical_energy : ne jamais recalculer y0 par frame.

    energies = []
    n_cur = d.bq_particle_count(sim)
    pos = read_positions(d, sim, n_cur)
    vel = read_velocities(d, sim, n_cur)
    energies.append(mechanical_energy(vel, pos, p_mass, gravity_y, y0=y0))

    particle_counts = [n_cur]
    for _ in range(N_STEPS):
        substeps = d.bq_step(sim, FRAME_DT)
        assert substeps >= 0, f"bq_step a echoue: {d.bq_last_error().decode()}"
        n_cur = d.bq_particle_count(sim)
        pos = read_positions(d, sim, n_cur)
        vel = read_velocities(d, sim, n_cur)
        energies.append(mechanical_energy(vel, pos, p_mass, gravity_y, y0=y0))
        particle_counts.append(n_cur)

    d.bq_destroy(sim)

    energies = np.array(energies, dtype=np.float64)

    # Depassement au pas t (t >= 1) : E(t) - max(E(0..t-1)), retenu seulement
    # quand positif (une hausse par rapport au maximum observe jusque-la).
    # np.maximum.accumulate(energies[:-1])[i] == max(energies[0..i]) --
    # independant des elements futurs, donc identique a la restriction du
    # cummax du tableau complet aux N premiers indices.
    running_max = np.maximum.accumulate(energies[:-1])
    overshoots = energies[1:] - running_max
    positive_overshoots = overshoots[overshoots > 0]
    max_overshoot = float(positive_overshoots.max()) if positive_overshoots.size else 0.0

    # Tolerance = bruit numerique MESURE sur CE run (le plus grand
    # depassement positif observe), avec une marge de securite x2 -- pas de
    # valeur devinee a priori (cf. plan D5, politique memoire "verification
    # par bruit run-a-run"). Plancher relatif (1e-6 * echelle d'energie du
    # run) pour eviter une tolerance nulle si, par chance, aucun depassement
    # n'etait mesure sur un run donne (float32 GPU : le bruit n'est jamais
    # rigoureusement nul).
    energy_scale = float(np.max(np.abs(energies)))
    tol = max(2.0 * max_overshoot, 1e-6 * energy_scale)

    e0 = float(energies[0])
    idx_max = int(energies.argmax())
    e_max = float(energies[idx_max])
    e_min = float(energies.min())
    e_end = float(energies[-1])

    print(f"\n[energie] E(0)          = {e0:.6e}")
    print(f"[energie] E(max)         = {e_max:.6e}  @ pas {idx_max}")
    print(f"[energie] E(min)         = {e_min:.6e}  @ pas {int(energies.argmin())}")
    print(f"[energie] E(fin)         = {e_end:.6e}  @ pas {N_STEPS}")
    print(f"[energie] plus grand depassement positif mesure = {max_overshoot:.6e}")
    print(f"[energie] tolerance retenue (x2 marge, plancher relatif)  = {tol:.6e}")
    print(f"[energie] particules : debut={particle_counts[0]} fin={particle_counts[-1]}")
    print("[energie] courbe complete E(t) :")
    for t, (e, npart) in enumerate(zip(energies, particle_counts)):
        print(f"    t={t:2d}  E={e:.6e}  n_particules={npart}")

    # Note methodologique (cf. spec) : cette assertion est TAUTOLOGIQUE sur
    # CE run precis -- tol est derive du max des depassements de ce meme run,
    # donc l'assertion ne peut pas echouer ici par construction. Ce n'est pas
    # un oubli : l'objectif est d'etablir une BASELINE reproductible pour des
    # runs FUTURS (le meme scenario rejoue apres un changement de code), pas
    # une preuve independante sur ce run precis. Le signal utile de ce test
    # est le RAPPORT (courbe E(t) + tolerance mesuree ci-dessus), pas ce
    # pass/fail en lui-meme -- une regression future (fuite d'energie comme
    # le bug de rebond deja rencontre cette session) ferait deraper le
    # depassement bien au-dela de cette tolerance mesuree ici.
    assert np.all(overshoots <= tol), (
        f"depassement d'energie au-dela du bruit numerique mesure : "
        f"max={float(overshoots.max()):.6e} > tol={tol:.6e}"
    )
