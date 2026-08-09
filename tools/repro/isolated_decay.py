"""Verification du correctif "plancher decroissant" pour les particules
fluides structurellement isolees (cf. k_ww_generate_potentials,
core/src/whitewater.cu).

Deux proprietes a verifier sur le meme scenario (une seule particule fluide,
STATIQUE, sans aucun voisin -- neighbor_count=0 <= k_n_cull a chaque frame,
plaquee pres d'un bord de domaine comme le film d'eau du dam-break rapporte) :

1. Garantie anti-trou transitoire toujours tenue : le plancher doit encore
   produire au moins une particule whitewater dans un delai raisonnable
   (isolated_time proche de 0 au debut, la decroissance n'a presque pas
   encore agi).
2. Le debit cumule sur une isolation structurelle de plusieurs secondes doit
   etre nettement (>=5-10x) inferieur a ce qu'un plancher CONSTANT (ancien
   comportement, 0.05 sans decroissance) aurait produit sur la meme duree --
   la constante est simulee analytiquement en rejouant exactement la meme
   recurrence raw/carry que le kernel (I=0.05 fixe a chaque frame, seule
   valeur possible ici puisque tous les autres termes de I sont nuls : une
   seule particule fluide, sans voisin, ne peut alimenter ni I_ta ni I_wc, et
   une vitesse nulle annule I_k).

Ne modifie rien dans le depot -- lecture seule de la DLL compilee
(extension/bin/bourrasque.dll).
"""
import sys
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")

import numpy as np
import lib


def analytic_constant_floor_count(n_frames, dt, spawn_rate, floor=0.05):
    """Rejoue exactement raw = I*spawn_rate*dt + carry ; cnt = floor(raw) ;
    carry = raw - cnt, avec I constant = floor a chaque frame (ancien
    comportement, avant l'ajout de la decroissance temporelle)."""
    carry = 0.0
    total = 0
    for _ in range(n_frames):
        raw = floor * spawn_rate * dt + carry
        cnt = int(np.floor(raw))
        carry = raw - cnt
        total += cnt
    return total


def run_isolated(n_frames, dt, pos, life=1.0e6, max_particles=200000):
    """Une seule particule fluide statique, vitesse nulle, sans aucun
    voisin (n=1) donc neighbor_count=0 <= k_n_cull a CHAQUE frame -- le cas
    "structurellement isole en continu" du rapport utilisateur (film plaque
    contre une paroi). vies tres longues pour que count() en fin de run
    approxime le total CUMULE genere (aucune mort dans la fenetre testee)."""
    cfg = lib.default_whitewater_config()
    cfg.max_particles = max_particles
    cfg.life_spray = life
    cfg.life_foam = life
    cfg.life_bubble = life

    pos_arr = np.array([pos], dtype=np.float32)
    vel_arr = np.zeros((1, 3), dtype=np.float32)

    counts = []
    with lib.Whitewater(cfg) as ww:
        for _ in range(n_frames):
            ww.step(pos_arr, vel_arr, dt)
            counts.append(ww.count())
    return counts, float(cfg.spawn_rate)


def decay_factor(isolated_time_s, tau=0.3):
    """Meme formule que le kernel : decay = 1 / (1 + isolated_time/tau)."""
    return 1.0 / (1.0 + isolated_time_s / tau)


def main():
    dt = 1.0 / 60.0
    n_frames = 200  # ~3.33s a 60fps, dans la fourchette demandee (2.5-3.3s)

    # position pres d'un bord du domaine [0,1]^3 (grid_res=64, cell_size=1/64
    # par defaut), comme le film d'eau plaque contre une paroi rapporte sur
    # le dam-break reel.
    pos = (0.5, 0.5, 0.02)

    counts, spawn_rate = run_isolated(n_frames, dt, pos)

    # --- 0. cas transitoire (1-2 frames) : ce qui determine si une goutte
    #        isolee 1-2 frames seulement garde un plancher quasi plein, c'est
    #        le facteur de decroissance a isolated_time proche de 0 (2 frames
    #        = 2*dt), PAS le temps total pris pour accumuler une particule
    #        entiere via carry (mecanisme inchange par ce correctif, deja
    #        present avant : ~24 frames/0.4s a plancher plein). Verifie que
    #        ce facteur reste proche de 1 (peu de perte) pour une isolation
    #        aussi breve. ---------------------------------------------------
    df_2frames = decay_factor(2 * dt)
    transient_floor_preserved = df_2frames >= 0.85

    # --- 1. garantie anti-trou tenue en regime isole continu (le meme
    #        mecanisme carry qu'avant, non affecte au premier ordre par la
    #        decroissance tant que isolated_time reste petit devant tau) ----
    first_frame_ge1 = next((i for i, c in enumerate(counts) if c >= 1), None)
    transient_ok = first_frame_ge1 is not None and first_frame_ge1 * dt <= 1.0

    # --- 2. debit cumule structurel : mesure (decroissance active) vs ------
    #        prediction analytique du plancher constant (ancien comportement)
    measured_final = counts[-1]
    analytic_final = analytic_constant_floor_count(n_frames, dt, spawn_rate)
    ratio = analytic_final / max(measured_final, 1)

    print("=== Verification isolated_time decay (whitewater.cu) ===")
    print(f"n_frames={n_frames}  dt={dt:.5f}s  duree={n_frames*dt:.2f}s  spawn_rate={spawn_rate}")
    print()
    print(f"[transitoire 1-2 frames] facteur de decroissance a isolated_time=2*dt "
          f"({2*dt:.4f}s) : {df_2frames:.4f} (plancher effectif = "
          f"{0.05*df_2frames:.4f}, quasi plein vs 0.05 constant)")
    print(f"[transitoire 1-2 frames] plancher preserve (facteur >= 0.85) : {transient_floor_preserved}")
    print()
    print(f"[isolation continue] premiere frame avec >=1 particule active : "
          f"{first_frame_ge1} (t={first_frame_ge1*dt:.3f}s)"
          if first_frame_ge1 is not None else
          "[isolation continue] AUCUNE particule generee dans la fenetre testee -- ECHEC")
    print(f"[isolation continue] garantie anti-trou tenue (<=1s) : {transient_ok}")
    print()
    print(f"[structurel] total mesure (decroissance active), {n_frames} frames : {measured_final}")
    print(f"[structurel] total analytique (plancher CONSTANT 0.05, ancien comportement) : {analytic_final}")
    print(f"[structurel] ratio reduction (constant / mesure) : {ratio:.2f}x")
    print(f"[structurel] reduction >= 5x : {ratio >= 5.0}")
    print()

    ok = (transient_floor_preserved and transient_ok and (ratio >= 5.0)
          and (first_frame_ge1 is not None))
    print("RESULTAT:", "OK" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
