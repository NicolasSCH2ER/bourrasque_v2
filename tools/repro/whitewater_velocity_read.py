"""Verification physique du nouveau parametre `vel` de `bq_whitewater_read`
(cf. core/include/bourrasque.h, core/src/whitewater.cu, extension/lib.py).

Cf. la formule de dvection SPRAY de `k_ww_advect` (core/src/whitewater.cu) :

    v_major = |v| + |gravity_y|*dt
    n_sub   = clamp(ceil(v_major*dt / (0.5*R)), 1, 256)
    h       = dt / n_sub
    par sous-pas (n_sub fois) :
        v.x += (0        - drag_spray*v.x*|v|) * h
        v.y += (gravity_y - drag_spray*v.y*|v|) * h
        v.z += (0        - drag_spray*v.z*|v|) * h

Ce script fait naitre UNE particule secondaire de type SPRAY avec une vitesse
initiale connue (celle de la particule fluide source, a un jitter d'emission
pres de 5% -- cf. k_ww_emit), la laisse s'eloigner de sa source fluide (donc
hors du rayon d'influence R -> classee SPRAY en continu, cf. ww_classify),
puis compare a chaque frame la vitesse lue via `Whitewater.read()` a la
prediction analytique ci-dessus, rejouee cote hote en float32. Une simple
non-nullite de la vitesse ne suffirait pas a distinguer un bug de decalage
memoire (lire un autre buffer par erreur) d'une vitesse correcte -- cette
comparaison quantitative si.

Ne modifie rien dans le depot -- lecture seule de la DLL compilee
(extension/bin/bourrasque.dll).
"""
import sys

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")

import numpy as np
import lib


def predict_spray_substep(v, R, dt, gravity_y, drag_spray):
    """Rejoue exactement la boucle de sous-pas SPRAY de k_ww_advect, en
    float32 pour rester au plus pres de la precision GPU."""
    v = np.asarray(v, dtype=np.float32).copy()
    v_major = np.float32(np.linalg.norm(v) + abs(gravity_y) * dt)
    n_sub = int(np.ceil(v_major * dt / (np.float32(0.5) * R)))
    n_sub = min(max(n_sub, 1), 256)
    h = np.float32(dt / n_sub)
    for _ in range(n_sub):
        vl = np.float32(np.linalg.norm(v))
        v[0] += (np.float32(0.0) - drag_spray * v[0] * vl) * h
        v[1] += (gravity_y - drag_spray * v[1] * vl) * h
        v[2] += (np.float32(0.0) - drag_spray * v[2] * vl) * h
    return v


def main():
    cfg = lib.default_whitewater_config()
    cfg.max_particles = 1000
    cfg.life_spray = 10.0
    cfg.life_foam = 10.0
    cfg.life_bubble = 10.0
    # Domaine largement surdimensionne (128 m/axe) pour que la particule
    # suivie ne touche jamais un mur absorbant pendant la fenetre testee.
    cfg.grid_res[0] = cfg.grid_res[1] = cfg.grid_res[2] = 64
    cfg.cell_size = 2.0

    R = cfg.influence_radius
    gravity_y = cfg.gravity_y
    drag_spray = cfg.drag_spray
    dt = 1.0 / 60.0

    # particule fluide source, FIXE, vitesse connue -- assez rapide pour
    # depasser ke_min (=1 par defaut, I_k = 0.5*|v|^2) et declencher une
    # generation en quelques frames via le budget carry.
    fluid_pos = np.array([[60.0, 60.0, 60.0]], dtype=np.float32)
    fluid_vel = np.array([[3.0, 0.0, 0.0]], dtype=np.float32)

    with lib.Whitewater(cfg) as ww:
        # --- 1. attendre la premiere generation (budget carry, cf.
        #        isolated_decay.py pour le meme mecanisme). ------------------
        spawn_frame = None
        for frame in range(60):
            ww.step(fluid_pos, fluid_vel, dt)
            if ww.count() >= 1:
                spawn_frame = frame
                break
        if spawn_frame is None:
            print("RESULTAT: ECHEC -- aucune particule generee en 60 frames")
            return 1

        pos, type_, size, age, vel = ww.read()
        print(f"[spawn] frame {spawn_frame} : n={pos.shape[0]}, "
              f"type={type_[0]} (0=SPRAY,1=FOAM,2=BUBBLE), vel0={vel[0]}")

        # --- 2. frames tampon : laisser la particule (indice 0, celle qui
        #        survit le plus longtemps donc reste en tete apres
        #        compaction, cf. k_ww_compact) s'eloigner de la source
        #        fluide fixe au-dela de R, jusqu'a classification SPRAY
        #        stable sur une frame entiere. --------------------------
        warmup_frames = 6
        for _ in range(warmup_frames):
            ww.step(fluid_pos, fluid_vel, dt)

        pos, type_, size, age, vel = ww.read()
        dist0 = float(np.linalg.norm(pos[0] - fluid_pos[0]))
        print(f"[apres tampon] n={pos.shape[0]}, dist a la source fluide="
              f"{dist0:.4f} m (R={R:.4f} m), type0={type_[0]}")

        if type_[0] != 0:
            print("RESULTAT: ECHEC -- particule 0 pas classee SPRAY (0) "
                  f"apres tampon (type={type_[0]}), scenario a ajuster")
            return 1
        if dist0 <= R:
            print("RESULTAT: ECHEC -- particule 0 encore dans le rayon "
                  f"d'influence ({dist0:.4f} <= {R:.4f}), tampon insuffisant")
            return 1

        # --- 3. fenetre de verification : a chaque frame, predire la
        #        vitesse via la formule SPRAY rejouee cote hote a partir de
        #        la vitesse lue au debut de la frame, comparer a la vitesse
        #        relue apres bq_whitewater_step. --------------------------
        n_verify = 6
        max_abs_err = 0.0
        max_rel_err = 0.0
        all_ok = True
        for k in range(n_verify):
            v_start = vel[0].copy()
            v_pred = predict_spray_substep(v_start, R, dt, gravity_y, drag_spray)

            ww.step(fluid_pos, fluid_vel, dt)
            pos, type_, size, age, vel = ww.read()

            if pos.shape[0] < 1 or type_[0] != 0:
                print(f"[verif frame {k}] ECHEC -- particule 0 absente ou "
                      f"plus SPRAY (n={pos.shape[0]}, type="
                      f"{type_[0] if pos.shape[0] else 'n/a'})")
                all_ok = False
                break

            v_actual = vel[0]
            abs_err = np.abs(v_actual - v_pred)
            rel_err = abs_err / np.maximum(np.abs(v_pred), 1e-6)
            max_abs_err = max(max_abs_err, float(np.max(abs_err)))
            max_rel_err = max(max_rel_err, float(np.max(rel_err)))

            print(f"[verif frame {k}] v_pred={v_pred}, v_actual={v_actual}, "
                  f"err_abs_max={float(np.max(abs_err)):.6f}, "
                  f"err_rel_max={float(np.max(rel_err)):.6f}")

        print()
        print(f"erreur absolue max sur la fenetre : {max_abs_err:.6f} m/s")
        print(f"erreur relative max sur la fenetre : {max_rel_err:.6f}")

        # tolerance large devant la precision numerique attendue (memes
        # operations, float32 des deux cotes) : detecte un vrai probleme de
        # cablage (mauvais buffer, unites, decalage), pas un bruit
        # d'arrondi. Cf. verification-par-bruit-run-a-run (memoire) --
        # seuil choisi pour etre tolerant a l'ordre des operations flottantes,
        # pas pour cacher un ecart structurel.
        physically_consistent = all_ok and max_abs_err < 0.01 and max_rel_err < 0.02

    print()
    print("RESULTAT:", "OK" if physically_consistent else "ECHEC")
    return 0 if physically_consistent else 1


if __name__ == "__main__":
    raise SystemExit(main())
