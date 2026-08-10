"""M17 — le couplage AMORTIT-il la reponse du corps ?

A7 a etabli que la force est correcte : elle suit proportionnellement le volume
immerge (8 % d'ecart entre 96 % et 67 % d'immersion). Le defaut est donc dans la
REPONSE du corps, pas dans la force. Deux candidats :

  D1  le couplage implicite soude le corps a la vitesse du fluide en contact
      (choc inelastique) : avec S_m >> m_b, le corps devient un traceur advecte
      et n'accelere quasiment pas, quelle que soit la force.
  D2  l'eau qui reste sur le dessus l'alourdit.

Ce script separe les deux, et il est court : on ne mesure que la PREMIERE
reponse, avant tout deplacement notable, la ou la force est encore celle qu'A7 a
mesuree sur le meme montage.

Protocole : meme cuve quasi-2D qu'A7, corps 8 cellules a 96 % d'immersion.
1. on laisse le fluide se tasser avec le corps MAINTENU (masse enorme, sans
   gravite propre) -- sinon on mesurerait un transitoire de tassement ;
2. on redeclare le meme corps LIBRE (masse reelle, gravite active) : cela
   reinitialise l'etat des corps, jamais celui du fluide ;
3. on lit sa vitesse apres 1 et 2 frames, et on la compare a `a = F_net / m`.

D2 ne peut pas agir en une frame (il faut le temps que l'eau monte sur le
dessus) : un ecart des la premiere frame accuse donc D1 sans ambiguite.

    python tools/repro/diag_reponse_corps_libre.py
"""

import ctypes
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("bq_lib", ROOT / "extension" / "lib.py")
lib = importlib.util.module_from_spec(_spec)
sys.modules["bq_lib"] = lib
_spec.loader.exec_module(lib)

GRAVITY, RHO = -9.81, 1000.0
BULK, GAMMA = 2.0e5, 7.0
CELL = 0.02
RES = (48, 48, 16)
WATER_LO, WATER_HI = (0.08, 0.08, 0.08), (0.88, 0.70, 0.24)
SIDE, SPAN_Z = 0.16, 0.20
SPACING = CELL / 2.0
FRAME_DT = 1.0 / 24.0
SETTLE = 130
CY = 0.46                      # 96 % d'immersion sur la surface reelle (~0.547)
DENSITE_CORPS = 500.0
V_CORPS = SIDE * SIDE * (WATER_HI[2] - WATER_LO[2])
F_POUSSEE = 94.0               # N, MESUREE par A7 sur ce montage exact


def box_tris(c):
    c = np.asarray(c, float)
    h = np.array([SIDE / 2.0, SIDE / 2.0, SPAN_Z])
    sg = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    v = c + sg * h
    f = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return np.array([[v[a], v[b], v[c_]] for a, b, c_ in f], dtype=np.float32)


def make_body(center, mass, gravite):
    inv_i = np.zeros(9, np.float32)
    inv_i[0] = inv_i[4] = inv_i[8] = 6.0 / (mass * SIDE * SIDE)
    b = lib.BqRigidBody()
    b.dynamic, b.mass = 1, mass
    b.inv_inertia = (ctypes.c_float * 9)(*inv_i)
    b.x = (ctypes.c_float * 3)(*center)
    b.q = (ctypes.c_float * 4)(1.0, 0.0, 0.0, 0.0)
    b.use_gravity = 1 if gravite else 0
    b.added_mass = 0.0
    # verrous lateraux : on ne veut mesurer que la reponse VERTICALE, sans
    # qu'une derive laterale ou une rotation ne brouille le chiffre.
    b.lock_lin = (ctypes.c_int * 3)(1, 0, 1)
    b.lock_ang = (ctypes.c_int * 3)(1, 1, 1)
    return b


def main():
    cfg = lib.BqConfig()
    cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2] = RES
    cfg.cell_size, cfg.gravity_y, cfg.cfl = CELL, GRAVITY, 0.4
    cfg.ppc_axis, cfg.max_particles = 2, 400000

    center = (0.48, CY, RES[2] * CELL / 2.0)
    ax = [np.arange(lo + SPACING / 2, hi, SPACING)
          for lo, hi in zip(WATER_LO, WATER_HI)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    half = np.array([SIDE / 2.0 + SPACING, SIDE / 2.0 + SPACING, 1e9])
    pts = np.ascontiguousarray(
        g[~np.all(np.abs(g - np.asarray(center)) < half, axis=1)], dtype=np.float32)

    tri = box_tris(center)
    n_tri = tri.shape[0]
    masse = DENSITE_CORPS * V_CORPS
    poids = masse * (-GRAVITY)
    f_net = F_POUSSEE - poids
    a_att = f_net / masse

    print(f"corps : {SIDE*100:.0f} cm de cote, densite {DENSITE_CORPS:.0f}, "
          f"masse {masse:.3f} kg")
    print(f"  poussee mesuree par A7  {F_POUSSEE:6.1f} N")
    print(f"  poids                   {poids:6.1f} N")
    print(f"  force nette             {f_net:6.1f} N  ->  a attendue "
          f"{a_att:6.2f} m/s^2")
    print(f"  vitesse attendue apres 1 frame : {a_att*FRAME_DT:.3f} m/s\n")

    with lib.Sim(cfg) as sim:
        sim.add_material(lib.BQ_MODEL_WATER, RHO, bulk=BULK, gamma=GAMMA)
        sim.emit_points(0, pts)
        # 1. tassement, corps MAINTENU (masse enorme, sans gravite propre)
        sim.set_collider_bodies([make_body(center, 1.0e5, False)])
        sim.set_colliders(tri, np.zeros_like(tri),
                          np.zeros(n_tri, np.float32), np.zeros(n_tri, np.int32))
        for _ in range(SETTLE):
            sim.step(FRAME_DT)
        v_rms = float(np.sqrt(
            (np.linalg.norm(sim.read_velocities(), axis=1) ** 2).mean()))
        print(f"  fluide tasse : |v|_rms = {v_rms:.4f} m/s")

        # 2. on redeclare le MEME corps, libre. Cela reinitialise l'etat des
        #    CORPS uniquement -- le fluide, lui, garde son etat tasse.
        sim.set_collider_bodies([make_body(center, masse, True)])

        # 3. reponse, frame par frame
        print(f"\n  {'frame':>6} {'y (m)':>9} {'vy (m/s)':>10} {'vy attendue':>12} {'ratio':>7}")
        for f in range(1, 5):
            sim.step(FRAME_DT)
            st = sim.read_collider_bodies()[0]
            vy, y = float(st[8]), float(st[1])
            att = a_att * FRAME_DT * f
            print(f"  {f:6d} {y:9.4f} {vy:10.4f} {att:12.3f} "
                  f"{vy/att if abs(att) > 1e-9 else float('nan'):7.3f}")
            if f == 1:
                r1 = vy / att if abs(att) > 1e-9 else float("nan")

    print("\n================ VERDICT ================")
    print(f"  ratio a la 1re frame : {r1:.3f}")
    if r1 > 0.6:
        print("  -> le corps repond a la force : PAS d'amortissement dominant.")
        print("     Le defaut serait alors la surcharge d'eau (D2), qui met")
        print("     plusieurs frames a s'installer.")
    elif r1 < 0.25:
        print("  -> D1 : le corps ne repond quasiment PAS a la force qu'il recoit.")
        print("     Le couplage implicite le soude a la vitesse du fluide ; c'est")
        print("     la formulation du solve, pas la recolte, qu'il faut revoir.")
    else:
        print(f"  -> amortissement PARTIEL (ratio {r1:.2f}) : les deux effets")
        print("     coexistent, aucun n'est dominant.")




# ---------------------------------------------------------------------------
# Porte du jalon : un corps libre atteint-il l'equilibre d'ARCHIMEDE ?
# ---------------------------------------------------------------------------

def flotte(densite=500.0, frames=260):
    """Corps libre lache immerge, laisse aller jusqu'a stabilisation. La
    fraction immergee finale doit valoir densite/1000.

    C'est la mesure que le couplage soude ne pouvait pas passer : il n'avait
    aucun equilibre propre, il figeait le corps la ou etait le fluide. La forme
    de masse ajoutee, elle, s'annule quand poussee = poids -- donc a Archimede."""
    cfg = lib.BqConfig()
    cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2] = RES
    cfg.cell_size, cfg.gravity_y, cfg.cfl = CELL, GRAVITY, 0.4
    cfg.ppc_axis, cfg.max_particles = 2, 400000

    center = (0.48, CY, RES[2] * CELL / 2.0)
    ax = [np.arange(lo + SPACING / 2, hi, SPACING)
          for lo, hi in zip(WATER_LO, WATER_HI)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    half = np.array([SIDE / 2.0 + SPACING, SIDE / 2.0 + SPACING, 1e9])
    pts = np.ascontiguousarray(
        g[~np.all(np.abs(g - np.asarray(center)) < half, axis=1)], dtype=np.float32)
    tri = box_tris(center)
    n_tri = tri.shape[0]
    masse = densite * V_CORPS

    print(f"\n===== EQUILIBRE : densite {densite:.0f}, fraction attendue "
          f"{densite/1000.0:.2f} =====")
    with lib.Sim(cfg) as sim:
        sim.add_material(lib.BQ_MODEL_WATER, RHO, bulk=BULK, gamma=GAMMA)
        sim.emit_points(0, pts)
        sim.set_collider_bodies([make_body(center, 1.0e5, False)])
        sim.set_colliders(tri, np.zeros_like(tri),
                          np.zeros(n_tri, np.float32), np.zeros(n_tri, np.int32))
        for _ in range(SETTLE):
            sim.step(FRAME_DT)

        sim.set_collider_bodies([make_body(center, masse, True)])
        for f in range(frames):
            st = sim.read_collider_bodies()[0]
            c = (float(st[0]), float(st[1]), float(st[2]))
            # geometrie du collider retransmise a la pose courante : le champ
            # vu par le fluide n'est rafraichi qu'une fois par frame (D4)
            sim.set_colliders(box_tris(c), np.zeros((n_tri, 3, 3), np.float32),
                              np.zeros(n_tri, np.float32), np.zeros(n_tri, np.int32))
            sim.step(FRAME_DT)
            if (f + 1) % 65 == 0:
                st = sim.read_collider_bodies()[0]
                print(f"    frame {f+1:4d} : y = {st[1]:.4f}  vy = {st[8]:+.4f}")
        st = sim.read_collider_bodies()[0]
        pos = sim.read_positions().copy()

    cy_fin = float(st[1])
    r = np.abs(pos[:, 0] - 0.48)
    loin = pos[r > SIDE / 2.0 + 2 * CELL]
    surf = float(np.percentile(loin[:, 1], 98))
    frac = float(np.clip((surf - (cy_fin - SIDE / 2.0)) / SIDE, 0.0, 1.0))
    print(f"  surface libre {surf:.4f}   centre du corps {cy_fin:.4f}")
    print(f"  fraction immergee MESUREE {frac:.3f}   attendue {densite/1000.0:.3f}")
    print(f"  vitesse finale {float(st[8]):+.4f} m/s")
    return frac


if __name__ == "__main__":
    main()
    flotte(500.0)
