"""Continuite du jet dans le REGIME DE L'UTILISATEUR : plusieurs couches dues
par frame (c'est la que la superposition faisait le degat).

Mesure l'histogramme des positions le long de l'axe d'ecoulement a une frame ou
le jet est encore en vol, et cherche des TROUS : un chapelet de galettes laisse
des intervalles vides larges devant `spacing`, un jet continu n'en laisse aucun.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")
import lib

GRID_RES, PPC = 64, 2
DOMAIN = 4.0
DX = DOMAIN / GRID_RES
SPACING = DX / PPC
FPS = 24.0
FRAME_DT = 1.0 / FPS

# Vitesse choisie pour que ~5 couches soient dues par frame.
SPEED = 5.0 * SPACING / FRAME_DT
print(f"spacing={SPACING:.5f}  vitesse={SPEED:.3f} m/s  "
      f"-> {SPEED * FRAME_DT / SPACING:.2f} couches/frame")

cfg = lib.default_config()
cfg.grid_res, cfg.ppc_axis, cfg.domain = GRID_RES, PPC, DOMAIN
cfg.cfl, cfg.gravity_y = 0.3, -9.8

with lib.Sim(cfg) as sim:
    m = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)

    # Emetteur : petite boite haut placee, jet dirige vers le bas (-Y solveur).
    face_y = 3.4
    lo = np.array([1.9, face_y, 1.9])
    hi = np.array([2.1, face_y + SPACING, 2.1])
    vel = np.array([0.0, -SPEED, 0.0])
    g = np.array([0.0, cfg.gravity_y, 0.0])

    # Reseau transverse de la couche source.
    def lattice(a, b):
        k0 = max(0, int(np.ceil((a - SPACING / 2) / SPACING)))
        k1 = int(np.floor((b - SPACING / 2) / SPACING))
        return SPACING / 2 + np.arange(k0, k1 + 1) * SPACING

    ax, ay, az = lattice(lo[0], hi[0]), lattice(lo[1], hi[1]), lattice(lo[2], hi[2])
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    layer = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1).astype(np.float32)
    print(f"couche source : {layer.shape[0]} points")

    D = 0.0
    for _ in range(25):
        D_prev, D = D, D + SPEED * FRAME_DT
        m0 = int(np.floor(D_prev / SPACING)) + 1
        m1 = int(np.floor(D / SPACING))
        for mm in range(m0, m1 + 1):
            tau = (D - mm * SPACING) / SPEED
            pts = layer + (vel * tau + 0.5 * g * tau * tau).astype(np.float32)
            v_em = vel + g * tau
            keep = np.all((pts > 3 * DX) & (pts < DOMAIN - 3 * DX), axis=1)
            if keep.any():
                sim.emit_points(m, np.ascontiguousarray(pts[keep]), tuple(v_em))
        sim.step(FRAME_DT)

    pos = sim.read_positions()

# Analyse : uniquement le jet en vol, au-dessus du sol.
y = np.sort(pos[:, 1])
y = y[y > 0.6]
gaps = np.diff(np.unique(np.round(y / (SPACING / 8)) * (SPACING / 8)))
gaps = gaps[gaps > 1e-6]
print(f"\nparticules totales : {pos.shape[0]}, en vol : {y.size}")
print(f"ecart median  : {np.median(gaps) / SPACING:6.2f} x spacing")
print(f"ecart max     : {gaps.max() / SPACING:6.2f} x spacing")
print(f"trous > 3x spacing : {(gaps > 3 * SPACING).sum()}")
print("\n=> un chapelet montre des dizaines de trous larges ; "
      "un jet continu n'en montre aucun.")
