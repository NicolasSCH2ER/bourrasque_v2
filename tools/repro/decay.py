"""Le bruit blanc par particule survit-il au transfert P2G ?

Emet un bloc de particules avec un bruit de vitesse gaussien connu, puis
mesure l'ecart-type des DEPLACEMENTS par step (proxy de la vitesse) au fil
des steps. Si le MPM filtre le bruit haute frequence, il s'effondre.
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")
import lib

RES, PPC, DOM = 64, 2, 4.0
DX = DOM / RES
SPACING = DX / PPC
DT = 1.0 / 24.0

cfg = lib.default_config()
cfg.grid_res, cfg.ppc_axis, cfg.domain = RES, PPC, DOM
cfg.cfl, cfg.gravity_y = 0.3, 0.0          # gravite nulle : on isole le bruit

rng = np.random.default_rng(0)

for label, sigma in (("sans bruit", 0.0), ("bruit sigma=1.0 m/s", 1.0)):
    with lib.Sim(cfg) as sim:
        m = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
        ax = SPACING / 2 + np.arange(8, 24) * SPACING
        gx, gy, gz = np.meshgrid(ax + 1.5, ax + 1.5, ax + 1.5, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1).astype(np.float32)
        vel = rng.normal(0.0, sigma, size=pts.shape).astype(np.float32)
        sim.emit_points_vel(m, pts, vel)

        prev = sim.read_positions().copy()
        print(f"\n{label} ({pts.shape[0]} particules)")
        for k in range(1, 13):
            sim.step(DT)
            cur = sim.read_positions()
            d = (cur - prev) / DT           # vitesse moyenne sur la frame
            # ecart-type autour de la moyenne = amplitude du bruit residuel
            print(f"  frame {k:2d} : sigma_v = {float(np.std(d - d.mean(0))):.4f} m/s")
            prev = cur.copy()
