"""Un bruit SPATIALEMENT COHERENT survit-il la ou le bruit blanc meurt ?

Compare, a energie injectee identique (sigma = 1.0 m/s) :
  - bruit blanc par particule (l'implementation actuelle)
  - bruit coherent a l'echelle L (champ grossier interpole trilineairement)
  - curl noise a l'echelle L (rotationnel d'un potentiel -> divergence nulle)
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")
import lib

RES, PPC, DOM = 64, 2, 4.0
DX = DOM / RES
SPACING = DX / PPC
DT = 1.0 / 24.0


def coarse_field(rng, n, shape3):
    """Champ vectoriel aleatoire sur une grille grossiere (n+1)^3."""
    return rng.normal(0.0, 1.0, size=(n + 1, n + 1, n + 1, shape3))


def sample_trilinear(field, pts, lo, cell):
    """Echantillonne `field` (grille reguliere, pas `cell`, origine `lo`)."""
    g = (pts - lo) / cell
    i0 = np.floor(g).astype(int)
    n = field.shape[0] - 1
    i0 = np.clip(i0, 0, n - 1)
    f = g - i0
    out = np.zeros((pts.shape[0], field.shape[3]))
    for dx_ in (0, 1):
        for dy_ in (0, 1):
            for dz_ in (0, 1):
                w = ((f[:, 0] if dx_ else 1 - f[:, 0])
                     * (f[:, 1] if dy_ else 1 - f[:, 1])
                     * (f[:, 2] if dz_ else 1 - f[:, 2]))
                out += w[:, None] * field[i0[:, 0] + dx_, i0[:, 1] + dy_, i0[:, 2] + dz_]
    return out


def make_noise(kind, pts, lo, L, rng):
    if kind == "blanc":
        return rng.normal(0.0, 1.0, size=pts.shape)
    n = int(np.ceil((pts.max(0) - lo).max() / L)) + 2
    if kind == "coherent":
        return sample_trilinear(coarse_field(rng, n, 3), pts, lo, L)
    # curl : v = rot(psi), par differences finies sur le potentiel grossier
    psi = coarse_field(rng, n, 3)
    cur = np.zeros_like(psi)
    d = lambda a, ax: (np.roll(a, -1, ax) - np.roll(a, 1, ax)) / (2 * L)
    cur[..., 0] = d(psi[..., 2], 1) - d(psi[..., 1], 2)
    cur[..., 1] = d(psi[..., 0], 2) - d(psi[..., 2], 0)
    cur[..., 2] = d(psi[..., 1], 0) - d(psi[..., 0], 1)
    return sample_trilinear(cur, pts, lo, L)


cfg = lib.default_config()
cfg.grid_res, cfg.ppc_axis, cfg.domain = RES, PPC, DOM
cfg.cfl, cfg.gravity_y = 0.3, 0.0

ax = SPACING / 2 + np.arange(8, 24) * SPACING + 1.5
gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
PTS = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1).astype(np.float32)
LO = PTS.min(0) - 1e-3

for kind in ("blanc", "coherent", "curl"):
    for mult in ((1,) if kind == "blanc" else (2, 4, 8)):
        rng = np.random.default_rng(0)
        v = make_noise(kind, PTS.astype(np.float64), LO, mult * DX, rng)
        v *= 1.0 / np.std(v)          # renormalise a sigma = 1.0 m/s injecte
        with lib.Sim(cfg) as sim:
            m = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
            sim.emit_points_vel(m, PTS, v.astype(np.float32))
            prev = sim.read_positions().copy()
            kept = []
            for k in range(6):
                sim.step(DT)
                cur = sim.read_positions()
                d_ = (cur - prev) / DT
                kept.append(float(np.std(d_ - d_.mean(0))))
                prev = cur.copy()
        tag = kind if kind == "blanc" else f"{kind} L={mult}*dx"
        print(f"{tag:18s} : f1={kept[0]:.4f}  f3={kept[2]:.4f}  f6={kept[5]:.4f} m/s")
