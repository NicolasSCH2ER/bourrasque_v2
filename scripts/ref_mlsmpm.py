#!/usr/bin/env python3
"""Reference NumPy du solveur MLS-MPM 3D de Bourrasque.

Role : specification executable de l'algorithme. Le code CUDA de core/
doit etre une transcription ligne a ligne de ce fichier. Toute divergence
entre la DLL et cette reference est un bug de la DLL.

Algorithme : MLS-MPM (Hu et al. 2018) avec B-splines quadratiques.
Materiaux : elastique corotationnel fixe (M1), eau EOS de Tait J-based (M2).

Usage :
    python ref_mlsmpm.py jelly   # M1 : cube de gelee qui tombe
    python ref_mlsmpm.py dam     # M2 : dam break eau
    python ref_mlsmpm.py splash  # M1+M2 : cube elastique dans l'eau
"""
import sys
import time

import numpy as np

# ---------------------------------------------------------------- parametres
N_GRID = 32                  # noeuds par axe (domaine = cube unite)
DX = 1.0 / N_GRID
INV_DX = float(N_GRID)
GRAVITY = -9.8
BOUND = 3                    # marge de noeuds pour les conditions aux limites
PPC_AXIS = 2                 # particules par cellule et par axe (8 / cellule)

ELASTIC, WATER = 0, 1

# Materiaux : (rho, E, nu) pour l'elastique ; (rho, bulk k, gamma) pour l'eau.
MAT_ELASTIC = dict(rho=1000.0, E=5.0e4, nu=0.2)
MAT_WATER = dict(rho=1000.0, k=4.0e4, gamma=3.0)


def sound_speed(mat):
    if "E" in mat:
        return np.sqrt(mat["E"] / mat["rho"])
    return np.sqrt(mat["k"] / mat["rho"])


class Sim:
    def __init__(self):
        self.x = np.zeros((0, 3), np.float64)
        self.v = np.zeros((0, 3), np.float64)
        self.C = np.zeros((0, 3, 3), np.float64)
        self.F = np.zeros((0, 3, 3), np.float64)
        self.Jw = np.zeros((0,), np.float64)   # J scalaire (eau)
        self.mat = np.zeros((0,), np.int32)
        spacing = DX / PPC_AXIS
        self.p_vol = spacing ** 3
        # dt fixe par la CFL acoustique du materiau le plus raide
        c = max(sound_speed(MAT_ELASTIC), sound_speed(MAT_WATER))
        self.dt = 0.3 * DX / c
        print(f"dt = {self.dt:.3e}  (c_max = {c:.1f} m/s)")

    def emit_box(self, mat_id, lo, hi, vel=(0, 0, 0)):
        spacing = DX / PPC_AXIS
        axes = [np.arange(l + spacing / 2, h, spacing) for l, h in zip(lo, hi)]
        g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        n = len(g)
        self.x = np.concatenate([self.x, g])
        self.v = np.concatenate([self.v, np.tile(np.asarray(vel, float), (n, 1))])
        self.C = np.concatenate([self.C, np.zeros((n, 3, 3))])
        self.F = np.concatenate([self.F, np.tile(np.eye(3), (n, 1, 1))])
        self.Jw = np.concatenate([self.Jw, np.ones(n)])
        self.mat = np.concatenate([self.mat, np.full(n, mat_id, np.int32)])
        print(f"emit {n} particules (mat {mat_id})")

    # ------------------------------------------------------------- substep
    def substep(self):
        dt, n = self.dt, len(self.x)
        grid_mv = np.zeros((N_GRID, N_GRID, N_GRID, 3))
        grid_m = np.zeros((N_GRID, N_GRID, N_GRID))

        base = (self.x * INV_DX - 0.5).astype(np.int32)          # (n,3)
        fx = self.x * INV_DX - base                              # in [0.5, 1.5]
        # Poids B-spline quadratique, par axe : (3, n, 3)
        w = np.stack([0.5 * (1.5 - fx) ** 2,
                      0.75 - (fx - 1.0) ** 2,
                      0.5 * (fx - 0.5) ** 2])

        # --- mise a jour de F (elastique) puis contrainte de Cauchy
        el = self.mat == ELASTIC
        wa = self.mat == WATER
        self.F[el] = (np.eye(3) + dt * self.C[el]) @ self.F[el]

        stress = np.zeros((n, 3, 3))
        if el.any():
            rho, E, nu = MAT_ELASTIC["rho"], MAT_ELASTIC["E"], MAT_ELASTIC["nu"]
            mu = E / (2 * (1 + nu))
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            Fe = self.F[el]
            U, S, Vt = np.linalg.svd(Fe)
            # correction de reflexion (det(R) doit valoir +1)
            det_uv = np.linalg.det(U @ Vt)
            Vt[det_uv < 0, -1, :] *= -1.0
            S[det_uv < 0, -1] *= -1.0
            R = U @ Vt
            J = np.prod(S, axis=-1)
            # corotationnel fixe : P(F) F^T = 2mu (F - R) F^T + lam J (J-1) I
            stress[el] = (2 * mu * (Fe - R) @ np.swapaxes(Fe, 1, 2)
                          + lam * (J * (J - 1))[:, None, None] * np.eye(3))
        if wa.any():
            k, gamma = MAT_WATER["k"], MAT_WATER["gamma"]
            Jw = self.Jw[wa]
            p = (k / gamma) * (Jw ** (-gamma) - 1.0)     # Tait ; p>0 si comprime
            stress[wa] = -p[:, None, None] * np.eye(3)   # sigma = -p I

        p_mass = np.where(el, MAT_ELASTIC["rho"], MAT_WATER["rho"]) * self.p_vol
        M = (-dt * self.p_vol * 4.0 * INV_DX * INV_DX) * stress
        affine = M + p_mass[:, None, None] * self.C

        # --- P2G : scatter masse + quantite de mouvement
        for i in range(3):
            for j in range(3):
                for kk in range(3):
                    off = np.array([i, j, kk], float)
                    dpos = (off - fx) * DX                       # (n,3) monde
                    weight = w[i, :, 0] * w[j, :, 1] * w[kk, :, 2]
                    mom = weight[:, None] * (p_mass[:, None] * self.v
                                             + np.einsum("nij,nj->ni", affine, dpos))
                    idx = base + np.array([i, j, kk])
                    flat = np.ravel_multi_index(idx.T, (N_GRID,) * 3)
                    np.add.at(grid_mv.reshape(-1, 3), flat, mom)
                    np.add.at(grid_m.reshape(-1), flat, weight * p_mass)

        # --- grille : v = mv/m, gravite, conditions aux limites (separating)
        mask = grid_m > 0
        grid_v = np.zeros_like(grid_mv)
        grid_v[mask] = grid_mv[mask] / grid_m[mask, None]
        grid_v[mask, 1] += dt * GRAVITY
        for axis in range(3):
            sl_lo = [slice(None)] * 3
            sl_hi = [slice(None)] * 3
            sl_lo[axis] = slice(0, BOUND)
            sl_hi[axis] = slice(N_GRID - BOUND, N_GRID)
            vlo = grid_v[tuple(sl_lo)]
            vlo[..., axis] = np.maximum(vlo[..., axis], 0.0)
            vhi = grid_v[tuple(sl_hi)]
            vhi[..., axis] = np.minimum(vhi[..., axis], 0.0)

        # --- G2P : gather vitesse + matrice affine C, advection
        new_v = np.zeros_like(self.v)
        new_B = np.zeros_like(self.C)
        for i in range(3):
            for j in range(3):
                for kk in range(3):
                    off = np.array([i, j, kk], float)
                    dpos = (off - fx) * DX
                    weight = w[i, :, 0] * w[j, :, 1] * w[kk, :, 2]
                    idx = base + np.array([i, j, kk])
                    gv = grid_v[idx[:, 0], idx[:, 1], idx[:, 2]]
                    new_v += weight[:, None] * gv
                    new_B += weight[:, None, None] * np.einsum("ni,nj->nij", gv, dpos)
        self.v = new_v
        self.C = 4.0 * INV_DX * INV_DX * new_B
        self.x = np.clip(self.x + dt * self.v,
                         BOUND * DX, 1.0 - BOUND * DX)
        # eau : J suit la divergence du champ de vitesse (trace de C)
        tr = np.trace(self.C[wa], axis1=1, axis2=2)
        self.Jw[wa] = np.clip(self.Jw[wa] * (1.0 + dt * tr), 0.5, 1.5)

    def frame(self, frame_dt=1.0 / 24):
        steps = int(np.ceil(frame_dt / self.dt))
        for _ in range(steps):
            self.substep()
        return steps


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else "jelly"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    sim = Sim()
    if scene == "jelly":
        sim.emit_box(ELASTIC, (0.35, 0.55, 0.35), (0.65, 0.85, 0.65))
    elif scene == "dam":
        sim.emit_box(WATER, (0.10, 0.10, 0.10), (0.35, 0.60, 0.90))
    elif scene == "splash":
        sim.emit_box(WATER, (0.10, 0.10, 0.10), (0.90, 0.30, 0.90))
        sim.emit_box(ELASTIC, (0.40, 0.60, 0.40), (0.60, 0.80, 0.60))
    else:
        sys.exit(f"scene inconnue : {scene}")

    t0 = time.time()
    for f in range(frames):
        steps = sim.frame()
        ke = 0.5 * sim.p_vol * 1000 * (sim.v ** 2).sum()
        ymin, ymax = sim.x[:, 1].min(), sim.x[:, 1].max()
        bad = np.isnan(sim.x).any()
        print(f"frame {f:3d} | {steps} substeps | KE {ke:9.4f} | "
              f"y [{ymin:.3f}, {ymax:.3f}] | NaN={bad}")
        if bad:
            sys.exit("EXPLOSION — divergence numerique")
    print(f"OK — {frames} frames en {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
