"""M17 — la poussee suit-elle le VOLUME IMMERGE pres de la surface libre ?

Etat des connaissances avant ce script (cf. docs/plan-milestone-17.md, A6) :
 - la poussee existe et est independante de la profondeur -> la recolte voit
   bien le gradient de pression ;
 - elle est surestimee ~4x sur un corps de 6 cellules, ~1.07x a 12 cellules :
   c'est un effet de RESOLUTION, pas de formulation ;
 - un corps libre lache immerge REMONTE (13 cm en 1 s) puis redescend et finit
   immerge : ce n'est donc ni une force absente, ni un corps inerte.

Reste une seule question : le corps n'arrive pas a TENIR un equilibre au
voisinage de la surface libre. Deux causes possibles, exclusives :

  C1  la force est correcte a toute immersion partielle, et le defaut est
      DYNAMIQUE (l'eau qui deferle sur le dessus l'alourdit, ou le couplage
      amortit trop). La formulation serait alors saine.
  C2  la force ne suit pas le volume immerge quand le corps affleure : elle
      s'effondre plus vite que l'immersion. Le defaut serait alors dans la
      recolte pres de la surface.

Protocole : corps MAINTENU (masse enorme, gravite propre coupee) a plusieurs
hauteurs a cheval sur la surface. On compare la force verticale mesuree a
rho * V_immerge * g, ou V_immerge est calcule sur la surface libre REELLE
mesuree sur les particules -- pas sur la hauteur d'eau nominale, que le corps
lui-meme fait monter en deplacant du volume.

Et, comme au jalon precedent : on refuse de lire une force tant que le fluide
n'est pas au repos. Un balayage entier a deja du etre jete pour l'avoir oublie.

    python tools/repro/diag_flottaison_surface.py
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

GRAVITY = -9.81
RHO = 1000.0
BULK, GAMMA = 2.0e5, 7.0

# --- Domaine QUASI-2D : une tranche mince en z ------------------------------
#
# Premiere version de ce script, jetee : cuve 0.48 x 0.48 avec un corps de
# 0.24 m. Le corps occupait une fraction enorme du bassin, son fond n'etait qu'a
# 4 cm du plancher, et la surface libre tombait a 0.32 m au lieu de 0.44 -- ce
# n'etait pas une experience de flottaison mais un piston dans un cylindre. Les
# rapports mesures (2.97 / 1.71 / 122) n'y voulaient rien dire, le dernier
# n'etant qu'une division par un zero.
#
# Il faut un corps PETIT devant la cuve ET bien resolu : contradictoire a ce
# cout en 3D (il faudrait ~10^6 particules). D'ou la tranche mince : le rapport
# corps/cuve redevient correct pour six fois moins de particules, et la poussee
# reste Archimede par unite de profondeur. Les parois z du domaine sont
# glissantes, ce qui est exactement la condition voulue en quasi-2D.
CELL = 0.02
# La bande de bord (bound = 3 cellules = 0.06 m de chaque cote) mange
# l'epaisseur : une tranche de 10 cellules n'en laissait que 4 d'utilisables, et
# l'eau, emise sur 4 cm seulement, s'etalait dans le reste -- le niveau chutait
# de moitie et toutes les fractions immergees ressortaient nulles. La tranche
# fait donc 16 cellules, et l'eau la remplit sur toute son epaisseur utile.
RES = (48, 48, 16)               # 0.96 x 0.96 x 0.32 m
WATER_LO = (0.08, 0.08, 0.08)
WATER_TOP = 0.70
WATER_HI = (0.88, WATER_TOP, 0.24)
SIDE = 0.16                      # 8 cellules de cote, contre 40 de large pour la cuve
SPAN_Z = 0.20                    # le corps traverse toute la tranche
SPACING = CELL / 2.0
FRAME_DT = 1.0 / 24.0
SETTLE, MEASURE = 130, 25
REPOS = 0.05


def cube_tris(c, s):
    """Boite fermee, carree en (x, y) et traversant toute la tranche en z."""
    c = np.asarray(c, float)
    h = np.array([s / 2.0, s / 2.0, SPAN_Z])   # deborde en z : quasi-2D
    sg = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                   [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    v = c + sg * h
    f = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
         (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return np.array([[v[a], v[b], v[c_]] for a, b, c_ in f], dtype=np.float32)


def water_points(center):
    ax = [np.arange(lo + SPACING / 2, hi, SPACING)
          for lo, hi in zip(WATER_LO, WATER_HI)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    half = np.array([SIDE / 2.0 + SPACING, SIDE / 2.0 + SPACING, 1e9])
    return np.ascontiguousarray(
        g[~np.all(np.abs(g - np.asarray(center)) < half, axis=1)], dtype=np.float32)


def surface_reelle(pos, center):
    """Niveau de la surface libre mesure LOIN du corps : le corps deplace du
    volume et fait monter l'eau, utiliser la hauteur nominale fausserait le
    volume immerge de reference."""
    r = np.abs(pos[:, 0] - center[0])   # z est traverse par le corps
    loin = pos[r > SIDE / 2.0 + 2 * CELL]
    return float(np.percentile(loin[:, 1], 98)) if len(loin) > 200 else float("nan")


def mesure(cy):
    cfg = lib.BqConfig()
    cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2] = RES
    cfg.cell_size, cfg.gravity_y, cfg.cfl = CELL, GRAVITY, 0.4
    cfg.ppc_axis, cfg.max_particles = 2, 400000

    # centre en x du bassin (eau de 0.08 a 0.88), et au milieu de la tranche en z
    center = (0.48, cy, RES[2] * CELL / 2.0)
    pts = water_points(center)
    tri = cube_tris(center, SIDE)
    n_tri = tri.shape[0]

    mass = 1.0e5
    inv_i = np.zeros(9, np.float32)
    inv_i[0] = inv_i[4] = inv_i[8] = 6.0 / (mass * SIDE * SIDE)  # ordre de grandeur suffit

    b = lib.BqRigidBody()
    b.dynamic, b.mass = 1, mass
    b.inv_inertia = (ctypes.c_float * 9)(*inv_i)
    b.x = (ctypes.c_float * 3)(*center)
    b.q = (ctypes.c_float * 4)(1.0, 0.0, 0.0, 0.0)
    b.use_gravity = 0          # on ne veut QUE la force du fluide
    b.added_mass = 0.0

    with lib.Sim(cfg) as sim:
        sim.add_material(lib.BQ_MODEL_WATER, RHO, bulk=BULK, gamma=GAMMA)
        sim.emit_points(0, pts)
        sim.set_collider_bodies([b])
        sim.set_colliders(tri, np.zeros_like(tri),
                          np.zeros(n_tri, np.float32), np.zeros(n_tri, np.int32))
        for _ in range(SETTLE):
            sim.step(FRAME_DT)
        fy = []
        for _ in range(MEASURE):
            sub = sim.step(FRAME_DT)
            fy.append(sim.read_collider_wrench()[0][1] / (FRAME_DT / sub))
        pos = sim.read_positions().copy()
        v_rms = float(np.sqrt((np.linalg.norm(sim.read_velocities(), axis=1) ** 2).mean()))

    fy = np.array(fy)
    surf = surface_reelle(pos, center)
    bas, haut = cy - SIDE / 2.0, cy + SIDE / 2.0
    frac = float(np.clip((surf - bas) / SIDE, 0.0, 1.0))
    # le corps traverse la tranche, mais seule l'epaisseur REMPLIE D'EAU
    # deplace du volume
    v_corps = SIDE * SIDE * (WATER_HI[2] - WATER_LO[2])
    f_att = RHO * (v_corps * frac) * (-GRAVITY)
    ratio = fy.mean() / f_att if f_att > 1e-9 else float("nan")

    print(f"\n--- centre y = {cy:.3f} m ---")
    print(f"  surface libre mesuree {surf:.4f} m (nominale {WATER_TOP:.2f})")
    print(f"  fraction immergee     {frac:.3f}")
    print(f"  F attendue            {f_att:8.3f} N")
    print(f"  F mesuree             {fy.mean():8.3f} N  (1re moitie "
          f"{fy[:len(fy)//2].mean():.2f} / 2e {fy[len(fy)//2:].mean():.2f})")
    print(f"  RAPPORT               {ratio:7.3f}")
    print(f"  |v|_rms {v_rms:.4f} m/s  ->  {'au repos' if v_rms < REPOS else 'EN MOUVEMENT'}")
    if v_rms >= REPOS:
        print("  /!\\ fluide non pose : mesure a ne pas exploiter.")
    return cy, frac, ratio, v_rms < REPOS


if __name__ == "__main__":
    # Hauteurs calees sur la surface libre REELLE (~0.547 m), pas sur la
    # nominale (0.70). L'eau emise sur 0.80 x 0.16 s'etale dans les
    # 0.84 x 0.20 utilisables du domaine : 0.62 * 0.762 = 0.47 m de colonne
    # au-dessus du plancher a 0.08. Le premier jeu de hauteurs, cale sur 0.70,
    # placait deux corps sur trois entierement HORS de l'eau -- force nulle,
    # ce qui est le bon resultat mais n'apprend rien.
    res = [mesure(cy) for cy in (0.46, 0.52, 0.57)]

    print("\n================ VERDICT ================")
    print(f"  {'centre y':>9} {'immerge':>9} {'rapport':>9} {'repos':>7}")
    for cy, frac, r, ok in res:
        print(f"  {cy:9.3f} {frac:9.3f} {r:9.3f} {'oui' if ok else 'NON':>7}")
    bons = [r for _, _, r, ok in res if ok and np.isfinite(r)]
    if len(bons) < 2:
        print("\n  Trop de mesures invalides : verdict impossible.")
        sys.exit(1)
    ecart = (max(bons) - min(bons)) / max(min(bons), 1e-9)
    print(f"\n  dispersion du rapport : {ecart*100:.0f} %")
    if ecart < 0.35:
        print("  -> C1 : la force SUIT le volume immerge a toute immersion.")
        print("     La formulation est saine ; le defaut est DYNAMIQUE")
        print("     (drainage sur le dessus, ou amortissement du couplage).")
    else:
        print("  -> C2 : la force NE suit PAS le volume immerge.")
        print("     Le defaut est dans la recolte pres de la surface libre.")
