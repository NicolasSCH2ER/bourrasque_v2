"""M18 / S4 — porte du jalon : l'angle de talus suit-il l'angle de frottement ?

Une colonne de sable lachee sur un sol plat s'effondre en un tas dont la pente
vaut, en theorie, l'angle de frottement interne du materiau. C'est l'equivalent
d'Archimede pour le sable : mesurable, comparable a un parametre analytique, et
directement visible a l'ecran.

LECON DU JALON PRECEDENT, appliquee ici d'emblee : un tas encore en mouvement
n'est pas un tas. Une campagne entiere de mesures de poussee d'Archimede a du
etre jetee parce qu'elle lisait un regime transitoire. On instrumente donc
l'agitation residuelle des particules, et on REFUSE de lire un angle tant
qu'elle n'est pas retombee.

On ne demande pas l'egalite exacte a l'analytique -- la discretisation, la
friction de paroi et la taille finie du tas biaisent toujours. On demande une
dependance MONOTONE et du bon ordre de grandeur.

Lancement :
    python tools/repro/diag_angle_talus.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# `extension/__init__.py` importe bpy : on charge lib.py PAR CHEMIN.
_spec = importlib.util.spec_from_file_location("bq_lib", ROOT / "extension" / "lib.py")
lib = importlib.util.module_from_spec(_spec)
sys.modules["bq_lib"] = lib
_spec.loader.exec_module(lib)

CELL = 0.02
RES = (48, 48, 48)                 # 0.96 m de cote
BOUND = 3                          # bande de bord du solveur

# Le sol doit etre RUGUEUX. Premiere version de ce test : on utilisait la paroi
# du domaine comme sol, pour eviter de meler la dette D6 (contrainte de contact
# qui ne met pas F a jour) a la mesure. Erreur de protocole : cette paroi est
# parfaitement GLISSANTE -- k_grid_update annule la composante normale et ne
# touche pas a la tangentielle. Le tas s'etalait alors jusqu'a 0.55 m de rayon
# pour 4 cm de haut, avec des angles de 4 a 7 deg non monotones : on mesurait le
# frottement du SOL (nul), pas celui du sable.
#
# D'ou un vrai collider, et FERME (boite pleine, pas un quad) : un quad ouvert
# casse la propagation de signe du SDF et rend le champ inutilisable.
FLOOR_LO, FLOOR_HI = 0.08, 0.16    # boite-sol
FLOOR = FLOOR_HI                   # le sable repose dessus
FLOOR_FRICTION = 1.0               # sol rugueux : c'est le sable qui doit ceder
SPACING = CELL / 2.0               # ppc_axis = 2
FRAME_DT = 1.0 / 24.0

COL_R = 0.15                       # colonne cylindrique lachee
COL_H = 0.30
CENTER = (0.48, 0.48)
DOM_LO, DOM_HI = 0.08, 0.88              # x, z

RHO, YOUNG, POISSON = 1600.0, 3.5e5, 0.3

MAX_FRAMES = 400
REPOS_SEUIL = 0.02                 # m/s : sous ce seuil, le tas est considere pose


def boite_sol():
    """12 triangles d'une boite FERMEE servant de sol rugueux, normales
    sortantes (le signe du SDF collider en depend)."""
    lo = np.array([DOM_LO, FLOOR_LO, DOM_LO])
    hi = np.array([DOM_HI, FLOOR_HI, DOM_HI])
    v = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]],
                  [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
                  [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
             (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float32)


def colonne():
    ax = np.arange(FLOOR + SPACING / 2, FLOOR + COL_H, SPACING)
    lat = np.arange(-COL_R, COL_R, SPACING)
    X, Y, Z = np.meshgrid(lat, ax, lat, indexing="ij")
    pts = np.stack([X.ravel() + CENTER[0], Y.ravel(), Z.ravel() + CENTER[1]], axis=-1)
    r = np.hypot(pts[:, 0] - CENTER[0], pts[:, 2] - CENTER[1])
    return np.ascontiguousarray(pts[r <= COL_R], dtype=np.float32)


def angles_du_tas(pos):
    """Deux estimateurs INDEPENDANTS de la pente, rapportes tous les deux.

    Un seul estimateur peut etre biaise par sa propre definition ; s'ils
    divergent, c'est le signal qu'aucun des deux n'est fiable."""
    y = pos[:, 1] - FLOOR
    cx, cz = pos[:, 0].mean(), pos[:, 2].mean()
    r = np.hypot(pos[:, 0] - cx, pos[:, 2] - cz)

    # 1. cone equivalent : hauteur au sommet contre rayon du pied
    h = np.percentile(y, 99.5)
    rad = np.percentile(r, 99.0)
    a_cone = np.degrees(np.arctan2(h, rad))

    # 2. regression de la surface libre sur la moitie EXTERIEURE du tas, la ou
    #    la pente est reellement le talus (le sommet est arrondi, le pied
    #    s'etale : les deux bouts biaisent).
    bins = np.linspace(0, rad, 13)
    rc, hc = [], []
    for i in range(len(bins) - 1):
        m = (r >= bins[i]) & (r < bins[i + 1])
        if m.sum() >= 20:
            rc.append(0.5 * (bins[i] + bins[i + 1]))
            hc.append(np.percentile(y[m], 95))
    rc, hc = np.array(rc), np.array(hc)
    sel = rc >= 0.4 * rad
    if sel.sum() >= 3:
        pente = np.polyfit(rc[sel], hc[sel], 1)[0]
        a_fit = np.degrees(np.arctan(max(-pente, 0.0)))
    else:
        a_fit = float("nan")
    return a_cone, a_fit, h, rad


def run(phi):
    cfg = lib.BqConfig()
    cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2] = RES
    cfg.cell_size = CELL
    cfg.gravity_y = -9.81
    cfg.cfl = 0.4
    cfg.ppc_axis = 2
    cfg.max_particles = 300000

    pts = colonne()
    print(f"\n===== phi = {phi:.0f} deg   ({pts.shape[0]} particules) =====")
    with lib.Sim(cfg) as sim:
        sim.add_material(lib.BQ_MODEL_SAND, RHO, E=YOUNG, nu=POISSON,
                         friction_angle=phi)
        tri = boite_sol()
        n_tri = tri.shape[0]
        sim.set_colliders(tri, np.zeros_like(tri),
                          np.full(n_tri, FLOOR_FRICTION, dtype=np.float32),
                          np.zeros(n_tri, dtype=np.int32))
        sim.emit_points(0, pts)

        pose_a = None
        for f in range(MAX_FRAMES):
            sim.step(FRAME_DT)
            if (f + 1) % 10 == 0:
                v = np.linalg.norm(sim.read_velocities(), axis=1).mean()
                if (f + 1) % 50 == 0:
                    print(f"    frame {f+1:4d} : |v| moyen {v:.4f} m/s")
                if v < REPOS_SEUIL:
                    pose_a = f + 1
                    print(f"    repos atteint frame {pose_a} (|v| = {v:.4f} m/s)")
                    break
        v_fin = float(np.linalg.norm(sim.read_velocities(), axis=1).mean())
        pos = sim.read_positions().copy()

    a_cone, a_fit, h, rad = angles_du_tas(pos)
    repos = v_fin < REPOS_SEUIL
    print(f"  hauteur {h:.4f} m   rayon {rad:.4f} m")
    print(f"  angle (cone equivalent) {a_cone:6.2f} deg")
    print(f"  angle (regression surface) {a_fit:6.2f} deg")
    print(f"  |v| final {v_fin:.4f} m/s  ->  {'AU REPOS' if repos else 'ENCORE EN MOUVEMENT'}")
    if not repos:
        print("  /!\\  tas non pose : l'angle lu ci-dessus ne vaut RIEN.")
    return phi, a_cone, a_fit, repos


if __name__ == "__main__":
    res = [run(p) for p in (25.0, 35.0, 45.0)]

    print("\n================ VERDICT ================")
    print(f"  {'phi vise':>9} {'cone':>8} {'regression':>12} {'repos ?':>9}")
    for phi, ac, af, ok in res:
        print(f"  {phi:9.0f} {ac:8.2f} {af:12.2f} {'oui' if ok else 'NON':>9}")

    if not all(ok for *_, ok in res):
        print("\n  Au moins un tas n'etait pas pose : verdict IMPOSSIBLE a rendre.")
        sys.exit(1)

    cones = [ac for _, ac, _, _ in res]
    fits = [af for _, _, af, _ in res]
    mono_c = all(b > a for a, b in zip(cones, cones[1:]))
    mono_f = all(b > a for a, b in zip(fits, fits[1:]))
    print(f"\n  monotone (cone)       : {'oui' if mono_c else 'NON'}")
    print(f"  monotone (regression) : {'oui' if mono_f else 'NON'}")
    if mono_c and mono_f:
        print("  -> l'angle de talus SUIT l'angle de frottement : porte franchie.")
    else:
        print("  -> pas de dependance monotone : porte NON franchie.")
