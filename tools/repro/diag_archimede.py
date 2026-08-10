"""M17 / A6 — la poussee d'Archimede est-elle CALCULEE par le solveur ?

Une seule question, deux hypotheses exclusives :

  H1  la force est bonne, c'est la reponse du corps qui est tuee par le solve
      implicite (le corps est soude a la vitesse du fluide en contact).
  H2  la force n'est pas la : la recolte ne voit pas le gradient de pression.

Protocole : un cube ENTIEREMENT IMMERGE dans une cuve au repos, de masse
enorme (1e5 kg) et sans gravite propre. La masse enorme joue le role d'un
encastrement, mais SANS utiliser les verrous d'axe : ceux-ci sont appliques
APRES la resolution du systeme 6x6, on ne saurait donc pas si l'impulsion
rapportee par bq_read_collider_wrench est celle d'avant ou d'apres. Avec
m_b -> infini, v_new -> v_b et m_b*(v_new - v_b) tend vers l'impulsion
reellement transmise : la lecture est non ambigue.

On mesure a DEUX profondeurs : la poussee d'Archimede n'en depend pas. Si la
mesure, elle, en depend fortement, c'est un indice fort en faveur de H2.

Lancement (interpreteur qui a numpy) :
    python tools/repro/diag_archimede.py
"""

import ctypes
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

GRAVITY = -9.81
RHO_WATER = 1000.0
BULK = 2.0e5          # c = 14 m/s : ~2 % de compression sous 0.46 m, eau credible
GAMMA = 7.0

CELL = 0.02
RES = (32, 48, 32)                      # 0.64 x 0.96 x 0.64 m
# La bande de bord du solveur (bound = 3 cellules) interdit les 0.06 m de
# pourtour : emettre dedans est refuse par bq_emit_points.
WATER_LO = (0.08, 0.08, 0.08)
WATER_HI = (0.56, 0.50, 0.56)           # surface libre a y = 0.50
BODY_L = 0.12                           # cube plein
SPACING = CELL / 2.0                    # ppc_axis = 2

SETTLE_FRAMES = 150                      # ~1.25 s : etablissement hydrostatique
MEASURE_FRAMES = 30
FRAME_DT = 1.0 / 24.0


def cube_triangles(center, side):
    """12 triangles d'un cube plein, normales SORTANTES (indispensable : le
    signe du SDF collider en depend)."""
    c = np.asarray(center, dtype=np.float64)
    h = side / 2.0
    s = np.array([[-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
                  [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]],
                 dtype=np.float64)
    v = c + s * h
    faces = [
        (0, 3, 2), (0, 2, 1),   # z-
        (4, 5, 6), (4, 6, 7),   # z+
        (0, 1, 5), (0, 5, 4),   # y-
        (3, 7, 6), (3, 6, 2),   # y+
        (0, 4, 7), (0, 7, 3),   # x-
        (1, 2, 6), (1, 6, 5),   # x+
    ]
    return np.array([[v[a], v[b], v[c_]] for a, b, c_ in faces], dtype=np.float32)


def water_points(body_center, side):
    """Reseau de particules remplissant la cuve, EXCLUANT le volume du corps
    (plus une marge d'un espacement : des particules nees dans le solide
    seraient expulsees violemment et pollueraient la mesure)."""
    axes = [np.arange(lo + SPACING / 2, hi, SPACING)
            for lo, hi in zip(WATER_LO, WATER_HI)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    half = side / 2.0 + SPACING
    inside = np.all(np.abs(grid - np.asarray(body_center)) < half, axis=1)
    return np.ascontiguousarray(grid[~inside], dtype=np.float32)


def measure(body_center, label, side=BODY_L):
    cfg = lib.BqConfig()
    cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2] = RES
    cfg.cell_size = CELL
    cfg.gravity_y = GRAVITY
    cfg.cfl = 0.4
    cfg.ppc_axis = 2
    cfg.max_particles = 400000

    pts = water_points(body_center, side)
    tri = cube_triangles(body_center, side)
    n_tri = tri.shape[0]

    mass = 1.0e5                                  # encastrement "par la masse"
    inertia = mass * side * side / 6.0        # cube plein
    inv_i = np.zeros(9, dtype=np.float32)
    inv_i[0] = inv_i[4] = inv_i[8] = 1.0 / inertia

    body = lib.BqRigidBody()
    body.dynamic = 1
    body.mass = mass
    body.inv_inertia = (ctypes.c_float * 9)(*inv_i)
    body.x = (ctypes.c_float * 3)(*body_center)
    body.q = (ctypes.c_float * 4)(1.0, 0.0, 0.0, 0.0)
    body.use_gravity = 0                          # on ne veut QUE la force du fluide
    body.added_mass = 0.0
    body.restitution = 0.0

    with lib.Sim(cfg) as sim:
        sim.add_material(lib.BQ_MODEL_WATER, RHO_WATER, bulk=BULK, gamma=GAMMA)
        sim.emit_points(0, pts)
        sim.set_collider_bodies([body])
        sim.set_colliders(tri,
                          np.zeros_like(tri),
                          np.zeros(n_tri, dtype=np.float32),
                          np.zeros(n_tri, dtype=np.int32))

        for i in range(SETTLE_FRAMES):
            sim.step(FRAME_DT)
            if (i + 1) % 25 == 0:
                sp = np.linalg.norm(sim.read_velocities(), axis=1)
                print(f"    tassement frame {i+1:4d} : |v| moyen {sp.mean():.4f} "
                      f"m/s, max {sp.max():.3f}")

        fy, smass, sub_n = [], [], 0
        for _ in range(MEASURE_FRAMES):
            substeps = sim.step(FRAME_DT)
            sub_n = substeps
            w = sim.read_collider_wrench()[0]
            dt = FRAME_DT / substeps
            fy.append(w[1] / dt)                  # impulsion verticale -> force
            smass.append(w[6])

        state = sim.read_collider_bodies()[0]
        sp = np.linalg.norm(sim.read_velocities(), axis=1)
        v_rms = float(np.sqrt((sp ** 2).mean()))

    fy = np.array(fy)
    v_body = side ** 3
    f_arch = RHO_WATER * v_body * (-GRAVITY)

    print(f"\n===== {label} : centre y = {body_center[1]:.3f} m "
          f"cote {side:.2f} m ({side/CELL:.0f} cellules) =====")
    print(f"  particules            {pts.shape[0]}   sous-pas/frame {sub_n}")
    print(f"  volume du corps       {v_body:.6f} m^3")
    print(f"  F_archimede attendue  {f_arch:.3f} N")
    print(f"  F_y mesuree           {fy.mean():.3f} N   "
          f"(min {fy.min():.3f} / max {fy.max():.3f})")
    print(f"  RAPPORT mesure/attendu  {fy.mean() / f_arch:.4f}   <<<<<")
    print(f"  S_m (fluide en contact) {np.mean(smass):.4f} kg")
    h = len(fy) // 2
    print(f"  F_y 1re moitie {fy[:h].mean():8.3f} N  |  2e moitie "
          f"{fy[h:].mean():8.3f} N   <- doit etre stable")
    print(f"  agitation residuelle du fluide : |v|_rms = {v_rms:.4f} m/s")
    if v_rms > 0.05:
        print("  /!\  fluide NON au repos : la mesure ne vaut rien.")
    print(f"  deplacement du corps    {state[1] - body_center[1]:+.6f} m")
    return fy.mean() / f_arch


# ---------------------------------------------------------------------------
# Test de flottaison de bout en bout : le critere de sortie de la phase A.
# ---------------------------------------------------------------------------

def float_test(density, side, y0, water_top):
    """Corps LIBRE (masse reelle, gravite active) lache entierement immerge.
    Doit remonter et se stabiliser avec une fraction immergee ~ density/1000.

    Contrairement aux mesures de force ci-dessus, on n'encastre PAS le corps :
    c'est sa REPONSE que l'on teste, celle que le solve implicite est soupconne
    de tuer en soudant le corps a la vitesse du fluide en contact."""
    global WATER_HI
    WATER_HI = (0.56, water_top, 0.56)

    cfg = lib.BqConfig()
    cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2] = RES
    cfg.cell_size = CELL
    cfg.gravity_y = GRAVITY
    cfg.cfl = 0.4
    cfg.ppc_axis = 2
    cfg.max_particles = 400000

    center = (0.32, y0, 0.32)
    pts = water_points(center, side)
    tri = cube_triangles(center, side)
    n_tri = tri.shape[0]

    mass = density * side ** 3
    inertia = mass * side * side / 6.0
    inv_i = np.zeros(9, dtype=np.float32)
    inv_i[0] = inv_i[4] = inv_i[8] = 1.0 / inertia

    body = lib.BqRigidBody()
    body.dynamic = 1
    body.mass = mass
    body.inv_inertia = (ctypes.c_float * 9)(*inv_i)
    body.x = (ctypes.c_float * 3)(*center)
    body.q = (ctypes.c_float * 4)(1.0, 0.0, 0.0, 0.0)
    body.use_gravity = 1
    body.added_mass = 0.0
    body.restitution = 0.0

    print(f"\n===== FLOTTAISON : densite {density:.0f}, cote {side:.2f} m "
          f"({side/CELL:.0f} cellules), surface libre y = {water_top:.2f} =====")
    print(f"  masse {mass:.3f} kg   fraction immergee attendue "
          f"{density/RHO_WATER:.2f}")

    traj = []
    with lib.Sim(cfg) as sim:
        sim.add_material(lib.BQ_MODEL_WATER, RHO_WATER, bulk=BULK, gamma=GAMMA)
        sim.emit_points(0, pts)
        sim.set_collider_bodies([body])
        for f in range(220):
            # Geometrie du collider retransmise a la position COURANTE du corps :
            # le champ vu par le fluide n'est rafraichi qu'une fois par frame
            # (D4), sans quoi le corps se detache de sa propre geometrie.
            st = sim.read_collider_bodies()[0]
            c = (float(st[0]), float(st[1]), float(st[2]))
            sim.set_colliders(cube_triangles(c, side), np.zeros((n_tri, 3, 3), np.float32),
                              np.zeros(n_tri, np.float32), np.zeros(n_tri, np.int32))
            sim.step(FRAME_DT)
            if (f + 1) % 20 == 0:
                st = sim.read_collider_bodies()[0]
                traj.append((f + 1, float(st[1]), float(st[7 + 1])))
                print(f"    frame {f+1:4d} : y = {st[1]:.4f} m   vy = {st[8]:+.4f} m/s")

    y_fin = traj[-1][1]
    top, bot = y_fin + side / 2, y_fin - side / 2
    imm = float(np.clip((min(top, water_top) - bot) / side, 0.0, 1.0))
    print(f"  y initial {y0:.4f} -> y final {y_fin:.4f}  (surface {water_top:.2f})")
    print(f"  fraction immergee mesuree {imm:.3f}   attendue "
          f"{density/RHO_WATER:.3f}")
    return imm


if __name__ == "__main__":
    # Le corps est bien resolu (12 cellules), la ou la sur-poussee mesuree
    # tombe a ~1.07x l'analytique. Densite 500 : il doit flotter a mi-immersion.
    float_test(density=500.0, side=0.24, y0=0.22, water_top=0.44)
