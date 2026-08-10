"""M17 / B5 -- verification de bq_set_body_pose : mise a jour de la pose ET de
la vitesse d'un corps CINEMATIQUE (dynamic == 0) par sous-pas/frame, sans quoi
rien ne faisait jamais avancer x/q d'un corps cinematique (k_body_predict et
k_advance_bodies ignorent explicitement dynamic == 0, cf. la tache).

Reutilise l'infrastructure ctypes de verify_contact_solve_b3b.py (memes
structures BqRigidBody/BqConfig, memes helpers box_body/set_bodies/read_bodies)
plutot que de la dupliquer -- seule bq_set_body_pose est un binding nouveau,
absent de ce module puisque hors ABI (aucune struct/signature existante
modifiee, BQ_ABI_VERSION reste a 13).

Lancement (interpreteur avec numpy) :
    python tools/repro/test_body_pose_b5.py
"""

import ctypes
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_contact_solve_b3b as b3b  # noqa: E402  reutilise dll, structures, helpers

dll = b3b.dll
BqRigidBody = b3b.BqRigidBody
make_sim = b3b.make_sim
box_body = b3b.box_body
set_bodies = b3b.set_bodies
read_bodies = b3b.read_bodies
read_sleep = b3b.read_sleep
check = b3b.check
DT_FRAME = b3b.DT_FRAME

# ---------------------------------------------------------------------------
# Binding ctypes de bq_set_body_pose (nouveau, cf. bourrasque.h)
# ---------------------------------------------------------------------------

F3 = ctypes.c_float * 3
F4 = ctypes.c_float * 4

dll.bq_set_body_pose.argtypes = [
    ctypes.c_void_p, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
]
dll.bq_set_body_pose.restype = ctypes.c_int


def set_body_pose(sim, body, x, q, v, w):
    xa = F3(*[float(c) for c in x])
    qa = F4(*[float(c) for c in q])
    va = F3(*[float(c) for c in v])
    wa = F3(*[float(c) for c in w])
    return dll.bq_set_body_pose(sim, body, xa, qa, va, wa)


def main():
    print(f"ABI = {dll.bq_abi_version()} (attendu 13, AUCUN bump pour ce lot)")
    all_ok = dll.bq_abi_version() == 13
    check(all_ok, "BQ_ABI_VERSION reste a 13")

    target_cell = 0.01

    # -----------------------------------------------------------------
    print("\n=== 3 (fait en premier, ne perturbe rien) -- REFUS ===")
    sim = make_sim()
    wall = box_body(sim, 0, (0.1, 0.3, 0.3), (0.02, 0.1, 0.1), dynamic=0,
                    target_cell=target_cell, n_per_face=4)
    box = box_body(sim, 1, (0.5, 0.3, 0.3), (0.05, 0.05, 0.05), dynamic=1, mass=1.0,
                  target_cell=target_cell)
    set_bodies(sim, [wall, box])

    rc_dyn = set_body_pose(sim, 1, (0.6, 0.3, 0.3), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))
    msg_dyn = dll.bq_last_error().decode()
    print(f"  appel sur corps DYNAMIQUE (indice 1) -> rc={rc_dyn}, message: {msg_dyn!r}")
    ok3a = check(rc_dyn == -1 and len(msg_dyn) > 0, "refuse un corps dynamic=1, message non vide")

    rc_oob_hi = set_body_pose(sim, 2, (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))
    msg_oob_hi = dll.bq_last_error().decode()
    print(f"  appel sur indice hors bornes (2, n_bodies=2) -> rc={rc_oob_hi}, message: {msg_oob_hi!r}")
    ok3b = check(rc_oob_hi == -1 and len(msg_oob_hi) > 0, "refuse un indice >= n_bodies")

    rc_oob_neg = set_body_pose(sim, -1, (0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (0, 0, 0))
    msg_oob_neg = dll.bq_last_error().decode()
    print(f"  appel sur indice negatif (-1) -> rc={rc_oob_neg}, message: {msg_oob_neg!r}")
    ok3c = check(rc_oob_neg == -1 and len(msg_oob_neg) > 0, "refuse un indice negatif")
    all_ok &= ok3a & ok3b & ok3c
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 4 -- ISOLATION : un appel sur le corps cinematique ne touche pas au corps voisin ===")
    sim = make_sim()
    wall = box_body(sim, 0, (0.1, 0.3, 0.3), (0.02, 0.1, 0.1), dynamic=0,
                    target_cell=target_cell, n_per_face=4)
    # corps dynamique voisin dote d'un etat NON trivial (v, w, q non identite)
    # pour qu'une corruption memoire ait une chance d'etre visible.
    angle = math.radians(17.0)
    qz = (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))
    box = box_body(sim, 1, (0.5, 0.3, 0.3), (0.05, 0.05, 0.05), dynamic=1, mass=1.0,
                  target_cell=target_cell, q=qz, v=(0.3, -0.2, 0.1))
    box.w[0] = 0.4; box.w[1] = -0.1; box.w[2] = 0.05
    set_bodies(sim, [wall, box])

    # NB : pas de bq_step entre les deux lectures -- isole l'ecriture memoire
    # de bq_set_body_pose de toute evolution physique, seule facon de
    # distinguer une fuite d'ecriture d'un effet du solveur.
    before = read_bodies(sim, 2)[1]
    sleep_before = read_sleep(sim, 2)

    rc = set_body_pose(sim, 0, (0.4, 0.3, 0.3), (0.7071068, 0.0, 0.7071068, 0.0),
                       (0.5, 0.0, 0.0), (0.0, 1.2, 0.0))
    ok4a = check(rc == 0, "bq_set_body_pose reussit sur le corps cinematique")

    after = read_bodies(sim, 2)[1]
    sleep_after = read_sleep(sim, 2)

    dx = np.max(np.abs(after["x"] - before["x"]))
    dq = np.max(np.abs(after["q"] - before["q"]))
    dv = np.max(np.abs(after["v"] - before["v"]))
    dw = np.max(np.abs(after["w"] - before["w"]))
    print(f"  corps 1 (dynamique) -- ecart max apres l'appel sur le corps 0 :")
    print(f"    x : {dx:.3e}  q : {dq:.3e}  v : {dv:.3e}  w : {dw:.3e}")
    print(f"  sommeil corps 0/1 avant={sleep_before} apres={sleep_after}")
    ok4b = check(dx == 0.0 and dq == 0.0 and dv == 0.0 and dw == 0.0,
                "etat du corps dynamique voisin EXACTEMENT inchange (bit a bit)")
    ok4c = check(sleep_before == sleep_after, "etat de sommeil des deux corps inchange")

    wall_state = read_bodies(sim, 2)[0]
    print(f"  corps 0 (cinematique) apres l'appel : x={wall_state['x']}, q={wall_state['q']}, "
          f"v={wall_state['v']}, w={wall_state['w']}")
    ok4d = check(np.allclose(wall_state["x"], [0.4, 0.3, 0.3], atol=1e-6),
                "la pose du corps cinematique lui-meme EST bien mise a jour")
    all_ok &= ok4a & ok4b & ok4c & ok4d
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 1 -- MUR qui avance pousse une caisse dynamique, sans fluide ===")
    sim = make_sim()
    wall_h = (0.02, 0.1, 0.1)
    box_h = (0.05, 0.05, 0.05)
    wall_x0 = 0.10
    box_x0 = 0.10 + wall_h[0] + 0.02 + box_h[0]  # gap initial = 2 cm (2 voxels)
    speed = 0.3  # m/s
    wall = box_body(sim, 0, (wall_x0, 0.3, 0.3), wall_h, dynamic=0,
                    target_cell=target_cell, n_per_face=4)
    box = box_body(sim, 1, (box_x0, 0.3, 0.3), box_h, dynamic=1, mass=1.0,
                  target_cell=target_cell, restitution=0.0, friction=0.4, use_gravity=0)
    box.lock_ang[0] = box.lock_ang[1] = box.lock_ang[2] = 1
    box.lock_lin[1] = box.lock_lin[2] = 1  # isole la poussee horizontale (axe du mur)
    set_bodies(sim, [wall, box])

    n_frames = 48  # 2 s
    box_x0_actual = read_bodies(sim, 2)[1]["x"][0]
    for f in range(1, n_frames + 1):
        wall_x = wall_x0 + speed * DT_FRAME * f
        rc = set_body_pose(sim, 0, (wall_x, 0.3, 0.3), (1.0, 0.0, 0.0, 0.0),
                           (speed, 0.0, 0.0), (0.0, 0.0, 0.0))
        if rc != 0:
            raise RuntimeError(dll.bq_last_error().decode())
        dll.bq_step(sim, DT_FRAME)

    final = read_bodies(sim, 2)
    wall_x_final = final[0]["x"][0]
    box_x_final = final[1]["x"][0]
    wall_dist = wall_x_final - wall_x0
    box_dist = box_x_final - box_x0_actual
    penetration = box_x_final - box_h[0] - (wall_x_final + wall_h[0])

    print(f"  distance parcourue par le mur   : {wall_dist*1000:.3f} mm")
    print(f"  distance parcourue par la caisse: {box_dist*1000:.3f} mm")
    print(f"  ratio caisse/mur                : {box_dist/wall_dist:.4f}")
    print(f"  penetration residuelle (jeu si >0): {penetration*1000:.4f} mm "
          f"(voxel = {target_cell*1000:.2f} mm)")
    ok1a = check(box_dist > 0.6 * wall_dist, "la caisse est entrainee sur une distance comparable au mur")
    ok1b = check(abs(penetration) < target_cell, "penetration/jeu residuel < 1 voxel")
    all_ok &= ok1a & ok1b
    dll.bq_destroy(sim)

    # -----------------------------------------------------------------
    print("\n=== 2 -- PALE qui tourne deplace une caisse posee dessus (rotation, pas seulement translation) ===")
    sim = make_sim(grid_res=48, cell_size=1.0 / 48.0)
    paddle_h = (0.3, 0.02, 0.3)
    box_h2 = (0.04, 0.04, 0.04)
    cx, cz = 0.5, 0.5
    r0 = 0.15
    omega = 1.2  # rad/s autour de l'axe MONDE y (vertical), sens +
    paddle = box_body(sim, 0, (cx, 0.0, cz), paddle_h, dynamic=0,
                      target_cell=target_cell, n_per_face=6, friction=0.7)
    box2 = box_body(sim, 1, (cx + r0, paddle_h[1] + box_h2[1] + 0.002, cz), box_h2, dynamic=1,
                   mass=0.3, target_cell=target_cell, restitution=0.0, friction=0.7)
    set_bodies(sim, [paddle, box2])

    n_frames2 = 96  # 4 s
    for f in range(1, n_frames2 + 1):
        theta = omega * DT_FRAME * f
        qy = (math.cos(theta / 2.0), 0.0, math.sin(theta / 2.0), 0.0)
        rc = set_body_pose(sim, 0, (cx, 0.0, cz), qy, (0.0, 0.0, 0.0), (0.0, omega, 0.0))
        if rc != 0:
            raise RuntimeError(dll.bq_last_error().decode())
        dll.bq_step(sim, DT_FRAME)

    st = read_bodies(sim, 2)[1]
    dx, dz = st["x"][0] - cx, st["x"][2] - cz
    angle_final = math.degrees(math.atan2(dz, dx))  # depart a 0 deg (dx=r0,dz=0)
    radius_final = math.hypot(dx, dz)
    total_paddle_deg = math.degrees(omega * DT_FRAME * n_frames2)
    print(f"  rotation totale imposee a la pale : {total_paddle_deg:.1f} deg")
    print(f"  position finale caisse (relatif centre) : dx={dx:.4f} dz={dz:.4f}, "
          f"rayon={radius_final*1000:.2f} mm (depart {r0*1000:.1f} mm)")
    print(f"  angle balaye par la caisse : {angle_final:.2f} deg (signe attendu: positif, comme omega)")
    ok2a = check(angle_final > 10.0, "la caisse a ete entrainee en rotation dans le bon sens (angle > 10 deg)")
    ok2b = check(radius_final < 3 * r0, "la caisse reste a proximite de la pale (pas ejectee)")
    all_ok &= ok2a & ok2b
    dll.bq_destroy(sim)

    print("\n" + ("TOUT OK" if all_ok else "AU MOINS UN ECHEC"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
