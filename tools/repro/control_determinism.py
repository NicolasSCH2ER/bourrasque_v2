"""Controle : le solveur est-il deterministe d'une execution a l'autre ?

Si deux executions identiques divergent du meme ordre de grandeur que l'ecart
frontend/API-directe mesure en V4 (2.61e-04), alors cet ecart vient des
atomicAdd du P2G et non du frontend Blender.
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")
import lib


def run():
    cfg = lib.default_config()
    cfg.grid_res = 64
    cfg.ppc_axis = 2
    cfg.domain = 1.0
    cfg.cfl = 0.3
    cfg.gravity_y = -9.8
    with lib.Sim(cfg) as sim:
        m = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
        sim.emit_box(m, (0.10, 0.10, 0.10), (0.35, 0.60, 0.90), (0.0, 0.0, 0.0))
        frames = []
        for _ in range(10):
            sim.step(1.0 / 24.0)
            frames.append(np.array(sim.read_positions(), copy=True))
        return sim.particle_count, frames


n1, a = run()
n2, b = run()
print(f"particules : run1={n1} run2={n2}")
assert n1 == n2

worst = 0.0
for i, (fa, fb) in enumerate(zip(a, b)):
    d = float(np.max(np.abs(fa - fb)))
    worst = max(worst, d)
    print(f"frame {i:2d} : ecart max = {d:.3e}")

print(f"\necart max sur 10 frames, meme code, deux executions : {worst:.3e}")
print(f"ecart frontend Blender / API directe (mesure V4)      : 2.610e-04")
if worst == 0.0:
    print("\n=> solveur DETERMINISTE : l'ecart V4 vient bien du frontend, a investiguer.")
elif worst >= 2.61e-04 / 10.0:
    print("\n=> non-determinisme du meme ordre : l'ecart V4 est du bruit atomicAdd, pas le frontend.")
else:
    print("\n=> non-determinisme trop faible pour expliquer l'ecart V4, a investiguer.")
