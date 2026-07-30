import math
import os
import sys

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")

import lib  # noqa: E402

grid_res = 64
size = 1.0
ppc_axis = 2
dx = size / grid_res
step = dx / ppc_axis


def estimate(lo, hi):
    count = 1
    for axis in range(3):
        extent = hi[axis] - lo[axis]
        n = max(0, math.ceil(extent / step - 0.5))
        count *= n
    return count


cases = [
    ((0.10, 0.10, 0.10), (0.35, 0.60, 0.90)),   # dam scene, ne tombe pas juste
    ((0.0, 0.0, 0.0), (0.13, 0.13, 0.13)),       # petite boite, non alignee
    ((0.05, 0.05, 0.05), (0.9137, 0.201, 0.6003)),
    ((0.2, 0.2, 0.2), (0.2005, 0.9, 0.9)),       # extent tres petit sur un axe
    ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),          # domaine complet
    ((0.01, 0.5, 0.3), (0.99, 0.5001, 0.700001)),
]

cfg = lib.default_config()
cfg.grid_res = grid_res
cfg.domain = size
cfg.ppc_axis = ppc_axis
cfg.max_particles = 3_000_000

print(f"{'lo':>28} {'hi':>28} {'estime':>10} {'reel':>10} {'match':>7}")
all_ok = True
for lo, hi in cases:
    sim = lib.Sim(cfg)
    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4e4, gamma=3.0)
    n_real = sim.emit_box(mat_id, lo, hi)
    sim.destroy()
    n_est = estimate(lo, hi)
    ok = n_est == n_real
    all_ok &= ok
    print(f"{str(lo):>28} {str(hi):>28} {n_est:>10} {n_real:>10} {'OK' if ok else 'MISMATCH':>7}")

print()
print("TOUS CONCORDENT" if all_ok else "DIVERGENCE DETECTEE")
