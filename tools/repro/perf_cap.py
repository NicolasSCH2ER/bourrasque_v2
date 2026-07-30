import sys, time
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
from extension import ops, props, register, unregister
from extension import lib
from extension.props import domain_transform, domain_usable_bounds, emitter_bounds_solver, world_to_solver_dir
import bpy, numpy as np

exec(open(r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\validate_turbulence.py").read().split("def main()")[0])

register()
try:
    for turb in (0.0, 0.3):
        sc = make_scene()
        sph = setup_sphere_emitter(turbulence=turb, seed=1)
        t0 = time.perf_counter()
        pos, emitted, spacing, vel, axis_pt = run_inflow_bake(sc, sph, n_frames=60)
        t1 = time.perf_counter()
        print(f"turb={turb}: total_time={t1-t0:.3f}s total_emitted={sum(emitted)}")
finally:
    unregister()
