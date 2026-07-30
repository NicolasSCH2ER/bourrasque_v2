import sys
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
from extension import ops, props, register, unregister
from extension import lib
from extension.props import domain_transform, domain_usable_bounds, emitter_bounds_solver, world_to_solver_dir
import bpy, numpy as np

exec(open(r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\validate_turbulence.py").read().split("def main()")[0])

# monkeypatch _CurlNoise.sample to white noise (independent per particle, per call)
_orig_sample = ops._CurlNoise.sample
call_counter = {"i": 0}
def white_sample(self, points, t):
    rng = np.random.default_rng([self.seed, self.emitter_index, call_counter["i"]])
    call_counter["i"] += 1
    return rng.normal(0.0, 1.0, size=(points.shape[0], 3))
ops._CurlNoise.sample = white_sample

register()
try:
    for turb in (0.0, 0.3):
        call_counter["i"] = 0
        sc = make_scene()
        sph = setup_sphere_emitter(turbulence=turb, seed=1)
        pos, emitted, spacing, vel, axis_pt = run_inflow_bake(sc, sph, n_frames=60)
        emitted = np.array(emitted)
        print(f"[WHITE] turb={turb}: total={emitted.sum()} mean={emitted.mean():.1f}")
finally:
    unregister()
