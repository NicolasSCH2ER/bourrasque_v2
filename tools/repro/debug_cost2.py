import sys, time
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
from extension import ops, props, register, unregister
from extension import lib
from extension.props import domain_transform, domain_usable_bounds, emitter_bounds_solver, world_to_solver_dir
import bpy, numpy as np

exec(open(r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\validate_turbulence.py").read().split("def main()")[0])

register()
try:
    from extension import sampling
    for turb in (0.0, 0.3):
        sc = make_scene(grid_res=64, ppc=2)
        sph = setup_sphere_emitter(radius=0.4, turbulence=turb, seed=1)
        cfg, origin, size = default_config(sc)
        sim = lib.Sim(cfg)
        mat_id = sim.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
        dx = size / sc.bourrasque.grid_res
        spacing = dx / sc.bourrasque.ppc_axis
        op = sph.bourrasque
        vel = world_to_solver_dir(tuple(op.initial_velocity))
        sites = sampling.sample_mesh_interior(sph, origin, size, sc.bourrasque.grid_res, sc.bourrasque.ppc_axis)
        frame_dt = sc.render.fps_base / sc.render.fps
        bbox_lo, bbox_hi = sites.min(axis=0), sites.max(axis=0)
        noise = ops._CurlNoise(op.turbulence_seed, 0, bbox_lo, bbox_hi, dx, vel, spacing, frame_dt)
        state = ops._InflowState(sph.name, mat_id, vel, spacing, sites, dx=dx,
                                  turbulence=op.turbulence, turbulence_seed=op.turbulence_seed,
                                  emitter_index=0, noise=noise)
        usable = props.domain_usable_bounds(sc)
        harness = _Harness(sim, [state], usable, frame_dt)
        harness._frame_index = 0
        harness._emit_inflow_sites()
        sim.step(frame_dt)
        times = []
        for f in range(1, 40):
            harness._frame_index = f
            t0 = time.perf_counter()
            harness._emit_inflow_sites()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            sim.step(frame_dt)
        times = np.array(times)
        print(f"turb={turb} n sites={sites.shape[0]}: mean={times.mean():.3f}ms  max={times.max():.3f}ms")
        sim.destroy()
finally:
    unregister()
