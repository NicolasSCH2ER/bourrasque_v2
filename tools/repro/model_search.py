"""Explore plusieurs modeles de vitesse de sortie pour estimate_inflow_count,
avec le PLAFOND en place (ops.py de production actuel). Ne modifie rien dans
le depot -- reimplemente l'estimation en Python pur, parametree par le choix
de v_exit, et compare au compte REEL mesure par le harnais (identique a
validate_gravity_fix_capcompat.py)."""
import math
import sys

import bpy

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

from extension import ops, props, register, unregister
from extension import lib
from extension.props import emitter_bounds_solver, world_to_solver_dir, domain_transform


class _Harness:
    _emit_inflow_sites = ops.BQ_OT_bake._emit_inflow_sites

    def __init__(self, sim, inflow_states, usable_bounds, frame_dt=0.0):
        self._sim = sim
        self._inflow_states = inflow_states
        self._usable_bounds = usable_bounds
        self._saturated = False
        self._frame_index = 0
        self._frame_dt = frame_dt
        self.reports = []

    def report(self, level, msg):
        self.reports.append((level, msg))


def make_scene(gravity, domain_size=4.0, frame_end=15, grid_res=48):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1.0

    bpy.ops.mesh.primitive_cube_add(size=domain_size, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = grid_res
    scene.bourrasque.ppc_axis = 2
    scene.bourrasque.gravity = gravity
    scene.bourrasque.cfl = 0.4
    scene.bourrasque.max_particles = 2_000_000
    scene.bourrasque.frame_start = 1
    scene.bourrasque.frame_end = frame_end
    return scene


def setup_box_emitter(scene, size=(0.3, 0.3, 0.3), location=(0.0, 0.0, 1.0), speed=(0.0, 0.0, -0.5)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.scale = size
    bpy.context.view_layer.update()
    obj.name = "BoxEmitter"
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.model = "WATER"
    obj.bourrasque.emit_mode = "INFLOW"
    obj.bourrasque.emit_source = "BOUNDS"
    obj.bourrasque.initial_velocity = speed
    return obj


def default_config(scene):
    origin, size = domain_transform(scene)
    cfg = lib.default_config()
    cfg.grid_res = scene.bourrasque.grid_res
    cfg.domain = size
    cfg.gravity_y = scene.bourrasque.gravity
    cfg.cfl = scene.bourrasque.cfl
    cfg.ppc_axis = scene.bourrasque.ppc_axis
    cfg.max_particles = scene.bourrasque.max_particles
    return cfg, origin, size


def run_bake(scene, obj, n_frames):
    cfg, origin, size = default_config(scene)
    sim = lib.Sim(cfg)
    mat_id = sim.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)

    dx = size / scene.bourrasque.grid_res
    spacing = dx / scene.bourrasque.ppc_axis
    vel = world_to_solver_dir(tuple(obj.bourrasque.initial_velocity))
    lo, hi = emitter_bounds_solver(obj, origin, size)
    sites = ops._lattice_points_in_box(lo, hi, spacing)
    state = ops._InflowState(obj.name, mat_id, vel, spacing, sites)

    usable = props.domain_usable_bounds(scene)
    frame_dt = scene.render.fps_base / scene.render.fps
    harness = _Harness(sim, [state], usable, frame_dt=frame_dt)

    for f in range(n_frames):
        harness._frame_index = f
        harness._emit_inflow_sites()
        sim.step(frame_dt)

    real = sim.particle_count
    positions = sim.read_positions()
    y_range = (float(positions[:, 1].min()), float(positions[:, 1].max())) if positions.shape[0] else (0.0, 0.0)
    sim.destroy()
    print(f"  [etendue Y solveur : {y_range}, cube solveur [0, {size:.3f}]]")
    return real


def estimate_model(obj, origin, size, grid_res, ppc_axis, duration, gravity, exit_speed_fn):
    """Reimplementation de estimate_inflow_count, exit_speed_fn(speed, g_par,
    traversal_length) -> v_exit, injectable pour comparer des modeles."""
    dx = size / grid_res
    spacing = dx / ppc_axis
    if spacing <= 0 or duration <= 0:
        return 0, 0, 0

    lo, hi = emitter_bounds_solver(obj, origin, size)
    ex, ey, ez = hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]
    initial_fill = max(0, math.ceil(ex * ey * ez / (spacing ** 3) - 0.5))

    v_solver = world_to_solver_dir(tuple(obj.bourrasque.initial_velocity))
    speed = math.sqrt(sum(c * c for c in v_solver))
    if speed < 1e-6:
        return int(initial_fill), int(initial_fill), 0

    vx, vy, vz = v_solver
    area = (ex * ey * abs(vz) + ey * ez * abs(vx) + ex * ez * abs(vy)) / speed
    particles_per_layer = max(0, math.ceil(area / (spacing * spacing) - 0.5))

    volume = ex * ey * ez
    if area > 0.0:
        traversal_length = volume / area
        uy = vy / speed
        g_par = gravity * uy
        v_exit = exit_speed_fn(speed, g_par, traversal_length)
    else:
        v_exit = speed

    n_layers = v_exit * duration / spacing
    steady_state = particles_per_layer * n_layers
    total = int(initial_fill + steady_state)
    return total, int(initial_fill), int(steady_state)


def v_nominal(speed, g_par, L):
    return speed


def v_out_torricelli(speed, g_par, L):
    return math.sqrt(max(speed * speed, speed * speed + 2.0 * g_par * L))


def v_average(speed, g_par, L):
    vout = v_out_torricelli(speed, g_par, L)
    return 0.5 * (speed + vout)


def v_out_half_length(speed, g_par, L):
    """Torricelli applique sur L/2 : le plafond reemet a un site VACANT
    n'importe ou dans le volume (pas seulement a la face d'entree), donc la
    distance MOYENNE restant a parcourir jusqu'a la sortie, pour une
    particule fraichement emise, est L/2 (milieu du volume en moyenne), pas
    L (traversee complete depuis la face d'entree)."""
    return math.sqrt(max(speed * speed, speed * speed + 2.0 * g_par * (L / 2.0)))


MODELS = {
    "nominal (v)": v_nominal,
    "v_out (Torricelli, actuel)": v_out_torricelli,
    "v_avg = (v + v_out)/2": v_average,
    "v_out(L/2) [refill uniforme]": v_out_half_length,
}


def scenario(name, gravity, domain_size=4.0, frame_end=15, grid_res=48,
             emitter_location=(0.0, 0.0, 1.0), emitter_speed=(0.0, 0.0, -0.5)):
    print("=" * 70)
    print(f"Scenario: {name} (gravity={gravity})")
    print("=" * 70)
    scene = make_scene(gravity, domain_size=domain_size, frame_end=frame_end, grid_res=grid_res)
    obj = setup_box_emitter(scene, location=emitter_location, speed=emitter_speed)
    origin, size = domain_transform(scene)

    fps_base = scene.render.fps_base or 1.0
    fps = scene.render.fps / fps_base if fps_base else 0.0
    n_frames = max(0, scene.bourrasque.frame_end - scene.bourrasque.frame_start + 1)
    duration = n_frames / fps if fps > 0 else 0.0

    real = run_bake(scene, obj, n_frames=n_frames)
    print(f"reel (avec plafond) = {real}")
    for label, fn in MODELS.items():
        total, initial_fill, steady = estimate_model(
            obj, origin, size, scene.bourrasque.grid_res, scene.bourrasque.ppc_axis,
            duration, gravity, fn,
        )
        ratio = total / real if real else float("nan")
        print(f"  {label:30s} : initial_fill={initial_fill} steady_state={steady} total={total}  ratio={ratio:.4f}")
    return real


register()
try:
    scenario("gravite forte (calibrage original)", -9.8)
    print()
    scenario("gravite nulle (calibrage original)", 0.0)
    print()
    scenario("gravite moderee", -4.9)
    print()
    scenario("gravite forte, domaine agrandi, court", -9.8,
              frame_end=15, domain_size=8.0, grid_res=96,
              emitter_location=(0.0, 0.0, 3.0), emitter_speed=(0.0, 0.0, -0.5))
    print()
    scenario("gravite forte, domaine agrandi, moyen", -9.8,
              frame_end=25, domain_size=8.0, grid_res=96,
              emitter_location=(0.0, 0.0, 3.0), emitter_speed=(0.0, 0.0, -0.5))
    print()
    scenario("gravite forte, domaine agrandi, long", -9.8,
              frame_end=35, domain_size=8.0, grid_res=96,
              emitter_location=(0.0, 0.0, 3.0), emitter_speed=(0.0, 0.0, -0.5))
finally:
    unregister()
