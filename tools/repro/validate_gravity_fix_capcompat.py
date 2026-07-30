"""Validation --background de la correction de gravite sur
`estimate_inflow_count` (extension/props.py).

Compare, pour un emetteur BOUNDS en mode INFLOW, l'estimation annoncee par
`estimate_inflow_count` au compte REEL obtenu en faisant tourner un bake
manuel (meme structure que `validate_inflow.py` : reutilise les VRAIES
methodes non-modales de `BQ_OT_bake` via un harnais leger, sans passer par
l'operateur modal qui ne fonctionne pas en --background).

Lance avec :
    blender.exe --background --python validate_gravity_fix.py
"""
import sys

import bpy

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

from extension import ops, props, register, unregister  # noqa: E402
from extension import lib  # noqa: E402


class _Harness:
    """PATCH local (scratchpad uniquement) : le plafond de conservation du
    nombre ajoute a `_emit_inflow_sites` consomme desormais `self._frame_index`
    (meme generateur que le jitter) meme a turbulence=0 -- attribut absent du
    harnais d'origine, qui datait d'avant ce correctif. Ajout minimal pour
    rendre le harnais a nouveau executable contre le `ops.py` de production,
    sans toucher a la logique de comparaison estime/reel elle-meme."""
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
        print(f"  [{level}] {msg}")


def make_scene(cache_dir, gravity, domain_size=4.0, frame_end=40):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1.0

    bpy.ops.mesh.primitive_cube_add(size=domain_size, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = 48
    scene.bourrasque.ppc_axis = 2
    scene.bourrasque.gravity = gravity
    scene.bourrasque.cfl = 0.4
    scene.bourrasque.max_particles = 2_000_000
    scene.bourrasque.frame_start = 1
    scene.bourrasque.frame_end = frame_end
    scene.bourrasque.cache_dir = cache_dir
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


def build_inflow_state(scene, obj):
    from extension.props import (
        domain_transform,
        emitter_bounds_solver,
        world_to_solver_dir,
    )

    origin, size = domain_transform(scene)
    dx = size / scene.bourrasque.grid_res
    spacing = dx / scene.bourrasque.ppc_axis

    op = obj.bourrasque
    vel = world_to_solver_dir(tuple(op.initial_velocity))
    lo, hi = emitter_bounds_solver(obj, origin, size)
    pts = ops._lattice_points_in_box(lo, hi, spacing)
    return vel, spacing, pts


def default_config(scene):
    origin, size = props.domain_transform(scene)
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

    vel, spacing, sites = build_inflow_state(scene, obj)
    state = ops._InflowState(obj.name, mat_id, vel, spacing, sites)

    usable = props.domain_usable_bounds(scene)
    frame_dt = scene.render.fps_base / scene.render.fps

    harness = _Harness(sim, [state], usable, frame_dt=frame_dt)

    for f in range(n_frames):
        harness._frame_index = f
        harness._emit_inflow_sites()
        sim.step(frame_dt)

    real_count = sim.particle_count
    positions = sim.read_positions()
    solver_y_range = (float(positions[:, 1].min()), float(positions[:, 1].max())) if positions.shape[0] else (0.0, 0.0)
    sim.destroy()
    return real_count, solver_y_range, size


def estimate(scene, obj, origin, size):
    fps_base = scene.render.fps_base or 1.0
    fps = scene.render.fps / fps_base if fps_base else 0.0
    n_frames = max(0, scene.bourrasque.frame_end - scene.bourrasque.frame_start + 1)
    duration = n_frames / fps if fps > 0 else 0.0
    return props.estimate_inflow_count(
        obj,
        origin,
        size,
        scene.bourrasque.grid_res,
        scene.bourrasque.ppc_axis,
        duration,
        scene.bourrasque.gravity,
    )


def estimate_no_gravity_arg(scene, obj, origin, size):
    """Appel SANS le parametre `gravity` (defaut 0.0) : sert a verifier
    l'invariant a gravite nulle (meme resultat qu'avant la modification,
    quand la fonction n'avait pas ce parametre du tout)."""
    fps_base = scene.render.fps_base or 1.0
    fps = scene.render.fps / fps_base if fps_base else 0.0
    n_frames = max(0, scene.bourrasque.frame_end - scene.bourrasque.frame_start + 1)
    duration = n_frames / fps if fps > 0 else 0.0
    return props.estimate_inflow_count(
        obj,
        origin,
        size,
        scene.bourrasque.grid_res,
        scene.bourrasque.ppc_axis,
        duration,
    )


def estimate_old_formula(obj, origin, size, grid_res, ppc_axis, duration):
    """Reimplementation independante de l'ANCIENNE formule (avant
    correction, `n_layers = speed * duration / spacing`) pour mesurer le
    ratio AVANT correction, sans modifier `props.py`."""
    import math

    from extension.props import emitter_bounds_solver, world_to_solver_dir

    dx = size / grid_res
    spacing = dx / ppc_axis
    if spacing <= 0 or duration <= 0:
        return 0

    lo, hi = emitter_bounds_solver(obj, origin, size)
    ex = hi[0] - lo[0]
    ey = hi[1] - lo[1]
    ez = hi[2] - lo[2]

    initial_fill = max(0, math.ceil(ex * ey * ez / (spacing**3) - 0.5))

    v_solver = world_to_solver_dir(tuple(obj.bourrasque.initial_velocity))
    speed = math.sqrt(sum(c * c for c in v_solver))
    if speed < 1e-6:
        return int(initial_fill)

    vx, vy, vz = v_solver
    area = (ex * ey * abs(vz) + ey * ez * abs(vx) + ex * ez * abs(vy)) / speed
    particles_per_layer = max(0, math.ceil(area / (spacing * spacing) - 0.5))

    n_layers = speed * duration / spacing
    steady_state = particles_per_layer * n_layers

    return int(initial_fill + steady_state)


def scenario(
    name,
    gravity,
    cache_dir,
    domain_size=4.0,
    frame_end=40,
    emitter_location=(0.0, 0.0, 1.0),
    emitter_speed=(0.0, 0.0, -0.5),
):
    print("=" * 70)
    print(f"Scenario: {name}  (gravity={gravity}, domain_size={domain_size}, frame_end={frame_end})")
    print("=" * 70)
    scene = make_scene(cache_dir, gravity, domain_size=domain_size, frame_end=frame_end)
    obj = setup_box_emitter(scene, location=emitter_location, speed=emitter_speed)
    origin, size = props.domain_transform(scene)

    fps_base = scene.render.fps_base or 1.0
    fps = scene.render.fps / fps_base if fps_base else 0.0
    n_frames_est = max(0, scene.bourrasque.frame_end - scene.bourrasque.frame_start + 1)
    duration = n_frames_est / fps if fps > 0 else 0.0

    est = estimate(scene, obj, origin, size)
    est_old = estimate_old_formula(
        obj, origin, size, scene.bourrasque.grid_res, scene.bourrasque.ppc_axis, duration
    )
    real, solver_y_range, solver_size = run_bake(scene, obj, n_frames=n_frames_est)
    ratio = est / real if real else float("nan")
    ratio_old = est_old / real if real else float("nan")
    print(f"estime AVANT correction={est_old}  estime APRES correction={est}  reel={real}")
    print(f"ratio AVANT (estime/reel)={ratio_old:.4f}  ratio APRES (estime/reel)={ratio:.4f}")
    print(
        f"etendue Y solveur des positions finales : {solver_y_range} "
        f"(cube solveur : [0, {solver_size:.3f}]) -- verifie que le fluide "
        "n'a pas atteint le fond du domaine (confondrait l'effet mesure "
        "avec une collision de paroi)"
    )
    return est, est_old, real, ratio, ratio_old


def main():
    cache_dir = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\bourrasque_cache_gravity_fix"

    # Domaine (4 m, emetteur en z=1, speed=0.5) et frame_end=15 (0.625 s) :
    # a v0=0.5 et g=-9.8, la chute max en 0.625 s est
    # 0.5*0.625+4.9*0.625^2 ~= 2.53 m, sous la distance emetteur (z=1) ->
    # fond (z=-2) = 3 m. Le jet reste dans le domaine sur toute la duree du
    # bake (verifie via `solver_y_range` imprime par `scenario`) : la mesure
    # isole l'effet du correctif, sans confondre avec une collision de
    # paroi (voir le test avec domaine=4/frame_end=40, mene en premier essai
    # de ce script, ou le jet touchait le fond et faussait la comparaison).
    domain_size = 4.0
    frame_end = 15
    emitter_location = (0.0, 0.0, 1.0)
    emitter_speed = (0.0, 0.0, -0.5)

    register()
    try:
        print()
        est_strong, est_strong_old, real_strong, ratio_strong, ratio_strong_old = scenario(
            "gravite forte",
            -9.8,
            cache_dir,
            domain_size=domain_size,
            frame_end=frame_end,
            emitter_location=emitter_location,
            emitter_speed=emitter_speed,
        )
        print()
        est_zero, est_zero_old, real_zero, ratio_zero, ratio_zero_old = scenario(
            "gravite nulle",
            0.0,
            cache_dir,
            domain_size=domain_size,
            frame_end=frame_end,
            emitter_location=emitter_location,
            emitter_speed=emitter_speed,
        )

        print()
        print("=" * 70)
        print("INVARIANT gravite nulle : estimation inchangee vs signature sans le")
        print("parametre gravity (comportement d'avant la modification)")
        print("=" * 70)
        scene0 = make_scene(cache_dir, 0.0, domain_size=domain_size, frame_end=frame_end)
        obj0 = setup_box_emitter(scene0, location=emitter_location, speed=emitter_speed)
        origin0, size0 = props.domain_transform(scene0)
        est_with_param = estimate(scene0, obj0, origin0, size0)
        est_without_param = estimate_no_gravity_arg(scene0, obj0, origin0, size0)
        print(f"avec gravity=0.0 explicite : {est_with_param}")
        print(f"sans argument gravity (defaut) : {est_without_param}")
        assert est_with_param == est_without_param, "INVARIANT VIOLE"
        print("invariant OK : identique")

        print()
        print("=" * 70)
        print("RESUME")
        print("=" * 70)
        print(
            f"gravite forte (-9.8) : reel={real_strong}  "
            f"AVANT: estime={est_strong_old} ratio={ratio_strong_old:.4f}  "
            f"APRES: estime={est_strong} ratio={ratio_strong:.4f}"
        )
        print(
            f"gravite nulle (0.0)  : reel={real_zero}  "
            f"AVANT: estime={est_zero_old} ratio={ratio_zero_old:.4f}  "
            f"APRES: estime={est_zero} ratio={ratio_zero:.4f}"
        )
        assert est_zero == est_zero_old, "gravite nulle : AVANT/APRES doivent coincider"
        print("gravite nulle : estime AVANT == estime APRES -> OK")
    finally:
        unregister()


if __name__ == "__main__":
    main()
