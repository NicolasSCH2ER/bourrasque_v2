"""Validation --background de la turbulence a l'emission (M4 correctif).

Lance avec :
    "C:\\Program Files\\Blender Foundation\\Blender 5.2\\blender.exe" --background --factory-startup --python validate_turbulence.py

Ne passe jamais par l'operateur modal BQ_OT_bake.invoke()/modal() (le modal
exige un cycle d'evenements bpy absent en --background) : reproduit le
setup a la main, comme `validate_inflow.py`, en reutilisant les VRAIES
briques de production (`ops._turbulent_emission`, `ops._turbulence_rng`,
`ops._box_lattice_points`, `ops._InflowState`,
`ops.BQ_OT_bake._emit_inflow_sites`).
"""
import sys

import bpy
import numpy as np

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

from extension import ops, props, register, unregister  # noqa: E402
from extension import lib  # noqa: E402
from extension.props import (  # noqa: E402
    domain_transform,
    domain_usable_bounds,
    emitter_bounds_solver,
    world_to_solver_dir,
)


class _Harness:
    """Reutilise _emit_inflow_sites, la VRAIE methode de production."""

    _emit_inflow_sites = ops.BQ_OT_bake._emit_inflow_sites

    def __init__(self, sim, inflow_states, usable_bounds, frame_dt):
        self._sim = sim
        self._inflow_states = inflow_states
        self._usable_bounds = usable_bounds
        self._frame_dt = frame_dt
        self._frame_index = 0
        self._saturated = False
        self.reports = []

    def report(self, level, msg):
        self.reports.append((level, msg))


def make_scene(grid_res=32, ppc=2, gravity=-0.5, frame_end=40):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1.0

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = grid_res
    scene.bourrasque.ppc_axis = ppc
    scene.bourrasque.gravity = gravity
    scene.bourrasque.cfl = 0.4
    scene.bourrasque.max_particles = 4_000_000
    scene.bourrasque.frame_start = 1
    scene.bourrasque.frame_end = frame_end
    return scene


def setup_box_emitter(location=(0.0, 0.0, 0.6), size=(0.3, 0.3, 0.3), speed=-0.15,
                       mode="INFLOW", source="BOUNDS", turbulence=0.0, seed=0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.scale = size
    bpy.context.view_layer.update()
    obj.name = "BoxEmitter"
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.model = "WATER"
    obj.bourrasque.emit_mode = mode
    obj.bourrasque.emit_source = source
    obj.bourrasque.initial_velocity = (0.0, 0.0, speed)
    obj.bourrasque.turbulence = turbulence
    obj.bourrasque.turbulence_seed = seed
    return obj


def setup_sphere_emitter(location=(0.0, 0.0, 0.6), radius=0.3, speed=-0.15,
                          mode="INFLOW", turbulence=0.0, seed=0):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=24, ring_count=12)
    obj = bpy.context.active_object
    obj.name = "SphereEmitter"
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.model = "WATER"
    obj.bourrasque.emit_mode = mode
    obj.bourrasque.emit_source = "MESH"
    obj.bourrasque.initial_velocity = (0.0, 0.0, speed)
    obj.bourrasque.turbulence = turbulence
    obj.bourrasque.turbulence_seed = seed
    return obj


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


# ---------------------------------------------------------------------------
# INFLOW : bake manuel via les VRAIES briques (ops._InflowState + la VRAIE
# methode ops.BQ_OT_bake._emit_inflow_sites, qui applique la turbulence).
# ---------------------------------------------------------------------------


def run_inflow_bake(scene, obj, n_frames):
    from extension import sampling

    cfg, origin, size = default_config(scene)
    sim = lib.Sim(cfg)
    mat_id = sim.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)

    dx = size / scene.bourrasque.grid_res
    spacing = dx / scene.bourrasque.ppc_axis
    op = obj.bourrasque
    vel = world_to_solver_dir(tuple(op.initial_velocity))

    if op.emit_source == "MESH":
        sites = sampling.sample_mesh_interior(
            obj, origin, size, scene.bourrasque.grid_res, scene.bourrasque.ppc_axis
        )
    else:
        lo, hi = emitter_bounds_solver(obj, origin, size)
        sites = ops._lattice_points_in_box(lo, hi, spacing)

    frame_dt = scene.render.fps_base / scene.render.fps
    bbox_lo, bbox_hi = sites.min(axis=0), sites.max(axis=0)
    noise = ops._CurlNoise(
        op.turbulence_seed, 0, bbox_lo, bbox_hi, dx, vel, spacing, frame_dt,
    )
    state = ops._InflowState(
        obj.name, mat_id, vel, spacing, sites,
        dx=dx,
        turbulence=op.turbulence, turbulence_seed=op.turbulence_seed,
        emitter_index=0, noise=noise,
    )

    usable = props.domain_usable_bounds(scene)
    harness = _Harness(sim, [state], usable, frame_dt)

    per_frame_emitted = []
    for f in range(n_frames):
        harness._frame_index = f
        n_emitted = harness._emit_inflow_sites()
        sim.step(frame_dt)
        per_frame_emitted.append(n_emitted)

    positions = sim.read_positions()
    axis_point = sites.mean(axis=0).astype(np.float64) if sites.shape[0] else None
    sim.destroy()
    return positions, per_frame_emitted, spacing, vel, axis_point


def transverse_dispersion(positions, vel, axis_point, downstream_frac=0.7):
    """Ecart-type des positions, PERPENDICULAIREMENT a l'axe d'ecoulement,
    restreint aux particules 'bien en aval' (memes conventions que
    `verif_jet.jet_radius`) : le nuage COMPLET inclut le volume statique de
    l'emetteur (ensemencement volumique, INFLOW), dont l'etendue propre
    (rayon de la sphere) domine tout effet de turbulence si on ne filtre
    pas — seul le jet en vol, hors du volume source, doit etre mesure."""
    vel = np.array(vel, dtype=np.float64)
    speed = np.linalg.norm(vel)
    if speed < 1e-9 or positions.shape[0] == 0:
        return 0.0
    axis_dir = vel / speed
    s = positions.astype(np.float64) @ axis_dir
    smin, smax = s.min(), s.max()
    if smax - smin < 1e-9:
        return 0.0
    threshold = smin + downstream_frac * (smax - smin)
    mask = s >= threshold
    downstream = positions[mask]
    if downstream.shape[0] == 0:
        return 0.0
    rel = downstream.astype(np.float64) - axis_point
    proj = np.outer(rel @ axis_dir, axis_dir)
    perp = rel - proj
    dist = np.linalg.norm(perp, axis=1)
    return float(np.std(dist))


# ---------------------------------------------------------------------------
# BLOCK/BOUNDS : reproduit EXACTEMENT les 3 branches de BQ_OT_bake.invoke()
# (turbulence <= 0 => emit_box strict ; sinon => _box_lattice_points +
# _turbulent_emission + emit_points_vel), pour mesurer le compte de
# particules obtenu par les VRAIES briques de production.
# ---------------------------------------------------------------------------


def run_block_bounds(scene, obj, frame_dt, emitter_index=0):
    cfg, origin, size = default_config(scene)
    sim = lib.Sim(cfg)
    mat_id = sim.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)

    dx = size / scene.bourrasque.grid_res
    spacing = dx / scene.bourrasque.ppc_axis
    op = obj.bourrasque
    vel = world_to_solver_dir(tuple(op.initial_velocity))
    lo, hi = emitter_bounds_solver(obj, origin, size)

    if op.turbulence <= 0.0:
        n = sim.emit_box(mat_id, lo, hi, vel=vel)
    else:
        lattice = ops._box_lattice_points(lo, hi, spacing)
        rng = ops._turbulence_rng(op.turbulence_seed, 0, emitter_index)
        noise = ops._CurlNoise(op.turbulence_seed, emitter_index, lo, hi, dx, vel, spacing, frame_dt)
        pts, vels = ops._turbulent_emission(lattice, vel, op.turbulence, spacing, frame_dt, rng, dx, 0.0, noise)
        n = sim.emit_points_vel(mat_id, pts, vels)

    positions = sim.read_positions()
    sim.destroy()
    return n, positions


def run_block_mesh(scene, obj, frame_dt, emitter_index=0):
    from extension import sampling

    cfg, origin, size = default_config(scene)
    sim = lib.Sim(cfg)
    mat_id = sim.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)

    op = obj.bourrasque
    vel = world_to_solver_dir(tuple(op.initial_velocity))
    mesh_points = sampling.sample_mesh_interior(
        obj, origin, size, scene.bourrasque.grid_res, scene.bourrasque.ppc_axis
    )
    dx = size / scene.bourrasque.grid_res
    spacing = dx / scene.bourrasque.ppc_axis
    rng = ops._turbulence_rng(op.turbulence_seed, 0, emitter_index)
    bbox_lo, bbox_hi = ops._points_bbox(mesh_points)
    noise = ops._CurlNoise(op.turbulence_seed, emitter_index, bbox_lo, bbox_hi, dx, vel, spacing, frame_dt)
    pts, vels = ops._turbulent_emission(mesh_points, vel, op.turbulence, spacing, frame_dt, rng, dx, 0.0, noise)
    n = sim.emit_points_vel(mat_id, pts, vels)

    positions = sim.read_positions()
    sim.destroy()
    return n, positions


def main():
    print("=" * 70)
    print("1) NON-REGRESSION A TURBULENCE NULLE — comptes")
    print("=" * 70)

    register()
    try:
        # --- INFLOW, turbulence = 0 : compte doit etre inchange -----------
        scene = make_scene()
        sphere0 = setup_sphere_emitter(turbulence=0.0)
        pos0, emitted0, spacing0, vel0, axis0 = run_inflow_bake(scene, sphere0, n_frames=40)
        total0 = sum(emitted0)
        print(f"INFLOW turbulence=0 : total emis={total0}, n_final={pos0.shape[0]}")

        # --- BLOCK/BOUNDS, turbulence = 0 : DOIT emprunter emit_box seul --
        scene_bb = make_scene()
        box_bb = setup_box_emitter(mode="BLOCK", source="BOUNDS", turbulence=0.0)
        frame_dt = scene_bb.render.fps_base / scene_bb.render.fps
        n_bb0, pos_bb0 = run_block_bounds(scene_bb, box_bb, frame_dt)

        # Reference independante : appel emit_box DIRECT, sans passer par
        # aucune branche modifiee de ce jalon.
        cfg, origin, size = default_config(scene_bb)
        sim_ref = lib.Sim(cfg)
        mat_ref = sim_ref.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
        lo, hi = emitter_bounds_solver(box_bb, origin, size)
        vel_ref = world_to_solver_dir(tuple(box_bb.bourrasque.initial_velocity))
        n_ref = sim_ref.emit_box(mat_ref, lo, hi, vel=vel_ref)
        sim_ref.destroy()
        print(f"BLOCK/BOUNDS turbulence=0 : n={n_bb0}, reference emit_box directe={n_ref}")
        assert n_bb0 == n_ref, "REGRESSION : le compte BLOCK/BOUNDS a turbulence 0 a change"

        # --- BLOCK/BOUNDS, turbulence = 0.3 : compte doit rester EGAL ------
        scene_bb3 = make_scene()
        box_bb3 = setup_box_emitter(mode="BLOCK", source="BOUNDS", turbulence=0.3)
        n_bb3, pos_bb3 = run_block_bounds(scene_bb3, box_bb3, frame_dt)
        print(f"BLOCK/BOUNDS turbulence=0.3 : n={n_bb3}")
        assert n_bb3 == n_ref, "le jitter positionnel a change le COMPTE BLOCK/BOUNDS (ne devrait jamais arriver, jitter applique APRES generation du reseau)"

        # --- BLOCK/MESH, turbulence = 0 vs 0.3 : compte inchange ----------
        scene_bm0 = make_scene()
        sphere_bm0 = setup_sphere_emitter(mode="BLOCK", turbulence=0.0)
        # emit_source pour une sphere est "MESH" par construction de setup_sphere_emitter
        n_bm0, pos_bm0 = run_block_mesh(scene_bm0, sphere_bm0, frame_dt)

        scene_bm3 = make_scene()
        sphere_bm3 = setup_sphere_emitter(mode="BLOCK", turbulence=0.3)
        n_bm3, pos_bm3 = run_block_mesh(scene_bm3, sphere_bm3, frame_dt)
        print(f"BLOCK/MESH turbulence=0 : n={n_bm0}   turbulence=0.3 : n={n_bm3}")
        assert n_bm0 == n_bm3, "REGRESSION : le compte BLOCK/MESH change avec la turbulence"

        # --- Enveloppe des positions (BOUNDS) reste dans le meme ordre ----
        print(f"BOUNDS turbulence=0   : min={pos_bb0.min(axis=0)} max={pos_bb0.max(axis=0)}")
        print(f"BOUNDS turbulence=0.3 : min={pos_bb3.min(axis=0)} max={pos_bb3.max(axis=0)}")

        print()
        print("=" * 70)
        print("2) EFFET MESURABLE — dispersion transverse du jet INFLOW")
        print("=" * 70)
        dispersions = {}
        for turb in (0.0, 0.1, 0.3):
            sc = make_scene()
            sph = setup_sphere_emitter(turbulence=turb, seed=42)
            pos, emitted, spacing, vel, axis_pt = run_inflow_bake(sc, sph, n_frames=40)
            disp = transverse_dispersion(pos, vel, axis_pt)
            dispersions[turb] = disp
            print(f"turbulence={turb:.1f} : dispersion transverse = {disp:.5f} m (n={pos.shape[0]})")
        assert dispersions[0.0] < dispersions[0.1] < dispersions[0.3], (
            f"dispersion NON monotone : {dispersions}"
        )
        print("croissance monotone : OK")

        print()
        print("=" * 70)
        print("3) DEBIT INFLOW — turbulence=0 vs 0.3")
        print("=" * 70)
        sc_a = make_scene()
        sph_a = setup_sphere_emitter(turbulence=0.0, seed=1)
        pos_a, emitted_a, *_ = run_inflow_bake(sc_a, sph_a, n_frames=60)
        total_a = sum(emitted_a)

        sc_b = make_scene()
        sph_b = setup_sphere_emitter(turbulence=0.3, seed=1)
        pos_b, emitted_b, *_ = run_inflow_bake(sc_b, sph_b, n_frames=60)
        total_b = sum(emitted_b)

        pct = 100.0 * (total_b - total_a) / total_a if total_a else float("nan")
        print(f"total emis turbulence=0   : {total_a}")
        print(f"total emis turbulence=0.3 : {total_b}")
        print(f"ecart relatif : {pct:+.2f} %")

        print()
        print("=" * 70)
        print("4) REPRODUCTIBILITE")
        print("=" * 70)
        sc_s1a = make_scene()
        sph_s1a = setup_sphere_emitter(turbulence=0.3, seed=7)
        pos_s1a, emitted_s1a, *_ = run_inflow_bake(sc_s1a, sph_s1a, n_frames=30)

        sc_s1b = make_scene()
        sph_s1b = setup_sphere_emitter(turbulence=0.3, seed=7)
        pos_s1b, emitted_s1b, *_ = run_inflow_bake(sc_s1b, sph_s1b, n_frames=30)

        sc_s2 = make_scene()
        sph_s2 = setup_sphere_emitter(turbulence=0.3, seed=99)
        pos_s2, emitted_s2, *_ = run_inflow_bake(sc_s2, sph_s2, n_frames=30)

        same_seed_counts_match = emitted_s1a == emitted_s1b
        same_seed_positions_match = (
            pos_s1a.shape == pos_s1b.shape and np.allclose(pos_s1a, pos_s1b)
        )
        diff_seed_positions_differ = not (
            pos_s1a.shape == pos_s2.shape and np.allclose(pos_s1a, pos_s2)
        )
        print(f"meme graine (7,7)   : comptes/frame identiques = {same_seed_counts_match}, positions identiques = {same_seed_positions_match}")
        print(f"graines differentes (7,99) : positions differentes = {diff_seed_positions_differ}")
        assert same_seed_counts_match
        assert same_seed_positions_match
        assert diff_seed_positions_differ

    finally:
        unregister()

    print()
    print("TOUTES LES VERIFICATIONS SONT PASSEES.")


if __name__ == "__main__":
    main()
