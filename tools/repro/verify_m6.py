"""Verification du jalon M6 (colliders animes), a lancer avec :

    blender --background --factory-startup --python verify_m6.py

Reprend le harnais des scripts de validation precedents (`validate_milestone.py`,
`validate_continuity.py`) : un objet `_FakeSelf` minimal porte l'etat
d'instance de `BQ_OT_bake`, mais les methodes numeriquement sensibles
(`_advance_frame`, `_advance_scene_frame`, `_update_colliders`,
`_emit_inflow_sites`, `_cleanup`) sont les VRAIES methodes de production
liees via `types.MethodType` -- c'est exactement le code livre dans
extension/ops.py qui est exerce, pas une reimplementation parallele.

`BQ_OT_bake.invoke()` ne peut pas etre appele directement en
`--background` (pas de fenetre, `wm.event_timer_add` echoue) : ce script
reproduit donc la partie "mise en place" de `invoke()` (creation de la Sim,
materiaux, emission du bloc d'eau, collecte des colliders) en appelant les
memes fonctions de production (`props.py`, `lib.py`, `ops._ColliderState`),
sans dupliquer la logique numerique elle-meme.
"""

import math
import os
import sys
import types

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

ROOT = r"C:\Users\nicol\Code\bourrasque_v2"
sys.path.insert(0, ROOT)

import extension  # noqa: E402

extension.register()

from extension import cache, lib, ops  # noqa: E402
from extension.props import (  # noqa: E402
    domain_resolution,
    domain_transform,
    emitter_bounds_solver,
    world_to_solver_dir,
)

import numpy as np  # noqa: E402

SCRATCHPAD = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRATCHPAD, "cache_out_m6")
os.makedirs(CACHE_DIR, exist_ok=True)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class _FakeSelf:
    """Mime le sous-ensemble d'attributs de BQ_OT_bake requis par les
    methodes de production reutilisees telles quelles."""

    def __init__(self, scene):
        self._scene = scene
        self._sim = None
        self._writer = None
        self._pos_buffer = None
        self._frame_dt = scene.render.fps_base / scene.render.fps
        self._inflow_states = []
        self._saturated = False
        self._usable_bounds = None
        self._domain_transform = None
        self._collider_states = []
        self._frame_index = 0
        self._start_frame = scene.frame_current
        self._timer = None
        self.reports = []

        self._advance_frame = types.MethodType(ops.BQ_OT_bake._advance_frame, self)
        self._advance_scene_frame = types.MethodType(
            ops.BQ_OT_bake._advance_scene_frame, self
        )
        self._update_colliders = types.MethodType(
            ops.BQ_OT_bake._update_colliders, self
        )
        self._emit_inflow_sites = types.MethodType(
            ops.BQ_OT_bake._emit_inflow_sites, self
        )
        self._cleanup = types.MethodType(ops.BQ_OT_bake._cleanup, self)

    def report(self, tags, msg):
        self.reports.append((tuple(tags), msg))
        print("   report:", tags, msg)


def fresh_scene():
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.bourrasque.domain_object = None
    return scene


def make_domain(scene, size=1.0, grid_res=32, ppc_axis=2, gravity=-9.8, cfl=0.3):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = grid_res
    scene.bourrasque.ppc_axis = ppc_axis
    scene.bourrasque.gravity = gravity
    scene.bourrasque.cfl = cfl
    scene.bourrasque.max_particles = 2_000_000
    scene.bourrasque.frame_start = 1
    return domain


def build_sim(scene):
    origin, size = domain_transform(scene)
    res, dx = domain_resolution(scene)
    cfg = lib.default_config()
    cfg.grid_res[:] = res
    cfg.cell_size = dx
    cfg.gravity_y = scene.bourrasque.gravity
    cfg.cfl = scene.bourrasque.cfl
    cfg.ppc_axis = scene.bourrasque.ppc_axis
    cfg.max_particles = scene.bourrasque.max_particles
    sim = lib.Sim(cfg)
    return sim, origin, size, dx


def add_water_block(sim, origin, size, location, dims, name="Water", vel=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = dims
    bpy.context.view_layer.update()
    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    lo, hi = emitter_bounds_solver(obj, origin, size)
    vel_solver = world_to_solver_dir(vel)
    n = sim.emit_box(mat_id, lo, hi, vel=vel_solver)
    return obj, mat_id, n


def add_static_collider(location, radius, name="Sphere", friction=0.2):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, location=location, segments=48, ring_count=24
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.bourrasque.role = "COLLIDER"
    obj.bourrasque.friction = friction
    bpy.context.view_layer.update()
    return obj


def add_animated_collider(loc_start, loc_end, frame_start, frame_end, radius,
                           name="MovingSphere", friction=0.2):
    obj = add_static_collider(loc_start, radius, name=name, friction=friction)
    obj.location = loc_start
    obj.keyframe_insert(data_path="location", frame=frame_start)
    obj.location = loc_end
    obj.keyframe_insert(data_path="location", frame=frame_end)
    # Blender 4.4+/5.x : les F-curves d'une Action "layered" vivent dans un
    # ActionChannelbag (obj.animation_data.action.fcurves n'existe plus).
    action = obj.animation_data.action
    channelbag = action.layers[0].strips[0].channelbag(
        obj.animation_data.action_slot
    )
    for fcurve in channelbag.fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = "LINEAR"
    obj.location = loc_start
    bpy.context.view_layer.update()
    return obj


def run_bake(scene, origin, size, dx, sim, collider_objs, n_frames):
    """Rejoue la boucle de bq.bake._advance_frame pour `n_frames`, avec les
    colliders fournis (liste de `ops._ColliderState`). Renvoie (fake, positions
    finales monde n'existe pas ici -- positions SOLVEUR)."""
    fake = _FakeSelf(scene)
    fake._sim = sim
    fake._domain_transform = (origin, size)
    fake._collider_states = collider_objs

    bqd_path, _ = cache.cache_paths(CACHE_DIR, scene.name + f"_{id(fake)}")
    cache.ensure_cache_dir(CACHE_DIR)
    fake._writer = cache.CacheWriter(bqd_path, sim.particle_count)

    for i in range(n_frames):
        fake._frame_index = i
        fake._advance_frame()

    return fake


def barycenter_world(sim, origin, size):
    from extension.transform import solver_to_world_array

    pos = sim.read_positions()
    world = solver_to_world_array(pos.astype(np.float32), origin, size)
    return world.mean(axis=0), pos


# ---------------------------------------------------------------------------
# 1) Non-regression : scene SANS collider, avec emetteur INFLOW.
# ---------------------------------------------------------------------------
print("\n=== 1) Non-regression (sans collider, inflow) ===")


def run_no_collider_inflow(seed_note):
    scene = fresh_scene()
    make_domain(scene, size=1.0, grid_res=24, ppc_axis=2, gravity=-9.8)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -0.35))
    obj = bpy.context.active_object
    obj.name = "Inflow"
    obj.scale = (0.15, 0.15, 0.15)
    bpy.context.view_layer.update()
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.emit_mode = "INFLOW"
    obj.bourrasque.emit_source = "BOUNDS"
    obj.bourrasque.model = "WATER"
    obj.bourrasque.initial_velocity = (0.0, 0.5, 0.0)

    sim, origin, size, dx = build_sim(scene)
    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)

    from extension import sampling  # noqa: F401 (garde la coherence des imports)
    from extension.props import domain_usable_bounds
    from extension.ops import _lattice_points_in_box, _InflowState, _CurlNoise

    props = scene.bourrasque
    spacing = dx / props.ppc_axis
    vel = world_to_solver_dir((0.0, 0.5, 0.0))
    lo, hi = emitter_bounds_solver(obj, origin, size)
    sites = _lattice_points_in_box(lo, hi, spacing)
    noise = _CurlNoise(0, 0, sites.min(axis=0), sites.max(axis=0), dx, vel, spacing,
                        scene.render.fps_base / scene.render.fps)

    fake = _FakeSelf(scene)
    fake._sim = sim
    fake._domain_transform = (origin, size)
    fake._usable_bounds = domain_usable_bounds(scene)
    fake._inflow_states = [
        _InflowState("Inflow", mat_id, vel, spacing, sites, dx=dx, noise=noise)
    ]
    bqd_path, _ = cache.cache_paths(CACHE_DIR, f"noregr_{seed_note}")
    cache.ensure_cache_dir(CACHE_DIR)
    fake._writer = cache.CacheWriter(bqd_path, sim.particle_count)

    starting_frame = scene.frame_current
    counts = []
    for i in range(30):
        fake._frame_index = i
        before = sim.particle_count
        fake._advance_frame()
        counts.append(sim.particle_count - before)

    n_sites = sites.shape[0]
    final_count = sim.particle_count
    fake._cleanup(bpy.context)
    frame_restored = scene.frame_current == starting_frame
    sim.destroy()
    return counts, n_sites, final_count, frame_restored


counts_a, n_sites_a, final_a, restored_a = run_no_collider_inflow("run1")
counts_b, n_sites_b, final_b, restored_b = run_no_collider_inflow("run2")

check("non-regression: nuage de sites inflow identique entre deux runs",
      n_sites_a == n_sites_b, f"{n_sites_a} vs {n_sites_b}")
check("non-regression: comptes emis par frame identiques (deterministe, sans turbulence)",
      counts_a == counts_b, f"a={counts_a}\n b={counts_b}")
check("non-regression: particle_count final identique entre deux runs",
      final_a == final_b, f"{final_a} vs {final_b}")
check("non-regression: frame restauree apres bake (run1)", restored_a)
check("non-regression: frame restauree apres bake (run2)", restored_b)
print(f"  particle_count final : run1={final_a} run2={final_b}")
print(f"  debit (particules/frame), run1 : {counts_a}")


# ---------------------------------------------------------------------------
# 2) Collider statique etanche : colonne d'eau tombant sur une sphere.
# ---------------------------------------------------------------------------
print("\n=== 2) Collider statique etanche ===")

scene = fresh_scene()
make_domain(scene, size=1.0, grid_res=64, ppc_axis=2, gravity=-9.8)
sim, origin, size, dx = build_sim(scene)
water_obj, mat_id, n_emit = add_water_block(
    sim, origin, size, location=(0.0, 0.0, 0.25), dims=(0.3, 0.3, 0.3)
)
sphere_radius = 0.15
sphere = add_static_collider((0.0, 0.0, -0.15), sphere_radius, name="StaticSphere")

collider_state = ops._ColliderState(sphere, sphere.bourrasque.friction)

from extension.transform import solver_to_world_array

fake = _FakeSelf(scene)
fake._sim = sim
fake._domain_transform = (origin, size)
fake._collider_states = [collider_state]
bqd_path, _ = cache.cache_paths(CACHE_DIR, "static_sphere")
cache.ensure_cache_dir(CACHE_DIR)
fake._writer = cache.CacheWriter(bqd_path, sim.particle_count)

sphere_center = np.array(sphere.location, dtype=np.float64)
penetration_trace = []
for i in range(90):
    fake._frame_index = i
    fake._advance_frame()
    if i % 15 == 14:
        pos_solver = sim.read_positions()
        pos_world = solver_to_world_array(pos_solver.astype(np.float32), origin, size)
        dist = np.linalg.norm(pos_world.astype(np.float64) - sphere_center, axis=1)
        pen = float(sphere_radius - dist.min()) if pos_world.shape[0] else 0.0
        penetration_trace.append((i + 1, pen))

print(f"  trace penetration (frame, penetration_max) : {penetration_trace}")

# Verification DIRECTE du champ de distance (bq_read_sdf, M6/refonte coeur) :
# independante de la reponse de collision MPM (les particules), ceci
# verifie que le champ lui-meme represente correctement la sphere -- au
# centre (profondement negatif), pres de la surface (proche de 0), loin
# (positif, ~egal a la distance analytique moins le rayon).
res, grid_dx = domain_resolution(scene)
sdf = sim.read_sdf()
check("sdf : forme == grid_res", sdf.shape == tuple(res), f"sdf.shape={sdf.shape} res={res}")

idx = np.indices(sdf.shape, dtype=np.float64)
cell_centers_solver = np.stack(
    [(idx[a] + 0.5) * grid_dx for a in range(3)], axis=-1
).reshape(-1, 3)
cell_centers_world = solver_to_world_array(
    cell_centers_solver.astype(np.float32), origin, size
).astype(np.float64)
analytic_dist = np.linalg.norm(cell_centers_world - sphere_center, axis=1) - sphere_radius
sdf_flat = sdf.reshape(-1).astype(np.float64)

# Cellules dans la bande etroite (narrow-band) effectivement peuplee : sonde
# prealable (probe_sdf.py) montrant une transition nette vers une valeur
# sentinelle (~999999.9) des ~3*dx de distance a la surface (coherent avec
# le commentaire coeur "pad AABB fait 3*dx d'epaisseur"). Au-dela, un champ
# a bande etroite est LIBRE de laisser cette sentinelle -- non un bug.
near_mask = np.abs(analytic_dist) < 2.5 * grid_dx
same_sign = np.sign(sdf_flat[near_mask]) == np.sign(analytic_dist[near_mask])
frac_same_sign = float(np.mean(same_sign)) if near_mask.any() else 1.0
check(
    "sdf : signe coherent avec la distance analytique (bande proche < 2.5 dx)",
    frac_same_sign > 0.99,
    f"fraction signe correct = {frac_same_sign:.4f} (n={int(near_mask.sum())})",
)

# Bande peuplee : la sentinelle (~999999.9) n'y apparait pas (voir sonde
# prealable), la comparaison a la distance analytique peut donc etre
# STRICTE (tolerance 0.25*dx, la precision reelle mesuree est de l'ordre de
# 0.03-0.04*dx -- tolerance tres large par rapport a la precision reelle).
err_mag = np.abs(sdf_flat[near_mask] - analytic_dist[near_mask])
max_err_mag = float(err_mag.max()) if near_mask.any() else 0.0
check(
    "sdf : magnitude proche de la distance analytique (bande proche < 2.5 dx)",
    max_err_mag < 0.25 * grid_dx,
    f"erreur max = {max_err_mag:.5f} m (0.25*dx = {0.25*grid_dx:.5f} m)",
)

n_sentinel = int(np.sum(sdf_flat > 1e5))
print(f"  cellules a valeur sentinelle (narrow-band, hors bande) : {n_sentinel} / {sdf_flat.shape[0]}")

pos_solver = sim.read_positions()
pos_world = solver_to_world_array(pos_solver.astype(np.float32), origin, size)
dist = np.linalg.norm(pos_world.astype(np.float64) - sphere_center, axis=1)
n_inside = int(np.sum(dist < sphere_radius))
penetration = float(sphere_radius - dist.min()) if pos_world.shape[0] else 0.0

check("collider statique : au plus une poignee de particules a l'interieur",
      n_inside <= max(5, int(0.01 * pos_world.shape[0])),
      f"n_inside={n_inside} / total={pos_world.shape[0]}")
check("collider statique : penetration < demi-cellule",
      penetration < 0.5 * dx, f"penetration={penetration:.5f} dx={dx:.5f}")
print(f"  n_inside={n_inside}, penetration_max={penetration:.5f} m (dx={dx:.5f} m)")

fake._cleanup(bpy.context)
sim.destroy()


# ---------------------------------------------------------------------------
# 3) Collider ANIME : deplace le fluide, compare au cas statique.
# ---------------------------------------------------------------------------
print("\n=== 3) Collider anime : deplacement du fluide ===")

N_FRAMES_3 = 40


def run_moving_scenario(animated, run_tag):
    scene = fresh_scene()
    make_domain(scene, size=1.0, grid_res=28, ppc_axis=2, gravity=-9.8)
    sim, origin, size, dx = build_sim(scene)
    # Eau au repos, deja tassee en bas du domaine, occupe une large tranche.
    water_obj, mat_id, n_emit = add_water_block(
        sim, origin, size, location=(0.0, 0.0, -0.15), dims=(0.7, 0.7, 0.3)
    )

    radius = 0.12
    if animated:
        sphere = add_animated_collider(
            (-0.4, 0.0, -0.15), (0.4, 0.0, -0.15),
            frame_start=1, frame_end=N_FRAMES_3, radius=radius,
            name=f"Mover_{run_tag}",
        )
    else:
        sphere = add_static_collider((-0.4, 0.0, -0.15), radius, name=f"Static_{run_tag}")

    collider_state = ops._ColliderState(sphere, sphere.bourrasque.friction)
    fake = run_bake(scene, origin, size, dx, sim, [collider_state], n_frames=N_FRAMES_3)
    barycenter, pos_solver = barycenter_world(sim, origin, size)
    fake._cleanup(bpy.context)
    sim.destroy()
    return barycenter


bary_static = run_moving_scenario(animated=False, run_tag="s")
bary_animated = run_moving_scenario(animated=True, run_tag="a")
# Bruit de reference : deux runs STATIQUES independants (meme scenario, meme
# binaire) pour comparer l'ecart anime/statique a un bruit du meme ordre que
# celui invoque par le protocole de verification.
bary_static2 = run_moving_scenario(animated=False, run_tag="s2")

diff_static_noise = float(np.linalg.norm(bary_static - bary_static2))
diff_animated_vs_static = float(np.linalg.norm(bary_animated - bary_static))

print(f"  barycentre statique   (run A) : {bary_static}")
print(f"  barycentre statique   (run B, bruit) : {bary_static2}")
print(f"  barycentre anime               : {bary_animated}")
print(f"  |statique - statique| (bruit)  : {diff_static_noise:.6f} m")
print(f"  |anime - statique|             : {diff_animated_vs_static:.6f} m")

check(
    "collider anime : deplace le barycentre bien au-dela du bruit run/run",
    diff_animated_vs_static > 5.0 * max(diff_static_noise, 1e-6),
    f"anime_vs_static={diff_animated_vs_static:.6f} bruit={diff_static_noise:.6f}",
)


# ---------------------------------------------------------------------------
# 4) Vitesse dans le bon repere : collider anime le long d'un axe MONDE.
# ---------------------------------------------------------------------------
print("\n=== 4) Vitesse dans le bon repere (axe monde -> axe solveur) ===")

scene = fresh_scene()
make_domain(scene, size=1.0, grid_res=20, ppc_axis=2, gravity=0.0)
sim, origin, size, dx = build_sim(scene)
water_obj, mat_id, n_emit = add_water_block(
    sim, origin, size, location=(0.0, 0.0, 0.0), dims=(0.2, 0.2, 0.2)
)

DELTA_Y_WORLD = 0.6
N_FRAMES_4 = 5
mover = add_animated_collider(
    (0.3, -0.3, 0.0), (0.3, -0.3 + DELTA_Y_WORLD, 0.0),
    frame_start=1, frame_end=N_FRAMES_4, radius=0.05, name="AxisMover",
)
collider_state = ops._ColliderState(mover, mover.bourrasque.friction)

captured = []
_orig_set_colliders = lib.Sim.set_colliders


def _spy_set_colliders(self, triangles, velocities, frictions):
    captured.append((np.array(triangles), np.array(velocities), np.array(frictions)))
    return _orig_set_colliders(self, triangles, velocities, frictions)


lib.Sim.set_colliders = _spy_set_colliders
try:
    fake = run_bake(scene, origin, size, dx, sim, [collider_state], n_frames=N_FRAMES_4)
finally:
    lib.Sim.set_colliders = _orig_set_colliders

fake._cleanup(bpy.context)
sim.destroy()

frame_dt = scene.render.fps_base / scene.render.fps
expected_speed_world_y = DELTA_Y_WORLD / (N_FRAMES_4 - 1) / frame_dt

# Frame 0 : pas de position precedente -> vitesse nulle (attendu).
tri0, vel0, fric0 = captured[0]
check("frame 0 : vitesse collider nulle (pas de position precedente)",
      bool(np.allclose(vel0, 0.0)), f"max|vel0|={np.abs(vel0).max():.4f}")

# Frame >= 1 : la vitesse solveur attendue est (0, 0, -expected_speed_world_y)
# sur CHAQUE sommet (deplacement RIGIDE, +Y monde pur).
tri1, vel1, fric1 = captured[1]
sample = vel1[0, 0]  # premier triangle, premier sommet
print(f"  echantillon vitesse solveur (triangle 0, sommet 0), frame 1 : {sample}")
print(f"  vitesse |Y monde| attendue : {expected_speed_world_y:.4f} m/s")

check("axe : vitesse solveur composante sx ~ 0",
      bool(np.allclose(vel1[..., 0], 0.0, atol=1e-3)), f"max={np.abs(vel1[...,0]).max():.4f}")
check("axe : vitesse solveur composante sy ~ 0",
      bool(np.allclose(vel1[..., 1], 0.0, atol=1e-3)), f"max={np.abs(vel1[...,1]).max():.4f}")
check(
    "axe : vitesse solveur composante sz == -vitesse Y monde (mapping correct)",
    bool(np.allclose(vel1[..., 2], -expected_speed_world_y, atol=1e-2)),
    f"sz sample={vel1[0,0,2]:.4f} attendu={-expected_speed_world_y:.4f}",
)
check("friction transmise correctement (constante par collider)",
      bool(np.allclose(fric1, mover.bourrasque.friction)),
      f"friction attendue={mover.bourrasque.friction}, recu={fric1[:3]}")


# ---------------------------------------------------------------------------
# 5) Frame restauree apres bake termine ET apres bake annule.
# ---------------------------------------------------------------------------
print("\n=== 5) Frame restauree (termine / annule) ===")

scene = fresh_scene()
make_domain(scene, size=1.0, grid_res=16, ppc_axis=1, gravity=-9.8)
scene.frame_set(7)
start_frame = scene.frame_current
sim, origin, size, dx = build_sim(scene)
water_obj, mat_id, n_emit = add_water_block(
    sim, origin, size, location=(0.0, 0.0, 0.0), dims=(0.2, 0.2, 0.2)
)
fake = run_bake(scene, origin, size, dx, sim, [], n_frames=10)
fake._cleanup(bpy.context)
check("frame restauree apres bake TERMINE", scene.frame_current == start_frame,
      f"start={start_frame} current={scene.frame_current}")
sim.destroy()

# Annulation : simule un ESC en plein milieu (quelques frames seulement,
# puis _cleanup direct — c'est exactement le chemin que suit BQ_OT_bake.modal
# sur ESC/cancel_requested).
scene2 = fresh_scene()
make_domain(scene2, size=1.0, grid_res=16, ppc_axis=1, gravity=-9.8)
scene2.frame_set(42)
start_frame2 = scene2.frame_current
sim2, origin2, size2, dx2 = build_sim(scene2)
water_obj2, mat_id2, n_emit2 = add_water_block(
    sim2, origin2, size2, dims=(0.2, 0.2, 0.2), location=(0.0, 0.0, 0.0)
)
fake2 = _FakeSelf(scene2)
fake2._sim = sim2
fake2._domain_transform = (origin2, size2)
bqd_path2, _ = cache.cache_paths(CACHE_DIR, "cancel_test")
cache.ensure_cache_dir(CACHE_DIR)
fake2._writer = cache.CacheWriter(bqd_path2, sim2.particle_count)
for i in range(3):
    fake2._frame_index = i
    fake2._advance_frame()
# "ESC" : cleanup immediat, sans finir les 10 frames prevues.
fake2._cleanup(bpy.context)
check("frame restauree apres bake ANNULE", scene2.frame_current == start_frame2,
      f"start={start_frame2} current={scene2.frame_current}")
sim2.destroy()


# ---------------------------------------------------------------------------
# RESULTAT
# ---------------------------------------------------------------------------
print("\n=== RESULTAT ===")
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
else:
    print("Toutes les verifications sont passees.")
sys.exit(1 if FAILURES else 0)
