"""Verification du jalon M17, phase B, tache B4 (cablage extension du
contact solide<->solide) -- a lancer avec :

    blender --background --factory-startup --python tools/repro/verify_contact_bake_b4.py

Meme harnais que `verify_rigidbody_bake.py` (phase A) : un `_FakeSelf`
minimal porte l'etat d'instance de `BQ_OT_bake`, les methodes numeriquement
sensibles etant les VRAIES methodes de production liees via
`types.MethodType`. `_setup_bake` reproduit ICI la partie "mise en place"
COMPLETE de `BQ_OT_bake.invoke` depuis M17/phase B : `_setup_collider_body`
pour TOUS les colliders (dynamiques ET fixes, D12), `_build_rigid_bodies`,
`set_collider_bodies`, PUIS `build_body_sdf`/`set_body_samples` pour tout
corps dont le maillage est ferme -- exactement l'ordre cable dans
`BQ_OT_bake.invoke`.

Couvre, cas par cas :
  1. LE CAS QUI COMPTE (verification 1 de la tache) : un cube dynamique
     lache au-dessus d'une boite fermee tres aplatie (collider STATIQUE)
     doit s'arreter dessus, pas la traverser. Sans fluide (isole le contact
     solide<->solide de tout couplage fluide<->solide).
  2. Meme scene, mais le collider est un PLANE (maillage ouvert) : le bake
     aboutit, un avertissement nommant l'objet est emis, ET le cube traverse
     (comportement attendu, desormais explique).
  3. Collider STATIQUE ANIME (keyframes) : verifie que son mouvement est
     bien suivi par le contact -- via `Sim.set_body_pose` (`bq_set_body_pose`,
     ABI 13), appelee CHAQUE FRAME pour tout collider cinematique dont le
     SDF de contact a ete construit (`_kinematic_pose_frame`, `ops.py`),
     par opposition a `Sim.set_collider_bodies` (une seule fois, au risque
     d'ecraser l'etat des corps dynamiques/le sommeil/le warm starting du
     contact -- voir sa docstring). Position ET vitesse (difference finie
     entre frames, linéaire et angulaire) sont transmises : sans la
     vitesse, un mur anime penetrerait puis serait repousse par la seule
     correction de position (contact mou).
  4. Non-regression : une scene de FLUIDE avec un collider STATIQUE et AUCUN
     corps dynamique se bake comme avant (positions finies, pas
     d'exception).
"""

import os
import sys
import types

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

ROOT = r"C:\Users\nicol\Code\bourrasque_v2"
sys.path.insert(0, ROOT)

if "scipy" not in sys.modules:
    _s = types.ModuleType("scipy")
    _sp = types.ModuleType("scipy.spatial")
    _sp.cKDTree = object
    _s.spatial = _sp
    sys.modules["scipy"] = _s
    sys.modules["scipy.spatial"] = _sp

import extension  # noqa: E402

extension.register()

import numpy as np  # noqa: E402

from extension import cache, lib, ops, rigidbody  # noqa: E402
from extension.props import domain_resolution, domain_transform  # noqa: E402

SCRATCHPAD = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRATCHPAD, "cache_out_m17_b4")
os.makedirs(CACHE_DIR, exist_ok=True)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Harnais (memes briques que verify_rigidbody_bake.py, phase A)
# ---------------------------------------------------------------------------


class _FakeSelf:
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
        self._dynamic_collider_states = []
        self._body_track = None
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
        self._post_keyframes = types.MethodType(
            ops.BQ_OT_bake._post_keyframes, self
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


def make_domain(scene, size=0.64, grid_res=64, ppc_axis=2, gravity=-9.8, cfl=0.3):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = grid_res
    scene.bourrasque.ppc_axis = ppc_axis
    scene.bourrasque.gravity = gravity
    scene.bourrasque.cfl = cfl
    scene.bourrasque.max_particles = 500_000
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
    from extension.props import emitter_bounds_solver, world_to_solver_dir

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


def add_collider(
    location,
    dims,
    name="Collider",
    dynamic=False,
    density=500.0,
    friction=0.4,
    use_gravity=True,
    lock_location=(False, False, False),
    lock_rotation=(False, False, False),
    restitution=0.0,
):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = dims
    bpy.context.view_layer.update()
    op = obj.bourrasque
    op.role = "COLLIDER"
    op.friction = friction
    op.dynamic = dynamic
    op.density = density
    op.use_gravity = use_gravity
    op.lock_location = lock_location
    op.lock_rotation = lock_rotation
    op.restitution = restitution
    return obj


def add_plane_collider(location, name="OpenPlane", size=0.9):
    """Plane -- ouvert par construction. Collider STATIQUE (dynamic=False,
    D14 : jamais refuse au bake meme ouvert)."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=location)
    obj = bpy.context.active_object
    obj.name = name
    bpy.context.view_layer.update()
    op = obj.bourrasque
    op.role = "COLLIDER"
    op.dynamic = False
    op.friction = 0.4
    return obj


def add_animated_floor(loc_start, loc_end, dims, frame_start, frame_end,
                        name="RisingFloor", friction=0.4):
    obj = add_collider(loc_start, dims, name=name, dynamic=False, friction=friction)
    obj.location = loc_start
    obj.keyframe_insert(data_path="location", frame=frame_start)
    obj.location = loc_end
    obj.keyframe_insert(data_path="location", frame=frame_end)
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


def _setup_bake(scene, sim, origin, size, dx, collider_objs, n_frames):
    """Reproduit la partie "mise en place colliders" de `BQ_OT_bake.invoke`
    depuis M17/phase B (D9, D12) : `_setup_collider_body` pour TOUS les
    colliders, `_build_rigid_bodies`/`set_collider_bodies`, PUIS
    `build_body_sdf`/`set_body_samples` pour tout corps dont le maillage
    est ferme -- meme ordre que la production.

    Renvoie `(fake, error_message_or_None, open_static_names)`.
    """
    fake = _FakeSelf(scene)
    fake._sim = sim
    fake._domain_transform = (origin, size)

    collider_states = [
        ops._ColliderState(obj, obj.bourrasque.friction) for obj in collider_objs
    ]
    for idx, state in enumerate(collider_states):
        state.body_index = idx
    dynamic_states = [s for s in collider_states if s.dynamic]

    fake._collider_states = collider_states
    fake._dynamic_collider_states = dynamic_states

    bqd_path, _ = cache.cache_paths(CACHE_DIR, scene.name + f"_{id(fake)}")
    cache.ensure_cache_dir(CACHE_DIR)
    fake._writer = cache.CacheWriter(bqd_path, sim.particle_count)

    open_static_names = []
    if collider_states:
        fake._frame_index = 0
        fake._advance_scene_frame()
        for state in collider_states:
            try:
                ops._setup_collider_body(state, fake._depsgraph, origin, size)
            except ValueError as exc:
                return fake, str(exc), open_static_names
            if not state.dynamic and not state.mesh_closed:
                open_static_names.append(state.obj.name)

        if open_static_names:
            # Meme message que BQ_OT_bake.invoke (point 3 de B4) --
            # reproduit ici pour exercer le CONTENU du message, pas
            # seulement la detection sous-jacente.
            joined = ", ".join(f"« {n} »" for n in open_static_names)
            fake.report(
                {"WARNING"},
                f"Maillage non fermé, contact solide↔solide "
                f"désactivé pour : {joined}. Ces colliders restent "
                "valides pour le fluide, mais ne peuvent pas "
                "arrêter un solide dynamique : utilisez une boîte "
                "fermée (même très aplatie) si vous voulez qu'ils "
                "en arrêtent un.",
            )

        bodies = ops._build_rigid_bodies(collider_states)
        sim.set_collider_bodies(bodies)

        for state in collider_states:
            if not state.body_sdf_built:
                continue
            tri_local = state.rest_offsets[state.rest_tris].astype(np.float32)
            sim.build_body_sdf(state.body_index, tri_local, dx, max_res=128)
            samples = rigidbody.surface_samples(
                state.rest_offsets, state.rest_tris, dx, 20000
            )
            sim.set_body_samples(state.body_index, samples)

        fake._body_track = []

    for i in range(n_frames):
        fake._frame_index = i
        fake._advance_frame()
        if fake._body_track is not None:
            fake._body_track.append(sim.read_collider_bodies().copy())

    return fake, None, open_static_names


# ---------------------------------------------------------------------------
# 1) LE CAS QUI COMPTE : cube dynamique sur boite fermee aplatie
# ---------------------------------------------------------------------------
print("\n=== 1) Cube dynamique sur boîte fermée aplatie (LE CAS QUI COMPTE) ===")

# Domaine 0.64 m, grille 64 -> dx = 0.01 m (10 mm), meme ordre de grandeur
# que la mesure coeur (penetration residuelle 3,1 mm a dx=10mm).
FLOOR_TOP_Z = -0.25
FLOOR_DIMS = (0.5, 0.5, 0.03)  # 3 cm d'epaisseur -- "tres aplatie"
FLOOR_CENTER_Z = FLOOR_TOP_Z - FLOOR_DIMS[2] / 2.0
CUBE_SIZE = 0.10
CUBE_START_Z = 0.05
EXPECTED_REST_Z = FLOOR_TOP_Z + CUBE_SIZE / 2.0
N_FRAMES_SETTLE = 96  # 4 s a 24 fps


def test_cube_stops_on_closed_floor():
    scene = fresh_scene()
    make_domain(scene)
    sim, origin, size, dx = build_sim(scene)

    sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)  # aucune emission -- requis par bq_set_colliders
    floor = add_collider(
        (0, 0, FLOOR_CENTER_Z), FLOOR_DIMS, name="Floor", dynamic=False,
    )
    cube = add_collider(
        (0, 0, CUBE_START_Z), (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), name="Cube",
        dynamic=True, density=500.0, use_gravity=True,
    )

    fake, error, _open = _setup_bake(
        scene, sim, origin, size, dx, [floor, cube], N_FRAMES_SETTLE
    )
    check("bake aboutit sans exception", error is None, detail=str(error))
    if error is not None:
        fake._cleanup(bpy.context)
        sim.destroy()
        return

    body_state = sim.read_collider_bodies()
    cube_state = fake._collider_states[[s.obj for s in fake._collider_states].index(cube)]
    x_solver = body_state[cube_state.body_index, 0:3]
    from extension.transform import solver_to_world

    x_world = solver_to_world(tuple(x_solver), origin, size)
    final_z = x_world[2]

    penetration = EXPECTED_REST_Z - final_z  # >0 si le cube s'enfonce
    v_solver = body_state[cube_state.body_index, 7:10]
    speed = float(np.linalg.norm(v_solver))

    print(f"   hauteur finale (z monde)  = {final_z:.5f} m")
    print(f"   hauteur attendue (repos)  = {EXPECTED_REST_Z:.5f} m")
    print(f"   pénétration               = {penetration * 1000.0:.3f} mm (dx = {dx * 1000.0:.2f} mm)")
    print(f"   |v| finale                = {speed:.5f} m/s")

    check(
        "le cube s'est arrêté SUR la boîte (pas à travers)",
        abs(penetration) < 2.0 * dx and final_z > FLOOR_TOP_Z,
        f"final_z={final_z:.5f}, attendu≈{EXPECTED_REST_Z:.5f}, floor_top={FLOOR_TOP_Z:.5f}",
    )
    check(
        "le cube est immobile (endormi ou quasi)",
        speed < 0.05,
        f"|v|={speed:.5f} m/s",
    )

    fake._cleanup(bpy.context)
    sim.destroy()


test_cube_stops_on_closed_floor()


# ---------------------------------------------------------------------------
# 2) Cube dynamique au-dessus d'un PLANE (maillage ouvert) : traverse
# ---------------------------------------------------------------------------
print("\n=== 2) Cube dynamique au-dessus d'un plane (maillage ouvert) ===")


def test_cube_passes_through_open_plane():
    scene = fresh_scene()
    make_domain(scene)
    sim, origin, size, dx = build_sim(scene)

    sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)  # aucune emission -- requis par bq_set_colliders
    plane = add_plane_collider((0, 0, FLOOR_TOP_Z), name="OpenPlane", size=0.9)
    cube = add_collider(
        (0, 0, CUBE_START_Z), (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), name="Cube2",
        dynamic=True, density=500.0, use_gravity=True,
    )

    fake, error, open_names = _setup_bake(
        scene, sim, origin, size, dx, [plane, cube], N_FRAMES_SETTLE
    )
    check("bake aboutit sans exception (pas de refus)", error is None, detail=str(error))

    check(
        "avertissement émis, nommant le plane",
        any(plane.name in msg for _tags, msg in fake.reports),
        f"reports={[msg for _t, msg in fake.reports]!r}",
    )
    check(
        "le plane est bien détecté ouvert (mesh_closed=False)",
        plane.name in [
            s.obj.name for s in fake._collider_states
            if not s.dynamic and not s.mesh_closed
        ],
    )

    if error is None:
        body_state = sim.read_collider_bodies()
        cube_state = fake._collider_states[
            [s.obj for s in fake._collider_states].index(cube)
        ]
        x_solver = body_state[cube_state.body_index, 0:3]
        from extension.transform import solver_to_world

        final_z = solver_to_world(tuple(x_solver), origin, size)[2]
        print(f"   hauteur finale (z monde) = {final_z:.5f} m (attendu très sous {FLOOR_TOP_Z:.5f})")
        check(
            "le cube a traversé le plane (comportement attendu, expliqué)",
            final_z < FLOOR_TOP_Z - 0.03,
            f"final_z={final_z:.5f}",
        )

    fake._cleanup(bpy.context)
    sim.destroy()


test_cube_passes_through_open_plane()


# ---------------------------------------------------------------------------
# 3) Collider STATIQUE ANIMÉ : le contact suit-il sa pose ?
# ---------------------------------------------------------------------------
print("\n=== 3) Collider statique animé (keyframes) : le contact suit-il ? ===")


def test_animated_static_collider_pushes_cube():
    scene = fresh_scene()
    make_domain(scene)
    sim, origin, size, dx = build_sim(scene)

    # Le sol monte de FLOOR_TOP_Z a FLOOR_TOP_Z + 0.15 sur les frames
    # [1, N_FRAMES_SETTLE] -- un cube pose dessus devrait etre pousse vers
    # le haut d'autant si le contact suit la pose animee.
    sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)  # aucune emission -- requis par bq_set_colliders
    rise = 0.15
    floor = add_animated_floor(
        loc_start=(0, 0, FLOOR_CENTER_Z),
        loc_end=(0, 0, FLOOR_CENTER_Z + rise),
        dims=FLOOR_DIMS,
        frame_start=1, frame_end=N_FRAMES_SETTLE,
        name="RisingFloor",
    )
    cube = add_collider(
        (0, 0, EXPECTED_REST_Z), (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE), name="Cube3",
        dynamic=True, density=500.0, use_gravity=True,
    )

    fake, error, _open = _setup_bake(
        scene, sim, origin, size, dx, [floor, cube], N_FRAMES_SETTLE
    )
    check("bake aboutit sans exception", error is None, detail=str(error))
    if error is not None:
        fake._cleanup(bpy.context)
        sim.destroy()
        return

    body_state = sim.read_collider_bodies()
    cube_state = fake._collider_states[
        [s.obj for s in fake._collider_states].index(cube)
    ]
    x_solver = body_state[cube_state.body_index, 0:3]
    from extension.transform import solver_to_world

    final_z = solver_to_world(tuple(x_solver), origin, size)[2]
    expected_if_tracked = EXPECTED_REST_Z + rise

    print(f"   sol : {FLOOR_TOP_Z:.5f} m -> {FLOOR_TOP_Z + rise:.5f} m (monte de {rise*1000:.0f} mm)")
    print(f"   cube : hauteur initiale = {EXPECTED_REST_Z:.5f} m")
    print(f"   cube : hauteur finale   = {final_z:.5f} m")
    print(f"   attendu SI le contact suit l'animation : ≈ {expected_if_tracked:.5f} m")
    tracked = abs(final_z - expected_if_tracked) < 2.0 * dx
    check(
        "le cube a été poussé par le sol animé (contact suit la pose)",
        tracked,
        f"final_z={final_z:.5f}, attendu≈{expected_if_tracked:.5f} (Sim.set_body_pose par frame)",
    )

    fake._cleanup(bpy.context)
    sim.destroy()


test_animated_static_collider_pushes_cube()


# ---------------------------------------------------------------------------
# 4) Non-régression : fluide + collider statique, aucun corps dynamique
# ---------------------------------------------------------------------------
print("\n=== 4) Non-régression : fluide + collider statique, sans corps dynamique ===")


def test_fluid_static_collider_no_dynamic_unchanged():
    scene = fresh_scene()
    make_domain(scene, size=1.0, grid_res=28)
    sim, origin, size, dx = build_sim(scene)

    add_water_block(sim, origin, size, location=(0, 0, 0.05), dims=(0.4, 0.4, 0.4))
    floor = add_collider((0, 0, -0.45), (0.9, 0.9, 0.06), name="FloorNoDyn", dynamic=False)

    fake, error, _open = _setup_bake(scene, sim, origin, size, dx, [floor], 6)
    check("bake aboutit sans exception", error is None, detail=str(error))
    finite = error is None and bool(np.all(np.isfinite(sim.read_positions())))
    check("positions finies (pas de divergence)", finite)

    fake._cleanup(bpy.context)
    sim.destroy()


test_fluid_static_collider_no_dynamic_unchanged()


print()
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
    sys.exit(1)
print("Toutes les verifications M17 (phase B, B4, cote extension) sont passees.")
