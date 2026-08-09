"""Verification du jalon M17, phase A (colliders dynamiques) cote extension,
a lancer avec :

    blender --background --factory-startup --python tools/repro/verify_rigidbody_bake.py

Reprend le harnais de `verify_m6.py` : un objet `_FakeSelf` minimal porte
l'etat d'instance de `BQ_OT_bake`, mais les methodes numeriquement sensibles
(`_advance_frame`, `_advance_scene_frame`, `_update_colliders`,
`_emit_inflow_sites`, `_post_keyframes`, `_cleanup`) sont les VRAIES methodes
de production liees via `types.MethodType` -- c'est le code livre dans
extension/ops.py qui est exerce, pas une reimplementation parallele.
`BQ_OT_bake.invoke()` ne peut pas etre appele directement en `--background`
(pas de fenetre, `wm.event_timer_add` echoue) : ce script reproduit donc la
partie "mise en place" (creation de la Sim, materiaux, emission, collecte
des colliders, `_setup_dynamic_collider`/`_build_rigid_bodies`/
`set_collider_bodies`) en appelant les memes fonctions de production.

Couvre, cas par cas :
  1. Bake complet (domaine + emetteur + un collider STATIQUE + un collider
     DYNAMIQUE) : aboutit sans exception, clés posees UNIQUEMENT sur le
     dynamique, invariant de repos verifie, `bq.clear_rigid_keys` les retire.
  2. Non-regression : une scene SANS AUCUN collider dynamique (colliders
     tous fixes) produit un bake au comportement inchange (pas d'exception,
     aucune clé posee nulle part).
  3. Un collider dynamique au maillage OUVERT fait echouer le bake avec un
     message nommant l'objet.
  4. Un collider FIXE ANIME (deplace par keyframes) continue de transmettre
     sa vitesse au fluide (difference finie, chemin M6 inchange), et
     `bq.clear_rigid_keys` NE retire RIEN de son animation (piège du
     jalon : l'operateur cible les corps dynamiques, jamais tous les
     colliders).
"""

import os
import sys
import types

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

ROOT = r"C:\Users\nicol\Code\bourrasque_v2"
sys.path.insert(0, ROOT)

# scipy est fourni en wheel bundle, installe par le gestionnaire d'extensions
# de Blender -- absent quand on importe l'arbre source en --factory-startup
# (meme garde que verify_material_ops.py). Aucun des chemins testes ici ne
# le touche (display.py l'importe transitivement via foam.py).
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

from extension import cache, lib, ops  # noqa: E402
from extension.props import domain_resolution, domain_transform  # noqa: E402
from extension.transform import solver_to_world_array  # noqa: E402

SCRATCHPAD = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRATCHPAD, "cache_out_m17")
os.makedirs(CACHE_DIR, exist_ok=True)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------------------
# Harnais (memes briques que verify_m6.py)
# ---------------------------------------------------------------------------


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


def make_domain(scene, size=1.0, grid_res=28, ppc_axis=2, gravity=-9.8, cfl=0.3):
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
    friction=0.2,
    use_gravity=True,
    added_mass=0.0,
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
    op.added_mass = added_mass
    op.lock_location = lock_location
    op.lock_rotation = lock_rotation
    op.restitution = restitution
    return obj


def add_open_mesh_collider(location, name="OpenPlane", dynamic=True):
    """Un simple quad -- ouvert par construction (2 aretes de bord)."""
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    bpy.context.view_layer.update()
    op = obj.bourrasque
    op.role = "COLLIDER"
    op.dynamic = dynamic
    op.density = 500.0
    return obj


def add_animated_fixed_collider(loc_start, loc_end, frame_start, frame_end,
                                 dims=(0.1, 0.1, 0.4), name="MovingWall",
                                 friction=0.2):
    obj = add_collider(loc_start, dims, name=name, dynamic=False, friction=friction)
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


def _setup_bake(scene, sim, origin, size, collider_objs, n_frames):
    """Reproduit la partie "mise en place colliders" de `BQ_OT_bake.invoke`
    (assignation de body_index, capture du repos des dynamiques,
    declaration des corps aupres du solveur) puis avance `n_frames` via
    `_advance_frame` (methode de production), en accumulant `body_track`
    exactement comme `_bake_worker`.

    `collider_objs` : liste d'objets Blender deja marques COLLIDER.
    Renvoie `(fake, error_message_or_None)` -- une ValueError levee par
    `_setup_dynamic_collider` (maillage ouvert / volume degenere) est
    capturee et renvoyee comme message plutot que de se propager, pour que
    l'appelant verifie le refus sans un try/except duplique a chaque site
    d'appel.
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

    if collider_states:
        fake._frame_index = 0
        fake._advance_scene_frame()
        for state in dynamic_states:
            try:
                ops._setup_dynamic_collider(state, fake._depsgraph, origin, size)
            except ValueError as exc:
                return fake, str(exc)
        bodies = ops._build_rigid_bodies(collider_states)
        sim.set_collider_bodies(bodies)
        fake._body_track = []

    for i in range(n_frames):
        fake._frame_index = i
        fake._advance_frame()
        if fake._body_track is not None:
            fake._body_track.append(sim.read_collider_bodies().copy())

    return fake, None


def fcurve_count(obj):
    """Nombre de F-curves de l'action courante de `obj` -- gere aussi bien
    les actions "legacy" (Blender < 4.4, `action.fcurves` direct) que les
    actions "layered" (Blender 4.4+/5.x, F-curves dans un ActionChannelbag,
    meme motif que `add_animated_fixed_collider`)."""
    anim = obj.animation_data
    if anim is None or anim.action is None:
        return 0
    action = anim.action
    try:
        return len(action.fcurves)
    except AttributeError:
        pass
    total = 0
    for layer in action.layers:
        for strip in layer.strips:
            channelbag = strip.channelbag(anim.action_slot)
            if channelbag is not None:
                total += len(channelbag.fcurves)
    return total


def keyed_objects(scene):
    return {obj.name for obj in scene.objects if fcurve_count(obj) > 0}


# ---------------------------------------------------------------------------
# 1) Bake complet : un collider statique + un collider dynamique
# ---------------------------------------------------------------------------
print("\n=== 1) Bake complet (statique + dynamique) ===")


def test_full_bake():
    scene = fresh_scene()
    make_domain(scene, size=1.0, grid_res=28, ppc_axis=2, gravity=-9.8)
    sim, origin, size, dx = build_sim(scene)

    add_water_block(sim, origin, size, location=(0, 0, 0.05), dims=(0.5, 0.5, 0.5))

    static_obj = add_collider(
        (0, 0, -0.45), (0.9, 0.9, 0.06), name="Floor", dynamic=False, friction=0.3,
    )
    dynamic_obj = add_collider(
        (0.0, 0.0, 0.15), (0.12, 0.12, 0.12), name="Cube", dynamic=True,
        density=500.0, use_gravity=True,
    )
    m0_before = np.array(dynamic_obj.matrix_world, dtype=np.float64)

    n_frames = 5
    fake, error = _setup_bake(
        scene, sim, origin, size, [static_obj, dynamic_obj], n_frames
    )
    check("bake complet : aboutit sans exception", error is None, detail=str(error))
    if error is not None:
        fake._cleanup(bpy.context)
        sim.destroy()
        return

    # Capture AVANT _cleanup (qui reinitialise _dynamic_collider_states).
    from extension import rigidbody

    state = fake._dynamic_collider_states[0]
    com0_world = state.com0_world.copy()
    m0 = state.m0.copy()

    fake._post_keyframes(bpy.context)
    fake._cleanup(bpy.context)

    keyed = keyed_objects(scene)
    check(
        "clés posées SEULEMENT sur le collider dynamique",
        keyed == {"Cube"},
        f"got={keyed!r}",
    )

    # Invariant D8 : a l'etat DECLARE (x = com0, q = identite -- avant tout
    # pas de solveur), compose_body_transform rend m0 BIT A BIT. Verifie ici
    # que le cablage d'espaces de _post_keyframes (com0_world/m0 captures
    # par _setup_dynamic_collider) reproduit fidelement la transformation
    # d'origine de l'objet -- pas seulement l'invariant deja verifie de
    # maniere isolee dans rigidbody.py.
    m_rest = rigidbody.compose_body_transform(
        com0_world, (1.0, 0.0, 0.0, 0.0), com0_world, m0
    )
    check(
        "invariant de repos : compose_body_transform(état déclaré) == m0",
        bool(np.allclose(m_rest, m0_before, atol=1e-9)),
        f"max ecart={float(np.max(np.abs(m_rest - m0_before))):.3e}",
    )

    # Le cube est tombe sous la gravite : sa position finale doit s'ecarter
    # sensiblement de sa position initiale -- confirme que le pipeline a
    # reellement fait tourner la physique (pas un no-op qui passerait
    # l'invariant ci-dessus par accident).
    moved = np.linalg.norm(np.array(dynamic_obj.location) - m0_before[:3, 3])
    check(
        "le collider dynamique a réellement bougé sous la gravité",
        moved > 1e-4,
        f"déplacement={moved:.3e} m",
    )

    bpy.ops.bq.clear_rigid_keys()
    keyed_after = keyed_objects(scene)
    check(
        "bq.clear_rigid_keys retire les clés du collider dynamique",
        "Cube" not in keyed_after,
        f"got={keyed_after!r}",
    )

    sim.destroy()


test_full_bake()


# ---------------------------------------------------------------------------
# 2) Non-regression : aucun collider dynamique
# ---------------------------------------------------------------------------
print("\n=== 2) Non-régression (dynamic=False partout) ===")


def test_no_dynamic_collider_unchanged():
    scene = fresh_scene()
    make_domain(scene, size=1.0, grid_res=24, ppc_axis=2, gravity=-9.8)
    sim, origin, size, dx = build_sim(scene)

    add_water_block(sim, origin, size, location=(0, 0, 0.05), dims=(0.4, 0.4, 0.4))
    static_a = add_collider((0, 0, -0.45), (0.9, 0.9, 0.06), name="FloorA", dynamic=False)
    static_b = add_collider((0.3, 0.0, 0.0), (0.1, 0.4, 0.4), name="WallB", dynamic=False)

    n_frames = 4
    fake, error = _setup_bake(scene, sim, origin, size, [static_a, static_b], n_frames)
    check(
        "sans collider dynamique : bake aboutit sans exception",
        error is None,
        detail=str(error),
    )
    positions_finite = error is None and bool(
        np.all(np.isfinite(sim.read_positions()))
    )
    if error is None:
        fake._post_keyframes(bpy.context)
    fake._cleanup(bpy.context)  # detruit sim (self._sim) -- lire AVANT, pas apres

    keyed = keyed_objects(scene)
    check(
        "sans collider dynamique : aucune clé posée nulle part",
        len(keyed) == 0,
        f"got={keyed!r}",
    )
    check(
        "sans collider dynamique : positions finies (pas de divergence)",
        positions_finite,
    )


test_no_dynamic_collider_unchanged()


# ---------------------------------------------------------------------------
# 3) Collider dynamique au maillage ouvert : refus nommant l'objet
# ---------------------------------------------------------------------------
print("\n=== 3) Refus (maillage ouvert, collider dynamique) ===")


def test_open_mesh_dynamic_refused():
    scene = fresh_scene()
    make_domain(scene, size=1.0, grid_res=24, ppc_axis=2, gravity=-9.8)
    sim, origin, size, dx = build_sim(scene)

    add_water_block(sim, origin, size, location=(0, 0, 0.05), dims=(0.3, 0.3, 0.3))
    open_obj = add_open_mesh_collider((0.0, 0.0, 0.1), name="OuvertDyn", dynamic=True)

    fake, error = _setup_bake(scene, sim, origin, size, [open_obj], 1)
    check(
        "maillage ouvert dynamique : le bake est refusé",
        error is not None,
    )
    check(
        "maillage ouvert dynamique : le message nomme l'objet",
        error is not None and open_obj.name in error,
        f"message={error!r}",
    )

    fake._cleanup(bpy.context)
    sim.destroy()


test_open_mesh_dynamic_refused()


# ---------------------------------------------------------------------------
# 4) Collider fixe animé : transmet toujours sa vitesse, clear_rigid_keys
#    ne touche pas à son animation
# ---------------------------------------------------------------------------
print("\n=== 4) Collider fixe animé (piège clear_rigid_keys) ===")


def test_fixed_animated_collider_untouched():
    scene = fresh_scene()
    make_domain(scene, size=1.0, grid_res=24, ppc_axis=2, gravity=-9.8)
    sim, origin, size, dx = build_sim(scene)

    add_water_block(sim, origin, size, location=(0, 0, 0.05), dims=(0.3, 0.3, 0.3))
    wall = add_animated_fixed_collider(
        loc_start=(-0.3, 0.0, 0.0), loc_end=(0.3, 0.0, 0.0),
        frame_start=1, frame_end=10, name="MovingWall",
    )
    # Un collider DYNAMIQUE est aussi present dans cette scene : sans lui,
    # `bq.clear_rigid_keys` refuserait de s'executer (poll -- aucun corps a
    # nettoyer), ce qui ne permettrait pas d'exercer le VRAI piège
    # (discrimination dynamique/fixe dans execute()) sur une scène mixte,
    # le cas d'usage reel.
    floater = add_collider(
        (0.0, 0.3, 0.15), (0.1, 0.1, 0.1), name="FloatingCube", dynamic=True,
        density=500.0,
    )
    n_fcurves_before = fcurve_count(wall)
    check(
        "collider fixe animé : porte bien ses propres clés avant le bake",
        n_fcurves_before > 0,
    )

    fake, error = _setup_bake(scene, sim, origin, size, [wall, floater], 5)
    check("collider fixe animé : bake aboutit sans exception", error is None)

    # Vitesse transmise au fluide : difference finie MONDE (chemin M6
    # inchange, cf. _collect_collider_frame) -- non nulle des la 2e frame
    # (la 1re n'a pas de position precedente).
    fake._frame_index = 2
    fake._advance_scene_frame()
    origin_dt, size_dt = fake._domain_transform
    _tri, vel_all, _fric, _body = ops._collect_collider_frame(
        fake._collider_states, fake._depsgraph, origin_dt, size_dt,
        fake._frame_dt, fake.report,
    )
    check(
        "collider fixe animé : transmet une vitesse non nulle au fluide",
        vel_all.size > 0 and bool(np.any(np.abs(vel_all) > 1e-6)),
    )

    if error is None:
        fake._post_keyframes(bpy.context)
    fake._cleanup(bpy.context)

    # Le bake ne doit rien AJOUTER a ses cles (il n'est pas dynamique) : le
    # nombre de F-curves ne doit pas avoir change.
    n_fcurves_after_bake = fcurve_count(wall)
    check(
        "collider fixe animé : le bake ne pose aucune clé sur lui",
        n_fcurves_after_bake == n_fcurves_before,
        f"{n_fcurves_before} -> {n_fcurves_after_bake}",
    )
    check(
        "collider dynamique de la même scène : reçoit bien ses clés",
        "FloatingCube" in keyed_objects(scene),
    )

    bpy.ops.bq.clear_rigid_keys()
    n_fcurves_after_clear = fcurve_count(wall)
    check(
        "bq.clear_rigid_keys NE RETIRE RIEN de l'animation d'un collider fixe",
        n_fcurves_after_clear == n_fcurves_before,
        f"{n_fcurves_before} -> {n_fcurves_after_clear}",
    )
    check(
        "bq.clear_rigid_keys retire bien les clés du collider dynamique voisin",
        "FloatingCube" not in keyed_objects(scene),
    )

    sim.destroy()


test_fixed_animated_collider_untouched()


print()
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
    sys.exit(1)
print("Toutes les verifications M17 (phase A, cote extension) sont passees.")
