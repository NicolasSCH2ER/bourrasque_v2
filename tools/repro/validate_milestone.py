"""Script de validation du jalon inflow/mesh-emission, a lancer avec :

    blender --background --python validate_milestone.py

Rejoue la logique de mise en place de BQ_OT_bake.invoke() (setup des
emetteurs, incluant l'echantillonnage mesh et la construction des
_InflowState), mais appelle ensuite REELLEMENT les methodes factorisees
BQ_OT_bake._advance_frame / _emit_due_inflow_layers / etc. sur un objet
"self" minimal (pas un vrai bpy.types.Operator invoque via bpy.ops,
puisque le modal ne tourne pas en --background) : le coeur numerique
teste est exactement celui livre dans extension/ops.py.
"""

import math
import os
import sys
import traceback
import types

import bpy

# Un seul reset factory, tout au debut : `read_factory_settings` desenregistre
# et reenregistre tous les addons/extensions (dont une eventuelle
# installation officielle de "bourrasque" sous
# extensions/user_default/bourrasque), ce qui entre en conflit si on
# l'appelle APRES avoir importe et enregistre notre copie de travail. On ne
# reset donc qu'une fois, puis on reutilise/nettoie la meme scene entre les
# sections de ce script (voir `fresh_scene`).
bpy.ops.wm.read_factory_settings(use_empty=True)

ROOT = r"C:\Users\nicol\Code\bourrasque_v2"
sys.path.insert(0, ROOT)

import extension  # noqa: E402

extension.register()

from extension import cache, lib, ops, sampling  # noqa: E402
from extension.props import (  # noqa: E402
    domain_transform,
    emitter_bounds_solver,
    world_to_solver_dir,
)

CACHE_DIR = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\cache_out"
os.makedirs(CACHE_DIR, exist_ok=True)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class _FakeSelf:
    """Mime le sous-ensemble d'attributs de BQ_OT_bake utilise par
    _advance_frame / _emit_due_inflow_layers, sans passer par bpy.ops
    (le modal ne tourne pas en --background)."""

    def __init__(self):
        self._sim = None
        self._writer = None
        self._pos_buffer = None
        self._frame_dt = 1.0 / 24.0
        self._inflow_states = []
        self._saturated = False
        # Gravite nulle (voir build_sim : cfg.gravity_y = 0.0) et pas de
        # bornes utiles pour ce script (les scenarios ici restent dans le
        # domaine sans avoir besoin de filtrage) : voir BQ_OT_bake._gravity_y
        # / ._usable_bounds, desormais lus par _emit_due_inflow_layers.
        self._gravity_y = 0.0
        self._usable_bounds = None
        self.reports = []
        # Lie les VRAIES methodes de BQ_OT_bake a cette instance factice :
        # c'est exactement le code de production qui est exerce, pas une
        # reimplementation parallele pour les besoins du test.
        self._emit_due_inflow_layers = types.MethodType(
            ops.BQ_OT_bake._emit_due_inflow_layers, self
        )
        self._advance_frame = types.MethodType(ops.BQ_OT_bake._advance_frame, self)

    def report(self, tags, msg):
        self.reports.append((tuple(tags), msg))
        print("   report:", tags, msg)


def fresh_scene():
    """Nettoie la scene courante (objets, roles) plutot que de refaire un
    reset factory complet (voir commentaire en tete de script)."""
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.bourrasque.domain_object = None
    return scene


def make_domain(scene, size=1.0):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = 32
    scene.bourrasque.ppc_axis = 2
    scene.bourrasque.max_particles = 2_000_000
    return domain


def setup_emitter_bounds_box(name, location, size, emit_mode, emit_source="BOUNDS",
                              velocity=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = size
    bpy.context.view_layer.update()
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.emit_mode = emit_mode
    obj.bourrasque.emit_source = emit_source
    obj.bourrasque.model = "WATER"
    obj.bourrasque.initial_velocity = velocity
    return obj


def setup_emitter_torus(name, location, major=0.15, minor=0.06, emit_mode="BLOCK",
                         emit_source="MESH"):
    bpy.ops.mesh.primitive_torus_add(
        location=location, major_radius=major, minor_radius=minor,
        major_segments=32, minor_segments=16,
    )
    obj = bpy.context.active_object
    obj.name = name
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.emit_mode = emit_mode
    obj.bourrasque.emit_source = emit_source
    obj.bourrasque.model = "WATER"
    return obj


def build_sim(scene, domain_size=1.0):
    origin, size = domain_transform(scene)
    cfg = lib.default_config()
    cfg.grid_res = scene.bourrasque.grid_res
    cfg.domain = size
    cfg.gravity_y = 0.0  # gravite coupee pour la verification "pas d'explosion"/comptage
    cfg.cfl = scene.bourrasque.cfl
    cfg.ppc_axis = scene.bourrasque.ppc_axis
    cfg.max_particles = scene.bourrasque.max_particles
    sim = lib.Sim(cfg)
    return sim, origin, size


def build_bake_setup(sim, scene, origin, size, emitters, fake):
    """Reproduit la boucle de mise en place de BQ_OT_bake.invoke() (sans
    bpy.ops), en appelant les memes fonctions que ops.py."""
    props = scene.bourrasque
    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)

    dx = size / props.grid_res
    spacing = dx / props.ppc_axis

    total_emitted = 0
    fake._inflow_states = []

    for obj in emitters:
        op = obj.bourrasque
        vel = world_to_solver_dir(tuple(op.initial_velocity))

        mesh_points = None
        if op.emit_source == "MESH":
            closed, message = sampling.check_mesh_closed(obj)
            if not closed:
                raise RuntimeError(f"{obj.name}: {message}")
            mesh_points = sampling.sample_mesh_interior(
                obj, origin, size, props.grid_res, props.ppc_axis
            )
            if mesh_points.shape[0] == 0:
                raise RuntimeError(f"{obj.name}: 0 points echantillonnes")

        if op.emit_mode == "BLOCK":
            if op.emit_source == "MESH":
                total_emitted += sim.emit_points(mat_id, mesh_points, vel=vel)
            else:
                lo, hi = emitter_bounds_solver(obj, origin, size)
                total_emitted += sim.emit_box(mat_id, lo, hi, vel=vel)
            continue

        speed = math.sqrt(sum(c * c for c in vel))
        axis, sign = ops._dominant_axis_sign(vel)
        if op.emit_source == "MESH":
            layer_points = ops._mesh_inflow_layer(mesh_points, axis, sign, spacing)
        else:
            lo, hi = emitter_bounds_solver(obj, origin, size)
            layer_points = ops._bounds_inflow_layer(lo, hi, axis, sign, spacing)

        if layer_points.shape[0] == 0:
            raise RuntimeError(f"{obj.name}: couche d'inflow vide")

        fake._inflow_states.append(
            ops._InflowState(obj.name, mat_id, vel, speed, spacing, layer_points)
        )

    return total_emitted, mat_id, spacing


# ---------------------------------------------------------------------------
# 1) Emission par maillage : tore, aucune particule dans le trou central
# ---------------------------------------------------------------------------
print("\n=== 1) Emission par maillage (tore, BLOCK/MESH) ===")
try:
    scene = fresh_scene()
    make_domain(scene, size=1.0)
    torus = setup_emitter_torus("Torus", (0, 0, 0), major=0.15, minor=0.06,
                                 emit_mode="BLOCK", emit_source="MESH")
    sim, origin, size = build_sim(scene)
    fake = _FakeSelf()
    total, mat_id, spacing = build_bake_setup(sim, scene, origin, size, [torus], fake)
    check("tore : particules emises > 0", total > 0, f"total={total}")

    pos = sim.read_positions()
    check("tore : positions shape coherente", pos.shape[0] == total)

    # Reconvertit en coordonnees monde autour du centre du tore (0,0,0),
    # solveur origin/size connus (world_to_solver applique un decalage
    # additif : reconvertir chaque point solveur -> monde puis mesurer sa
    # distance a l'axe Z du tore, dans le plan XY monde).
    from extension.transform import solver_to_world_array

    world_pts = solver_to_world_array(pos.astype("float32"), origin, size)
    # centre du tore en monde = (0,0,0) ; le "trou" est le cylindre de
    # rayon (major_radius - minor_radius) = 0.09 autour de l'axe Z.
    import numpy as np

    r_xy = np.sqrt(world_pts[:, 0] ** 2 + world_pts[:, 1] ** 2)
    hole_radius = 0.15 - 0.06
    n_in_hole = int(np.sum((r_xy < hole_radius * 0.9) & (np.abs(world_pts[:, 2]) < 0.05)))
    check("tore : aucune particule dans le trou central", n_in_hole == 0,
          f"n_in_hole={n_in_hole} (rayon trou={hole_radius:.3f})")

    sim.destroy()
except Exception:
    traceback.print_exc()
    FAILURES.append("1) exception")


# ---------------------------------------------------------------------------
# 6) Non-regression BLOCK/BOUNDS
# ---------------------------------------------------------------------------
print("\n=== 6) Non-regression BLOCK/BOUNDS ===")
try:
    scene = fresh_scene()
    make_domain(scene, size=1.0)
    box = setup_emitter_bounds_box("Box", (0.0, 0.0, 0.0), (0.3, 0.3, 0.3),
                                    emit_mode="BLOCK", emit_source="BOUNDS")
    sim, origin, size = build_sim(scene)
    fake = _FakeSelf()
    total, mat_id, spacing = build_bake_setup(sim, scene, origin, size, [box], fake)

    lo, hi = emitter_bounds_solver(box, origin, size)
    dx = size / scene.bourrasque.grid_res
    step = dx / scene.bourrasque.ppc_axis
    expected = 1
    for axis in range(3):
        extent = hi[axis] - lo[axis]
        expected *= max(0, math.ceil(extent / step - 0.5))
    check("BLOCK/BOUNDS : compte == estimation deterministe", total == expected,
          f"total={total} expected={expected}")
    sim.destroy()
except Exception:
    traceback.print_exc()
    FAILURES.append("6) exception")


# ---------------------------------------------------------------------------
# 2/3) Inflow debit constant + pas d'explosion, 60 frames
# ---------------------------------------------------------------------------
print("\n=== 2/3) Inflow debit constant + pas d'explosion (60 frames) ===")
try:
    scene = fresh_scene()
    make_domain(scene, size=1.0)
    # box allongee sur X, en bas du domaine (proche origine solveur, en
    # marge des 3 cellules de bord), vitesse le long de +X solveur.
    box = setup_emitter_bounds_box(
        "Inflow", (0.0, 0.0, 0.0), (0.15, 0.15, 0.15),
        emit_mode="INFLOW", emit_source="BOUNDS",
        velocity=(0.0, 0.0, 0.0),
    )
    sim, origin, size = build_sim(scene)
    fake = _FakeSelf()
    _, mat_id, spacing = build_bake_setup(sim, scene, origin, size, [box], fake)
    # Vitesse solveur imposee directement (contournement de
    # world_to_solver_dir pour un cas de test simple) : +X, magnitude
    # choisie pour ~1 couche toutes les 2-3 frames a 24 fps.
    fake._inflow_states[0].vel = (0.6, 0.0, 0.0)
    fake._inflow_states[0].speed = 0.6

    fake._sim = sim
    bqd_path, _ = cache.cache_paths(CACHE_DIR, "inflow_test")
    cache.ensure_cache_dir(CACHE_DIR)
    fake._writer = cache.CacheWriter(bqd_path)
    fake._frame_dt = 1.0 / 24.0

    counts_per_frame = []
    prev_total = sim.particle_count
    for i in range(60):
        before = sim.particle_count
        fake._advance_frame()
        emitted = sim.particle_count - before
        counts_per_frame.append(emitted)

    fake._writer.write_materials(sim.read_materials())
    fake._writer.close()

    print("serie de comptes emis par frame :", counts_per_frame)
    nonzero = [c for c in counts_per_frame if c > 0]
    check("inflow : au moins une emission sur 60 frames", len(nonzero) > 0,
          f"nonzero frames={len(nonzero)}")
    if nonzero:
        layer_size = fake._inflow_states[0].layer_points.shape[0]
        check("inflow : chaque emission == taille d'une couche (ou multiple entier)",
              all(c % layer_size == 0 for c in counts_per_frame),
              f"layer_size={layer_size} counts={counts_per_frame}")
        # "au plus une rangee d'ecart d'une frame a l'autre" : les valeurs
        # non nulles valent soit 0 soit 1*layer_size soit (rarement) 2x.
        distinct_multiples = sorted(set(c // layer_size for c in counts_per_frame))
        check("inflow : debit stable (pas de battement, multiples adjacents)",
              max(distinct_multiples) - min(distinct_multiples) <= 1,
              f"multiples observes={distinct_multiples}")

    pos = sim.read_positions()
    check("inflow : aucun NaN", bool(np.all(np.isfinite(pos))))
    check("inflow : toutes les positions dans le domaine [0,size]",
          bool(np.all(pos >= -1e-4) and np.all(pos <= size + 1e-4)),
          f"min={pos.min(axis=0)} max={pos.max(axis=0)} size={size}")

    sim.destroy()
except Exception:
    traceback.print_exc()
    FAILURES.append("2/3) exception")


# ---------------------------------------------------------------------------
# 4) Saturation (max_particles bas)
# ---------------------------------------------------------------------------
print("\n=== 4) Saturation (max_particles bas) ===")
try:
    scene = fresh_scene()
    make_domain(scene, size=1.0)
    scene.bourrasque.max_particles = 50000
    box = setup_emitter_bounds_box(
        "InflowSat", (0.0, 0.0, 0.0), (0.4, 0.4, 0.4),
        emit_mode="INFLOW", emit_source="BOUNDS",
        velocity=(0.0, 0.0, 0.0),
    )
    sim, origin, size = build_sim(scene)
    fake = _FakeSelf()
    _, mat_id, spacing = build_bake_setup(sim, scene, origin, size, [box], fake)
    fake._inflow_states[0].vel = (1.5, 0.0, 0.0)
    fake._inflow_states[0].speed = 1.5

    fake._sim = sim
    bqd_path, _ = cache.cache_paths(CACHE_DIR, "inflow_sat_test")
    fake._writer = cache.CacheWriter(bqd_path)
    fake._frame_dt = 1.0 / 24.0

    for i in range(60):
        fake._advance_frame()
    warning_count = sum(1 for tags, msg in fake.reports if "WARNING" in tags)

    fake._writer.write_materials(sim.read_materials())
    frames_written = fake._writer.frames_written
    fake._writer.close()

    check("saturation : bake va au bout (60 frames ecrites)", frames_written == 60,
          f"frames_written={frames_written}")
    check("saturation : self._saturated est vrai", fake._saturated)
    check("saturation : avertissement emis exactement une fois", warning_count == 1,
          f"warning_count={warning_count}")
    check("saturation : particle_count <= max_particles", sim.particle_count <= 50000,
          f"particle_count={sim.particle_count}")

    # relecture
    reader = cache.CacheReader(bqd_path)
    check("saturation : cache relisible, is_variable", reader.is_variable)
    check("saturation : frame_count == 60", reader.frame_count == 60)
    last_frame = reader.read_frame(59)
    check("saturation : derniere frame non vide", last_frame.shape[0] > 0,
          f"n={last_frame.shape[0]}")
    reader.close()

    sim.destroy()
except Exception:
    traceback.print_exc()
    FAILURES.append("4) exception")


# ---------------------------------------------------------------------------
# 5) Cache v2 relisible (reprend le cache du test 2/3)
# ---------------------------------------------------------------------------
print("\n=== 5) Cache v2 relisible (inflow_test.bqd) ===")
try:
    bqd_path, mat_path = cache.cache_paths(CACHE_DIR, "inflow_test")
    reader = cache.CacheReader(bqd_path)
    check("cache v2 : is_variable", reader.is_variable)
    counts = [reader.particle_count(i) for i in range(reader.frame_count)]
    check("cache v2 : particle_count croissant (ou constant)",
          all(b >= a for a, b in zip(counts, counts[1:])),
          f"counts={counts}")
    ok_sizes = True
    for i in range(reader.frame_count):
        frame = reader.read_frame(i)
        if frame.shape[0] != counts[i]:
            ok_sizes = False
            break
    check("cache v2 : read_frame donne la bonne taille a chaque frame", ok_sizes)
    mats = reader.read_materials()
    check("cache v2 : sidecar .mat present et de taille == derniere frame",
          mats is not None and mats.shape[0] == counts[-1],
          f"mats={None if mats is None else mats.shape}")
    reader.close()
except Exception:
    traceback.print_exc()
    FAILURES.append("5) exception")


print("\n=== RESULTAT ===")
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
else:
    print("Toutes les verifications sont passees.")
sys.exit(1 if FAILURES else 0)
