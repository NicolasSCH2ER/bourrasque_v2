"""Validation --background de l'ensemencement volumique INFLOW (M4 correctif).

Lance avec :
    blender.exe --background --python validate_inflow.py

Ne passe jamais par l'operateur modal BQ_OT_bake (invoke()/modal() exigent
un cycle d'evenements bpy absent en --background) : on reproduit la meme
logique de setup, puis on pilote _emit_inflow_sites()/step() directement
via un harnais leger qui reutilise les VRAIES methodes non-modales de
BQ_OT_bake (elles ne dependent d'aucun etat bpy autre que self).
"""
import math
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

from extension import ops, props, register, unregister  # noqa: E402
from extension import lib  # noqa: E402


# ---------------------------------------------------------------------------
# Harnais : reutilise les methodes non-modales de BQ_OT_bake sans passer par
# bpy.types.Operator (impossible a instancier directement hors du systeme RNA
# de Blender).
# ---------------------------------------------------------------------------


class _Harness:
    _emit_inflow_sites = ops.BQ_OT_bake._emit_inflow_sites

    def __init__(self, sim, inflow_states, usable_bounds, frame_dt):
        self._sim = sim
        self._inflow_states = inflow_states
        self._usable_bounds = usable_bounds
        self._frame_dt = frame_dt
        self._saturated = False
        self.reports = []

    def report(self, level, msg):
        self.reports.append((level, msg))
        print(f"  [{level}] {msg}")


def make_scene(cache_dir):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps = 24
    scene.render.fps_base = 1.0

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = 32
    scene.bourrasque.ppc_axis = 2
    # Gravite volontairement faible : sur 40 frames a 24 fps (1.667 s), une
    # gravite realiste (9.8) ferait chuter le jet de ~16 m, tres au-dela
    # d'un domaine de quelques metres - le jet toucherait le fond et
    # s'etalerait en flaque bien avant la derniere frame, ce qui fausserait
    # la mesure de rayon de jet (on mesurerait la flaque, pas le jet). Une
    # gravite faible garde le jet en chute libre coherente jusqu'a la
    # derniere frame, seul regime que la mesure V1 est censee caracteriser.
    scene.bourrasque.gravity = -0.5
    scene.bourrasque.cfl = 0.4
    scene.bourrasque.max_particles = 2_000_000
    scene.bourrasque.frame_start = 1
    scene.bourrasque.frame_end = 40
    scene.bourrasque.cache_dir = cache_dir
    return scene


def setup_sphere_emitter(scene, radius=0.3, location=(0.0, 0.0, 0.6), speed=-0.15):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=24, ring_count=12)
    obj = bpy.context.active_object
    obj.name = "SphereEmitter"
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.model = "WATER"
    obj.bourrasque.emit_mode = "INFLOW"
    obj.bourrasque.emit_source = "MESH"
    # Vitesse monde vers le bas.
    obj.bourrasque.initial_velocity = (0.0, 0.0, speed)
    return obj


def setup_box_emitter(scene, size=(0.3, 0.3, 0.3), location=(0.0, 0.0, 0.6), speed=-0.15):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.scale = size
    # `obj.matrix_world` (utilise par `emitter_bounds_solver`) n'est pas
    # recalcule tant que le depsgraph n'a pas ete evalue : sans ce point de
    # synchronisation, `obj.scale` ne se reflete pas encore dans la bbox
    # monde lue juste apres par le setup du bake.
    bpy.context.view_layer.update()
    obj.name = "BoxEmitter"
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.model = "WATER"
    obj.bourrasque.emit_mode = "INFLOW"
    obj.bourrasque.emit_source = "BOUNDS"
    obj.bourrasque.initial_velocity = (0.0, 0.0, speed)
    return obj


def build_inflow_state(scene, obj, old_layer=False):
    """Reproduit le setup de BQ_OT_bake.invoke() pour UN emetteur INFLOW.

    `old_layer=True` reconstruit l'ANCIEN comportement (tranche aval) pour
    la mesure "avant correctif" du critere 1, en reimplementant a la main
    l'ancienne regle disparue de ops.py (elle n'existe plus dans le module).
    """
    from extension import sampling
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

    if op.emit_source == "MESH":
        pts = sampling.sample_mesh_interior(
            obj, origin, size, scene.bourrasque.grid_res, scene.bourrasque.ppc_axis
        )
    else:
        lo, hi = emitter_bounds_solver(obj, origin, size)
        pts = ops._lattice_points_in_box(lo, hi, spacing)

    if not old_layer:
        return vel, spacing, pts

    # --- ancienne regle (tranche aval), pour comparaison seulement --------
    comps = tuple(abs(c) for c in vel)
    axis = max(range(3), key=lambda i: comps[i])
    sign = 1 if vel[axis] >= 0.0 else -1
    coord = pts[:, axis]
    if sign >= 0:
        mask = coord >= coord.max() - spacing
    else:
        mask = coord <= coord.min() + spacing
    return vel, spacing, pts[mask]


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


def run_bake(scene, obj, n_frames, old_layer=False, track_counts=False):
    """Boucle de bake manuelle : reproduit BQ_OT_bake sans le modal."""
    cfg, origin, size = default_config(scene)
    sim = lib.Sim(cfg)
    mat_id = sim.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)

    vel, spacing, sites = build_inflow_state(scene, obj, old_layer=old_layer)
    state = ops._InflowState(obj.name, mat_id, vel, spacing, sites)

    usable = props.domain_usable_bounds(scene)
    frame_dt = scene.render.fps_base / scene.render.fps

    harness = _Harness(sim, [state], usable, frame_dt)

    lo_bbox = sites.min(axis=0) - spacing / 2.0 if sites.shape[0] else None
    hi_bbox = sites.max(axis=0) + spacing / 2.0 if sites.shape[0] else None

    per_frame_emitted = []
    per_frame_count = []
    per_frame_bbox_count = []
    per_frame_ms = []
    for f in range(n_frames):
        t0 = time.perf_counter()
        n_emitted = harness._emit_inflow_sites()
        t1 = time.perf_counter()
        sim.step(frame_dt)
        per_frame_ms.append((t1 - t0) * 1000.0)
        per_frame_emitted.append(n_emitted)
        if track_counts:
            per_frame_count.append(sim.particle_count)
            pos_now = sim.read_positions()
            if lo_bbox is not None and pos_now.shape[0] > 0:
                in_bbox = np.all((pos_now >= lo_bbox) & (pos_now <= hi_bbox), axis=1)
                per_frame_bbox_count.append(int(in_bbox.sum()))
            else:
                per_frame_bbox_count.append(0)

    positions = sim.read_positions()
    axis_point = sites.mean(axis=0).astype(np.float64) if sites.shape[0] else None
    sim.destroy()
    return (
        positions,
        per_frame_emitted,
        per_frame_count,
        per_frame_bbox_count,
        per_frame_ms,
        spacing,
        vel,
        axis_point,
    )


def jet_radius(positions, vel, spacing, axis_point, downstream_frac=0.7):
    """Rayon max a l'axe d'ecoulement (droite passant par `axis_point`,
    direction `vel`, tous deux en espace solveur) pour les particules
    'bien en aval'.

    `axis_point` DOIT etre un point reellement sur l'axe (le centre
    geometrique de l'emetteur, calcule a partir du nuage de sites AVANT
    toute emission) — utiliser un centroide derive des positions FINALES
    (donc du nuage deforme par l'ecoulement) biaiserait la reference et
    fausserait la distance perpendiculaire mesuree.
    """
    vel = np.array(vel, dtype=np.float64)
    speed = np.linalg.norm(vel)
    if speed < 1e-9 or positions.shape[0] == 0 or axis_point is None:
        return 0.0
    axis_dir = vel / speed
    # Coordonnee le long de l'axe (produit scalaire), origine arbitraire.
    s = positions @ axis_dir
    smin, smax = s.min(), s.max()
    if smax - smin < 1e-9:
        return 0.0
    threshold = smin + downstream_frac * (smax - smin)
    # "En aval" = du cote vers lequel vel pointe : s doit depasser threshold
    # dans le sens de axis_dir (deja garanti par s croissant dans ce sens).
    mask = s >= threshold
    downstream = positions[mask]
    if downstream.shape[0] == 0:
        return 0.0
    # Distance a l'axe (droite passant par `axis_point`, direction axis_dir).
    rel = downstream - axis_point
    proj = np.outer(rel @ axis_dir, axis_dir)
    perp = rel - proj
    dist = np.linalg.norm(perp, axis=1)
    return float(dist.max())


def continuity_gaps(positions, vel, spacing):
    """Ecarts entre positions consecutives triees le long de l'axe."""
    vel = np.array(vel, dtype=np.float64)
    speed = np.linalg.norm(vel)
    if speed < 1e-9 or positions.shape[0] < 2:
        return np.array([])
    axis_dir = vel / speed
    s = np.sort(positions @ axis_dir)
    # Ne garder que les positions DISTINCTES (particules a la meme profondeur
    # transverse, meme s -- on veut les trous le long de l'axe, pas la
    # densite locale).
    s_unique = np.unique(np.round(s, 6))
    return np.diff(s_unique)


def main():
    cache_dir = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\bourrasque_cache_validate"

    print("=" * 70)
    print("1) SYMPTOME — rayon de jet, sphere emettrice, AVANT/APRES")
    print("=" * 70)

    register()
    try:
        scene = make_scene(cache_dir)
        sphere = setup_sphere_emitter(scene, radius=0.3, location=(0.0, 0.0, 0.6), speed=-0.15)
        positions_before, emitted_before, _, _, _, spacing, vel, axis_pt = run_bake(
            scene, sphere, n_frames=40, old_layer=True
        )
        r_before = jet_radius(positions_before, vel, spacing, axis_pt)
        print(f"rayon de jet AVANT (ancienne regle, tranche aval) : {r_before:.5f} m (spacing={spacing:.5f})")

        # Nouvelle sphere fraiche (bake precedent a modifie le mesh actif) :
        scene2 = make_scene(cache_dir)
        sphere2 = setup_sphere_emitter(scene2, radius=0.3, location=(0.0, 0.0, 0.6), speed=-0.15)
        (
            positions_after,
            emitted_after,
            counts_after,
            bbox_counts_after,
            ms_after,
            spacing2,
            vel2,
            axis_pt2,
        ) = run_bake(scene2, sphere2, n_frames=40, old_layer=False, track_counts=True)
        r_after = jet_radius(positions_after, vel2, spacing2, axis_pt2)
        print(f"rayon de jet APRES (ensemencement volumique) : {r_after:.5f} m (spacing={spacing2:.5f}, rayon sphere=0.3)")
        print(f"critere : r_after doit etre ~O(0.3), r_before ~O(spacing={spacing:.5f})")

        print()
        print("=" * 70)
        print("2) CONTINUITE — ecarts le long de l'axe d'ecoulement (APRES)")
        print("=" * 70)
        gaps = continuity_gaps(positions_after, vel2, spacing2)
        if gaps.size:
            print(f"n_gaps={gaps.size}  max={gaps.max():.5f}  spacing={spacing2:.5f}  max/spacing={gaps.max()/spacing2:.2f}")
        else:
            print("aucun ecart calculable (trop peu de positions distinctes)")

        print()
        print("=" * 70)
        print("3) DENSITE — coincidences + stabilite du nombre de particules dans l'emetteur")
        print("=" * 70)
        # Coincidences EXACTES (tolerance 1e-5 m, bien en-dessous de spacing) :
        # deux particules distinctes ne doivent jamais partager la meme
        # position a la precision flottante pres.
        rounded = np.round(positions_after / 1e-5).astype(np.int64)
        _, counts = np.unique(rounded, axis=0, return_counts=True)
        n_exact_dupes = int((counts > 1).sum())
        print(f"groupes de positions EXACTEMENT coincidentes (tol. 1e-5 m) : {n_exact_dupes}")

        print("particules totales dans la simulation, par frame :")
        print(counts_after)
        print("particules DANS LA BBOX DE L'EMETTEUR, par frame (doit se stabiliser) :")
        print(bbox_counts_after)
        tail = bbox_counts_after[-10:]
        print(
            f"regime stable (10 dernieres frames) : min={min(tail)} max={max(tail)} "
            f"amplitude={max(tail) - min(tail)}"
        )

        print()
        print("=" * 70)
        print("4) DEBIT — particules emises par frame sur 40 frames (APRES)")
        print("=" * 70)
        print(emitted_after)

        print()
        print("=" * 70)
        print("5) MODE BOUNDS — rayon de section de jet")
        print("=" * 70)
        scene3 = make_scene(cache_dir)
        box = setup_box_emitter(scene3, size=(0.3, 0.3, 0.3), location=(0.0, 0.0, 0.6), speed=-0.15)
        (
            positions_box,
            emitted_box,
            counts_box,
            bbox_counts_box,
            ms_box,
            spacing3,
            vel3,
            axis_pt3,
        ) = run_bake(scene3, box, n_frames=40, old_layer=False, track_counts=True)
        r_box = jet_radius(positions_box, vel3, spacing3, axis_pt3)
        print(f"rayon de jet BOUNDS (demi-cote attendu ~0.15) : {r_box:.5f} m (spacing={spacing3:.5f})")
        print(f"particules emises par frame (BOUNDS) : {emitted_box}")

        print()
        print("=" * 70)
        print("SURCOUT PAR FRAME du test d'occupation")
        print("=" * 70)
        print(f"sphere : moyenne={np.mean(ms_after):.3f} ms, max={np.max(ms_after):.3f} ms (sur {len(ms_after)} frames)")
        print(f"boite  : moyenne={np.mean(ms_box):.3f} ms, max={np.max(ms_box):.3f} ms (sur {len(ms_box)} frames)")

    finally:
        unregister()

    print()
    print("=" * 70)
    print("6) CYCLE register/unregister/register")
    print("=" * 70)
    register()
    unregister()
    register()
    print("cycle propre : OK")
    unregister()


if __name__ == "__main__":
    main()
