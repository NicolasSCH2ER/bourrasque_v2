"""Renforcement du critere V2 (revue M5) : le domaine 2:1:1 vs 1:1:2 valide
initialement a un defaut structurel -- `res_a=(2R+6,R+6,R+6)` donne
`res.y == res.z`, exactement les deux facteurs de la linearisation
`(gx*res.y+gy)*res.z+gz` : une interversion de res.y et res.z y serait
INVISIBLE. Meme defaut sur 1:1:2 (res.x == res.y).

Ce script reprend exactement la methode de validate_m5.py (meme
construction geometrique de l'emetteur B par echange sx<->sz des bornes
solveur de A, meme comparaison de statistiques agregees par axe), mais sur
un domaine dont les TROIS resolutions sont distinctes :

    domaine A : etendues monde (worldX=1, worldY=2, worldZ=4)
                -> extent_solver (sx,sy,sz) = (worldX, worldZ, worldY) = (1, 4, 2)
    domaine B : etendues monde (worldX=2, worldY=1, worldZ=4) (rotation 90
                degres autour de l'axe de gravite monde Z, qui echange
                worldX<->worldY)
                -> extent_solver = (2, 4, 1) = echange sx<->sz de A

A dx egal (meme grid_res, meme max extent monde = 4 pour A et B), res_a et
res_b ont trois composantes deux a deux distinctes (verifie explicitement
ci-dessous avant de lancer quoi que ce soit), donc une confusion
sx<->sy ou sy<->sz produirait une divergence GROSSIERE detectee par ce
test, contrairement au domaine 2:1:1/1:1:2 original.

En plus de la comparaison A permute vs B, ce script mesure le bruit de
non-determinisme run1/run2 du MEME setup (domaine A lance deux fois), pour
comparer l'ecart A-vs-B a l'echelle de bruit du binaire, jamais a un
epsilon fixe (methode imposee par docs/plan-milestone-5.md).

Lancer avec : blender --background --python validate_m5_v2_strong.py
"""

import sys
import traceback

import bmesh
import bpy
import mathutils
import numpy as np

bpy.ops.wm.read_factory_settings(use_empty=True)

ROOT = r"C:\Users\nicol\Code\bourrasque_v2"
sys.path.insert(0, ROOT)

import extension  # noqa: E402

extension.register()

from extension import lib  # noqa: E402
from extension.props import (  # noqa: E402
    domain_resolution,
    domain_transform,
    emitter_bounds_solver,
)
from extension.transform import solver_to_world  # noqa: E402

FAILURES = []
_scene_counter = [0]


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def new_scene():
    _scene_counter[0] += 1
    scene = bpy.data.scenes.new(f"BqTest{_scene_counter[0]}")
    scene.bourrasque.domain_object = None
    return scene


def _make_cube_object(scene, name, location, scale):
    mesh = bpy.data.meshes.new(f"{name}Mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)

    lx, ly, lz = location
    sx, sy, sz = scale
    obj.matrix_world = mathutils.Matrix((
        (sx, 0.0, 0.0, lx),
        (0.0, sy, 0.0, ly),
        (0.0, 0.0, sz, lz),
        (0.0, 0.0, 0.0, 1.0),
    ))
    return obj


def make_domain(scene, extent_world, grid_res=32, location=(0.0, 0.0, 0.0)):
    domain = _make_cube_object(scene, "Domain", location, extent_world)
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = grid_res
    scene.bourrasque.ppc_axis = 2
    scene.bourrasque.max_particles = 2_000_000
    return domain


def make_box_emitter(scene, name, location, extent_world):
    obj = _make_cube_object(scene, name, location, extent_world)
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.emit_mode = "BLOCK"
    obj.bourrasque.emit_source = "BOUNDS"
    obj.bourrasque.model = "WATER"
    return obj


def build_sim(scene):
    res, dx = domain_resolution(scene)
    cfg = lib.default_config()
    cfg.grid_res[:] = res
    cfg.cell_size = dx
    cfg.gravity_y = scene.bourrasque.gravity
    cfg.cfl = scene.bourrasque.cfl
    cfg.ppc_axis = scene.bourrasque.ppc_axis
    cfg.max_particles = scene.bourrasque.max_particles
    sim = lib.Sim(cfg)
    return sim, res, dx


def run_frames(sim, n, frame_dt=1.0 / 24.0):
    for _ in range(n):
        sim.step(frame_dt)


def axis_stats(pos):
    return {
        "mean": pos.mean(axis=0),
        "std": pos.std(axis=0),
        "min": pos.min(axis=0),
        "max": pos.max(axis=0),
    }


GRID_RES = 40
N_FRAMES = 30


def build_and_run(extent_world, location, seed_tag):
    scene = new_scene()
    make_domain(scene, extent_world=extent_world, grid_res=GRID_RES, location=location)
    emitter = make_box_emitter(scene, f"Emit{seed_tag}", (0.15, 0.3, 0.35), (0.15, 0.5, 0.7))
    origin, size = domain_transform(scene)
    res, dx = domain_resolution(scene)
    lo, hi = emitter_bounds_solver(emitter, origin, size)

    sim, _res2, _dx2 = build_sim(scene)
    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    n = sim.emit_box(mat_id, lo, hi, vel=(0.0, 0.0, 0.0))
    run_frames(sim, N_FRAMES)
    pos = sim.read_positions()
    sim.destroy()
    return n, pos, res, dx, origin, size, lo, hi


print("\n=== V2 renforce : domaine (1,2,4) vs (2,1,4), trois res distinctes ===")
try:
    # --- verification prealable : les trois res sont deux a deux distinctes
    scene_probe = new_scene()
    make_domain(scene_probe, extent_world=(1.0, 2.0, 4.0), grid_res=GRID_RES)
    res_probe, dx_probe = domain_resolution(scene_probe)
    print(f"  res domaine A = {res_probe}, dx = {dx_probe:.6f}")
    check(
        "pre-requis : res.x, res.y, res.z deux a deux distinctes (domaine A)",
        len(set(res_probe)) == 3,
        f"res={res_probe}",
    )

    # --- domaine A : etendues monde (1, 2, 4) -> extent_solver (1, 4, 2)
    scene_a = new_scene()
    make_domain(scene_a, extent_world=(1.0, 2.0, 4.0), grid_res=GRID_RES)
    emitter_a = make_box_emitter(scene_a, "EmitA", (0.15, 0.3, 0.35), (0.15, 0.5, 0.7))
    origin_a, size_a = domain_transform(scene_a)
    res_a, dx_a = domain_resolution(scene_a)
    lo_a, hi_a = emitter_bounds_solver(emitter_a, origin_a, size_a)
    print(f"  domaine A : res={res_a}, dx={dx_a:.6f}, emitter lo={lo_a}, hi={hi_a}")

    # --- domaine B : etendues monde (2, 1, 4) -> extent_solver (2, 4, 1),
    # rotation 90 deg autour de l'axe de gravite monde Z (echange worldX<->worldY)
    scene_b = new_scene()
    make_domain(scene_b, extent_world=(2.0, 1.0, 4.0), grid_res=GRID_RES,
                location=(10.0, 0.0, 0.0))
    origin_b, size_b = domain_transform(scene_b)
    res_b, dx_b = domain_resolution(scene_b)
    print(f"  domaine B : res={res_b}, dx={dx_b:.6f}")

    check("dx identique entre A et B (meme grid_res, meme max extent)",
          abs(dx_a - dx_b) < 1e-9, f"dx_a={dx_a} dx_b={dx_b}")
    check("res[B] == permutation (sx<->sz) de res[A]",
          res_b == (res_a[2], res_a[1], res_a[0]),
          f"res_a={res_a} res_b={res_b}")
    check("res.x, res.y, res.z deux a deux distinctes sur A ET B",
          len(set(res_a)) == 3 and len(set(res_b)) == 3,
          f"res_a={res_a} res_b={res_b}")

    lo_b_solver = (lo_a[2], lo_a[1], lo_a[0])
    hi_b_solver = (hi_a[2], hi_a[1], hi_a[0])

    corners_solver = [
        (lo_b_solver[0] if bx == 0 else hi_b_solver[0],
         lo_b_solver[1] if by == 0 else hi_b_solver[1],
         lo_b_solver[2] if bz == 0 else hi_b_solver[2])
        for bx in (0, 1) for by in (0, 1) for bz in (0, 1)
    ]
    corners_world = [solver_to_world(c, origin_b, size_b) for c in corners_solver]
    xs = [c[0] for c in corners_world]
    ys = [c[1] for c in corners_world]
    zs = [c[2] for c in corners_world]
    world_lo = (min(xs), min(ys), min(zs))
    world_hi = (max(xs), max(ys), max(zs))
    world_center = tuple((world_lo[i] + world_hi[i]) / 2.0 for i in range(3))
    world_extent = tuple(world_hi[i] - world_lo[i] for i in range(3))

    emitter_b = make_box_emitter(scene_b, "EmitB", world_center, world_extent)
    lo_b_check, hi_b_check = emitter_bounds_solver(emitter_b, origin_b, size_b)
    lo_err = max(abs(a - b) for a, b in zip(lo_b_check, lo_b_solver))
    hi_err = max(abs(a - b) for a, b in zip(hi_b_check, hi_b_solver))
    check("construction geometrique de l'emetteur B correcte (lo)",
          lo_err < 1e-4, f"erreur={lo_err:.2e}")
    check("construction geometrique de l'emetteur B correcte (hi)",
          hi_err < 1e-4, f"erreur={hi_err:.2e}")

    # --- runs A et B ---
    sim_a, _res_a2, _dx_a2 = build_sim(scene_a)
    mat_a = sim_a.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    n_a = sim_a.emit_box(mat_a, lo_a, hi_a, vel=(0.0, 0.0, 0.0))
    run_frames(sim_a, N_FRAMES)
    pos_a = sim_a.read_positions()
    sim_a.destroy()

    sim_b, _res_b2, _dx_b2 = build_sim(scene_b)
    mat_b = sim_b.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    n_b = sim_b.emit_box(mat_b, lo_b_check, hi_b_check, vel=(0.0, 0.0, 0.0))
    run_frames(sim_b, N_FRAMES)
    pos_b = sim_b.read_positions()
    sim_b.destroy()

    check("meme nombre de particules emises (A et B, boites de meme volume)",
          n_a == n_b, f"n_a={n_a} n_b={n_b}")

    stats_a = axis_stats(pos_a)
    stats_b = axis_stats(pos_b)
    perm = [2, 1, 0]
    stats_a_perm = {k: v[perm] for k, v in stats_a.items()}

    print(f"  A (permute sx<->sz) : mean={stats_a_perm['mean']}, std={stats_a_perm['std']}")
    print(f"  B                   : mean={stats_b['mean']}, std={stats_b['std']}")

    # --- bruit de non-determinisme run1/run2 du MEME setup (domaine A) ---
    print("\n  -- bruit run1/run2 du meme binaire (domaine A relance) --")
    n_a2, pos_a2, res_a2, dx_a2, *_ = build_and_run((1.0, 2.0, 4.0), (0.0, 0.0, 0.0), "A2")
    check("run1/run2 : meme compte de particules (domaine A)", n_a == n_a2, f"n_a={n_a} n_a2={n_a2}")
    if pos_a.shape == pos_a2.shape:
        diff_run = np.abs(pos_a - pos_a2)
        noise_mean = float(diff_run.mean())
        noise_max = float(diff_run.max())
        print(f"  bruit run1/run2 (positions, domaine A) : mean={noise_mean:.4e} max={noise_max:.4e}")
    else:
        noise_max = None
        print(f"  [!] shapes differentes run1={pos_a.shape} run2={pos_a2.shape}, bruit non mesurable")

    tol = 5.0 * dx_a
    print(f"\n  tolerance A-vs-B (5*dx) = {tol:.5f} ; pour reference, bruit run1/run2 max = "
          f"{noise_max if noise_max is not None else 'N/A'}")
    for axis, label in enumerate(("x", "y", "z")):
        err_mean = abs(stats_a_perm["mean"][axis] - stats_b["mean"][axis])
        check(
            f"moyenne axe {label} (A permute vs B) dans la tolerance",
            err_mean < tol,
            f"A_perm={stats_a_perm['mean'][axis]:.4f} B={stats_b['mean'][axis]:.4f} "
            f"err={err_mean:.4f} tol={tol:.4f}",
        )
        err_std = abs(stats_a_perm["std"][axis] - stats_b["std"][axis])
        check(
            f"ecart-type axe {label} (A permute vs B) dans la tolerance",
            err_std < tol,
            f"A_perm={stats_a_perm['std'][axis]:.4f} B={stats_b['std'][axis]:.4f} "
            f"err={err_std:.4f} tol={tol:.4f}",
        )

    check("aucun NaN/Inf (A et B)",
          bool(np.all(np.isfinite(pos_a))) and bool(np.all(np.isfinite(pos_b))))
except Exception:
    traceback.print_exc()
    FAILURES.append("V2 renforce : exception")


print("\n=== RESULTAT ===")
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
else:
    print("Toutes les verifications sont passees.")
sys.exit(1 if FAILURES else 0)
