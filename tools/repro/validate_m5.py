"""Validation numerique du jalon M5 (domaine en pave), a lancer avec :

    blender --background --python validate_m5.py

Exercice le VRAI code de production (extension.props.domain_transform /
domain_resolution / emitter_bounds_solver, extension.lib.Sim sur la vraie
DLL) sur des scenes construites en memoire (pas de .blend requis).

Couvre V1 (cube inchange), V2 (symetrie par permutation d'axes — le test
le plus discriminant), V3 (parois d'un pave 4:1:1), V4 (memoire, ncell
proportionnel), V5 (emission pres d'un bord du grand axe).
"""

import math
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

FAILURES = []
_scene_counter = [0]


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def new_scene():
    """Cree une scene bpy INDEPENDANTE (pas la scene active reutilisee/
    videe) : deux domaines qui coexistent dans le meme test (V2) ont besoin
    de deux `scene.bourrasque` distincts, une seule scene partagee et videe
    entre les deux fait courir le risque de lire l'un pendant qu'on croit
    lire l'autre (piege rencontre lors de la premiere version de ce
    script : `build_sim(scene_a)` lisait en realite le domaine B, deja en
    place dans la MEME scene bpy sous-jacente)."""
    _scene_counter[0] += 1
    scene = bpy.data.scenes.new(f"BqTest{_scene_counter[0]}")
    scene.bourrasque.domain_object = None
    return scene


def _make_cube_object(scene, name, location, scale):
    """Cree un objet maillage cube (corners locaux +-0.5, comme
    `primitive_cube_add(size=1.0)`) directement via bpy.data/bmesh, et fixe
    `matrix_world` explicitement (translation + echelle diagonale) plutot
    que de passer par `obj.location`/`obj.scale` + une mise a jour du
    depsgraph : robuste meme quand `scene` n'est pas la scene active du
    contexte (cas de ce script, background mode, plusieurs scenes)."""
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


# ---------------------------------------------------------------------------
# V4 — memoire : ncell proportionnel a res[0]*res[1]*res[2], pas au cube du
# maximum
# ---------------------------------------------------------------------------
print("\n=== V4 : memoire (ncell proportionnel) ===")
try:
    scene = new_scene()
    make_domain(scene, extent_world=(4.0, 1.0, 1.0), grid_res=64)
    res, dx = domain_resolution(scene)
    ncell = res[0] * res[1] * res[2]
    cube_equiv = max(res) ** 3
    ratio = cube_equiv / ncell
    print(f"  res={res}, ncell={ncell}, cube englobant equivalent={cube_equiv}, ratio={ratio:.2f}x")
    check(
        "V4 : pave 4:1:1 alloue au moins 3x moins qu'un cube englobant equivalent",
        ratio >= 3.0,
        f"ratio={ratio:.2f}",
    )
    # bq_create alloue effectivement ncell (via SimParamsGpu.res) : la Sim
    # se cree sans erreur est deja une preuve indirecte que l'allocation
    # correspond a res[0]*res[1]*res[2] et pas au cube du max (sinon la
    # difference de VRAM serait flagrante mais invisible ici) ; le calcul
    # ci-dessus est la mesure directe demandee par le plan.
    sim, _res, _dx = build_sim(scene)
    check("V4 : creation de la Sim reussit sur un pave 4:1:1", sim is not None)
    sim.destroy()
except Exception:
    traceback.print_exc()
    FAILURES.append("V4 : exception")


# ---------------------------------------------------------------------------
# V1 — cube inchange : deux runs du meme setup cubique donnent le meme
# compte de particules et des positions dans le bruit de non-determinisme
# ---------------------------------------------------------------------------
print("\n=== V1 : cube inchange ===")
try:
    def cube_run():
        scene = new_scene()
        make_domain(scene, extent_world=(1.0, 1.0, 1.0), grid_res=32)
        emitter = make_box_emitter(scene, "Box", (0.0, 0.0, 0.1), (0.3, 0.3, 0.3))
        origin, size = domain_transform(scene)
        sim, res, dx = build_sim(scene)
        mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
        lo, hi = emitter_bounds_solver(emitter, origin, size)
        n = sim.emit_box(mat_id, lo, hi, vel=(0.0, 0.0, 0.0))
        run_frames(sim, 20)
        pos = sim.read_positions()
        sim.destroy()
        return n, pos, res, dx

    n1, pos1, res1, dx1 = cube_run()
    n2, pos2, res2, dx2 = cube_run()

    check("V1 : compte de particules identique entre 2 runs", n1 == n2, f"n1={n1} n2={n2}")
    check("V1 : resolution isotrope sur un cube", res1[0] == res1[1] == res1[2], f"res={res1}")
    check("V1 : dx identique entre 2 runs (deterministe)", abs(dx1 - dx2) < 1e-9)

    if pos1.shape == pos2.shape:
        diff = np.abs(pos1 - pos2)
        noise = float(diff.mean())
        noise_max = float(diff.max())
        print(f"  bruit de non-determinisme (cube) : mean={noise:.3e} max={noise_max:.3e}")
        # Pas d'epsilon fixe (methode du plan) : la seule chose qu'on peut
        # affirmer ici est que le bruit reste petit devant la taille du
        # domaine (positions bornees dans [0, size]) -- sinon ce serait une
        # divergence, pas du bruit d'ordonnancement.
        check(
            "V1 : le bruit run1 vs run2 reste petit devant la taille du domaine",
            noise_max < 0.1 * max(res1) * dx1,
            f"noise_max={noise_max:.4f}, domaine~{max(res1) * dx1:.4f}",
        )
        check("V1 : aucun NaN/Inf", bool(np.all(np.isfinite(pos1))) and bool(np.all(np.isfinite(pos2))))
    else:
        FAILURES.append("V1 : shapes differentes entre run1 et run2")
        print(f"[FAIL] V1 : pos1.shape={pos1.shape} pos2.shape={pos2.shape}")
except Exception:
    traceback.print_exc()
    FAILURES.append("V1 : exception")


# ---------------------------------------------------------------------------
# V2 — symetrie par permutation d'axes (test le plus discriminant)
# ---------------------------------------------------------------------------
#
# Domaine A : etendues MONDE (X=2, Y=1, Z=1) -> ratio solveur (sx:sy:sz) =
# (worldX:worldZ:worldY) = 2:1:1.
# Domaine B : etendues MONDE (X=1, Y=2, Z=1) -> ratio solveur 1:1:2.
# C'est exactement la scene tournee de 90 degres autour de l'axe VERTICAL
# (monde Z = axe de gravite = solveur sy, invariant sous cette rotation) :
# seuls les deux axes horizontaux (monde X, monde Y) sont permutes, ce qui
# correspond en espace solveur a un echange sx <-> sz (sy, l'axe de
# gravite, reste inchange).
#
# Plutot que de raisonner sur une rotation Blender explicite (risque
# d'erreur de signe), l'emetteur du domaine B est construit DIRECTEMENT en
# imposant que ses bornes SOLVEUR soient l'echange sx<->sz de celles de
# l'emetteur du domaine A (via solver_to_world), ce qui est la definition
# exacte de la symetrie testee : c'est le code de production
# (`emitter_bounds_solver`, `solver_to_world`) qui fait le lien avec
# l'espace monde, mais l'invariant verifie est purement solveur.
print("\n=== V2 : symetrie par permutation d'axes (2:1:1 vs 1:1:2) ===")
try:
    from extension.transform import solver_to_world

    GRID_RES = 40

    # --- domaine A (ratio solveur 2:1:1) ---
    scene_a = new_scene()
    make_domain(scene_a, extent_world=(2.0, 1.0, 1.0), grid_res=GRID_RES)
    # Emetteur asymetrique : decale vers +X monde (-> +sx), leger au-dessus
    # du centre en Z monde (-> sy, axe de gravite).
    emitter_a = make_box_emitter(scene_a, "EmitA", (0.5, 0.0, 0.15), (0.3, 0.3, 0.2))
    origin_a, size_a = domain_transform(scene_a)
    res_a, dx_a = domain_resolution(scene_a)
    lo_a, hi_a = emitter_bounds_solver(emitter_a, origin_a, size_a)
    print(f"  domaine A : res={res_a}, dx={dx_a:.5f}, emitter lo={lo_a}, hi={hi_a}")

    # --- domaine B (ratio solveur 1:1:2), meme grid_res -> meme dx ---
    scene_b = new_scene()
    make_domain(scene_b, extent_world=(1.0, 2.0, 1.0), grid_res=GRID_RES,
                location=(5.0, 0.0, 0.0))  # decale en monde pour ne pas se
                                            # superposer avec la scene A
                                            # (sans effet sur le calcul
                                            # solveur, qui est relatif a
                                            # l'origine du DOMAINE, pas du
                                            # monde)
    origin_b, size_b = domain_transform(scene_b)
    res_b, dx_b = domain_resolution(scene_b)
    print(f"  domaine B : res={res_b}, dx={dx_b:.5f}")

    check("V2 : dx identique entre A et B (meme grid_res, meme max extent)",
          abs(dx_a - dx_b) < 1e-9, f"dx_a={dx_a} dx_b={dx_b}")
    check("V2 : res[B] == permutation (sx<->sz) de res[A]",
          res_b == (res_a[2], res_a[1], res_a[0]),
          f"res_a={res_a} res_b={res_b}")

    # bornes solveur voulues pour l'emetteur B : echange sx<->sz de A.
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
    # Auto-verification de la construction : `emitter_bounds_solver` sur
    # l'objet B fraichement cree doit redonner (a l'epsilon flottant pres)
    # les bornes solveur voulues -- sinon la geometrie construite pour B ne
    # teste pas ce qu'on croit tester.
    lo_b_check, hi_b_check = emitter_bounds_solver(emitter_b, origin_b, size_b)
    lo_err = max(abs(a - b) for a, b in zip(lo_b_check, lo_b_solver))
    hi_err = max(abs(a - b) for a, b in zip(hi_b_check, hi_b_solver))
    check("V2 : construction geometrique de l'emetteur B correcte (lo)",
          lo_err < 1e-4, f"erreur={lo_err:.2e}")
    check("V2 : construction geometrique de l'emetteur B correcte (hi)",
          hi_err < 1e-4, f"erreur={hi_err:.2e}")

    # --- runs ---
    sim_a, _res_a2, _dx_a2 = build_sim(scene_a)
    mat_a = sim_a.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    n_a = sim_a.emit_box(mat_a, lo_a, hi_a, vel=(0.0, 0.0, 0.0))
    run_frames(sim_a, 30)
    pos_a = sim_a.read_positions()
    sim_a.destroy()

    sim_b, _res_b2, _dx_b2 = build_sim(scene_b)
    mat_b = sim_b.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    n_b = sim_b.emit_box(mat_b, lo_b_check, hi_b_check, vel=(0.0, 0.0, 0.0))
    run_frames(sim_b, 30)
    pos_b = sim_b.read_positions()
    sim_b.destroy()

    check("V2 : meme nombre de particules emises (A et B, boites de meme volume)",
          n_a == n_b, f"n_a={n_a} n_b={n_b}")

    # Comparaison AGREGEE (pas particule a particule : l'ordre d'emission
    # de bq_emit_box depend de l'imbrication des boucles par axe, qui n'est
    # PAS simplement permutee entre A et B -- seul l'ENSEMBLE des points
    # l'est). Moyenne/ecart-type/min/max par axe sont invariants a l'ordre
    # et suffisent a detecter une confusion d'axes : une erreur de
    # linearisation romprait ces statistiques de facon grossiere, pas dans
    # le bruit.
    def axis_stats(pos):
        return {
            "mean": pos.mean(axis=0),
            "std": pos.std(axis=0),
            "min": pos.min(axis=0),
            "max": pos.max(axis=0),
        }

    stats_a = axis_stats(pos_a)
    stats_b = axis_stats(pos_b)
    # Permute les stats de A (sx<->sz) pour les rendre comparables a B.
    perm = [2, 1, 0]
    stats_a_perm = {k: v[perm] for k, v in stats_a.items()}

    print(f"  A (permute sx<->sz) : mean={stats_a_perm['mean']}, std={stats_a_perm['std']}")
    print(f"  B                   : mean={stats_b['mean']}, std={stats_b['std']}")

    # Tolerance : bruit de non-determinisme + differences de discretisation
    # residuelles (res_a/res_b different d'au plus l'arrondi), evaluee en
    # fraction de dx (grandeur physique commune aux deux domaines).
    tol = 5.0 * dx_a
    for axis, label in enumerate(("x", "y", "z")):
        err_mean = abs(stats_a_perm["mean"][axis] - stats_b["mean"][axis])
        check(
            f"V2 : moyenne axe {label} (A permute vs B) dans la tolerance",
            err_mean < tol,
            f"A_perm={stats_a_perm['mean'][axis]:.4f} B={stats_b['mean'][axis]:.4f} "
            f"err={err_mean:.4f} tol={tol:.4f}",
        )
        err_std = abs(stats_a_perm["std"][axis] - stats_b["std"][axis])
        check(
            f"V2 : ecart-type axe {label} (A permute vs B) dans la tolerance",
            err_std < tol,
            f"A_perm={stats_a_perm['std'][axis]:.4f} B={stats_b['std'][axis]:.4f} "
            f"err={err_std:.4f} tol={tol:.4f}",
        )

    check("V2 : aucun NaN/Inf (A et B)",
          bool(np.all(np.isfinite(pos_a))) and bool(np.all(np.isfinite(pos_b))))
except Exception:
    traceback.print_exc()
    FAILURES.append("V2 : exception")


# ---------------------------------------------------------------------------
# V3 — parois d'un pave 4:1:1 : fluide au repos, ne fuit par aucune face
# ---------------------------------------------------------------------------
print("\n=== V3 : parois d'un pave 4:1:1 (fluide au repos) ===")
try:
    scene = new_scene()
    make_domain(scene, extent_world=(4.0, 1.0, 1.0), grid_res=48)
    # Bloc d'eau centre, loin des parois au depart, gravite normale : sur
    # quelques frames courtes il tombe et se tasse sans jamais fuir hors
    # de [0, size] sur aucun axe.
    emitter = make_box_emitter(scene, "Box", (0.0, 0.0, 0.0), (2.0, 0.5, 0.5))
    origin, size = domain_transform(scene)
    res, dx = domain_resolution(scene)
    scene.bourrasque.gravity = -9.8

    sim, _res2, _dx2 = build_sim(scene)
    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    lo, hi = emitter_bounds_solver(emitter, origin, size)
    n = sim.emit_box(mat_id, lo, hi, vel=(0.0, 0.0, 0.0))
    check("V3 : emission non vide", n > 0, f"n={n}")

    max_abs_overflow = [0.0, 0.0, 0.0]
    for i in range(40):
        sim.step(1.0 / 24.0)
        pos = sim.read_positions()
        for axis in range(3):
            below = -pos[:, axis].min() if pos.shape[0] else 0.0
            above = pos[:, axis].max() - size[axis] if pos.shape[0] else 0.0
            max_abs_overflow[axis] = max(max_abs_overflow[axis], below, above)

    print(f"  depassement max observe par axe (doit etre <= 0) : {max_abs_overflow}")
    for axis, label in enumerate(("sx", "sy", "sz")):
        check(
            f"V3 : aucune fuite sur l'axe {label} (pave 4:1:1)",
            max_abs_overflow[axis] <= 1e-4,
            f"depassement={max_abs_overflow[axis]:.6f}",
        )

    pos_final = sim.read_positions()
    check("V3 : aucun NaN/Inf apres 40 frames", bool(np.all(np.isfinite(pos_final))))
    sim.destroy()
except Exception:
    traceback.print_exc()
    FAILURES.append("V3 : exception")


# ---------------------------------------------------------------------------
# V5 (emission) — emetteur pres d'une extremite du grand axe d'un domaine
# non cubique : emission correcte, pas de rejet de bornes.
# ---------------------------------------------------------------------------
print("\n=== V5 (emission) : emetteur pres d'un bord du grand axe ===")
try:
    from extension.props import emitter_overflow

    scene = new_scene()
    make_domain(scene, extent_world=(6.0, 1.0, 1.0), grid_res=48)
    origin, size = domain_transform(scene)
    res, dx = domain_resolution(scene)
    # Emetteur cale pres de l'extremite +X (grand axe), a l'interieur de la
    # zone utile (juste apres la marge de stencil).
    margin = 3 * dx
    x_center = 3.0 - (margin + 0.05)  # bord +X du domaine (world X in
                                       # [-3,3], boite domaine centree)
    emitter = make_box_emitter(scene, "EdgeEmit", (x_center, 0.0, 0.0), (0.08, 0.3, 0.3))

    overflow = emitter_overflow(emitter, origin, size, dx)
    check("V5 : emetteur pres du bord du grand axe n'est PAS rejete",
          overflow is None, f"overflow={overflow}")

    sim, _res2, _dx2 = build_sim(scene)
    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    lo, hi = emitter_bounds_solver(emitter, origin, size)
    n = sim.emit_box(mat_id, lo, hi, vel=(0.0, 0.0, 0.0))
    check("V5 : emission pres du bord reussit (n > 0)", n > 0, f"n={n}")
    pos = sim.read_positions()
    check("V5 : positions dans [0, size] sur les 3 axes",
          bool(np.all(pos[:, 0] >= -1e-4)) and bool(np.all(pos[:, 0] <= size[0] + 1e-4))
          and bool(np.all(pos[:, 1] >= -1e-4)) and bool(np.all(pos[:, 1] <= size[1] + 1e-4))
          and bool(np.all(pos[:, 2] >= -1e-4)) and bool(np.all(pos[:, 2] <= size[2] + 1e-4)))
    sim.destroy()
except Exception:
    traceback.print_exc()
    FAILURES.append("V5 (emission) : exception")


print("\n=== RESULTAT ===")
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
else:
    print("Toutes les verifications sont passees.")
sys.exit(1 if FAILURES else 0)
