"""test_sampling_triangles.py — verification de `sampling.evaluated_world_mesh`
/ `evaluated_world_triangles` (extraction de triangles monde d'un objet
evalue), et du calcul de vitesse par sommet d'un collider (ops.py, M6).

Depend de bpy (bmesh/BVHTree) : ne peut PAS s'executer via `python
test_sampling_triangles.py`, contrairement aux six suites `python`-only du
dossier. A lancer via :

    blender --background --factory-startup --python extension/tests/test_sampling_triangles.py

Les six suites `python`-only restent inchangees et executables telles
quelles ; ce fichier est un septieme test, specifique au jalon M6, qui
necessite l'interpreteur Python de Blender pour bpy/bmesh.
"""

import os
import sys

import bpy
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

bpy.ops.wm.read_factory_settings(use_empty=True)

import extension  # noqa: E402

extension.register()

from extension import sampling  # noqa: E402
from extension.transform import world_to_solver_dir_array  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def approx_eq(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# 1) evaluated_world_mesh / evaluated_world_triangles : forme et coherence
#    avec matrix_world.
# ---------------------------------------------------------------------------

scene = bpy.context.scene
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(5.0, -3.0, 1.0))
cube = bpy.context.active_object
cube.name = "TriTestCube"
cube.rotation_euler = (0.0, 0.0, 0.7)
cube.scale = (1.5, 1.0, 0.5)
bpy.context.view_layer.update()

depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = cube.evaluated_get(depsgraph)

verts_world, tris = sampling.evaluated_world_mesh(obj_eval)
check(
    "evaluated_world_mesh: verts_world shape (n,3)",
    verts_world.ndim == 2 and verts_world.shape[1] == 3 and verts_world.shape[0] > 0,
    f"shape={verts_world.shape}",
)
check(
    "evaluated_world_mesh: tris shape (n,3), au moins 12 triangles (cube)",
    tris.ndim == 2 and tris.shape[1] == 3 and tris.shape[0] >= 12,
    f"shape={tris.shape}",
)
check(
    "evaluated_world_mesh: indices de tris valides (< n_verts)",
    bool(np.all(tris < verts_world.shape[0])) and bool(np.all(tris >= 0)),
)

tris_world = sampling.evaluated_world_triangles(obj_eval)
check(
    "evaluated_world_triangles: forme (n_tri, 3, 3)",
    tris_world.shape == (tris.shape[0], 3, 3),
    f"shape={tris_world.shape}",
)
check(
    "evaluated_world_triangles == verts_world[tris]",
    bool(np.allclose(tris_world, verts_world[tris])),
)

# Coherence avec matrix_world : le centroide de TOUS les sommets monde doit
# coincider avec le centre de l'objet (cube centre a l'origine locale,
# translate uniquement -- rotation/echelle ne deplacent pas le centroide).
centroid = verts_world.mean(axis=0)
expected_centroid = np.array(cube.matrix_world.translation)
err_centroid = float(np.max(np.abs(centroid - expected_centroid)))
check(
    "evaluated_world_mesh: centroide des sommets == translation de matrix_world",
    err_centroid < 1e-5,
    f"centroid={centroid} expected={expected_centroid} err={err_centroid:.2e}",
)

# La demi-diagonale du cube local est sqrt(3) (taille 2 -> demi-arete 1) ;
# apres matrix_world (rotation + echelle non uniforme), chaque sommet monde
# doit rester a la distance EXACTE que donnerait une transformation directe
# du sommet local correspondant -- verifie sur un sommet arbitraire (le
# premier) en comparant a une transformation manuelle via matrix_world.
mat = np.array(cube.matrix_world, dtype=np.float64)
mesh_local = obj_eval.to_mesh()
try:
    co_local = np.array(mesh_local.vertices[0].co, dtype=np.float64)
finally:
    obj_eval.to_mesh_clear()
expected_v0 = mat[:3, :3] @ co_local + mat[:3, 3]
err_v0 = float(np.max(np.abs(verts_world[0] - expected_v0)))
check(
    "evaluated_world_mesh: sommet 0 == matrix_world appliquee au sommet local",
    err_v0 < 1e-5,
    f"got={verts_world[0]} expected={expected_v0} err={err_v0:.2e}",
)

bpy.data.objects.remove(cube, do_unlink=True)


# ---------------------------------------------------------------------------
# 2) Vitesse par sommet d'un collider anime (logique de _update_colliders,
#    ops.py) : deplacement le long d'un axe MONDE connu -> vitesse le long
#    de l'axe SOLVEUR correspondant, magnitude correcte.
# ---------------------------------------------------------------------------

bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
mover = bpy.context.active_object
mover.name = "TriTestMover"
bpy.context.view_layer.update()

depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = mover.evaluated_get(depsgraph)
verts_prev, _tris = sampling.evaluated_world_mesh(obj_eval)

# Deplace de +2m le long de +Y MONDE.
DELTA_Y = 2.0
FRAME_DT = 1.0 / 24.0
mover.location = (0.0, DELTA_Y, 0.0)
bpy.context.view_layer.update()

depsgraph = bpy.context.evaluated_depsgraph_get()
obj_eval = mover.evaluated_get(depsgraph)
verts_cur, _tris = sampling.evaluated_world_mesh(obj_eval)

vel_world = (verts_cur - verts_prev) / FRAME_DT
vel_solver = world_to_solver_dir_array(vel_world)

expected_speed = DELTA_Y / FRAME_DT
check(
    "vitesse par sommet : deplacement +Y monde -> vitesse world uniforme",
    bool(np.allclose(vel_world[:, 1], expected_speed, atol=1e-6))
    and bool(np.allclose(vel_world[:, 0], 0.0, atol=1e-6))
    and bool(np.allclose(vel_world[:, 2], 0.0, atol=1e-6)),
    f"vel_world sample={vel_world[0]}",
)

# world_to_solver_dir : (vx, vy, vz) -> (vx, vz, -vy). +Y monde pur donne
# donc (0, 0, -expected_speed) en espace solveur : deplacement +Y monde ->
# vitesse le long de -sz solveur, JAMAIS de composante sx/sy parasite (le
# piege documente du jalon : confondre direction et position).
check(
    "vitesse par sommet : +Y monde -> -sz solveur (mapping d'axe correct)",
    bool(np.allclose(vel_solver[:, 0], 0.0, atol=1e-6))
    and bool(np.allclose(vel_solver[:, 1], 0.0, atol=1e-6))
    and bool(np.allclose(vel_solver[:, 2], -expected_speed, atol=1e-6)),
    f"vel_solver sample={vel_solver[0]} expected_speed={expected_speed}",
)

bpy.data.objects.remove(mover, do_unlink=True)


# ---------------------------------------------------------------------------
# RESULTAT
# ---------------------------------------------------------------------------

print()
if FAILURES:
    print(f"{len(FAILURES)} test(s) failed: {FAILURES}")
    sys.exit(1)
else:
    print("Tous les tests sont passes.")
    sys.exit(0)
