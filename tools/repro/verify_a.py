import sys
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
import math
import bpy
import numpy as np
from mathutils import Euler

from extension import sampling
from extension.transform import solver_to_world

origin = (-5.0, -5.0, -5.0)
size = 10.0
grid_res = 64
ppc_axis = 2

cases = [
    ("origine", (0, 0, 0), (0, 0, 0), (1, 1, 1)),
    ("translate (0,0,3)", (0, 0, 3), (0, 0, 0), (1, 1, 1)),
    ("translate (2,-1,3)", (2, -1, 3), (0, 0, 0), (1, 1, 1)),
    ("rotation 45deg", (0, 0, 0), (0, 0, math.radians(45)), (1, 1, 1)),
    ("echelle uniforme 2", (0, 0, 0), (0, 0, 0), (2, 2, 2)),
    ("echelle non uniforme (2,1,0.5)", (0, 0, 0), (0, 0, 0), (2, 1, 0.5)),
]

print(f"{'cas':35s} {'n_pts':>8s} {'vol_reel':>10s} {'vol_est_par_pts':>16s} {'centre_obj':>24s} {'centre_nuage':>24s} {'ecart':>10s}")

for name, loc, rot, scale in cases:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, segments=32, ring_count=16)
    obj = bpy.context.active_object
    obj.location = loc
    obj.rotation_euler = Euler(rot)
    obj.scale = scale
    bpy.context.view_layer.update()

    pts_solver = sampling.sample_mesh_interior(obj, origin, size, grid_res, ppc_axis)
    n = pts_solver.shape[0]

    dx = size / grid_res
    spacing = dx / ppc_axis
    cell_vol = spacing ** 3
    vol_est = n * cell_vol

    # vrai volume de la sphere transformee : (4/3) pi r^3 * sx*sy*sz
    real_vol = (4.0 / 3.0) * math.pi * (1.0 ** 3) * scale[0] * scale[1] * scale[2]

    if n > 0:
        pts_world = np.array([solver_to_world(tuple(p), origin, size) for p in pts_solver])
        centroid_world = pts_world.mean(axis=0)
    else:
        centroid_world = np.array([float('nan')] * 3)

    obj_center_world = np.array(obj.matrix_world @ obj.location.__class__((0, 0, 0))) if False else np.array(loc, dtype=float)
    # objet centre = matrix_world @ origin_local (sphere centree en 0 local)
    obj_center_world = np.array(obj.matrix_world.translation)

    ecart = np.linalg.norm(centroid_world - obj_center_world)

    print(f"{name:35s} {n:8d} {real_vol:10.4f} {vol_est:16.4f} {str(np.round(obj_center_world,3)):>24s} {str(np.round(centroid_world,3)):>24s} {ecart:10.4f}  (pas reseau dx={spacing:.4f})")

print("OK")
