"""Hypothese : BVHTree.FromObject travaille en espace LOCAL, or sampling.py
lui envoie des points monde. Si c'est vrai, deplacer l'emetteur ne deplace pas
le nuage echantillonne.
"""
import sys

import bpy
import numpy as np

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
from extension import sampling

# Domaine : cube de 10 m centre sur l'origine -> bbox monde [-5, 5]^3
ORIGIN = (-5.0, -5.0, -5.0)
SIZE = 10.0
GRID_RES, PPC = 64, 2


def sample_at(z):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0.0, 0.0, z))
    obj = bpy.context.active_object
    pts = sampling.sample_mesh_interior(obj, ORIGIN, SIZE, GRID_RES, PPC)
    if pts.shape[0] == 0:
        return obj, pts, None
    # centre du nuage, ramene en monde (sy solveur = bz monde - mz)
    centre_sz_monde = float(pts[:, 1].mean()) + ORIGIN[2]
    return obj, pts, centre_sz_monde


for z in (0.0, 3.0):
    obj, pts, centre = sample_at(z)
    attendu = z
    print(f"sphere a z={z:+.1f} : {pts.shape[0]:6d} points, "
          f"centre monde z = {centre if centre is None else f'{centre:+.3f}'}, "
          f"attendu {attendu:+.1f}")

print()
print("Si le centre reste a 0 quand la sphere monte a z=3, l'hypothese est confirmee.")
