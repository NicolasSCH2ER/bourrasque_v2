import importlib, sys, types, time, math
EXT_ROOT = r"C:\Users\nicol\Code\bourrasque_v2"
_pkg = types.ModuleType("extension")
_pkg.__path__ = [EXT_ROOT + r"\extension"]
sys.modules["extension"] = _pkg
sampling = importlib.import_module("extension.sampling")

import bpy

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

# Grand domaine, tore de taille "raisonnable pour une scene" mais avec un
# reseau dense (ppc_axis eleve) pour approcher qqes centaines de milliers
# de points candidats.
ORIGIN = (-10.0, -10.0, -10.0)
SIZE = 20.0
GRID_RES = 128
PPC_AXIS = 3

bpy.ops.mesh.primitive_torus_add(location=(0,0,0), major_radius=3.0, minor_radius=1.0,
                                  major_segments=64, minor_segments=32)
obj = bpy.context.active_object

t0 = time.perf_counter()
pts = sampling.sample_mesh_interior(obj, ORIGIN, SIZE, GRID_RES, PPC_AXIS)
dt = time.perf_counter() - t0
print(f"grid_res={GRID_RES} ppc_axis={PPC_AXIS} torus R=3 r=1 -> n_points={pts.shape[0]} temps={dt:.3f}s")
