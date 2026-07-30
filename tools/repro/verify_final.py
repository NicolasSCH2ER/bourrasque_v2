"""Verification globale finale : bake complet hors modal, sphere emettrice
MESH deplacee hors de l'origine -- les particules doivent apparaitre DANS
la sphere a la premiere frame (symptome rapporte par l'utilisateur)."""
import sys
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
import math
import bpy
import numpy as np

import extension
extension.register()

from extension import ops as bq_ops, props, cache

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

bpy.ops.mesh.primitive_cube_add(size=6.0, location=(0, 0, 0))
domain = bpy.context.active_object
domain.name = "Domain"
domain.bourrasque.role = "DOMAIN"

scene = bpy.context.scene
scene.bourrasque.domain_object = domain
scene.bourrasque.grid_res = 64
scene.bourrasque.ppc_axis = 2
scene.bourrasque.frame_start = 1
scene.bourrasque.frame_end = 3
scene.bourrasque.cache_dir = (
    r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2"
    r"\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\_bq_final_cache\\"
)

# Sphere emettrice DEPLACEE hors de l'origine (le symptome rapporte).
sphere_center = (1.0, -0.5, 1.5)
sphere_radius = 1.0
bpy.ops.mesh.primitive_uv_sphere_add(
    radius=sphere_radius, location=sphere_center, segments=32, ring_count=16
)
emitter = bpy.context.active_object
emitter.name = "Sphere"
emitter.bourrasque.role = "EMITTER"
emitter.bourrasque.emit_mode = "BLOCK"
emitter.bourrasque.emit_source = "MESH"
emitter.bourrasque.model = "WATER"

class _FakeOperator:
    pass

for name in dir(bq_ops.BQ_OT_bake):
    if name.startswith("__"):
        continue
    setattr(_FakeOperator, name, getattr(bq_ops.BQ_OT_bake, name))


def _no_cleanup(self, context):
    # On veut inspecter la Sim/le writer APRES l'echec attendu du cablage
    # RNA final (modal_handler_add, qui exige un vrai bpy.types.Operator,
    # impossible a obtenir hors interaction reelle -- voir verify_c.py) :
    # court-circuite le vrai _cleanup pour ne pas detruire la Sim avant
    # inspection. Nettoyage manuel fait a la fin du script.
    pass


_FakeOperator._cleanup = _no_cleanup

reports = []
inst = _FakeOperator()
inst.report = lambda level, msg: reports.append((level, msg))
context = bpy.context
event = type("E", (), {"type": "TIMER"})()

result = bq_ops.BQ_OT_bake.invoke(inst, context, event)
print("invoke ->", result)
for level, msg in reports:
    print(" report", level, msg)

sim = getattr(inst, "_sim", None)
assert sim is not None, "la Sim GPU aurait du etre creee (validation OK attendue)"

n = sim.particle_count
print(f"particules emises a la premiere frame = {n}")
assert n > 0, "aucune particule emise"

positions_solver = sim.read_positions()
origin, solver_size = props.domain_transform(scene)
positions_world = np.array(
    [props.solver_to_world(tuple(p), origin, solver_size) for p in positions_solver]
)

center = np.array(sphere_center)
dist = np.linalg.norm(positions_world - center, axis=1)
max_dist = dist.max()
n_outside = int(np.sum(dist > sphere_radius + 1e-3))

print(f"centre sphere (monde) = {sphere_center}")
print(f"distance max au centre de la sphere parmi les particules = {max_dist:.5f} (rayon = {sphere_radius})")
print(f"particules hors sphere (tolerance 1e-3) = {n_outside} / {n}")

assert max_dist <= sphere_radius + 1e-2, (
    "des particules sont apparues HORS de la sphere : le bug de repere "
    "(correctif A) n'est pas resolu"
)
assert n_outside == 0

print("VERIFICATION GLOBALE : particules bien DANS la sphere deplacee -- OK")

writer = getattr(inst, "_writer", None)
if writer is not None:
    try:
        writer.close()
    except Exception:
        pass
sim.destroy()
