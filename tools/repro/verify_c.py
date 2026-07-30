"""Verification du correctif C, hors modal reel.

`bpy.ops.bq.bake('INVOKE_DEFAULT')` echoue systematiquement en
`--background` (verifie sur un operateur modal trivial de sanite : Blender
refuse d'attacher un modal handler hors boucle d'evenements interactive,
message « Invalid operator call »). On contourne cette limitation
d'ENVIRONNEMENT (pas du code teste) en creant un objet-relais qui porte les
memes methodes que `BQ_OT_bake` (recuperees directement depuis la classe)
et en appelant `invoke()` dessus avec un vrai `context` : toute la logique
de validation / echantillonnage / emission testee ici est le vrai code de
`ops.py`, seul le cablage RNA final (`modal_handler_add`, qui exige un
VRAI `bpy.types.Operator`) est court-circuite.
"""
import sys
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
import bpy

import extension
extension.register()

from extension import ops as bq_ops, props


class _FakeOperator:
    cancel_requested = False
    _active_instance = None


for _name in dir(bq_ops.BQ_OT_bake):
    if _name.startswith("__"):
        continue
    _attr = getattr(bq_ops.BQ_OT_bake, _name)
    if callable(_attr) and not isinstance(_attr, type):
        setattr(_FakeOperator, _name, _attr)

# Court-circuite uniquement le cablage RNA final (timer + modal handler),
# impossible hors interaction reelle : on garde la Sim GPU vivante pour
# pouvoir l'inspecter, et on la detruira nous-memes a la fin.
def _fake_finalize(self, context):
    pass


_FakeOperator._cleanup = _fake_finalize


def make_scene(emit_source, cube_size_factor):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"

    scene = bpy.context.scene
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = 64
    scene.bourrasque.ppc_axis = 2
    scene.bourrasque.frame_start = 1
    scene.bourrasque.frame_end = 2

    origin, solver_size = props.domain_transform(scene)
    dx = solver_size / scene.bourrasque.grid_res
    spacing = dx / scene.bourrasque.ppc_axis
    cube_size = spacing * cube_size_factor

    bpy.ops.mesh.primitive_cube_add(size=cube_size, location=(0, 0, 0))
    emitter = bpy.context.active_object
    emitter.name = "TinyEmitter"
    emitter.bourrasque.role = "EMITTER"
    emitter.bourrasque.emit_mode = "BLOCK"
    emitter.bourrasque.emit_source = emit_source
    emitter.bourrasque.model = "WATER"

    return scene, emitter, spacing, cube_size


def run_case(emit_source):
    print(f"--- emit_source = {emit_source} ---")
    scene, emitter, spacing, cube_size = make_scene(emit_source, 1.0 / 3.0)
    print(f"spacing = {spacing:.6f} m (espace solveur)")
    print(f"cube_size (emetteur) = {cube_size:.6f} m ( = spacing/3 = {spacing/3:.6f} m)")

    estimated = props.estimate_particle_count(scene)
    print(f"estimate_particle_count -> {estimated} (attendu >= 1)")

    reports = []
    inst = _FakeOperator()
    inst.report = lambda level, msg: reports.append((level, msg))

    context = bpy.context
    event = type("E", (), {"type": "TIMER"})()

    result = bq_ops.BQ_OT_bake.invoke(inst, context, event)
    print(f"invoke() -> {result}")

    for level, msg in reports:
        print(f"  report {level}: {msg}")

    sim = getattr(inst, "_sim", None)
    n = sim.particle_count if sim is not None else None
    print(f"sim.particle_count = {n}")

    has_warning = any(
        "WARNING" in level and "plus petit que le pas du réseau" in msg
        for level, msg in reports
    )
    print(f"avertissement 'plus petit que le pas du reseau' present : {has_warning}")

    # Nettoyage manuel de la Sim GPU (le vrai `_cleanup` n'a pas ete
    # atteint puisque le cablage RNA final a leve une exception).
    if sim is not None:
        sim.destroy()

    return result, n, has_warning


results = {}
for emit_source in ("BOUNDS", "MESH"):
    results[emit_source] = run_case(emit_source)
    print()

print("Resume :")
for k, (result, n, has_warning) in results.items():
    print(f"  {k}: sim cree = {result != {'CANCELLED'} or n is not None}, "
          f"particules = {n}, avertissement = {has_warning}")
