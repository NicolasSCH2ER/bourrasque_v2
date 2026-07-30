import sys
import bpy

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

import extension.props as props_mod

# register/unregister/register cycle
for cls in props_mod.classes:
    try:
        bpy.utils.unregister_class(cls)
    except Exception:
        pass

props_mod.register()
props_mod.unregister()
props_mod.register()
print("register/unregister/register cycle OK")

# Clean scene (register() already active from the cycle above)
bpy.ops.wm.read_factory_settings(use_empty=True)

scene = bpy.context.scene

# Domain
bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
domain_obj = bpy.context.active_object
domain_obj.name = "Domain"
domain_obj.bourrasque.role = "DOMAIN"
scene.bourrasque.domain_object = domain_obj
scene.bourrasque.grid_res = 32
scene.bourrasque.ppc_axis = 2
scene.frame_start = 1
scene.frame_end = 120
scene.render.fps = 24
scene.render.fps_base = 1.0
scene.bourrasque.frame_start = 1
scene.bourrasque.frame_end = 120

def make_emitter(name, loc, size, emit_mode, velocity):
    bpy.ops.mesh.primitive_cube_add(size=size, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    obj.bourrasque.role = "EMITTER"
    obj.bourrasque.emit_mode = emit_mode
    obj.bourrasque.emit_source = "BOUNDS"
    obj.bourrasque.initial_velocity = velocity
    return obj

e1 = make_emitter("E_block", (0, 0, 0), 0.5, "BLOCK", (0, 0, 0))
count1 = props_mod.estimate_particle_count(scene)
print("BLOCK/BOUNDS only:", count1)

e1.bourrasque.role = "NONE"

e2 = make_emitter("E_inflow_zero", (0, 0, 0), 0.5, "INFLOW", (0, 0, 0))
count2 = props_mod.estimate_particle_count(scene)
print("INFLOW/BOUNDS zero velocity:", count2)

e2.bourrasque.role = "NONE"

e3 = make_emitter("E_inflow_v", (0, 0, 0), 0.5, "INFLOW", (0, 1.0, 0))
count3 = props_mod.estimate_particle_count(scene)
print("INFLOW/BOUNDS v=(0,1,0), frame_end=120:", count3)

scene.bourrasque.frame_end = 240
count3b = props_mod.estimate_particle_count(scene)
print("INFLOW/BOUNDS v=(0,1,0), frame_end=240:", count3b)

assert count3b > count3, "inflow count should grow with duration"

# Test sampling module absence: ensure no crash for MESH source
e3.bourrasque.role = "NONE"
e4 = make_emitter("E_mesh_block", (0, 0, 0), 0.5, "BLOCK", (0, 0, 0))
e4.bourrasque.emit_source = "MESH"
try:
    count4 = props_mod.estimate_particle_count(scene)
    print("BLOCK/MESH (sampling absent, fallback to bbox):", count4)
except Exception as ex:
    print("FAILED: exception raised when sampling module absent:", ex)
    raise

# Simulate an "incomplete" sampling module present but broken (raises on call).
import types

fake_sampling = types.ModuleType("extension.sampling")

def _broken(*args, **kwargs):
    raise RuntimeError("stub incomplet, simule un module casse")

fake_sampling.estimate_mesh_sample_count = _broken
sys.modules["extension.sampling"] = fake_sampling

try:
    count5 = props_mod.estimate_particle_count(scene)
    print("BLOCK/MESH (sampling present but broken, fallback to bbox):", count5)
    assert count5 == count4, "fallback should match plain bbox estimate"
except Exception as ex:
    print("FAILED: exception raised when sampling module is broken:", ex)
    raise
finally:
    del sys.modules["extension.sampling"]

print("ALL CHECKS PASSED")
