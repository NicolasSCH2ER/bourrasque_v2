import sys
import bpy

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

import extension.props as props_mod
import extension.ui as ui_mod

props_mod.register()
ui_mod.register()
print("ui register OK")

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
domain_obj = bpy.context.active_object
domain_obj.bourrasque.role = "DOMAIN"
scene = bpy.context.scene
scene.bourrasque.domain_object = domain_obj

bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0))
emitter = bpy.context.active_object
emitter.bourrasque.role = "EMITTER"
emitter.bourrasque.emit_mode = "INFLOW"
emitter.bourrasque.emit_source = "MESH"
emitter.bourrasque.initial_velocity = (0.0, 0.0, 0.0)
bpy.context.view_layer.objects.active = emitter

# Simulate mimimal draw call using a UILayout from a temporary window if
# possible; drawing panels fully requires a UI context which --background
# doesn't have. Instead, directly exercise the draw() body logic we can
# reach without a real layout by using bpy.types.UILayout is not
# constructible standalone. So just check poll() and the property reads
# used throughout draw() manually mirror what draw() does, without a
# layout object -- verifies no AttributeError on property names.

panel = ui_mod.BQ_PT_material
print("poll:", panel.poll(bpy.context))

obj_props = emitter.bourrasque
_ = obj_props.preset
_ = obj_props.model
_ = obj_props.rho
_ = obj_props.initial_velocity
_ = obj_props.emit_mode
_ = obj_props.emit_source
print("all referenced properties exist and are readable")

print("_is_vector_zero:", ui_mod._is_vector_zero(obj_props.initial_velocity))
print("_is_object_animated:", ui_mod._is_object_animated(emitter))

ui_mod.unregister()
props_mod.unregister()
print("ALL UI CHECKS PASSED")
