import sys
import bpy

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")

import extension.props as props_mod
import extension.ui as ui_mod

props_mod.register()
ui_mod.register()

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
domain_obj = bpy.context.active_object
domain_obj.bourrasque.role = "DOMAIN"
scene = bpy.context.scene
scene.bourrasque.domain_object = domain_obj
scene.bourrasque.grid_res = 32
scene.bourrasque.ppc_axis = 2

# closed mesh (icosphere) emitter, MESH source, BLOCK mode
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.4, location=(0, 0, 0))
emitter = bpy.context.active_object
emitter.bourrasque.role = "EMITTER"
emitter.bourrasque.emit_mode = "BLOCK"
emitter.bourrasque.emit_source = "MESH"

count = props_mod.estimate_particle_count(scene)
print("BLOCK/MESH icosphere real sampling count:", count)

from extension.sampling import check_mesh_closed
closed, msg = check_mesh_closed(emitter)
print("check_mesh_closed (icosphere):", closed, repr(msg))

# open mesh (plane) emitter
bpy.ops.mesh.primitive_plane_add(size=0.4, location=(0, 0, 0))
plane = bpy.context.active_object
plane.bourrasque.role = "EMITTER"
plane.bourrasque.emit_source = "MESH"
closed2, msg2 = check_mesh_closed(plane)
print("check_mesh_closed (plane, open):", closed2, repr(msg2))

print("DONE")
