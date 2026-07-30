"""Reproduit le chemin d'activation reel de l'extension (addon_utils.enable),
qui est celui ou bpy.data est un _RestrictData.
"""
import addon_utils
import bpy

MOD = "bl_ext.user_default.bourrasque"

for cycle in (1, 2):
    ok = addon_utils.enable(MOD, default_set=True, persistent=True)
    print(f"cycle {cycle} : enable -> {ok!r}")
    if not ok:
        raise SystemExit(f"ECHEC activation au cycle {cycle}")
    addon_utils.disable(MOD, default_set=True)
    print(f"cycle {cycle} : disable OK")

# Reactive et laisse le timer differe s'executer.
addon_utils.enable(MOD, default_set=True, persistent=True)
print("is_baking apres activation :",
      [s.bourrasque.is_baking for s in bpy.data.scenes])
print("load_post :", [h.__name__ for h in bpy.app.handlers.load_post])
print("frame_change_post :", [h.__name__ for h in bpy.app.handlers.frame_change_post])
addon_utils.disable(MOD, default_set=True)
print("load_post apres disable :", [h.__name__ for h in bpy.app.handlers.load_post])
print("frame_change_post apres disable :",
      [h.__name__ for h in bpy.app.handlers.frame_change_post])
print("TOUT OK")
