import sys

import bpy

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")


def handler_names():
    return [fn.__name__ for fn in bpy.app.handlers.frame_change_post]


before = len(bpy.app.handlers.frame_change_post)
print("handlers before:", before, handler_names())

import extension
from extension import overlay

extension.register()
after_reg1 = len(bpy.app.handlers.frame_change_post)
overlay_h1 = overlay._draw_handler
print("handlers after register #1:", after_reg1, handler_names())
print("overlay draw handler after register #1:", overlay_h1)
assert overlay_h1 is not None, "overlay draw handler not set after register #1"

extension.unregister()
after_unreg = len(bpy.app.handlers.frame_change_post)
overlay_h_unreg = overlay._draw_handler
print("handlers after unregister:", after_unreg, handler_names())
print("overlay draw handler after unregister:", overlay_h_unreg)
assert overlay_h_unreg is None, "overlay draw handler not cleared after unregister"

extension.register()
after_reg2 = len(bpy.app.handlers.frame_change_post)
overlay_h2 = overlay._draw_handler
print("handlers after register #2:", after_reg2, handler_names())
print("overlay draw handler after register #2:", overlay_h2)

# Simule un rechargement pendant le developpement : register() est appele
# une 2e fois sans unregister() prealable. La garde de overlay.register()
# doit retirer l'ancien handler avant d'en ajouter un nouveau : aucune
# duplication, meme si l'identite Python change (nouvelle closure).
overlay.register()
overlay_h3 = overlay._draw_handler
print("overlay draw handler after direct 2nd overlay.register():", overlay_h3)
assert overlay_h3 is not overlay_h2, "expected a fresh handler reference"

names = handler_names()
dupes = len(names) - len(set(names))
print("duplicate frame_change_post handler names:", dupes)

extension.unregister()
overlay_h_final = overlay._draw_handler
print("overlay draw handler after final unregister:", overlay_h_final)
assert overlay_h_final is None, "overlay draw handler not cleared on final unregister"

assert before == after_unreg, f"handler leak: before={before} after_unreg={after_unreg}"
assert after_reg1 == after_reg2, f"inconsistent handler count across register calls: {after_reg1} != {after_reg2}"
assert dupes == 0, "duplicate frame_change_post handlers found"

print("ALL CHECKS PASSED")
