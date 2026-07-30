import sys

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

ROOT = r"C:\Users\nicol\Code\bourrasque_v2"
sys.path.insert(0, ROOT)

import extension  # noqa: E402

extension.register()
extension.unregister()
extension.register()
extension.unregister()
extension.register()

print("REGISTER_CYCLE_OK")
sys.exit(0)
