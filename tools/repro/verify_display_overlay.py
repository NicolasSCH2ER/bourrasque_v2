"""Verification runtime de display.py / overlay.py sous
`blender --background --factory-startup --python verify_display_overlay.py`.
"""
import os
import sys
import time
import traceback

import numpy as np
import bpy

REPO = r"C:\Users\nicol\Code\bourrasque_v2"
SCRATCH = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad"

sys.path.insert(0, REPO)

FAILURES = []


def check(cond, msg):
    if not cond:
        FAILURES.append(msg)
        print(f"FAIL: {msg}")
    else:
        print(f"OK: {msg}")


# --- import extension package as 'extension' -----------------------------
import importlib

if "extension" in sys.modules:
    del sys.modules["extension"]
ext_names = [n for n in list(sys.modules) if n.startswith("extension.")]
for n in ext_names:
    del sys.modules[n]

import extension as bq_ext  # noqa: E402

bq_ext.register()

from extension import cache, display, overlay, props, transform  # noqa: E402

# ---------------------------------------------------------------------------
# Build a minimal scene: domain object + bourrasque props
# ---------------------------------------------------------------------------
scene = bpy.context.scene
scene.name = "TestScene"

bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
domain_obj = bpy.context.active_object
domain_obj.name = "TestDomain"
domain_obj.bourrasque.role = "DOMAIN"
scene.bourrasque.domain_object = domain_obj
scene.bourrasque.frame_start = 1
scene.bourrasque.cache_dir = SCRATCH

origin, size = props.domain_transform(scene)
print("domain transform:", origin, size)

# ---------------------------------------------------------------------------
# TASK 1 — v2 variable-count cache
# ---------------------------------------------------------------------------

bqd_path_v2 = os.path.join(SCRATCH, "TestScene.bqd")
mat_path_v2 = os.path.join(SCRATCH, "TestScene.mat")
for p in (bqd_path_v2, mat_path_v2):
    if os.path.exists(p):
        os.remove(p)

counts = [100, 500, 2000, 8000]
rng = np.random.default_rng(42)
frames_positions = []
with cache.CacheWriter(bqd_path_v2) as w:
    for c in counts:
        pos = rng.uniform(0.0, size, size=(c, 3)).astype(np.float32)
        frames_positions.append(pos)
        w.append_frame(pos)
    mat = rng.integers(0, 3, size=counts[-1], dtype=np.uint8)
    w.write_materials(mat)

n_max = counts[-1]

ctx = bpy.context
obj = display.ensure_particle_object(ctx, n_max)
check(len(obj.data.vertices) == n_max, f"v2: mesh allocated at n_max={n_max}")

display.write_material_attribute(obj, mat)

vcount_before = len(obj.data.vertices)

reader_check = cache.CacheReader(bqd_path_v2)

timings = []
for i, c in enumerate(counts):
    scene.frame_set(1 + i)
    t0 = time.perf_counter()
    display.refresh(scene)
    dt = time.perf_counter() - t0
    timings.append(dt)

    mesh = bpy.data.objects.get("Bourrasque_Particles").data
    check(len(mesh.vertices) == n_max, f"v2 frame {i}: no reallocation, len={len(mesh.vertices)}")

    co = np.empty(n_max * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n_max, 3)

    expected_world = transform.solver_to_world_array(frames_positions[i], origin, size)
    ok_valid = np.allclose(co[:c], expected_world, atol=1e-4)
    check(ok_valid, f"v2 frame {i}: first {c} vertices match cache positions (world space)")

    if c < n_max:
        fallback = co[c:n_max]
        first = fallback[0]
        ok_fallback = np.all(np.all(fallback == first, axis=1)) and fallback.shape[0] == n_max - c
        check(ok_fallback, f"v2 frame {i}: excess {n_max - c} vertices all at single fallback position {first}")
    else:
        print(f"v2 frame {i}: count == n_max, no excess vertices to check")

reader_check.close()

print("v2 refresh timings (s):", timings)
print(f"v2 refresh mean: {sum(timings)/len(timings)*1000:.3f} ms, max: {max(timings)*1000:.3f} ms")

# ---------------------------------------------------------------------------
# TASK 1 — v1 non-regression (dam.bqd at repo root)
# ---------------------------------------------------------------------------

dam_bqd = os.path.join(REPO, "dam.bqd")
dam_mat = os.path.join(REPO, "dam.mat")
if os.path.isfile(dam_bqd):
    scene2_name = "TestSceneV1"
    bpy.ops.scene.new(type="EMPTY")
    scene2 = bpy.context.scene
    scene2.name = scene2_name

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 0))
    domain_obj2 = bpy.context.active_object
    domain_obj2.name = "TestDomainV1"
    domain_obj2.bourrasque.role = "DOMAIN"
    scene2.bourrasque.domain_object = domain_obj2
    scene2.bourrasque.frame_start = 1
    scene2.bourrasque.cache_dir = REPO  # dam.bqd is at repo root, name "dam"

    origin2, size2 = props.domain_transform(scene2)

    import shutil
    dst_bqd = os.path.join(REPO, f"{scene2_name}.bqd")
    dst_mat = os.path.join(REPO, f"{scene2_name}.mat")
    shutil.copyfile(dam_bqd, dst_bqd)
    if os.path.isfile(dam_mat):
        shutil.copyfile(dam_mat, dst_mat)

    reader_v1 = cache.CacheReader(dst_bqd)
    n_v1 = reader_v1.n_particles
    frame_count_v1 = reader_v1.frame_count
    check(not reader_v1.is_variable, "v1: is_variable is False")

    obj_v1 = display.ensure_particle_object(bpy.context, n_v1)
    check(len(obj_v1.data.vertices) == n_v1, f"v1: mesh allocated at n={n_v1}")

    v1_timings = []
    n_frames_to_test = min(3, frame_count_v1)
    for i in range(n_frames_to_test):
        scene2.frame_set(1 + i)
        t0 = time.perf_counter()
        display.refresh(scene2)
        dt = time.perf_counter() - t0
        v1_timings.append(dt)

        mesh_v1 = bpy.data.objects.get("Bourrasque_Particles").data
        check(len(mesh_v1.vertices) == n_v1, f"v1 frame {i}: no reallocation, len={len(mesh_v1.vertices)}")

        pos_i = reader_v1.read_frame(i)
        expected_world_v1 = transform.solver_to_world_array(np.asarray(pos_i), origin2, size2)

        co_v1 = np.empty(n_v1 * 3, dtype=np.float32)
        mesh_v1.vertices.foreach_get("co", co_v1)
        co_v1 = co_v1.reshape(n_v1, 3)

        ok_v1 = np.allclose(co_v1, expected_world_v1, atol=1e-3)
        check(ok_v1, f"v1 frame {i}: all {n_v1} vertices match (world space)")

    reader_v1.close()
    print("v1 refresh timings (s):", v1_timings)
    if v1_timings:
        print(f"v1 refresh mean: {sum(v1_timings)/len(v1_timings)*1000:.3f} ms")

else:
    print(f"SKIP v1 non-regression: {dam_bqd} not found")
    dst_bqd = dst_mat = None

# ---------------------------------------------------------------------------
# TASK 2 — overlay: check registration doesn't crash, and emit_source logic
# by directly exercising _object_world_bounds / iter_elements path
# indirectly via a dry-run of _draw() logic (headless, no viewport draw call
# possible without an actual GL context, so we validate only the filtering
# logic by inspecting source, and the register/unregister cycle below).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TASK 4 — register/unregister/register cycle: no duplicate/orphan handlers
# ---------------------------------------------------------------------------

def count_bq_frame_handlers():
    return sum(
        1
        for fn in bpy.app.handlers.frame_change_post
        if fn.__name__ == "_bq_frame_change_post"
    )


bq_ext.unregister()
check(count_bq_frame_handlers() == 0, "after unregister: no frame_change_post handler left")
check(overlay._draw_handler is None, "after unregister: overlay._draw_handler is None")

bq_ext.register()
check(count_bq_frame_handlers() == 1, "after register: exactly 1 frame_change_post handler")
check(overlay._draw_handler is not None, "after register: overlay._draw_handler set")

bq_ext.unregister()
bq_ext.register()
check(count_bq_frame_handlers() == 1, "after re-register cycle: still exactly 1 handler")

bq_ext.unregister()
check(count_bq_frame_handlers() == 0, "final unregister: 0 handlers")

# ---------------------------------------------------------------------------
# Cleanup temp v1 copy now that display's cached reader (closed by
# unregister()) no longer holds the file open.
# ---------------------------------------------------------------------------
for p in (dst_bqd, dst_mat):
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except PermissionError as e:
            print(f"WARN: could not remove {p}: {e}")

print("\n=== SUMMARY ===")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
