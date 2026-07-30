import sys
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
import bpy
import numpy as np

import extension
extension.register()

from extension import props, lib

# Cree une scene : domaine cube de cote 4, emetteur presque plein domaine.
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0, 0, 0))
domain = bpy.context.active_object
domain.name = "Domain"
domain.bourrasque.role = "DOMAIN"

scene = bpy.context.scene
scene.bourrasque.domain_object = domain
scene.bourrasque.grid_res = 64
scene.bourrasque.ppc_axis = 2

# Emetteur qui coincide EXACTEMENT avec la boite du domaine (meme cube).
bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0, 0, 0))
emitter = bpy.context.active_object
emitter.name = "Emitter"
emitter.bourrasque.role = "EMITTER"
emitter.bourrasque.emit_mode = "BLOCK"
emitter.bourrasque.emit_source = "BOUNDS"
emitter.bourrasque.model = "WATER"

transform = props.domain_transform(scene)
origin, solver_size = transform
grid_res = scene.bourrasque.grid_res
L = 4.0
marge_expected = 3.0 * solver_size / grid_res
solver_size_expected = L * grid_res / (grid_res - 6)

print(f"L (boite artiste)      = {L}")
print(f"solver_size (attendu)  = {solver_size_expected:.6f}")
print(f"solver_size (obtenu)   = {solver_size:.6f}")
print(f"marge (attendue)       = {marge_expected:.6f}")

lo, hi = props.emitter_bounds_solver(emitter, origin, solver_size)
usable_lo, usable_hi = props.domain_usable_bounds(scene)
print(f"emitter bounds solver  = lo={lo}, hi={hi}")
print(f"usable bounds solver   = [{usable_lo:.6f}, {usable_hi:.6f}]")

for a in range(3):
    err_lo = abs(lo[a] - usable_lo)
    err_hi = abs(hi[a] - usable_hi)
    print(f"axe {a}: err_lo={err_lo:.8f} err_hi={err_hi:.8f}")
    assert err_lo < 1e-5, f"axe {a} lo hors tolerance"
    assert err_hi < 1e-5, f"axe {a} hi hors tolerance"

overflow = props.emitter_overflow(emitter, origin, solver_size, grid_res)
print(f"emitter_overflow -> {overflow}  (attendu: None = accepte)")
assert overflow is None, "la validation du bake doit ACCEPTER cet emetteur"

# --- Simulation reelle : verifie que des particules atteignent les parois.
config = lib.default_config()
config.grid_res = grid_res
config.domain = solver_size
config.gravity_y = -9.8
config.cfl = 0.3
config.ppc_axis = 2
config.max_particles = 2_000_000

sim = lib.Sim(config)
mat_id = sim.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
n_emitted = sim.emit_box(mat_id, lo, hi, vel=(0.0, 0.0, 0.0))
print(f"particules emises = {n_emitted}")

dx = solver_size / grid_res
frame_dt = 1.0 / 24.0
n_frames = 60
min_dist_to_wall = None
for f in range(n_frames):
    sim.step(frame_dt)
    xs = sim.read_positions()
    # distance de chaque particule a la paroi de la BOITE DE L'ARTISTE (zone utile)
    dist_lo = xs - usable_lo
    dist_hi = usable_hi - xs
    dist = np.minimum(dist_lo, dist_hi).min()
    if min_dist_to_wall is None or dist < min_dist_to_wall:
        min_dist_to_wall = dist

print(f"dx = {dx:.6f}")
print(f"distance minimale atteinte a une paroi de la boite artiste apres {n_frames} frames : {min_dist_to_wall:.6f} (doit etre < dx={dx:.6f})")
assert min_dist_to_wall < dx, "le fluide n'atteint pas la paroi visible"

print("CORRECTIF B : OK")
