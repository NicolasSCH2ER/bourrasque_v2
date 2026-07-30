"""Mesure de densite (volume emetteur + jet aval) avec/sans plafond de
conservation du nombre, a turbulence=0. Purement diagnostique, ne modifie
rien dans le depot (le bloc de plafonnement est revert/reapplique dans une
COPIE temporaire du fichier de production, jamais le fichier livre lui-meme
au moment de la mesure finale)."""
import sys
sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2")
from extension import ops, props, register, unregister
from extension import lib
from extension.props import domain_transform, domain_usable_bounds, emitter_bounds_solver, world_to_solver_dir
import bpy, numpy as np

exec(open(r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad\validate_turbulence.py").read().split("def main()")[0])


def orthonormal_basis(axis_dir):
    # construit e1, e2 perpendiculaires a axis_dir
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis_dir, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis_dir, e1)
    return e1, e2


def measure(n_frames=90, warmup=30, seed=1, slice_offset=0.5, slice_half_thick=None, label=""):
    from extension import sampling
    sc = make_scene(frame_end=n_frames + 5)
    sph = setup_sphere_emitter(turbulence=0.0, seed=seed)
    cfg, origin, size = default_config(sc)
    sim = lib.Sim(cfg)
    mat_id = sim.add_material(model=lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    dx = size / sc.bourrasque.grid_res
    spacing = dx / sc.bourrasque.ppc_axis
    if slice_half_thick is None:
        slice_half_thick = spacing
    op = sph.bourrasque
    vel = world_to_solver_dir(tuple(op.initial_velocity))
    sites = sampling.sample_mesh_interior(sph, origin, size, sc.bourrasque.grid_res, sc.bourrasque.ppc_axis)
    frame_dt = sc.render.fps_base / sc.render.fps
    bbox_lo, bbox_hi = sites.min(axis=0), sites.max(axis=0)
    noise = ops._CurlNoise(op.turbulence_seed, 0, bbox_lo, bbox_hi, dx, vel, spacing, frame_dt)
    state = ops._InflowState(sph.name, mat_id, vel, spacing, sites, dx=dx,
                              turbulence=op.turbulence, turbulence_seed=op.turbulence_seed,
                              emitter_index=0, noise=noise)
    usable = props.domain_usable_bounds(sc)
    harness = _Harness(sim, [state], usable, frame_dt)

    n_sites = sites.shape[0]
    lo = sites.min(axis=0) - spacing / 2.0
    hi = sites.max(axis=0) + spacing / 2.0
    center = sites.mean(axis=0).astype(np.float64)
    vel_arr = np.array(vel, dtype=np.float64)
    speed = np.linalg.norm(vel_arr)
    axis_dir = vel_arr / speed
    e1, e2 = orthonormal_basis(axis_dir)

    # footprint transverse (perpendiculaire a l'axe d'ecoulement) du nuage
    # de sites : sert de gabarit FIXE (identique dans les deux regimes,
    # puisque `sites` ne change jamais) pour definir le volume de la
    # tranche jet en aval.
    rel_sites = sites.astype(np.float64) - center
    half_e1 = float(np.max(np.abs(rel_sites @ e1)))
    half_e2 = float(np.max(np.abs(rel_sites @ e2)))
    slab_volume = (2 * half_e1) * (2 * half_e2) * (2 * slice_half_thick)
    rest_density = 1.0 / (spacing ** 3)

    slice_center = center + slice_offset * axis_dir

    density_bbox = []
    density_jet_ratio = []
    front_s = []
    transverse_std = []
    for f in range(n_frames):
        harness._frame_index = f
        harness._emit_inflow_sites()
        sim.step(frame_dt)
        if f >= warmup:
            positions = sim.read_positions().astype(np.float64)
            in_bbox = np.all((positions >= lo) & (positions <= hi), axis=1)
            n_occ = int(in_bbox.sum())
            density_bbox.append(n_occ / n_sites)

            s = (positions - center) @ axis_dir
            front_s.append(float(s.max()))

            rel = positions - slice_center
            s_local = rel @ axis_dir
            e1_local = rel @ e1
            e2_local = rel @ e2
            slab_mask = (
                (np.abs(s_local) <= slice_half_thick)
                & (np.abs(e1_local) <= half_e1)
                & (np.abs(e2_local) <= half_e2)
            )
            n_slab = int(slab_mask.sum())
            num_density = n_slab / slab_volume
            density_jet_ratio.append(num_density / rest_density)

            # dispersion transverse du jet (tout le nuage, pas seulement le
            # gabarit) dans une bande fine autour de slice_center, pour un
            # rayon "a mi-hauteur" independant du gabarit fixe.
            thin_mask = np.abs(s_local) <= slice_half_thick
            if thin_mask.sum() > 0:
                r = np.linalg.norm(
                    np.stack([e1_local[thin_mask], e2_local[thin_mask]], axis=1), axis=1
                )
                transverse_std.append(float(np.std(r)))

    positions_final = sim.read_positions()
    n_final = positions_final.shape[0]
    sim.destroy()

    out = {
        "label": label,
        "n_sites": n_sites,
        "spacing": spacing,
        "density_bbox_mean": float(np.mean(density_bbox)),
        "density_bbox_min": float(np.min(density_bbox)),
        "density_bbox_max": float(np.max(density_bbox)),
        "slab_volume": slab_volume,
        "half_e1": half_e1,
        "half_e2": half_e2,
        "slice_half_thick": slice_half_thick,
        "density_jet_ratio_mean": float(np.mean(density_jet_ratio)) if density_jet_ratio else None,
        "front_s_mean": float(np.mean(front_s)) if front_s else None,
        "front_s_final": front_s[-1] if front_s else None,
        "transverse_std_mean": float(np.mean(transverse_std)) if transverse_std else None,
        "n_final": n_final,
    }
    print(f"--- {label} ---")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


register()
try:
    r = measure(label="ETAT ACTUEL DU DEPOT (a determiner: avec ou sans plafond selon ops.py courant)")
finally:
    unregister()
