"""Validation du champ de distance signee de k_collider_sdf contre la verite
analytique d'une sphere. Verifie :
  - la precision de la distance dans la bande proche de la surface
  - l'exactitude du signe partout, avec localisation des defauts eventuels
Fait varier le nombre de triangles pour trancher le doute sur la boucle
d'anneaux (bug suppose vs faux positif).
"""
import ctypes

import numpy as np

DLL = r"C:\Users\nicol\Code\bourrasque_v2\build\Release\bourrasque.dll"
d = ctypes.CDLL(DLL)

F = ctypes.POINTER(ctypes.c_float)


class Cfg(ctypes.Structure):
    _fields_ = [("grid_res", ctypes.c_int * 3), ("cell_size", ctypes.c_float),
                ("gravity_y", ctypes.c_float), ("cfl", ctypes.c_float),
                ("ppc_axis", ctypes.c_int), ("max_particles", ctypes.c_int)]


class Mat(ctypes.Structure):
    _fields_ = [("model", ctypes.c_int), ("rho", ctypes.c_float),
                ("E", ctypes.c_float), ("nu", ctypes.c_float),
                ("bulk", ctypes.c_float), ("gamma", ctypes.c_float)]


d.bq_abi_version.restype = ctypes.c_int
d.bq_default_config.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.argtypes = [ctypes.POINTER(Cfg)]
d.bq_create.restype = ctypes.c_void_p
d.bq_destroy.argtypes = [ctypes.c_void_p]
d.bq_add_material.argtypes = [ctypes.c_void_p, ctypes.POINTER(Mat)]
d.bq_add_material.restype = ctypes.c_int
d.bq_set_colliders.argtypes = [ctypes.c_void_p, F, F, F, ctypes.c_int]
d.bq_set_colliders.restype = ctypes.c_int
d.bq_read_sdf.argtypes = [ctypes.c_void_p, F]
d.bq_read_sdf.restype = ctypes.c_int
d.bq_last_error.restype = ctypes.c_char_p

print("ABI version :", d.bq_abi_version())


def sphere(n_sub, centre=(0.5, 0.5, 0.5), r=0.2):
    """Sphere UV triangulee, ~2*n_sub^2 triangles."""
    u = np.linspace(0, np.pi, n_sub + 1)
    v = np.linspace(0, 2 * np.pi, n_sub + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    P = np.stack([centre[0] + r * np.sin(uu) * np.cos(vv),
                  centre[1] + r * np.sin(uu) * np.sin(vv),
                  centre[2] + r * np.cos(uu)], -1)
    tris = []
    for i in range(n_sub):
        for j in range(n_sub):
            a, b, c, e = P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]
            tris.append([a, b, c])
            tris.append([a, c, e])
    return np.array(tris, dtype=np.float32)


CENTRE = np.array([0.5, 0.5, 0.5], dtype=np.float64)
RADIUS = 0.2


def run(n_sub):
    cfg = Cfg()
    d.bq_default_config(ctypes.byref(cfg))
    sim = d.bq_create(ctypes.byref(cfg))
    if not sim:
        raise SystemExit("bq_create a echoue")

    # Indispensable : bq_set_colliders lit s->prm (res, dx), renseigne
    # seulement par upload_params, declenche au premier bq_add_material.
    mat = Mat(model=1, rho=1000.0, E=0.0, nu=0.0, bulk=4.0e4, gamma=3.0)
    if d.bq_add_material(sim, ctypes.byref(mat)) < 0:
        raise SystemExit(d.bq_last_error().decode())

    tri = sphere(n_sub, centre=tuple(CENTRE), r=RADIUS)
    n_tri = tri.shape[0]
    vel = np.zeros_like(tri)
    fric = np.full(n_tri, 0.2, dtype=np.float32)
    rc = d.bq_set_colliders(sim, tri.ctypes.data_as(F), vel.ctypes.data_as(F),
                            fric.ctypes.data_as(F), n_tri)
    if rc != 0:
        raise SystemExit(d.bq_last_error().decode())

    rx, ry, rz = cfg.grid_res[0], cfg.grid_res[1], cfg.grid_res[2]
    dx = cfg.cell_size
    ncell = rx * ry * rz
    buf = (ctypes.c_float * ncell)()
    got = d.bq_read_sdf(sim, buf)
    if got != ncell:
        raise SystemExit(f"bq_read_sdf a renvoye {got}, attendu {ncell} ({d.bq_last_error().decode()})")
    phi = np.frombuffer(buf, dtype=np.float32, count=ncell).reshape(rx, ry, rz).astype(np.float64)

    # centres de cellules, meme convention que le kernel : (i+0.5)*dx
    ii, jj, kk = np.meshgrid(np.arange(rx), np.arange(ry), np.arange(rz), indexing="ij")
    px = (ii + 0.5) * dx
    py = (jj + 0.5) * dx
    pz = (kk + 0.5) * dx
    dist_centre = np.sqrt((px - CENTRE[0]) ** 2 + (py - CENTRE[1]) ** 2 + (pz - CENTRE[2]) ** 2)
    phi_exact = dist_centre - RADIUS

    d.bq_destroy(sim)

    return dict(n_tri=n_tri, rx=rx, ry=ry, rz=rz, dx=dx,
                phi=phi, phi_exact=phi_exact, ii=ii, jj=jj, kk=kk,
                px=px, py=py, pz=pz)


def analyse(res):
    n_tri = res["n_tri"]
    dx = res["dx"]
    phi = res["phi"]
    phi_exact = res["phi_exact"]

    print(f"\n=== n_tri = {n_tri} ===")

    # bande proche de la surface : |phi_exact| < 3*dx
    band = np.abs(phi_exact) < 3 * dx
    err = np.abs(phi[band] - phi_exact[band])
    print(f"bande proche (|phi_exact| < 3*dx = {3*dx:.5f}) : {band.sum()} cellules")
    print(f"  ecart max  : {err.max():.6f}  ({err.max()/dx:.3f} dx)")
    print(f"  ecart moyen: {err.mean():.6f}  ({err.mean()/dx:.3f} dx)")

    # signe
    sign_calc = np.sign(phi)
    sign_exact = np.sign(phi_exact)
    # ignorer les cellules hors bande active (phi==1e6, forcement exterieur,
    # sign +1, coherent avec phi_exact toujours positif loin du centre -> ok)
    mismatch = sign_calc != sign_exact
    n_mismatch = mismatch.sum()
    print(f"cellules a signe faux : {n_mismatch} / {phi.size}")

    if n_mismatch > 0:
        idx = np.argwhere(mismatch)
        rx, ry, rz = res["rx"], res["ry"], res["rz"]
        print("  localisation (i,j,k, phi_calc, phi_exact, dist au bord AABB en dx):")
        # AABB des triangles (dilatee par pad=3*dx dans le kernel, mais on
        # rapporte la position brute pour classer bord domaine / bord AABB / surface)
        for (i, j, k) in idx[:30]:
            pc = phi[i, j, k]
            pe = phi_exact[i, j, k]
            near_domain_edge = (i == 0 or i == rx - 1 or j == 0 or j == ry - 1
                                 or k == 0 or k == rz - 1)
            near_surface = abs(pe) < 3 * dx
            tag = []
            if near_surface:
                tag.append("pres surface")
            if near_domain_edge:
                tag.append("bord domaine")
            if not tag:
                tag.append("loin de tout")
            print(f"    ({i:3d},{j:3d},{k:3d})  calc={pc:+.5f}  exact={pe:+.5f}  [{'/'.join(tag)}]")
        if len(idx) > 30:
            print(f"    ... et {len(idx) - 30} de plus")

    return dict(n_tri=n_tri, band_err_max=err.max(), band_err_mean=err.mean(),
                n_mismatch=n_mismatch)


results = []
for n_sub in (50, 158):
    res = run(n_sub)
    results.append(analyse(res))

print("\n=== resume ===")
for r in results:
    print(r)
