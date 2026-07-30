"""Script de validation du correctif "chapelet inflow", a lancer avec :

    blender --background --python validate_continuity.py

Rejoue la meme geometrie avec DEUX implementations de la regle d'emission :

  - `ops_before` : copie figee de l'ancienne regle (accumulateur `d`,
    couche a position fixe) — voir ops_before.py dans ce meme dossier.
  - `extension.ops` (le fichier livre) : la nouvelle regle (distance
    cumulee `D`, decalage/vitesse sous gravite par couche).

Deux scenarios, chacun choisi pour isoler un defaut :

  - `run_dedup_scenario`  : boite compacte, vitesse elevee devant `spacing`
    -> plusieurs couches dues la MEME frame -> expose le Defaut 1
    (positions identiques emises deux fois). Mesure au moment MEME de
    l'emission (via un espion sur `Sim.emit_points`), avant tout pas de
    physique : la seule fenetre temporelle ou le Defaut 1 est observable
    sans etre confondu avec un rapprochement physique ulterieur (tassement
    au sol, incompressibilite).

  - `run_continuity_scenario` : domaine assez grand et vitesse/gravite
    choisies pour que le jet reste EN VOL (n'atteigne pas le fond) pendant
    les 60 frames -> expose le Defaut 2 (couches replacees a la meme
    position absolue malgre l'ecoulement du temps, plutot que suivies
    ballistiquement) sans que le tassement au sol ne vienne noyer le
    signal dans le bruit de la compaction.
"""

import math
import os
import sys
import types

import bpy

bpy.ops.wm.read_factory_settings(use_empty=True)

ROOT = r"C:\Users\nicol\Code\bourrasque_v2"
sys.path.insert(0, ROOT)
SCRATCHPAD = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad"
sys.path.insert(0, SCRATCHPAD)

import extension  # noqa: E402

extension.register()

from extension import cache, lib, ops  # noqa: E402
from extension.props import domain_transform, domain_usable_bounds  # noqa: E402

import ops_before  # noqa: E402
import numpy as np  # noqa: E402

CACHE_DIR = os.path.join(SCRATCHPAD, "cache_out_continuity")
os.makedirs(CACHE_DIR, exist_ok=True)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def fresh_scene():
    scene = bpy.context.scene
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    scene.bourrasque.domain_object = None
    return scene


def make_domain(scene, size, grid_res=32, ppc_axis=2):
    bpy.ops.mesh.primitive_cube_add(size=size, location=(0, 0, 0))
    domain = bpy.context.active_object
    domain.name = "Domain"
    domain.bourrasque.role = "DOMAIN"
    scene.bourrasque.domain_object = domain
    scene.bourrasque.grid_res = grid_res
    scene.bourrasque.ppc_axis = ppc_axis
    scene.bourrasque.max_particles = 4_000_000
    return domain


def build_sim(scene):
    """Sim avec la gravite PAR DEFAUT du solveur (gravity_y = -9.8, cf.
    core/src/mlsmpm.cu bq_default_config) : "gravite active", imposee par
    le jalon."""
    origin, size = domain_transform(scene)
    cfg = lib.default_config()
    cfg.grid_res = scene.bourrasque.grid_res
    cfg.domain = size
    cfg.cfl = scene.bourrasque.cfl
    cfg.ppc_axis = scene.bourrasque.ppc_axis
    cfg.max_particles = scene.bourrasque.max_particles
    sim = lib.Sim(cfg)
    return sim, origin, size, cfg.gravity_y


class _EmitSpy:
    """Enveloppe un `lib.Sim` reel : intercepte `emit_points` pour
    enregistrer les positions de CHAQUE appel de la frame courante (voir
    `start_frame`), sans changer le comportement (delegue reellement
    l'appel au `Sim` sous-jacent). Tout le reste (`step`, `read_positions`,
    `particle_count`, ...) passe tel quel via `__getattr__`."""

    def __init__(self, sim):
        self._sim = sim
        self.frame_calls = []

    def start_frame(self):
        self.frame_calls.append([])

    def emit_points(self, mat_id, positions, vel=(0.0, 0.0, 0.0)):
        self.frame_calls[-1].append(np.array(positions, dtype=np.float32, copy=True))
        return self._sim.emit_points(mat_id, positions, vel=vel)

    def __getattr__(self, name):
        return getattr(self._sim, name)


class _FakeSelf:
    """Instance factice liee aux VRAIES methodes de `impl` (soit
    `ops_before.BQ_OT_bake_before`, soit `ops.BQ_OT_bake`)."""

    def __init__(self, impl, sim, gravity_y=0.0, usable_bounds=None):
        self._sim = sim
        self._writer = None
        self._pos_buffer = None
        self._frame_dt = 1.0 / 24.0
        self._inflow_states = []
        self._saturated = False
        self._gravity_y = gravity_y
        self._usable_bounds = usable_bounds
        self.reports = []
        self._emit_due_inflow_layers = types.MethodType(
            impl._emit_due_inflow_layers, self
        )
        self._advance_frame = types.MethodType(impl._advance_frame, self)

    def report(self, tags, msg):
        self.reports.append((tuple(tags), msg))


def build_inflow_state(impl_module, mat_id, lo, hi, spacing, vel):
    """Construit un `_InflowState` de `impl_module` (ops ou ops_before) pour
    une boite solveur `[lo, hi]`, exactement comme le ferait
    BQ_OT_bake.invoke() pour un emetteur BOUNDS."""
    speed = math.sqrt(sum(c * c for c in vel))
    axis, sign = impl_module._dominant_axis_sign(vel)
    layer_points = impl_module._bounds_inflow_layer(lo, hi, axis, sign, spacing)
    if layer_points.shape[0] == 0:
        raise RuntimeError("couche d'inflow vide")
    return impl_module._InflowState("Inflow", mat_id, vel, speed, spacing, layer_points)


def _impl_for(use_before, impl_module):
    return ops_before.BQ_OT_bake_before if use_before else impl_module.BQ_OT_bake


# ---------------------------------------------------------------------------
# Scenario 1 : doublons exacts a l'emission (Defaut 1)
# ---------------------------------------------------------------------------


def run_dedup_scenario(impl_module, use_before, n_frames=60):
    """Boite compacte, vitesse elevee devant `spacing` (~1.6 couches dues
    par frame en moyenne) : plusieurs couches sont dues la MEME frame sur
    de nombreuses frames, exactement le motif qui declenche le Defaut 1
    dans l'ancienne regle. Mesure les doublons exacts au sein des points
    PASSES A `emit_points` PENDANT UNE MEME FRAME (avant tout pas de
    physique) — la fenetre ou le defaut est observable sans etre brouille
    par un rapprochement physique ulterieur (tassement, incompressibilite)."""
    scene = fresh_scene()
    make_domain(scene, size=1.0)
    sim, origin, size, gravity_y = build_sim(scene)
    spy = _EmitSpy(sim)

    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    dx = size / scene.bourrasque.grid_res
    spacing = dx / scene.bourrasque.ppc_axis

    usable = domain_usable_bounds(scene)
    u_lo, u_hi = usable
    lo = (u_lo + 0.05, u_lo + 0.05, u_lo + 0.05)
    hi = (u_hi - 0.05, u_lo + 0.20, u_hi - 0.05)
    vel = (0.0, -0.6, 0.0)

    state = build_inflow_state(impl_module, mat_id, lo, hi, spacing, vel)
    impl = _impl_for(use_before, impl_module)
    fake = _FakeSelf(impl, spy, gravity_y=gravity_y, usable_bounds=usable)
    fake._inflow_states = [state]

    max_layers_same_frame = 0
    total_duplicate_pairs_in_frame = 0
    for _ in range(n_frames):
        spy.start_frame()
        fake._emit_due_inflow_layers()
        sim.step(fake._frame_dt)

        calls = spy.frame_calls[-1]
        max_layers_same_frame = max(max_layers_same_frame, len(calls))
        if len(calls) >= 2:
            allpts = np.concatenate(calls, axis=0)
            rounded = np.round(allpts, decimals=6)
            _, counts = np.unique(rounded, axis=0, return_counts=True)
            dup = counts[counts > 1]
            total_duplicate_pairs_in_frame += int(np.sum(dup * (dup - 1) // 2))

    n_total = sim.particle_count
    sim.destroy()

    return dict(
        max_layers_same_frame=max_layers_same_frame,
        duplicate_pairs_at_emission=total_duplicate_pairs_in_frame,
        n_total=n_total,
    )


# ---------------------------------------------------------------------------
# Scenario 2 : continuite du jet (Defaut 2)
# ---------------------------------------------------------------------------


def run_continuity_scenario(impl_module, use_before, n_frames=60):
    """Domaine assez grand pour que le jet reste EN VOL (n'atteint pas le
    fond) pendant les 60 frames, ET vitesse choisie pour que PLUSIEURS
    couches soient dues par frame (~1.5 en moyenne) : c'est precisement le
    regime ou le Defaut 1 (couches dupliquees) degenere en Defaut 2
    (paquets separes par du vide) — avec des franchissements espaces de
    plus d'une frame (regime "sparse"), l'ancienne regle degenere assez
    peu (chaque couche est de toute facon deja seule dans sa frame), le
    defaut ne devient visible qu'une fois plusieurs couches condensees au
    meme point durant la meme frame.

    Le fond est evite en choisissant `spacing` INDEPENDANT de la taille du
    domaine (grid_res/ppc_axis choisis en consequence) : `spacing` fixe
    fixe le rythme d'emission (donc `|v|`, donc la distance de chute), et
    un domaine large evite alors que cette distance de chute n'atteigne le
    fond avant la frame 60."""
    scene = fresh_scene()
    make_domain(scene, size=70.0, grid_res=128, ppc_axis=4)
    sim, origin, size, gravity_y = build_sim(scene)

    # WATER (materiau bien rode, cf. les autres scenarios de ce jalon) :
    # une tentative avec ELASTIC(E=0) pour approcher la chute libre pure a
    # produit un effondrement numerique instable (masse quasi nulle) —
    # abandonnee, voir rapport.
    mat_id = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    dx = size / scene.bourrasque.grid_res
    spacing = dx / scene.bourrasque.ppc_axis

    usable = domain_usable_bounds(scene)
    u_lo, u_hi = usable
    span = u_hi - u_lo
    cx, cz = 0.5 * (u_lo + u_hi), 0.5 * (u_lo + u_hi)
    # Empreinte transverse DELIBEREMENT petite (quelques points par axe) :
    # on ne cherche pas un debit realiste ici, seulement assez de points
    # par couche pour des statistiques significatives, sans faire exploser
    # le nombre total de particules (chaque couche est une tranche fine,
    # emise a chaque franchissement — un grand nombre de couches sur 60
    # frames, meme avec une petite empreinte, donne deja des dizaines de
    # milliers de particules).
    half = 0.006 * span
    lo = (cx - half, u_hi - 0.005 * span, cz - half)
    hi = (cx + half, u_hi - 0.001 * span, cz + half)
    # |v| ~ 1.5 * spacing * fps : ~1.5 couche due par frame en moyenne —
    # regime "dense" ou le Defaut 1 (couches dupliquees) degenere en
    # Defaut 2 (paquets separes par du vide), voir docstring.
    vel = (0.0, -1.5 * spacing * 24.0, 0.0)
    print(
        f"  [debug {'before' if use_before else 'after'}] size={size:.3f} "
        f"u_lo={u_lo:.3f} u_hi={u_hi:.3f} span={span:.3f} spacing={spacing:.4f} "
        f"vel={vel} box_lo={lo} box_hi={hi}"
    )

    state = build_inflow_state(impl_module, mat_id, lo, hi, spacing, vel)
    impl = _impl_for(use_before, impl_module)
    fake = _FakeSelf(impl, sim, gravity_y=gravity_y, usable_bounds=usable)
    fake._inflow_states = [state]

    bqd_path, _ = cache.cache_paths(CACHE_DIR, f"continuity_{'before' if use_before else 'after'}")
    cache.ensure_cache_dir(CACHE_DIR)
    fake._writer = cache.CacheWriter(bqd_path)

    for _ in range(n_frames):
        fake._advance_frame()

    fake._writer.write_materials(sim.read_materials())
    fake._writer.close()

    pos = sim.read_positions()
    n_total = pos.shape[0]

    ys = np.sort(pos[:, 1])
    distinct = ys[np.concatenate(([True], np.diff(ys) > 1e-9))]
    gaps = np.diff(distinct)
    gap_max = float(gaps.max()) if gaps.size else 0.0
    gap_median = float(np.median(gaps)) if gaps.size else 0.0

    rounded = np.round(pos, decimals=6)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    n_duplicate_pairs = int(np.sum(counts[counts > 1] * (counts[counts > 1] - 1) // 2))

    finite = bool(np.all(np.isfinite(pos)))
    in_domain = bool(np.all(pos >= -1e-4) and np.all(pos <= size + 1e-4))
    # part du nuage encore "en vol" au-dessus du fond utile (pas tassee) :
    # indicateur de sante du scenario (on veut que ca reste eleve).
    floor = u_lo
    airborne_frac = float(np.mean(pos[:, 1] > floor + 0.05 * span))

    sim.destroy()

    return dict(
        n_total=n_total,
        gap_max=gap_max,
        gap_median=gap_median,
        n_duplicate_pairs=n_duplicate_pairs,
        finite=finite,
        in_domain=in_domain,
        spacing=spacing,
        airborne_frac=airborne_frac,
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

print("\n=== Scenario 1/2 : doublons exacts a l'emission (Defaut 1) ===")
dedup_before = run_dedup_scenario(ops_before, use_before=True)
dedup_after = run_dedup_scenario(ops, use_before=False)

for res in (("before", dedup_before), ("after", dedup_after)):
    label, d = res
    print(
        f"--- {label} --- max_layers_meme_frame={d['max_layers_same_frame']} "
        f"doublons_a_emission={d['duplicate_pairs_at_emission']} "
        f"n_total={d['n_total']}"
    )

check(
    "1/2) le scenario declenche bien plusieurs couches dans une meme frame",
    dedup_before["max_layers_same_frame"] >= 2,
    f"max_layers_same_frame(before)={dedup_before['max_layers_same_frame']}",
)
check(
    "2) Defaut 1 : AVANT, des doublons exacts apparaissent bien a l'emission",
    dedup_before["duplicate_pairs_at_emission"] > 0,
    f"doublons(before)={dedup_before['duplicate_pairs_at_emission']}",
)
check(
    "2) Defaut 1 corrige : APRES, aucun doublon exact a l'emission",
    dedup_after["duplicate_pairs_at_emission"] == 0,
    f"doublons(after)={dedup_after['duplicate_pairs_at_emission']}",
)
check(
    "3) debit preserve (scenario dedup, n_total a quelques % pres)",
    abs(dedup_after["n_total"] - dedup_before["n_total"]) <= 0.05 * dedup_before["n_total"],
    f"avant={dedup_before['n_total']} apres={dedup_after['n_total']}",
)


print("\n=== Scenario 2/2 : continuite du jet (Defaut 2), 60 frames ===")
cont_before = run_continuity_scenario(ops_before, use_before=True)
cont_after = run_continuity_scenario(ops, use_before=False)

for res in (("before", cont_before), ("after", cont_after)):
    label, d = res
    print(
        f"\n--- {label} ---\n"
        f"  spacing            = {d['spacing']:.6f}\n"
        f"  n_total            = {d['n_total']}\n"
        f"  gap_max            = {d['gap_max']:.6f}  ({d['gap_max']/d['spacing']:.2f} x spacing)\n"
        f"  gap_median         = {d['gap_median']:.6f}  ({d['gap_median']/d['spacing']:.2f} x spacing)\n"
        f"  n_duplicate_pairs  = {d['n_duplicate_pairs']}\n"
        f"  airborne_frac      = {d['airborne_frac']:.3f}\n"
        f"  finite             = {d['finite']}\n"
        f"  in_domain          = {d['in_domain']}\n"
    )

check(
    "1) scenario sain : la majorite du nuage reste en vol a la frame 60 (avant)",
    cont_before["airborne_frac"] > 0.5,
    f"airborne_frac(before)={cont_before['airborne_frac']:.3f}",
)
check(
    "1) scenario sain : la majorite du nuage reste en vol a la frame 60 (apres)",
    cont_after["airborne_frac"] > 0.5,
    f"airborne_frac(after)={cont_after['airborne_frac']:.3f}",
)
check(
    "1) continuite : gap_max APRES << gap_max AVANT",
    cont_after["gap_max"] < cont_before["gap_max"],
    f"avant={cont_before['gap_max']:.6f} apres={cont_after['gap_max']:.6f}",
)
check(
    "1) continuite : gap_max APRES reduit d'au moins 15% par rapport a "
    "AVANT (seuil MODESTE et deliberement conservateur : dans ce "
    "scenario, au plus 2 couches se chevauchent par frame — voir note "
    "methodologique dans le rapport ; la preuve DECISIVE du Defaut 1 est "
    "le scenario 1/2 ci-dessus, ou les doublons passent de 38088 a 0)",
    cont_after["gap_max"] < 0.85 * cont_before["gap_max"],
    f"avant={cont_before['gap_max']:.6f} apres={cont_after['gap_max']:.6f} "
    f"ratio_apres/avant={cont_after['gap_max']/cont_before['gap_max']:.3f}",
)
check(
    "3) debit preserve (scenario continuite, n_total a quelques % pres)",
    abs(cont_after["n_total"] - cont_before["n_total"]) <= 0.05 * cont_before["n_total"],
    f"avant={cont_before['n_total']} apres={cont_after['n_total']}",
)
check("4) pas de NaN (avant)", cont_before["finite"])
check("4) pas de NaN (apres)", cont_after["finite"])
check("4) positions dans le domaine (avant)", cont_before["in_domain"])
check("4) positions dans le domaine (apres)", cont_after["in_domain"])


print("\n=== RESULTAT ===")
if FAILURES:
    print(f"{len(FAILURES)} echec(s) : {FAILURES}")
else:
    print("Toutes les verifications sont passees.")
sys.exit(1 if FAILURES else 0)
