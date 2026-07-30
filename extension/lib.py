"""lib.py — binding ctypes de l'API C plate de Bourrasque.

C'est le SEUL module de l'extension qui parle directement a la DLL native
(bourrasque.dll / bourrasque.so). Tout le reste de l'extension (operateurs,
panneaux bpy) doit passer par la classe `Sim` definie ici plutot que par des
appels ctypes bruts.

Aucun import de `bpy` : ce module doit rester testable hors de Blender avec
`python lib.py`.

Choix numpy vs ctypes pur : on utilise numpy pour les buffers de sortie
(`read_positions` / `read_materials`). Numpy est disponible dans l'interpreteur
embarque par Blender depuis longtemps, et un `numpy.ndarray` float32 contigu
s'ecrit directement dans un fichier binaire (`ndarray.tofile` / `.tobytes()`)
et se passe tel quel a `bpy.types.Attribute.foreach_set` / `foreach_set` sur un
mesh, qui acceptent nativement des buffers numpy. Ca evite aussi une
reallocation Python par frame pendant un bake : on peut reutiliser le meme
ndarray via le parametre `out`.

Le chargement de la DLL est PARESSEUX (`load()` appele au premier usage) :
si on chargeait au moment de l'import, l'extension entiere echouerait a
s'enregistrer dans Blender des que la DLL est absente (build pas encore
lance) ou incompatible (runtime CUDA introuvable depuis le processus hote).
"""

import ctypes
import os

import numpy as np

# ---------------------------------------------------------------------------
# Erreurs
# ---------------------------------------------------------------------------


class BourrasqueError(Exception):
    """Erreur remontee par le solveur natif ou par le chargement de la DLL."""


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

BQ_MODEL_ELASTIC = 0
BQ_MODEL_WATER = 1
BQ_MAX_MATERIALS = 8

# Doit rester synchronise avec la macro BQ_ABI_VERSION de core/include/bourrasque.h.
BQ_ABI_VERSION = 4

_DLL_NAME = "bourrasque.dll" if os.name == "nt" else "libbourrasque.so"
_DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", _DLL_NAME)


# ---------------------------------------------------------------------------
# Structures ctypes — ordre et types EXACTEMENT ceux de core/include/bourrasque.h
# ---------------------------------------------------------------------------


class BqConfig(ctypes.Structure):
    _fields_ = [
        ("grid_res", ctypes.c_int * 3),
        ("cell_size", ctypes.c_float),
        ("gravity_y", ctypes.c_float),
        ("cfl", ctypes.c_float),
        ("ppc_axis", ctypes.c_int),
        ("max_particles", ctypes.c_int),
    ]


class BqMaterial(ctypes.Structure):
    _fields_ = [
        ("model", ctypes.c_int),
        ("rho", ctypes.c_float),
        ("E", ctypes.c_float),
        ("nu", ctypes.c_float),
        ("bulk", ctypes.c_float),
        ("gamma", ctypes.c_float),
    ]


class BqSim(ctypes.Structure):
    """Handle opaque cote C (struct BqSim). On ne dereference jamais son
    contenu depuis Python ; on ne manipule que des pointeurs dessus."""


BqSimPtr = ctypes.POINTER(BqSim)

# ---------------------------------------------------------------------------
# Chargement paresseux de la DLL
# ---------------------------------------------------------------------------

_dll = None


def load():
    """Charge la DLL native et declare les prototypes de toutes les fonctions.

    Idempotent : les appels suivants renvoient l'instance deja chargee.
    Leve BourrasqueError avec un message actionnable pour un artiste si le
    chargement echoue.
    """
    global _dll
    if _dll is not None:
        return _dll

    if not os.path.isfile(_DLL_PATH):
        raise BourrasqueError(
            "Le moteur de simulation Bourrasque est introuvable "
            f"({_DLL_PATH}).\n"
            "Il faut d'abord compiler le solveur (lancer le build CMake du "
            "projet) avant d'utiliser l'extension."
        )

    try:
        dll = ctypes.CDLL(_DLL_PATH)
    except OSError as exc:
        raise BourrasqueError(
            "Le moteur de simulation Bourrasque n'a pas pu etre charge "
            f"({_DLL_PATH}).\n"
            "C'est probablement du a un pilote NVIDIA ou un runtime CUDA "
            "manquant ou incompatible sur cette machine : verifiez que la "
            "carte graphique NVIDIA et ses pilotes sont a jour.\n"
            f"Detail technique : {exc}"
        )

    _check_abi_compat(dll)

    _declare_prototypes(dll)
    _dll = dll
    return _dll


def _check_abi_compat(dll):
    """Verifie que la DLL chargee est compatible avec ce code Python AVANT
    toute declaration de prototype ou tout appel.

    Ce garde-fou existe parce que ctypes garde une bibliotheque native
    chargee dans le processus une fois `CDLL()` appele : reinstaller
    l'extension (ou meme la desactiver/reactiver dans Blender) ne decharge
    PAS l'ancienne DLL tant que le processus Blender tourne encore. Sans ce
    controle, une DLL perimee est relue avec la mauvaise disposition de
    struct (ex. `BqConfig` passee de 24 a 32 octets entre deux versions), ce
    qui produit des valeurs aberrantes (ex. `max_particles` lu a l'offset de
    `cfl`) et des diagnostics completement trompeurs plus loin dans la chaine
    (ex. "memoire insuffisante" sur un `cudaMalloc` de 99 Gio).
    """
    try:
        dll.bq_abi_version.restype = ctypes.c_int
        dll.bq_abi_version.argtypes = []
        native_abi = dll.bq_abi_version()
    except AttributeError:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est perimee : elle "
            "ne fournit meme pas la fonction de verification de version "
            "d'ABI (bq_abi_version), introduite depuis.\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque.\n"
            f"Version d'ABI attendue par cette extension : {BQ_ABI_VERSION}."
        )

    if native_abi != BQ_ABI_VERSION:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est incompatible "
            "avec cette version de l'extension "
            f"(version d'ABI native = {native_abi}, "
            f"version d'ABI attendue = {BQ_ABI_VERSION}).\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque."
        )

    try:
        dll.bq_config_size.restype = ctypes.c_int
        dll.bq_config_size.argtypes = []
        native_config_size = dll.bq_config_size()
    except AttributeError:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est perimee : elle "
            "ne fournit meme pas la fonction de verification de taille de "
            "configuration (bq_config_size), introduite depuis.\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque."
        )

    expected_config_size = ctypes.sizeof(BqConfig)
    if native_config_size != expected_config_size:
        raise BourrasqueError(
            "La bibliotheque native Bourrasque chargee est incompatible "
            "avec cette version de l'extension : la structure de "
            "configuration (BqConfig) n'a pas la meme taille cote natif et "
            f"cote Python (taille native = {native_config_size} octets, "
            f"taille attendue = {expected_config_size} octets).\n"
            "Reinstallez l'extension Bourrasque, PUIS REDEMARREZ BLENDER : "
            "une simple desactivation/reactivation de l'extension ne suffit "
            "pas, car la bibliotheque native reste chargee en memoire dans "
            "le processus Blender tant qu'il tourne, meme apres avoir "
            "remplace le fichier sur le disque.\n"
            "Sans redemarrage, la configuration serait relue avec de "
            "mauvais decalages memoire, ce qui produit des valeurs "
            "aberrantes (ex. un nombre de particules delirant) et des "
            "diagnostics trompeurs (ex. 'memoire insuffisante')."
        )


def _declare_prototypes(dll):
    dll.bq_default_config.argtypes = [ctypes.POINTER(BqConfig)]
    dll.bq_default_config.restype = None

    dll.bq_create.argtypes = [ctypes.POINTER(BqConfig)]
    dll.bq_create.restype = BqSimPtr

    dll.bq_destroy.argtypes = [BqSimPtr]
    dll.bq_destroy.restype = None

    dll.bq_add_material.argtypes = [BqSimPtr, ctypes.POINTER(BqMaterial)]
    dll.bq_add_material.restype = ctypes.c_int

    dll.bq_emit_box.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.bq_emit_box.restype = ctypes.c_int

    dll.bq_emit_points.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    dll.bq_emit_points.restype = ctypes.c_int

    dll.bq_emit_points_vel.argtypes = [
        BqSimPtr,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    dll.bq_emit_points_vel.restype = ctypes.c_int

    dll.bq_set_colliders.argtypes = [
        BqSimPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
    ]
    dll.bq_set_colliders.restype = ctypes.c_int

    dll.bq_step.argtypes = [BqSimPtr, ctypes.c_float]
    dll.bq_step.restype = ctypes.c_int

    dll.bq_particle_count.argtypes = [BqSimPtr]
    dll.bq_particle_count.restype = ctypes.c_int

    dll.bq_read_positions.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_read_positions.restype = ctypes.c_int

    dll.bq_read_materials.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_uint8)]
    dll.bq_read_materials.restype = ctypes.c_int

    dll.bq_read_sdf.argtypes = [BqSimPtr, ctypes.POINTER(ctypes.c_float)]
    dll.bq_read_sdf.restype = ctypes.c_int

    dll.bq_last_error.argtypes = []
    dll.bq_last_error.restype = ctypes.c_char_p


def _last_error(dll):
    msg = dll.bq_last_error()
    if msg is None:
        return "(pas de detail disponible)"
    return msg.decode("utf-8", errors="replace")


def default_config():
    """Retourne un BqConfig rempli par bq_default_config."""
    dll = load()
    cfg = BqConfig()
    dll.bq_default_config(ctypes.byref(cfg))
    return cfg


def _vec3(seq):
    arr = (ctypes.c_float * 3)(*seq)
    return arr


# ---------------------------------------------------------------------------
# API pythonique
# ---------------------------------------------------------------------------


class Sim:
    """Encapsule un BqSim* et expose une API pythonique au-dessus de l'API C.

    Supporte le protocole de contexte : `with Sim(cfg) as sim: ...` garantit
    la liberation de la memoire GPU meme si une exception survient pendant
    le bake.
    """

    def __init__(self, config):
        self._dll = load()
        # Memorise la resolution de grille (espace hote, pas cote coeur) :
        # necessaire pour dimensionner le buffer de `read_sdf`, seule
        # methode dont la taille du resultat ne se lit pas via
        # `particle_count`.
        self._grid_res = (
            int(config.grid_res[0]),
            int(config.grid_res[1]),
            int(config.grid_res[2]),
        )
        self._handle = self._dll.bq_create(ctypes.byref(config))
        if not self._handle:
            raise BourrasqueError(
                "Impossible de creer la simulation Bourrasque : "
                f"{_last_error(self._dll)}"
            )

    def _check(self, code):
        if code < 0:
            raise BourrasqueError(_last_error(self._dll))
        return code

    def add_material(self, model, rho, E=0.0, nu=0.0, bulk=0.0, gamma=0.0):
        mat = BqMaterial(model=model, rho=rho, E=E, nu=nu, bulk=bulk, gamma=gamma)
        return self._check(self._dll.bq_add_material(self._handle, ctypes.byref(mat)))

    def emit_box(self, mat_id, lo, hi, vel=(0.0, 0.0, 0.0)):
        return self._check(
            self._dll.bq_emit_box(
                self._handle, mat_id, _vec3(lo), _vec3(hi), _vec3(vel)
            )
        )

    def emit_points(self, mat_id, positions, vel=(0.0, 0.0, 0.0)):
        """Emet des particules a des positions explicites (espace solveur).

        `positions` doit etre convertible en ndarray `(n, 3)`. On force
        `np.ascontiguousarray(..., dtype=np.float32)` avant de prendre le
        pointeur : un tableau non contigu (ex. une vue transposee ou un
        slice avec un pas) ou en float64 (le dtype par defaut de numpy)
        passerait silencieusement des octets errones a `bq_emit_points`,
        qui attend un buffer `float32` C-contigu de `count*3` elements.
        """
        arr = np.ascontiguousarray(positions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"emit_points: positions doit etre de forme (n, 3), "
                f"recu {arr.shape}"
            )
        count = arr.shape[0]
        if count == 0:
            return 0
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._check(
            self._dll.bq_emit_points(
                self._handle, mat_id, ptr, count, _vec3(vel)
            )
        )

    def emit_points_vel(self, mat_id, positions, velocities):
        """Emet des particules a des positions explicites, chacune avec sa
        propre vitesse (espace solveur).

        `positions` et `velocities` doivent etre convertibles en ndarray
        `(n, 3)`, meme n pour les deux. Meme justification que `emit_points`
        pour le `np.ascontiguousarray(..., dtype=np.float32)` applique aux
        DEUX tableaux : un tableau non contigu (ex. une vue transposee ou un
        slice avec un pas) ou en float64 (le dtype par defaut de numpy)
        passerait silencieusement des octets errones a `bq_emit_points_vel`,
        qui attend deux buffers `float32` C-contigus de `count*3` elements
        chacun.
        """
        pos_arr = np.ascontiguousarray(positions, dtype=np.float32)
        if pos_arr.ndim != 2 or pos_arr.shape[1] != 3:
            raise ValueError(
                f"emit_points_vel: positions doit etre de forme (n, 3), "
                f"recu {pos_arr.shape}"
            )
        vel_arr = np.ascontiguousarray(velocities, dtype=np.float32)
        if vel_arr.ndim != 2 or vel_arr.shape[1] != 3:
            raise ValueError(
                f"emit_points_vel: velocities doit etre de forme (n, 3), "
                f"recu {vel_arr.shape}"
            )
        if pos_arr.shape[0] != vel_arr.shape[0]:
            raise ValueError(
                "emit_points_vel: positions et velocities doivent avoir le "
                f"meme nombre de lignes (recu {pos_arr.shape[0]} et "
                f"{vel_arr.shape[0]})"
            )
        count = pos_arr.shape[0]
        if count == 0:
            return 0
        pos_ptr = pos_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vel_ptr = vel_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._check(
            self._dll.bq_emit_points_vel(
                self._handle, mat_id, pos_ptr, vel_ptr, count
            )
        )

    def set_colliders(self, triangles, velocities, frictions):
        """Remplace l'ensemble des colliders du solveur (espace solveur).

        `triangles` et `velocities` doivent etre convertibles en ndarray
        `(n_tri, 3, 3)` (triangle, sommet, xyz) de MEME forme : `velocities`
        est la vitesse par SOMMET, pas par triangle. `frictions` doit etre
        convertible en ndarray `(n_tri,)`, un coefficient par triangle.
        Meme discipline `np.ascontiguousarray(..., dtype=np.float32)` que
        `emit_points_vel` sur les trois tableaux : un tableau non contigu ou
        en float64 (le dtype par defaut de numpy) passerait silencieusement
        des octets errones a `bq_set_colliders`, qui attend des buffers
        `float32` C-contigus.

        `n_tri == 0` est un appel VALIDE (pas court-circuite cote Python,
        contrairement a `emit_points`/`emit_points_vel`) : c'est la
        convention du coeur pour effacer les colliders (voir
        `bourrasque.h`), a appeler explicitement des qu'un bake n'a plus de
        collider a transmettre pour la frame courante.
        """
        tri_arr = np.ascontiguousarray(triangles, dtype=np.float32)
        if tri_arr.ndim != 3 or tri_arr.shape[1:] != (3, 3):
            raise ValueError(
                f"set_colliders: triangles doit etre de forme (n, 3, 3), "
                f"recu {tri_arr.shape}"
            )
        vel_arr = np.ascontiguousarray(velocities, dtype=np.float32)
        if vel_arr.shape != tri_arr.shape:
            raise ValueError(
                "set_colliders: velocities doit avoir la meme forme que "
                f"triangles ({tri_arr.shape}), recu {vel_arr.shape}"
            )
        fric_arr = np.ascontiguousarray(frictions, dtype=np.float32)
        if fric_arr.ndim != 1 or fric_arr.shape[0] != tri_arr.shape[0]:
            raise ValueError(
                f"set_colliders: frictions doit etre de forme (n,) avec "
                f"n == {tri_arr.shape[0]} (nombre de triangles), recu "
                f"{fric_arr.shape}"
            )

        n_tri = tri_arr.shape[0]
        tri_ptr = tri_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        vel_ptr = vel_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        fric_ptr = fric_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        return self._check(
            self._dll.bq_set_colliders(
                self._handle, tri_ptr, vel_ptr, fric_ptr, n_tri
            )
        )

    def step(self, frame_dt):
        return self._check(self._dll.bq_step(self._handle, frame_dt))

    @property
    def particle_count(self):
        return self._check(self._dll.bq_particle_count(self._handle))

    def read_positions(self, out=None):
        """Renvoie un ndarray float32 de forme (n, 3) rempli in-place si
        `out` est fourni (reutilisation entre frames pendant un bake)."""
        n = self.particle_count
        if out is None or out.shape != (n, 3) or out.dtype != np.float32:
            out = np.empty((n, 3), dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._check(self._dll.bq_read_positions(self._handle, ptr))
        return out

    def read_materials(self, out=None):
        """Renvoie un ndarray uint8 de forme (n,) rempli in-place si `out`
        est fourni."""
        n = self.particle_count
        if out is None or out.shape != (n,) or out.dtype != np.uint8:
            out = np.empty((n,), dtype=np.uint8)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        self._check(self._dll.bq_read_materials(self._handle, ptr))
        return out

    def read_sdf(self, out=None):
        """Renvoie le champ de distance signee courant, ndarray float32 de
        forme `(grid_res[0], grid_res[1], grid_res[2])` (espace SOLVEUR),
        rempli in-place si `out` est fourni. Echantillonne AUX NOEUDS de la
        grille (`i*dx` sur chaque axe, meme convention que `k_grid_update`),
        pas au centre des cellules.

        Diagnostic et validation (voir `bq_read_sdf`, `bourrasque.h`) :
        permet de verifier depuis Python qu'un collider produit bien un
        champ coherent (signe negatif a l'interieur, distance nulle sur la
        surface), sans avoir a en deduire l'etat indirectement via les
        positions de particules.
        """
        shape = self._grid_res
        if out is None or out.shape != shape or out.dtype != np.float32:
            out = np.empty(shape, dtype=np.float32)
        ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._check(self._dll.bq_read_sdf(self._handle, ptr))
        return out

    def destroy(self):
        """Libere la memoire GPU. Idempotent."""
        if getattr(self, "_handle", None):
            self._dll.bq_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
        return False

    def __del__(self):
        # Filet de securite : ne pas s'appuyer dessus pour du code sensible,
        # mais evite une fuite GPU silencieuse si destroy() n'a pas ete
        # appele explicitement.
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Verification autonome : `python lib.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = default_config()
    print(
        "config par defaut :",
        "grid_res=(%d,%d,%d) cell_size=%.5f gravity_y=%.3f cfl=%.3f ppc_axis=%d max_particles=%d"
        % (
            cfg.grid_res[0],
            cfg.grid_res[1],
            cfg.grid_res[2],
            cfg.cell_size,
            cfg.gravity_y,
            cfg.cfl,
            cfg.ppc_axis,
            cfg.max_particles,
        ),
    )

    with Sim(cfg) as sim:
        mat_id = sim.add_material(BQ_MODEL_WATER, rho=1000.0, bulk=4e4, gamma=3.0)
        print("materiau eau cree, id =", mat_id)

        n_emit = sim.emit_box(
            mat_id,
            lo=(0.10, 0.10, 0.10),
            hi=(0.35, 0.60, 0.90),
        )
        print("particules emises :", n_emit)
        print("particle_count :", sim.particle_count)

        for i in range(3):
            substeps = sim.step(1.0 / 24.0)
            print("step %d -> %d substeps" % (i, substeps))

        pos = sim.read_positions()
        print("positions shape :", pos.shape, pos.dtype)
        print("min :", pos.min(axis=0))
        print("max :", pos.max(axis=0))

    print("OK — simulation detruite proprement.")
