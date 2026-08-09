"""whitewatercache.py — lecture/ecriture du cache whitewater au format `.bqw`.

Aucune dependance a `bpy` : ce module est pur, importable et testable en
dehors de Blender (voir `extension/tests/test_whitewatercache.py`,
executable avec `python extension/tests/test_whitewatercache.py`), au meme
titre que `extension/meshcache.py` dont il reprend les conventions.

Format `.bqw` — SPECIFICATION DE REFERENCE cote Python (voir
docs/plan-milestone-8.md, decision D7). Meme famille de choix que `.bqd`
(`extension/cache.py`) et `.bqm` (`extension/meshcache.py`), pour les memes
raisons : petit-boutien EXPLICITE (`<`, jamais l'ordre natif), magie de 4
octets puis numero de version, ecriture purement SEQUENTIELLE et
STREAMABLE (le bake whitewater est un vrai solveur qui simule frame par
frame, le nombre de frames est inconnu a l'ouverture, aucune frame deja
ecrite n'est jamais reecrite), table d'index en fin de fichier pour un
acces O(1) a une frame arbitraire au defilement de la timeline.

    fichier.bqw (v1) :
        4 octets  "BQW1"     magie
        int32     version    = 1
        int32     frames                  (reecrit a la fermeture)
        int32     canaux                  champ de bits ; 1<<0 = VITESSE
        int64     index_off               (reecrit a la fermeture)
        --- parametres de production, pour l'invalidation ---
        float32   influence_radius
        float32   spawn_rate
        float32   ta_min, ta_max, ta_weight
        float32   wc_min, wc_max, wc_weight
        float32   ke_min, ke_max, ke_weight
        float32   life_spray, life_foam, life_bubble
        float32   drag_spray, drag_foam
        float32   buoyancy_bubble
        int32     src_frames              nombre de frames du .bqd source
        int32     src_n_max               n_max du .bqd source
        --- repete `frames` fois ---
        int32     n_particles
        n*3 float32   positions
        n   int32     type (0 spray / 1 foam / 2 bubble)
        n   float32   taille
        n   float32   age
        [si canal VITESSE] n*3 float32   vitesses (vx, vy, vz)
        --- a index_off ---
        frames * int64       offset absolu du debut de chaque frame

Les parametres de production sont des VALEURS NOMMEES COMPARABLES UNE A
UNE, pas un hachage : quand le cache est declare caduc, l'interface doit
pouvoir dire a l'artiste quel parametre a change (meme raison que la
decision D1 du plan M7, reprise pour ce format). `diff_params` compare un
jeu de parametres stocke a un jeu de parametres courant et renvoie la
liste des noms de champs qui different (liste vide si le cache est
valide). Les champs flottants sont compares apres passage par
`numpy.float32`, parce que les valeurs stockees ont elles-memes transite
par un `float32` a l'ecriture : comparer un `float64` Python brut au
flottant relu produirait de faux positifs de simple imprecision de
conversion, pas un changement reel de parametre.

Une frame peut n'avoir aucune particule secondaire active (pas encore de
generation, ou tout est mort) : `n_particles = 0` est un cas NORMAL, ecrit
et relu sans erreur (voir
`test_whitewatercache.py::test_empty_frame_roundtrip`) — meme convention
qu'un maillage vide en `.bqm`.

Le canal vitesse est RESERVE (motion blur), comme le canal vitesse par
sommet de `.bqm` (D9 de M7) : non calcule cote GPU pour l'instant, mais
`WhitewaterCacheWriter` sait le serialiser si on le lui fournit.

Comme `meshcache.py`, ce module prefere numpy a `struct`/`array` pour les
gros tableaux : le consommateur final est un nuage de points Blender
(`mesh.vertices.foreach_set` / attributs `foreach_set`), qui accepte
nativement un ndarray contigu.
"""

import dataclasses
import os
import pathlib
import struct

import numpy as np

_MAGIC = b"BQW1"
_VERSION = 1

# magic(4s) version(i) frames(i) canaux(i) index_off(q)
#   influence_radius(f) spawn_rate(f)
#   ta_min(f) ta_max(f) ta_weight(f)
#   wc_min(f) wc_max(f) wc_weight(f)
#   ke_min(f) ke_max(f) ke_weight(f)
#   life_spray(f) life_foam(f) life_bubble(f)
#   drag_spray(f) drag_foam(f) buoyancy_bubble(f)
#   src_frames(i) src_n_max(i)
_HEADER_STRUCT = struct.Struct("<4siiiq" + "f" * 17 + "ii")
_HEADER_SIZE = _HEADER_STRUCT.size

_FRAME_HEADER_STRUCT = struct.Struct("<i")  # n_particles(i32)
_INDEX_ENTRY_STRUCT = struct.Struct("<q")

_BYTES_PER_POSITION = 3 * 4  # 3 float32
_BYTES_PER_TYPE = 4  # 1 int32
_BYTES_PER_SCALAR = 4  # 1 float32 (taille ou age)

CHANNEL_VELOCITY = 1 << 0

_PARAM_FIELDS = (
    "influence_radius",
    "spawn_rate",
    "ta_min",
    "ta_max",
    "ta_weight",
    "wc_min",
    "wc_max",
    "wc_weight",
    "ke_min",
    "ke_max",
    "ke_weight",
    "life_spray",
    "life_foam",
    "life_bubble",
    "drag_spray",
    "drag_foam",
    "buoyancy_bubble",
    "src_frames",
    "src_n_max",
)

_FLOAT_PARAM_FIELDS = frozenset(
    (
        "influence_radius",
        "spawn_rate",
        "ta_min",
        "ta_max",
        "ta_weight",
        "wc_min",
        "wc_max",
        "wc_weight",
        "ke_min",
        "ke_max",
        "ke_weight",
        "life_spray",
        "life_foam",
        "life_bubble",
        "drag_spray",
        "drag_foam",
        "buoyancy_bubble",
    )
)


@dataclasses.dataclass(frozen=True)
class WhitewaterProductionParams:
    """Parametres qui ont produit un `.bqw`, tels que stockes en en-tete.

    Champs NOMMES et comparables un a un (voir `diff_params`), a dessein :
    ce n'est pas un hachage, pour pouvoir dire a l'artiste quel parametre
    precis rend un cache caduc.
    """

    influence_radius: float
    spawn_rate: float
    ta_min: float
    ta_max: float
    ta_weight: float
    wc_min: float
    wc_max: float
    wc_weight: float
    ke_min: float
    ke_max: float
    ke_weight: float
    life_spray: float
    life_foam: float
    life_bubble: float
    drag_spray: float
    drag_foam: float
    buoyancy_bubble: float
    src_frames: int
    src_n_max: int


def diff_params(stored, current):
    """Compare deux `WhitewaterProductionParams` champ a champ.

    Renvoie la liste des noms de champs qui different (liste vide si
    `current` produirait le meme cache que `stored`, cache donc valide).

    Les champs flottants sont compares en `float32` : les valeurs stockees
    dans le fichier ont elles-memes transite par un `float32` a
    l'ecriture, comparer un flottant Python brut produirait de faux
    positifs de simple imprecision de conversion.
    """
    diffs = []
    for name in _PARAM_FIELDS:
        sv = getattr(stored, name)
        cv = getattr(current, name)
        if name in _FLOAT_PARAM_FIELDS:
            if np.float32(sv) != np.float32(cv):
                diffs.append(name)
        else:
            if int(sv) != int(cv):
                diffs.append(name)
    return diffs


def _pack_header(frames, channels, index_off, params):
    return _HEADER_STRUCT.pack(
        _MAGIC,
        _VERSION,
        frames,
        channels,
        index_off,
        float(params.influence_radius),
        float(params.spawn_rate),
        float(params.ta_min),
        float(params.ta_max),
        float(params.ta_weight),
        float(params.wc_min),
        float(params.wc_max),
        float(params.wc_weight),
        float(params.ke_min),
        float(params.ke_max),
        float(params.ke_weight),
        float(params.life_spray),
        float(params.life_foam),
        float(params.life_bubble),
        float(params.drag_spray),
        float(params.drag_foam),
        float(params.buoyancy_bubble),
        int(params.src_frames),
        int(params.src_n_max),
    )


def _unpack_header(header_bytes, path):
    (
        magic,
        version,
        frames,
        channels,
        index_off,
        influence_radius,
        spawn_rate,
        ta_min,
        ta_max,
        ta_weight,
        wc_min,
        wc_max,
        wc_weight,
        ke_min,
        ke_max,
        ke_weight,
        life_spray,
        life_foam,
        life_bubble,
        drag_spray,
        drag_foam,
        buoyancy_bubble,
        src_frames,
        src_n_max,
    ) = _HEADER_STRUCT.unpack(header_bytes)

    if magic != _MAGIC:
        raise ValueError(
            f"magie invalide : {path} n'est pas un fichier .bqw "
            f"(magie {magic!r}, {_MAGIC!r} attendue)"
        )
    if version != _VERSION:
        raise ValueError(
            f"cache {path} : version BQW inconnue ({version}), "
            f"seule la version {_VERSION} est supportee"
        )
    if frames < 0 or index_off < _HEADER_SIZE:
        raise ValueError(
            f"header .bqw incoherent : {path} frames={frames} "
            f"index_off={index_off}"
        )
    if channels & ~CHANNEL_VELOCITY:
        raise ValueError(
            f"header .bqw incoherent : {path} canaux={channels} "
            "contient des bits inconnus"
        )

    params = WhitewaterProductionParams(
        influence_radius=influence_radius,
        spawn_rate=spawn_rate,
        ta_min=ta_min,
        ta_max=ta_max,
        ta_weight=ta_weight,
        wc_min=wc_min,
        wc_max=wc_max,
        wc_weight=wc_weight,
        ke_min=ke_min,
        ke_max=ke_max,
        ke_weight=ke_weight,
        life_spray=life_spray,
        life_foam=life_foam,
        life_bubble=life_bubble,
        drag_spray=drag_spray,
        drag_foam=drag_foam,
        buoyancy_bubble=buoyancy_bubble,
        src_frames=src_frames,
        src_n_max=src_n_max,
    )
    return frames, channels, index_off, params


def read_params(path):
    """Lit uniquement les parametres de production d'un `.bqw`, sans
    charger la moindre particule secondaire (aucune position, type, taille,
    age ni table d'index n'est lue). Usage typique : verifier la validite
    du cache au moment d'ouvrir le panneau, avant de decider de charger une
    frame.
    """
    path = pathlib.Path(path)
    with open(path, "rb") as f:
        header_bytes = f.read(_HEADER_SIZE)
    if len(header_bytes) < _HEADER_SIZE:
        raise ValueError(
            f"cache tronque : {path} fait {len(header_bytes)} octets, "
            f"un header de {_HEADER_SIZE} octets est attendu"
        )
    _frames, _channels, _index_off, params = _unpack_header(header_bytes, path)
    return params


class WhitewaterCacheWriter:
    """Ecrit un fichier `.bqw` frame par frame.

    Meme discipline que `meshcache.MeshCacheWriter` : header provisoire
    (`frames = 0`, `index_off = 0`) ecrit a l'ouverture, chaque frame
    ecrite une fois de facon purement sequentielle, offset de debut retenu
    en memoire, table d'index et header final ecrits a la fermeture.
    Supporte le protocole de contexte : `close()` est appele dans
    `__exit__` meme sur exception, pour que le fichier reste relisible avec
    une table d'index coherente meme si le bake whitewater est annule en
    cours de route.

    `params` (un `WhitewaterProductionParams`) est fige a l'ouverture : ce
    sont les valeurs ecrites en en-tete pour l'invalidation du cache, elles
    ne changent pas en cours de bake.

    `velocity=True` declare le canal vitesse : `append_frame` exige alors
    un argument de vitesses a chaque appel.
    """

    def __init__(self, path, params, velocity=False):
        self._path = pathlib.Path(path)
        self._params = params
        self._channels = CHANNEL_VELOCITY if velocity else 0
        self._frame_offsets = []
        self._frames_written = 0
        self._closed = False
        self._f = open(self._path, "wb")
        self._f.write(_pack_header(0, self._channels, 0, self._params))

    @property
    def frames_written(self):
        return self._frames_written

    @property
    def has_velocity(self):
        return bool(self._channels & CHANNEL_VELOCITY)

    def append_frame(self, pos, type, size, age, velocity=None):
        """Ajoute une frame de particules secondaires.

        `pos` : `n*3` float32 (positions, ctypes ou numpy).
        `type` : `n` int32 (0 spray / 1 foam / 2 bubble).
        `size`/`age` : `n` float32.
        `n_particles = 0` est une valeur valide (aucune particule
        secondaire active sur cette frame, cas normal).

        `velocity` (`n*3` float32) est obligatoire si le canal vitesse a
        ete declare a l'ouverture (`velocity=True`), et interdit sinon :
        passer l'un sans l'autre leve `ValueError` plutot que de produire
        un fichier dont le contenu de frame ne correspond plus au champ
        `canaux` de l'en-tete.
        """
        wants_velocity = self.has_velocity
        if wants_velocity and velocity is None:
            raise ValueError(
                "append_frame: canal vitesse declare a l'ouverture "
                "(velocity=True), mais aucune vitesse fournie"
            )
        if not wants_velocity and velocity is not None:
            raise ValueError(
                "append_frame: des vitesses ont ete fournies mais le canal "
                "vitesse n'a pas ete declare a l'ouverture (velocity=True)"
            )

        pos_arr = np.asarray(pos, dtype=np.float32)
        if pos_arr.size % 3 != 0:
            raise ValueError(
                f"append_frame: buffer de positions de taille {pos_arr.size}, "
                "un multiple de 3 (x, y, z par particule) est attendu"
            )
        n = pos_arr.size // 3

        type_arr = np.asarray(type, dtype=np.int32)
        if type_arr.size != n:
            raise ValueError(
                f"append_frame: {type_arr.size} valeurs de type, "
                f"{n} attendues ({n} particules)"
            )

        size_arr = np.asarray(size, dtype=np.float32)
        if size_arr.size != n:
            raise ValueError(
                f"append_frame: {size_arr.size} valeurs de taille, "
                f"{n} attendues ({n} particules)"
            )

        age_arr = np.asarray(age, dtype=np.float32)
        if age_arr.size != n:
            raise ValueError(
                f"append_frame: {age_arr.size} valeurs d'age, "
                f"{n} attendues ({n} particules)"
            )

        vel_arr = None
        if wants_velocity:
            vel_arr = np.asarray(velocity, dtype=np.float32)
            if vel_arr.size != n * 3:
                raise ValueError(
                    f"append_frame: {vel_arr.size} valeurs de vitesse, "
                    f"{n * 3} attendues ({n} particules)"
                )

        # Une frame corrompue/tronquee corromprait tout le reste du fichier
        # sans erreur visible : on ecrit uniquement apres validation.
        offset = self._f.tell()
        self._f.write(_FRAME_HEADER_STRUCT.pack(n))
        self._f.write(np.ascontiguousarray(pos_arr).tobytes())
        self._f.write(np.ascontiguousarray(type_arr).tobytes())
        self._f.write(np.ascontiguousarray(size_arr).tobytes())
        self._f.write(np.ascontiguousarray(age_arr).tobytes())
        if wants_velocity:
            self._f.write(np.ascontiguousarray(vel_arr).tobytes())
        self._frame_offsets.append(offset)
        self._frames_written += 1

    def close(self):
        """Ecrit la table d'index puis reecrit le header, avant fermeture."""
        if self._closed:
            return
        index_off = self._f.tell()
        for offset in self._frame_offsets:
            self._f.write(_INDEX_ENTRY_STRUCT.pack(offset))

        self._f.seek(0)
        self._f.write(
            _pack_header(self._frames_written, self._channels, index_off, self._params)
        )
        self._f.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class WhitewaterCacheReader:
    """Lit un fichier `.bqw` frame par frame, sans jamais charger tout le
    fichier en memoire.

    A l'ouverture, lit et valide le header, les parametres de production,
    et la table d'index (avec un aller-retour leger sur chaque en-tete de
    frame pour verifier ses bornes) — pas les particules secondaires
    elles-memes, qui ne sont chargees que par `read_frame`, a la demande.
    """

    def __init__(self, path):
        self._path = pathlib.Path(path)
        self._f = open(self._path, "rb")
        self._closed = False
        try:
            self._read_header_and_index()
        except Exception:
            self._f.close()
            self._closed = True
            raise

    def _read_header_and_index(self):
        header_bytes = self._f.read(_HEADER_SIZE)
        if len(header_bytes) < _HEADER_SIZE:
            raise ValueError(
                f"cache tronque : {self._path} fait {len(header_bytes)} "
                f"octets, un header de {_HEADER_SIZE} octets est attendu"
            )
        frames, channels, index_off, params = _unpack_header(
            header_bytes, self._path
        )

        file_size = os.fstat(self._f.fileno()).st_size
        index_size = frames * _INDEX_ENTRY_STRUCT.size
        if index_off > file_size or index_off + index_size > file_size:
            raise ValueError(
                f"index_off incoherent : {self._path} fait {file_size} "
                f"octets, la table d'index a l'offset {index_off} "
                f"({index_size} octets, {frames} frames) deborde du fichier"
            )

        self._f.seek(index_off)
        index_bytes = self._f.read(index_size)
        if len(index_bytes) != index_size:
            raise ValueError(
                f"cache tronque : {self._path} table d'index incomplete "
                f"a l'offset {index_off}"
            )
        offsets = list(
            struct.unpack(f"<{frames}q", index_bytes) if frames else ()
        )

        has_velocity = bool(channels & CHANNEL_VELOCITY)

        # Bornes de la region "frames" : entre la fin du header et le
        # debut de la table d'index. Chaque offset doit y tomber, et les
        # offsets doivent etre strictement croissants (ecriture
        # sequentielle). n_particles = 0 (frame vide) est valide.
        prev_end = _HEADER_SIZE
        n_particles_list = []
        for i, off in enumerate(offsets):
            if off < prev_end or off >= index_off:
                raise ValueError(
                    f"table d'index incoherente : {self._path} offset de "
                    f"frame {i} ({off}) hors bornes [{prev_end}, {index_off})"
                )
            self._f.seek(off)
            fh_bytes = self._f.read(_FRAME_HEADER_STRUCT.size)
            if len(fh_bytes) != _FRAME_HEADER_STRUCT.size:
                raise ValueError(
                    f"cache tronque : {self._path} frame {i} illisible a "
                    f"l'offset {off}"
                )
            (n_particles,) = _FRAME_HEADER_STRUCT.unpack(fh_bytes)
            if n_particles < 0:
                raise ValueError(
                    f"frame {i} incoherente : {self._path} "
                    f"n_particles={n_particles}"
                )
            frame_end = (
                off
                + _FRAME_HEADER_STRUCT.size
                + n_particles * _BYTES_PER_POSITION
                + n_particles * _BYTES_PER_TYPE
                + n_particles * _BYTES_PER_SCALAR  # taille
                + n_particles * _BYTES_PER_SCALAR  # age
                + (n_particles * _BYTES_PER_POSITION if has_velocity else 0)
            )
            if frame_end > index_off:
                raise ValueError(
                    f"frame {i} deborde de la table d'index : {self._path} "
                    f"fin de frame {frame_end} > index_off {index_off}"
                )
            n_particles_list.append(n_particles)
            prev_end = frame_end

        self._frames = frames
        self._channels = channels
        self._params = params
        self._frame_offsets = offsets
        self._n_particles = n_particles_list

    @property
    def params(self):
        return self._params

    @property
    def frame_count(self):
        return self._frames

    @property
    def has_velocity(self):
        return bool(self._channels & CHANNEL_VELOCITY)

    def _check_index(self, index, caller):
        if not (0 <= index < self._frames):
            raise IndexError(
                f"{caller}: index {index} hors bornes [0, {self._frames})"
            )

    def particle_count(self, index):
        self._check_index(index, "particle_count")
        return self._n_particles[index]

    def read_frame(self, index):
        """Lit la frame `index` (0-based) en un seul seek.

        Renvoie `(pos, type, size, age)` : `pos` un ndarray `(n, 3)`
        float32, `type` un ndarray `(n,)` int32, `size`/`age` des ndarray
        `(n,)` float32, plus la vitesse (`(n, 3)` float32) en cinquieme
        valeur si `has_velocity` est vrai. `n_particles = 0` (frame vide)
        est un cas normal : les tableaux correspondants sont alors vides,
        pas `None`.
        """
        self._check_index(index, "read_frame")
        n = self._n_particles[index]
        offset = self._frame_offsets[index] + _FRAME_HEADER_STRUCT.size

        self._f.seek(offset)

        pos = np.empty((n, 3), dtype=np.float32)
        if n:
            n_read = self._f.readinto(pos)
            expected = n * _BYTES_PER_POSITION
            if n_read != expected:
                raise ValueError(
                    f"read_frame: {n_read} octets de positions lus a la "
                    f"frame {index}, {expected} attendus (fichier tronque ?)"
                )

        type_ = np.empty((n,), dtype=np.int32)
        if n:
            n_read = self._f.readinto(type_)
            expected = n * _BYTES_PER_TYPE
            if n_read != expected:
                raise ValueError(
                    f"read_frame: {n_read} octets de type lus a la frame "
                    f"{index}, {expected} attendus (fichier tronque ?)"
                )

        size = np.empty((n,), dtype=np.float32)
        if n:
            n_read = self._f.readinto(size)
            expected = n * _BYTES_PER_SCALAR
            if n_read != expected:
                raise ValueError(
                    f"read_frame: {n_read} octets de taille lus a la frame "
                    f"{index}, {expected} attendus (fichier tronque ?)"
                )

        age = np.empty((n,), dtype=np.float32)
        if n:
            n_read = self._f.readinto(age)
            expected = n * _BYTES_PER_SCALAR
            if n_read != expected:
                raise ValueError(
                    f"read_frame: {n_read} octets d'age lus a la frame "
                    f"{index}, {expected} attendus (fichier tronque ?)"
                )

        if not self.has_velocity:
            return pos, type_, size, age

        vel = np.empty((n, 3), dtype=np.float32)
        if n:
            n_read = self._f.readinto(vel)
            expected = n * _BYTES_PER_POSITION
            if n_read != expected:
                raise ValueError(
                    f"read_frame: {n_read} octets de vitesses lus a la "
                    f"frame {index}, {expected} attendus (fichier tronque ?)"
                )
        return pos, type_, size, age, vel

    def close(self):
        if self._closed:
            return
        self._f.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
