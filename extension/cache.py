"""cache.py — lecture/ecriture du cache de simulation au format `.bqd`.

Aucune dependance a `bpy` : ce module est pur, importable et testable en
dehors de Blender (voir `extension/tests/test_cache.py`, executable avec
`python extension/tests/test_cache.py`).

Format `.bqd` — SPECIFICATION DE REFERENCE cote Python.

Il existe TROIS versions du format, toutes en little-endian EXPLICITE
(`<`, jamais l'ordre natif) :

v1 — format historique, a compte de particules FIXE. Produit par
`core/headless/main.cpp` (C++) et relu par `scripts/view_dump.py`. Cette
extension ne l'ECRIT plus (voir v2 ci-dessous) mais continue de le LIRE :
les dumps `dam.bqd`, `jelly.bqd`, `splash.bqd` a la racine du depot sont des
v1 reels, produits par le binaire C++, et doivent rester lisibles tels quels.

    fichier.bqd (v1) :
        int32  n         nombre de particules, CONSTANT sur toute la sim
        int32  frames    nombre de frames effectivement ecrites
        puis frames * n * 3 float32 : les positions, frame par frame,
                                      chaque frame etant n triplets (x, y, z)

v2 — format courant, ecrit par `CacheWriter`. Le nombre de particules varie
d'une frame a l'autre (emission continue) : chaque frame porte son propre
compte, et une table d'index en fin de fichier permet un acces direct O(1) a
n'importe quelle frame sans relecture sequentielle.

    fichier.bqd (v2) :
        4 octets   "BQD2"      magie
        int32      version     = 2
        int32      frames      nombre de frames (reecrit a la fermeture)
        int32      n_max       nombre de particules a la DERNIERE frame
                                (reecrit a la fermeture)
        int64      index_off   offset absolu de la table d'index (reecrit
                                a la fermeture)
        -- repete `frames` fois, dans l'ordre --
        int32      count       nombre de particules de cette frame
        count*3    float32     positions (x, y, z) par particule
        -- a l'offset index_off --
        frames * int64         offset absolu du DEBUT de chaque frame (celui
                                du champ `count`), dans l'ordre des frames

    Pourquoi la table d'index est en fin de fichier plutot qu'au debut : le
    bake est modal et interruptible par l'artiste, le nombre de frames n'est
    donc pas connu a l'ouverture du fichier. Ecrire la table en fin de
    fichier garde l'ecriture purement sequentielle et streamable (chaque
    frame est ecrite une fois, jamais reecrite), tandis que la lecture d'une
    frame arbitraire au scrub de la timeline reste en O(1) via la table
    d'index (un seek, jamais de relecture sequentielle depuis le debut).

v3 — format courant, ecrit par `CacheWriter`. Identique a v2 (compte de
particules variable par frame, table d'index en fin de fichier) mais avec un
champ de bits `canaux` en en-tete, qui declare quelles donnees additionnelles
accompagnent chaque frame. Un seul canal existe pour l'instant : la vitesse
par particule (`CHANNEL_VELOCITY = 1 << 0`), necessaire au motion blur du
maillage (voir `extension/meshcache.py`). Le canal est desactive par defaut :
un v3 sans vitesse a exactement le meme contenu de frame qu'un v2 equivalent
(seul l'en-tete gagne le champ `canaux`, 4 octets).

    fichier.bqd (v3) :
        4 octets   "BQD3"      magie
        int32      version     = 3
        int32      frames      nombre de frames (reecrit a la fermeture)
        int32      n_max       nombre de particules a la DERNIERE frame
                                (reecrit a la fermeture)
        int32      canaux      champ de bits ; 1<<0 = VITESSE
        int64      index_off   offset absolu de la table d'index (reecrit
                                a la fermeture)
        -- repete `frames` fois, dans l'ordre --
        int32      count       nombre de particules de cette frame
        count*3    float32     positions (x, y, z) par particule
        [si canal VITESSE] count*3 float32   vitesses (vx, vy, vz) par
                                              particule, meme ordre que les
                                              positions
        -- a l'offset index_off --
        frames * int64         offset absolu du DEBUT de chaque frame (celui
                                du champ `count`), dans l'ordre des frames

`CacheReader` lit indifferemment v1, v2 et v3. `CacheWriter` n'ecrit
desormais que du v3 ; le canal vitesse se declare a l'ouverture
(`CacheWriter(path, velocity=True)`) et `append_frame` prend alors un
argument de vitesses obligatoire (erreur explicite sinon, dans les deux
sens : vitesses fournies sans canal declare, ou canal declare sans vitesses
fournies).

Dans toutes les versions :

    fichier.mat (sidecar, meme chemin que le .bqd avec l'extension
                 remplacee par `.mat`, optionnel) :
        n * uint8 : l'id de materiau de chaque particule, POUR LA DERNIERE
                    FRAME ecrite.

    Invariant dont depend ce sidecar : les particules ne sont JAMAIS
    retirees, seulement ajoutees en fin de tableau. La frame `i` (de compte
    `count_i <= n_max`) correspond donc toujours aux `count_i` PREMIERES
    entrees de `.mat`. Si une future evolution du solveur retire des
    particules (mort, fusion...), cette propriete casse et le sidecar `.mat`
    devra porter un id par frame, pas un seul pour toute la sim.

Les positions sont en ESPACE SOLVEUR (pave [0, size_x] x [0, size_y] x
[0, size_z], Y vers le haut).
Ce module ne fait AUCUNE conversion de repere : c'est le role de
`extension/transform.py`.

Choix numpy plutot que `struct`/`array` : le consommateur final cote Blender
est `mesh.vertices.foreach_set("co", buffer)`, qui accepte nativement un
ndarray float32 contigu (ou tout objet exposant le protocole buffer). numpy
donne aussi `tofile` / `fromfile` / `readinto` pour lire/ecrire des frames
sans passer par des tuples Python intermediaires, ce qui compte pour un bake
ou un scrub de timeline a haute frequence (voir docstring de `lib.py`, meme
choix pour la meme raison).

`append_frame` et `write_materials` acceptent aussi bien un ndarray qu'un
tableau ctypes (ex. renvoye par `lib.Sim.read_positions` / `read_materials`) :
les deux exposent le protocole buffer, `np.asarray` les enveloppe donc sans
copie tant que le dtype/la disposition memoire correspondent deja.
"""

import os
import pathlib
import struct

import numpy as np

# --- v1 (lecture seule) ------------------------------------------------
_V1_HEADER_STRUCT = struct.Struct("<ii")  # n (int32), frames (int32)
_V1_HEADER_SIZE = _V1_HEADER_STRUCT.size  # 8

# --- v2 (lecture seule) -------------------------------------------------
_MAGIC_V2 = b"BQD2"
_V2_VERSION = 2
# magic(4s) version(i) frames(i) n_max(i) index_off(q)
_V2_HEADER_STRUCT = struct.Struct("<4siiiq")
_V2_HEADER_SIZE = _V2_HEADER_STRUCT.size  # 24

# --- v3 (lecture + ecriture) ---------------------------------------------
_MAGIC_V3 = b"BQD3"
_V3_VERSION = 3
# magic(4s) version(i) frames(i) n_max(i) canaux(i) index_off(q)
_V3_HEADER_STRUCT = struct.Struct("<4siiiiq")
_V3_HEADER_SIZE = _V3_HEADER_STRUCT.size  # 28

CHANNEL_VELOCITY = 1 << 0

# communs aux formats v2 et v3 (meme disposition de frame et d'index)
_FRAME_COUNT_STRUCT = struct.Struct("<i")  # count (int32) en tete de frame
_INDEX_ENTRY_STRUCT = struct.Struct("<q")  # int64 par entree de la table

_BYTES_PER_PARTICLE = 3 * 4  # 3 float32 par particule et par frame


def cache_paths(cache_dir, name):
    """Construit les chemins `.bqd` et `.mat` pour une simulation `name`.

    Fonction PURE : aucun effet de bord sur le disque (ne cree pas
    `cache_dir`). Appelee sur le chemin chaud du scrub de timeline
    (`display.refresh`, a chaque `frame_change_post`), un effet de bord ici
    creerait un dossier de cache dans n'importe quel .blend ayant
    l'extension active, meme sans le moindre bake. Voir `ensure_cache_dir`
    pour la creation explicite, reservee a l'ecrivain (le bake).

    Renvoie `(bqd_path, mat_path)` sous forme de `pathlib.Path`.
    """
    cache_dir = pathlib.Path(cache_dir)
    bqd_path = cache_dir / f"{name}.bqd"
    mat_path = cache_dir / f"{name}.mat"
    return bqd_path, mat_path


def ensure_cache_dir(cache_dir):
    """Cree `cache_dir` (et ses parents) si necessaire.

    Seul point d'effet de bord sur le systeme de fichiers pour le dossier
    de cache : appele uniquement par l'ecrivain, au moment du bake
    (`ops.py`), jamais depuis le chemin chaud de lecture (`display.py`).
    """
    pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)


def _mat_path_for(bqd_path):
    return pathlib.Path(bqd_path).with_suffix(".mat")


class CacheWriter:
    """Ecrit un fichier `.bqd` frame par frame, au format v3 (compte de
    particules variable par frame, canal vitesse optionnel).

    Le nombre total de frames n'est pas connu a l'ouverture (le bake est
    modal et interruptible par l'artiste) : un header provisoire
    (`frames = 0`, `n_max = 0`, `index_off = 0`) est ecrit a l'ouverture.
    Chaque frame est ensuite ecrite de facon purement sequentielle (jamais
    de reecriture retroactive d'une frame deja ecrite), et son offset de
    debut est retenu en memoire. A la fermeture, la table d'index est
    ecrite a la suite de la derniere frame, puis le header est reecrit
    (`seek(0)`) avec le compte de frames reel, `n_max` (compte de la
    derniere frame) et l'offset reel de la table d'index.

    Supporte le protocole de contexte : `close()` est appele dans
    `__exit__` meme si le bloc `with` leve une exception, pour garantir que
    le fichier reste relisible — avec une table d'index coherente portant
    uniquement sur les frames effectivement ecrites — meme si le bake est
    annule en cours de route.

    `velocity=True` declare le canal vitesse a l'ouverture : `append_frame`
    exige alors un argument de vitesses a chaque appel. Le canal est
    desactive par defaut, et dans ce cas un fichier v3 a un contenu de
    frame identique octet pour octet a un v2 equivalent (seul l'en-tete
    gagne le champ `canaux`).
    """

    def __init__(self, path, n_particles=None, velocity=False):
        # n_particles est conserve pour compatibilite d'appel mais ignore :
        # chaque frame porte desormais son propre compte (voir append_frame).
        del n_particles
        self._path = pathlib.Path(path)
        self._frames_written = 0
        self._last_n = None
        self._frame_offsets = []
        self._closed = False
        self._channels = CHANNEL_VELOCITY if velocity else 0
        self._f = open(self._path, "wb")
        self._f.write(
            _V3_HEADER_STRUCT.pack(
                _MAGIC_V3, _V3_VERSION, 0, 0, self._channels, 0
            )
        )

    @property
    def frames_written(self):
        return self._frames_written

    @property
    def has_velocity(self):
        return bool(self._channels & CHANNEL_VELOCITY)

    def write_materials(self, mat_array):
        """Ecrit le sidecar `.mat` (id materiau par particule).

        Doit etre appele une fois que la derniere frame a ete ecrite : le
        compte attendu est celui de la derniere frame `append_frame`ee (voir
        l'invariant "particules jamais retirees" documente en tete de
        module).
        """
        if self._last_n is None:
            raise ValueError(
                "write_materials: aucune frame ecrite, n_particules inconnu"
            )
        arr = np.asarray(mat_array, dtype=np.uint8)
        if arr.size != self._last_n:
            raise ValueError(
                f"write_materials: {arr.size} materiaux fournis, "
                f"{self._last_n} attendus (particules de la derniere frame)"
            )
        mat_path = _mat_path_for(self._path)
        arr.tofile(str(mat_path))

    def append_frame(self, positions, velocities=None):
        """Ajoute une frame de positions (count*3 float32, ctypes ou numpy).

        `count` peut differer d'un appel a l'autre (emission continue).

        `velocities` (count*3 float32, meme convention) est obligatoire si
        le canal vitesse a ete declare a l'ouverture (`velocity=True`), et
        interdit sinon : passer l'un sans l'autre leve `ValueError` plutot
        que de produire un fichier dont le contenu de frame ne correspond
        plus au champ `canaux` de l'en-tete.
        """
        wants_velocity = bool(self._channels & CHANNEL_VELOCITY)
        if wants_velocity and velocities is None:
            raise ValueError(
                "append_frame: canal vitesse declare a l'ouverture "
                "(velocity=True), mais aucune vitesse fournie"
            )
        if not wants_velocity and velocities is not None:
            raise ValueError(
                "append_frame: des vitesses ont ete fournies mais le canal "
                "vitesse n'a pas ete declare a l'ouverture (velocity=True)"
            )

        arr = np.asarray(positions, dtype=np.float32)
        if arr.size % 3 != 0:
            raise ValueError(
                f"append_frame: buffer de taille {arr.size}, "
                "un multiple de 3 (x, y, z par particule) est attendu"
            )
        count = arr.size // 3

        vel_arr = None
        if wants_velocity:
            vel_arr = np.asarray(velocities, dtype=np.float32)
            if vel_arr.size != count * 3:
                raise ValueError(
                    f"append_frame: {vel_arr.size} valeurs de vitesse, "
                    f"{count * 3} attendues ({count} particules)"
                )

        # Une frame corrompue/tronquee corromprait tout le reste du fichier
        # sans erreur visible : on ecrit uniquement apres validation.
        offset = self._f.tell()
        self._f.write(_FRAME_COUNT_STRUCT.pack(count))
        self._f.write(np.ascontiguousarray(arr).tobytes())
        if wants_velocity:
            self._f.write(np.ascontiguousarray(vel_arr).tobytes())
        self._frame_offsets.append(offset)
        self._last_n = count
        self._frames_written += 1

    def close(self):
        """Ecrit la table d'index puis reecrit le header, avant fermeture."""
        if self._closed:
            return
        index_off = self._f.tell()
        for offset in self._frame_offsets:
            self._f.write(_INDEX_ENTRY_STRUCT.pack(offset))

        self._f.seek(0)
        n_max = self._last_n if self._last_n is not None else 0
        self._f.write(
            _V3_HEADER_STRUCT.pack(
                _MAGIC_V3,
                _V3_VERSION,
                self._frames_written,
                n_max,
                self._channels,
                index_off,
            )
        )
        self._f.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class CacheReader:
    """Lit un fichier `.bqd` frame par frame, sans jamais charger tout le
    fichier en memoire (un bake complet peut peser plusieurs centaines de
    Mo).

    Lit indifferemment les fichiers v1 (compte fixe, produits par le
    headless C++), v2 (compte variable, sans canaux) et v3 (compte
    variable, canal vitesse optionnel, ecrit par `CacheWriter`). La version
    est detectee par les 4 premiers octets du fichier : `b"BQD2"` signale un
    v2, `b"BQD3"` un v3, toute autre valeur est interpretee comme le champ
    `n` (int32) d'un header v1.
    """

    def __init__(self, path):
        self._path = pathlib.Path(path)
        self._f = open(self._path, "rb")
        self._closed = False
        self._channels = 0

        try:
            magic = self._f.read(4)
            if len(magic) < 4:
                raise ValueError(
                    f"cache tronque : {self._path} fait {len(magic)} octet(s), "
                    "un header d'au moins 4 octets est attendu"
                )

            if magic == _MAGIC_V3:
                self._read_header_v3()
            elif magic == _MAGIC_V2:
                self._read_header_v2()
            else:
                self._read_header_v1()
        except Exception:
            self._f.close()
            self._closed = True
            raise

    # -- v1 ---------------------------------------------------------------

    def _read_header_v1(self):
        self._f.seek(0)
        header = self._f.read(_V1_HEADER_SIZE)
        if len(header) < _V1_HEADER_SIZE:
            raise ValueError(
                f"cache tronque : {self._path} fait {len(header)} octets, "
                f"un header de {_V1_HEADER_SIZE} octets est attendu"
            )
        n, frames = _V1_HEADER_STRUCT.unpack(header)
        if n < 0 or frames < 0:
            raise ValueError(
                f"header v1 incoherent : {self._path} n={n} frames={frames}"
            )

        file_size = os.fstat(self._f.fileno()).st_size
        expected_size = _V1_HEADER_SIZE + frames * n * _BYTES_PER_PARTICLE
        if file_size != expected_size:
            raise ValueError(
                f"cache tronque ou incoherent : {self._path} fait "
                f"{file_size} octets, {expected_size} attendus pour "
                f"n={n} particules et frames={frames} (header = 8 + frames*n*12)"
            )

        self._version = 1
        self._n = n
        self._n_max = n
        self._frames = frames

    # -- v2 ---------------------------------------------------------------

    def _read_header_v2(self):
        rest = self._f.read(_V2_HEADER_SIZE - 4)
        if len(rest) < _V2_HEADER_SIZE - 4:
            raise ValueError(
                f"cache tronque : {self._path} n'a pas de header v2 complet "
                f"({_V2_HEADER_SIZE} octets attendus)"
            )
        _magic, version, frames, n_max, index_off = _V2_HEADER_STRUCT.unpack(
            _MAGIC_V2 + rest
        )
        if version != _V2_VERSION:
            raise ValueError(
                f"cache {self._path} : version BQD inconnue ({version}), "
                f"seule la version {_V2_VERSION} est supportee"
            )
        if frames < 0 or n_max < 0 or index_off < _V2_HEADER_SIZE:
            raise ValueError(
                f"header v2 incoherent : {self._path} frames={frames} "
                f"n_max={n_max} index_off={index_off}"
            )

        self._version = 2
        self._channels = 0
        self._parse_frames_and_index(
            header_size=_V2_HEADER_SIZE,
            frames=frames,
            n_max=n_max,
            index_off=index_off,
        )

    # -- v3 ---------------------------------------------------------------

    def _read_header_v3(self):
        rest = self._f.read(_V3_HEADER_SIZE - 4)
        if len(rest) < _V3_HEADER_SIZE - 4:
            raise ValueError(
                f"cache tronque : {self._path} n'a pas de header v3 complet "
                f"({_V3_HEADER_SIZE} octets attendus)"
            )
        _magic, version, frames, n_max, channels, index_off = (
            _V3_HEADER_STRUCT.unpack(_MAGIC_V3 + rest)
        )
        if version != _V3_VERSION:
            raise ValueError(
                f"cache {self._path} : version BQD inconnue ({version}), "
                f"seule la version {_V3_VERSION} est supportee"
            )
        if frames < 0 or n_max < 0 or index_off < _V3_HEADER_SIZE:
            raise ValueError(
                f"header v3 incoherent : {self._path} frames={frames} "
                f"n_max={n_max} index_off={index_off}"
            )
        if channels & ~CHANNEL_VELOCITY:
            raise ValueError(
                f"header v3 incoherent : {self._path} canaux={channels} "
                "contient des bits inconnus"
            )

        self._version = 3
        self._channels = channels
        self._parse_frames_and_index(
            header_size=_V3_HEADER_SIZE,
            frames=frames,
            n_max=n_max,
            index_off=index_off,
        )

    # -- commun v2/v3 -------------------------------------------------------

    def _parse_frames_and_index(self, header_size, frames, n_max, index_off):
        """Lit et valide la table d'index, puis relit le champ `count` de
        chaque frame pour en verifier les bornes. Partage par v2 et v3, qui
        ne different que par la taille du header et la presence eventuelle
        de vitesses a la suite des positions de chaque frame."""
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

        has_velocity = bool(self._channels & CHANNEL_VELOCITY)
        bytes_per_frame_particle = (
            2 * _BYTES_PER_PARTICLE if has_velocity else _BYTES_PER_PARTICLE
        )

        # Bornes de la region "frames" : entre la fin du header et le debut
        # de la table d'index. Chaque offset doit y tomber, et les offsets
        # doivent etre strictement croissants (ecriture sequentielle).
        prev_end = header_size
        counts = []
        for i, off in enumerate(offsets):
            if off < prev_end or off >= index_off:
                raise ValueError(
                    f"table d'index incoherente : {self._path} offset de "
                    f"frame {i} ({off}) hors bornes [{prev_end}, {index_off})"
                )
            self._f.seek(off)
            count_bytes = self._f.read(_FRAME_COUNT_STRUCT.size)
            if len(count_bytes) != _FRAME_COUNT_STRUCT.size:
                raise ValueError(
                    f"cache tronque : {self._path} frame {i} illisible a "
                    f"l'offset {off}"
                )
            (count,) = _FRAME_COUNT_STRUCT.unpack(count_bytes)
            if count < 0:
                raise ValueError(
                    f"frame {i} incoherente : {self._path} count={count}"
                )
            frame_end = off + _FRAME_COUNT_STRUCT.size + count * bytes_per_frame_particle
            if frame_end > index_off:
                raise ValueError(
                    f"frame {i} deborde de la table d'index : {self._path} "
                    f"fin de frame {frame_end} > index_off {index_off}"
                )
            counts.append(count)
            prev_end = frame_end

        if frames > 0 and counts[-1] != n_max:
            raise ValueError(
                f"header incoherent : {self._path} n_max={n_max} mais la "
                f"derniere frame contient {counts[-1]} particules"
            )

        self._frames = frames
        self._n_max = n_max
        self._frame_offsets = offsets
        self._counts = counts

    @property
    def n_particles(self):
        """Compte de particules. Constant pour un v1 ; pour un v2, expose
        le compte de la derniere frame (`n_max`). Voir `particle_count`
        pour le compte d'une frame arbitraire."""
        return self._n_max

    @property
    def frame_count(self):
        return self._frames

    @property
    def is_variable(self):
        """Vrai si le cache est au format v2 ou v3 (compte de particules
        variable d'une frame a l'autre)."""
        return self._version in (2, 3)

    @property
    def has_velocity(self):
        """Vrai si le canal vitesse est present (uniquement possible en
        v3). Constant pour tout le fichier : declare une fois en en-tete,
        pas par frame. Permet a l'appelant de savoir si `read_velocity` est
        utilisable sans avoir a inspecter l'en-tete lui-meme."""
        return bool(self._channels & CHANNEL_VELOCITY)

    def particle_count(self, frame_index):
        """Compte de particules de la frame `frame_index` (0-based).
        Constant (= `n_particles`) pour un v1."""
        if not (0 <= frame_index < self._frames):
            raise IndexError(
                f"particle_count: index {frame_index} hors bornes "
                f"[0, {self._frames})"
            )
        if self._version == 1:
            return self._n
        return self._counts[frame_index]

    def read_frame(self, index, out=None):
        """Lit la frame `index` (0-based).

        Renvoie un ndarray `(count, 3)` float32, `count` etant le compte
        REEL de cette frame (`particle_count(index)`) — pas necessairement
        celui de `out`. `out` n'est reutilise que s'il est assez grand pour
        cette frame (`out.shape[0] >= count` et dtype float32) : la valeur
        renvoyee est alors une VUE `out[:count]`, pas `out` en entier. En
        v1, le compte est constant et `out` est toujours reutilise tel
        quel si sa forme correspond.
        """
        if not (0 <= index < self._frames):
            raise IndexError(
                f"read_frame: index {index} hors bornes [0, {self._frames})"
            )

        if self._version == 1:
            count = self._n
            offset = _V1_HEADER_SIZE + index * self._n * _BYTES_PER_PARTICLE
        else:
            count = self._counts[index]
            offset = self._frame_offsets[index] + _FRAME_COUNT_STRUCT.size

        if (
            out is not None
            and out.dtype == np.float32
            and out.ndim == 2
            and out.shape[1] == 3
            and out.shape[0] >= count
        ):
            dest = out[:count]
        else:
            dest = np.empty((count, 3), dtype=np.float32)

        self._f.seek(offset)
        n_read = self._f.readinto(dest)
        expected = count * _BYTES_PER_PARTICLE
        if n_read != expected:
            raise ValueError(
                f"read_frame: {n_read} octets lus a l'offset {offset}, "
                f"{expected} attendus (fichier tronque ?)"
            )
        return dest

    def read_velocity(self, index, out=None):
        """Lit le canal vitesse de la frame `index` (0-based).

        Meme contrat que `read_frame` (ndarray `(count, 3)` float32,
        reutilisation de `out` sous les memes conditions), mais pour les
        vitesses : leve `ValueError` si le fichier n'a pas de canal vitesse
        (`has_velocity` est False), plutot que de lire des octets qui
        n'existent pas.
        """
        if not self.has_velocity:
            raise ValueError(
                f"read_velocity: {self._path} n'a pas de canal vitesse "
                "(has_velocity est False)"
            )
        if not (0 <= index < self._frames):
            raise IndexError(
                f"read_velocity: index {index} hors bornes [0, {self._frames})"
            )

        count = self._counts[index]
        offset = (
            self._frame_offsets[index]
            + _FRAME_COUNT_STRUCT.size
            + count * _BYTES_PER_PARTICLE
        )

        if (
            out is not None
            and out.dtype == np.float32
            and out.ndim == 2
            and out.shape[1] == 3
            and out.shape[0] >= count
        ):
            dest = out[:count]
        else:
            dest = np.empty((count, 3), dtype=np.float32)

        self._f.seek(offset)
        n_read = self._f.readinto(dest)
        expected = count * _BYTES_PER_PARTICLE
        if n_read != expected:
            raise ValueError(
                f"read_velocity: {n_read} octets lus a l'offset {offset}, "
                f"{expected} attendus (fichier tronque ?)"
            )
        return dest

    def read_materials(self):
        """Lit le sidecar `.mat` s'il existe, `None` sinon."""
        mat_path = _mat_path_for(self._path)
        if not mat_path.exists():
            return None
        return np.fromfile(str(mat_path), dtype=np.uint8)

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
