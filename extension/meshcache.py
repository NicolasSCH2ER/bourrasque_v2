"""meshcache.py — lecture/ecriture du cache de maillage au format `.bqm`.

Aucune dependance a `bpy` : ce module est pur, importable et testable en
dehors de Blender (voir `extension/tests/test_meshcache.py`, executable avec
`python extension/tests/test_meshcache.py`), au meme titre que
`extension/cache.py` dont il reprend les conventions.

Format `.bqm` — SPECIFICATION DE REFERENCE cote Python. Meme famille de
choix que `.bqd` (voir `extension/cache.py`), pour les memes raisons :
petit-boutien EXPLICITE (`<`, jamais l'ordre natif), magie de 4 octets puis
numero de version, ecriture purement SEQUENTIELLE et STREAMABLE (le bake de
maillage est modal et interruptible comme celui des particules, le nombre
de frames est inconnu a l'ouverture, aucune frame deja ecrite n'est jamais
reecrite), table d'index en fin de fichier pour un acces O(1) a une frame
arbitraire au defilement de la timeline.

    fichier.bqm (v2) :
        4 octets  "BQM1"     magie
        int32     version    = 2
        int32     frames                  (reecrit a la fermeture)
        int32     canaux                  champ de bits ; 1<<0 = VITESSE_SOMMET
        int64     index_off               (reecrit a la fermeture)
        --- parametres de production, pour l'invalidation ---
        int32[3]  mesh_res
        float32   cell_size
        float32   influence_radius
        float32   particle_radius
        float32   collider_offset
        int32     smoothing_iters
        int32     min_component_tris      (v2, cf. docs/plan-milestone-11.md)
        int32     src_frames              nombre de frames du .bqd source
        int32     src_n_max               n_max du .bqd source

    v1 -> v2 (docs/plan-milestone-11.md) : ajout de `min_component_tris`
    (suppression des petites composantes connexes, cf. `bourrasque.h`
    `BqMesherConfig`). Pas de compatibilite de lecture avec les fichiers v1 :
    `_unpack_header` rejette toute version differente de `_VERSION` (voir
    plus bas) — un `.bqm` v1 existant redevient un cache manquant plutot
    qu'un cache errone, comportement deja tolere par les appelants
    (`MeshCacheReader` leve, `ops.py`/`ui.py` retombent sur "aucun cache").
    Justifie ici parce que le changement cote coeur (M11) rend de toute
    facon caduque la geometrie de tout `.bqm` produit avant lui (nouvelle
    semantique Taubin de `smoothing_iters`, nouveau filtrage topologique) :
    un rebake est necessaire quoi qu'il arrive.
        --- repete `frames` fois ---
        int32     n_verts
        int32     n_tris
        n_verts*3 float32    positions des sommets (x, y, z)
        n_tris*3  int32      indices de triangles
        [si canal VITESSE_SOMMET] n_verts*3 float32   vitesses (vx, vy, vz)
        --- a index_off ---
        frames * int64       offset absolu du debut de chaque frame

Les parametres de production sont des VALEURS NOMMEES COMPARABLES UNE A
UNE, pas un hachage : quand le cache est declare caduc, l'interface doit
pouvoir dire a l'artiste quel parametre a change (voir la decision D1 du
plan M7). `diff_params` compare un jeu de parametres stocke a un jeu de
parametres courant et renvoie la liste des noms de champs qui different
(liste vide si le cache est valide). Les champs flottants sont compares
apres passage par `numpy.float32`, parce que les valeurs stockees ont
elles-memes transite par un `float32` a l'ecriture : comparer un `float64`
Python brut au flottant relu produirait de faux positifs de simple
imprecision de conversion, pas un changement reel de parametre.

Un maillage peut etre vide sur une frame (aucun fluide dans le domaine) :
`n_verts = 0` et `n_tris = 0` est un cas NORMAL, ecrit et relu sans erreur
(voir `test_meshcache.py::test_empty_frame_roundtrip`).

Comme `cache.py`, ce module prefere numpy a `struct`/`array` pour les gros
tableaux : le consommateur final est `mesh.vertices.foreach_set` /
`mesh.polygons.foreach_set` cote Blender, qui accepte nativement un ndarray
contigu.
"""

import dataclasses
import os
import pathlib
import struct

import numpy as np

_MAGIC = b"BQM1"
_VERSION = 2

# magic(4s) version(i) frames(i) canaux(i) index_off(q)
#   mesh_res(3i) cell_size(f) influence_radius(f) particle_radius(f)
#   collider_offset(f) smoothing_iters(i) min_component_tris(i)
#   src_frames(i) src_n_max(i)
_HEADER_STRUCT = struct.Struct("<4siiiq3iffffiiii")
_HEADER_SIZE = _HEADER_STRUCT.size  # 68

_FRAME_HEADER_STRUCT = struct.Struct("<ii")  # n_verts(i32) n_tris(i32)
_INDEX_ENTRY_STRUCT = struct.Struct("<q")

_BYTES_PER_VERTEX = 3 * 4  # 3 float32 (position ou vitesse)
_BYTES_PER_TRIANGLE = 3 * 4  # 3 int32 (indices)

CHANNEL_VERTEX_VELOCITY = 1 << 0

_PARAM_FIELDS = (
    "mesh_res",
    "cell_size",
    "influence_radius",
    "particle_radius",
    "collider_offset",
    "smoothing_iters",
    "min_component_tris",
    "src_frames",
    "src_n_max",
)

_FLOAT_PARAM_FIELDS = frozenset(
    ("cell_size", "influence_radius", "particle_radius", "collider_offset")
)


@dataclasses.dataclass(frozen=True)
class MeshProductionParams:
    """Parametres qui ont produit un `.bqm`, tels que stockes en en-tete.

    Champs NOMMES et comparables un a un (voir `diff_params`), a dessein :
    ce n'est pas un hachage, pour pouvoir dire a l'artiste quel parametre
    precis rend un cache caduc.
    """

    mesh_res: tuple  # (int, int, int)
    cell_size: float
    influence_radius: float
    particle_radius: float
    collider_offset: float
    smoothing_iters: int
    min_component_tris: int
    src_frames: int
    src_n_max: int


def diff_params(stored, current):
    """Compare deux `MeshProductionParams` champ a champ.

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
        if name == "mesh_res":
            if tuple(int(v) for v in sv) != tuple(int(v) for v in cv):
                diffs.append(name)
        elif name in _FLOAT_PARAM_FIELDS:
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
        int(params.mesh_res[0]),
        int(params.mesh_res[1]),
        int(params.mesh_res[2]),
        float(params.cell_size),
        float(params.influence_radius),
        float(params.particle_radius),
        float(params.collider_offset),
        int(params.smoothing_iters),
        int(params.min_component_tris),
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
        res_x,
        res_y,
        res_z,
        cell_size,
        influence_radius,
        particle_radius,
        collider_offset,
        smoothing_iters,
        min_component_tris,
        src_frames,
        src_n_max,
    ) = _HEADER_STRUCT.unpack(header_bytes)

    if magic != _MAGIC:
        raise ValueError(
            f"magie invalide : {path} n'est pas un fichier .bqm "
            f"(magie {magic!r}, {_MAGIC!r} attendue)"
        )
    if version != _VERSION:
        raise ValueError(
            f"cache {path} : version BQM inconnue ({version}), "
            f"seule la version {_VERSION} est supportee"
        )
    if frames < 0 or index_off < _HEADER_SIZE:
        raise ValueError(
            f"header .bqm incoherent : {path} frames={frames} "
            f"index_off={index_off}"
        )
    if channels & ~CHANNEL_VERTEX_VELOCITY:
        raise ValueError(
            f"header .bqm incoherent : {path} canaux={channels} "
            "contient des bits inconnus"
        )

    params = MeshProductionParams(
        mesh_res=(res_x, res_y, res_z),
        cell_size=cell_size,
        influence_radius=influence_radius,
        particle_radius=particle_radius,
        collider_offset=collider_offset,
        smoothing_iters=smoothing_iters,
        min_component_tris=min_component_tris,
        src_frames=src_frames,
        src_n_max=src_n_max,
    )
    return frames, channels, index_off, params


def read_params(path):
    """Lit uniquement les parametres de production d'un `.bqm`, sans
    charger la moindre geometrie (aucun sommet, triangle ni table
    d'index n'est lu). Usage typique : verifier la validite du cache au
    moment d'ouvrir le panneau, avant de decider de charger une frame.
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


class MeshCacheWriter:
    """Ecrit un fichier `.bqm` frame par frame.

    Meme discipline que `cache.CacheWriter` : header provisoire
    (`frames = 0`, `index_off = 0`) ecrit a l'ouverture, chaque frame
    ecrite une fois de facon purement sequentielle, offset de debut
    retenu en memoire, table d'index et header final ecrits a la
    fermeture. Supporte le protocole de contexte : `close()` est appele
    dans `__exit__` meme sur exception, pour que le fichier reste
    relisible avec une table d'index coherente meme si le bake de
    maillage est annule en cours de route.

    `params` (un `MeshProductionParams`) est fige a l'ouverture : ce sont
    les valeurs ecrites en en-tete pour l'invalidation du cache, elles ne
    changent pas en cours de bake.

    `velocity=True` declare le canal vitesse par sommet : `append_frame`
    exige alors un argument de vitesses a chaque appel.
    """

    def __init__(self, path, params, velocity=False):
        self._path = pathlib.Path(path)
        self._params = params
        self._channels = CHANNEL_VERTEX_VELOCITY if velocity else 0
        self._frame_offsets = []
        self._frames_written = 0
        self._closed = False
        self._f = open(self._path, "wb")
        self._f.write(_pack_header(0, self._channels, 0, self._params))

    @property
    def frames_written(self):
        return self._frames_written

    @property
    def has_vertex_velocity(self):
        return bool(self._channels & CHANNEL_VERTEX_VELOCITY)

    def append_frame(self, verts, tris, velocities=None):
        """Ajoute une frame de maillage.

        `verts` : `n_verts*3` float32 (positions, ctypes ou numpy).
        `tris`  : `n_tris*3` int32 (indices de triangles).
        `n_verts = 0` et `n_tris = 0` sont des valeurs valides (maillage
        vide, aucun fluide dans le domaine sur cette frame).

        `velocities` (`n_verts*3` float32) est obligatoire si le canal
        vitesse a ete declare a l'ouverture (`velocity=True`), et interdit
        sinon : passer l'un sans l'autre leve `ValueError` plutot que de
        produire un fichier dont le contenu de frame ne correspond plus au
        champ `canaux` de l'en-tete.
        """
        wants_velocity = self.has_vertex_velocity
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

        varr = np.asarray(verts, dtype=np.float32)
        if varr.size % 3 != 0:
            raise ValueError(
                f"append_frame: buffer de sommets de taille {varr.size}, "
                "un multiple de 3 (x, y, z par sommet) est attendu"
            )
        n_verts = varr.size // 3

        tarr = np.asarray(tris, dtype=np.int32)
        if tarr.size % 3 != 0:
            raise ValueError(
                f"append_frame: buffer de triangles de taille {tarr.size}, "
                "un multiple de 3 (indices par triangle) est attendu"
            )
        n_tris = tarr.size // 3

        vel_arr = None
        if wants_velocity:
            vel_arr = np.asarray(velocities, dtype=np.float32)
            if vel_arr.size != n_verts * 3:
                raise ValueError(
                    f"append_frame: {vel_arr.size} valeurs de vitesse, "
                    f"{n_verts * 3} attendues ({n_verts} sommets)"
                )

        # Une frame corrompue/tronquee corromprait tout le reste du fichier
        # sans erreur visible : on ecrit uniquement apres validation.
        offset = self._f.tell()
        self._f.write(_FRAME_HEADER_STRUCT.pack(n_verts, n_tris))
        self._f.write(np.ascontiguousarray(varr).tobytes())
        self._f.write(np.ascontiguousarray(tarr).tobytes())
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


class MeshCacheReader:
    """Lit un fichier `.bqm` frame par frame, sans jamais charger tout le
    fichier en memoire.

    A l'ouverture, lit et valide le header, les parametres de production,
    et la table d'index (avec un aller-retour leger sur chaque en-tete de
    frame pour verifier ses bornes) — pas la geometrie elle-meme, qui n'est
    chargee que par `read_frame`, a la demande.
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

        has_velocity = bool(channels & CHANNEL_VERTEX_VELOCITY)

        # Bornes de la region "frames" : entre la fin du header et le
        # debut de la table d'index. Chaque offset doit y tomber, et les
        # offsets doivent etre strictement croissants (ecriture
        # sequentielle). n_verts = n_tris = 0 (maillage vide) est valide.
        prev_end = _HEADER_SIZE
        n_verts_list = []
        n_tris_list = []
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
            n_verts, n_tris = _FRAME_HEADER_STRUCT.unpack(fh_bytes)
            if n_verts < 0 or n_tris < 0:
                raise ValueError(
                    f"frame {i} incoherente : {self._path} n_verts={n_verts} "
                    f"n_tris={n_tris}"
                )
            frame_end = (
                off
                + _FRAME_HEADER_STRUCT.size
                + n_verts * _BYTES_PER_VERTEX
                + n_tris * _BYTES_PER_TRIANGLE
                + (n_verts * _BYTES_PER_VERTEX if has_velocity else 0)
            )
            if frame_end > index_off:
                raise ValueError(
                    f"frame {i} deborde de la table d'index : {self._path} "
                    f"fin de frame {frame_end} > index_off {index_off}"
                )
            n_verts_list.append(n_verts)
            n_tris_list.append(n_tris)
            prev_end = frame_end

        self._frames = frames
        self._channels = channels
        self._params = params
        self._frame_offsets = offsets
        self._n_verts = n_verts_list
        self._n_tris = n_tris_list

    @property
    def params(self):
        return self._params

    @property
    def frame_count(self):
        return self._frames

    @property
    def has_vertex_velocity(self):
        return bool(self._channels & CHANNEL_VERTEX_VELOCITY)

    def _check_index(self, index, caller):
        if not (0 <= index < self._frames):
            raise IndexError(
                f"{caller}: index {index} hors bornes [0, {self._frames})"
            )

    def vertex_count(self, index):
        self._check_index(index, "vertex_count")
        return self._n_verts[index]

    def triangle_count(self, index):
        self._check_index(index, "triangle_count")
        return self._n_tris[index]

    def read_frame(self, index):
        """Lit la frame `index` (0-based) en un seul seek.

        Renvoie `(verts, tris, vel)` : `verts` un ndarray `(n_verts, 3)`
        float32, `tris` un ndarray `(n_tris, 3)` int32, et `vel` un
        ndarray `(n_verts, 3)` float32 si `has_vertex_velocity` est vrai,
        `None` sinon. `n_verts = 0` / `n_tris = 0` (maillage vide) est un
        cas normal : les tableaux correspondants sont alors vides, pas
        `None`.
        """
        self._check_index(index, "read_frame")
        n_verts = self._n_verts[index]
        n_tris = self._n_tris[index]
        offset = self._frame_offsets[index] + _FRAME_HEADER_STRUCT.size

        self._f.seek(offset)

        verts = np.empty((n_verts, 3), dtype=np.float32)
        if n_verts:
            n_read = self._f.readinto(verts)
            expected = n_verts * _BYTES_PER_VERTEX
            if n_read != expected:
                raise ValueError(
                    f"read_frame: {n_read} octets de sommets lus a la frame "
                    f"{index}, {expected} attendus (fichier tronque ?)"
                )

        tris = np.empty((n_tris, 3), dtype=np.int32)
        if n_tris:
            n_read = self._f.readinto(tris)
            expected = n_tris * _BYTES_PER_TRIANGLE
            if n_read != expected:
                raise ValueError(
                    f"read_frame: {n_read} octets de triangles lus a la "
                    f"frame {index}, {expected} attendus (fichier tronque ?)"
                )

        vel = None
        if self.has_vertex_velocity:
            vel = np.empty((n_verts, 3), dtype=np.float32)
            if n_verts:
                n_read = self._f.readinto(vel)
                expected = n_verts * _BYTES_PER_VERTEX
                if n_read != expected:
                    raise ValueError(
                        f"read_frame: {n_read} octets de vitesses lus a la "
                        f"frame {index}, {expected} attendus "
                        "(fichier tronque ?)"
                    )

        return verts, tris, vel

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
