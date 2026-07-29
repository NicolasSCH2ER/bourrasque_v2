"""overlay.py — dessin dans le viewport 3D des boites reellement simulees.

Un emetteur `bourrasque.emit_source == 'BOUNDS'` n'emet des particules que
dans sa boite englobante alignee sur les axes (`bq_emit_box`) : ce module
dessine alors, en complement du maillage affiche par `display.py`, la bbox
monde de l'emetteur, afin que l'artiste voie l'ecart entre la forme de son
objet et ce qui est effectivement simule. Un emetteur `emit_source ==
'MESH'` emet dans la forme reelle du maillage — l'artiste la voit deja,
c'est son objet — et dessiner son AABB serait trompeur (elle suggere un
volume qui ne sera pas rempli) : aucune boite n'est donc dessinee pour lui.

Le domaine est toujours dessine, quel que soit `emit_source` : c'est la
zone UTILE REELLEMENT simulee, c'est-a-dire le pave solveur
(`props.domain_transform`) reduit de sa marge de stencil sur chaque face
(`props.domain_usable_bounds`), reconverti en espace MONDE via
`transform.solver_to_world`. Depuis M5, `res[i] = round(extent/dx) +
2*SOLVER_STENCIL_BOUND` : l'arrondi introduit un ecart d'au plus `dx` par
axe entre cette zone utile et la bbox brute de l'objet domaine — ecart
petit mais qui doit rester VISIBLE, ce que dessiner directement la bbox de
l'objet (comme avant M5, ou l'ecart etait nul par construction) ne
montrerait pas. L'objet Blender de l'artiste n'est jamais redimensionne :
seul le TRACE change.

Un COLLIDER est toujours dessine comme sa boite ORIENTEE (les 8 coins de
`obj.bound_box` transformes par `matrix_world`, relies sans repasser par un
min/max par axe) et non comme un AABB monde : contrairement a un emetteur
BOUNDS, la geometrie REELLEMENT transmise au coeur pour un collider est son
maillage evalue tel quel (voir `sampling.py` / `ops.py::_update_colliders`),
rotation comprise -- une boite alignee sur les axes du monde ne suivrait pas
la rotation de l'objet et induirait l'artiste en erreur sur la forme de
l'obstacle qu'il a place.

Shader retenu : `POLYLINE_UNIFORM_COLOR`, confirme present en Blender 5.2 via
`gpu.init(); gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')` (les shaders
`2D_*`/`3D_*` legacy sont retires depuis 4.0). Attribut de sommet attendu :
`pos` (VEC3) ; uniformes `color`, `lineWidth`, `viewportSize`.

Ce module ne stocke aucun cache de bounding box entre deux dessins : avec au
plus quelques dizaines d'objets taggues, recalculer les coins a chaque appel
de `draw()` est negligeable, et c'est le seul moyen simple de rester correct
quand l'artiste deplace un emetteur ou le domaine en temps reel.
"""

import bpy
import gpu
import mathutils
from gpu_extras.batch import batch_for_shader

from .props import domain_transform, domain_usable_bounds, iter_elements
from .transform import solver_to_world

__all__ = ("register", "unregister")

_LINE_WIDTH = 1.5

_COLOR_WATER = (0.2, 0.5, 1.0, 1.0)
_COLOR_ELASTIC = (1.0, 0.55, 0.1, 1.0)
_COLOR_DOMAIN = (0.9, 0.9, 0.9, 1.0)
# Rouge, distinct des couleurs materiau (bleu eau / orange gelee) et du gris
# du domaine : les colliders sont des obstacles, pas de la matiere simulee.
_COLOR_COLLIDER = (0.9, 0.15, 0.15, 1.0)

# Alpha applique a la boite d'un emetteur selon son mode d'emission.
# `BLOCK` (matiere emise d'un coup) reste pleinement opaque ; `INFLOW`
# (matiere emise en continu, frame apres frame) est dessine plus
# transparent, pour suggerer une boite qui "laisse passer" de la matiere
# au fil du temps plutot qu'un volume rempli d'un coup. Reste un seul canal
# (l'alpha des couleurs existantes) pour ne pas ajouter de nouvelle
# variable visuelle a l'overlay.
_ALPHA_BLOCK = 1.0
_ALPHA_INFLOW = 0.45

# Reference module-level du handler de dessin, pour pouvoir le retirer avant
# d'en reajouter un lors d'un rechargement de l'extension (voir `register`).
_draw_handler = None

# Les 12 aretes d'une boite, exprimees comme 24 indices de coins (2 par
# arete) dans l'ordre des 8 coins de `obj.bound_box` / d'un produit
# cartesien {min,max}^3 range (x, y, z).
_BOX_EDGE_CORNER_INDICES = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # face du bas
    (4, 5), (5, 6), (6, 7), (7, 4),  # face du haut
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticales
)


def _box_corners(min_corner, max_corner):
    """8 coins d'une boite AABB, dans l'ordre attendu par
    `_BOX_EDGE_CORNER_INDICES` (celui de `Object.bound_box`)."""
    x0, y0, z0 = min_corner
    x1, y1, z1 = max_corner
    return (
        (x0, y0, z0),
        (x0, y0, z1),
        (x0, y1, z1),
        (x0, y1, z0),
        (x1, y0, z0),
        (x1, y0, z1),
        (x1, y1, z1),
        (x1, y1, z0),
    )


def _box_line_points(min_corner, max_corner):
    """Liste de points (paires consecutives) pour un batch en mode 'LINES'
    dessinant les 12 aretes de la boite `[min_corner, max_corner]`."""
    corners = _box_corners(min_corner, max_corner)
    points = []
    for i, j in _BOX_EDGE_CORNER_INDICES:
        points.append(corners[i])
        points.append(corners[j])
    return points


def _object_world_bounds(obj):
    """bbox monde de `obj` sous forme `(min_corner, max_corner)`."""
    mat = obj.matrix_world
    corners = [mat @ mathutils.Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _object_world_oriented_box_points(obj):
    """Points (paires consecutives, mode 'LINES') des 12 aretes de la boite
    ORIENTEE de `obj` : les 8 coins de `obj.bound_box` transformes par
    `matrix_world`, relies directement par `_BOX_EDGE_CORNER_INDICES` --
    SANS repasser par un min/max par axe. `obj.bound_box` est deja range
    dans l'ordre attendu par `_BOX_EDGE_CORNER_INDICES` (celui d'un produit
    cartesien {min,max}^3 local, voir la docstring de `_box_corners`), donc
    le transformer directement par `matrix_world` preserve cet ordre et la
    rotation de l'objet ; un passage par min/max (comme `_object_world_bounds`)
    reconstruit au contraire une boite alignee sur les axes du monde et
    perd la rotation par construction -- c'est exactement le bug que cette
    fonction corrige pour les colliders (voir docstring de module)."""
    mat = obj.matrix_world
    corners = [mat @ mathutils.Vector(c) for c in obj.bound_box]
    points = []
    for i, j in _BOX_EDGE_CORNER_INDICES:
        points.append((corners[i].x, corners[i].y, corners[i].z))
        points.append((corners[j].x, corners[j].y, corners[j].z))
    return points


def _domain_usable_world_bounds(scene):
    """Bbox MONDE `(min_corner, max_corner)` de la zone UTILE du domaine
    (le pave solveur reduit de sa marge de stencil sur chaque face), ou
    `None` si aucun domaine n'est defini.

    Convertit les 8 coins du pave `[lo, hi]` (espace solveur,
    `props.domain_usable_bounds`) en espace monde via
    `transform.solver_to_world`, puis reprend leur bbox : le mapping
    monde<->solveur est une isometrie (permutation d'axes + inversion d'un
    axe + translation), elle transforme donc bien un pave aligne sur les
    axes solveur en un pave aligne sur les axes monde, mais PAS
    componentwise (voir `transform.py`, l'axe sz est invertit par rapport a
    by) — d'ou le passage par les 8 coins plutot qu'un simple couple
    lo/hi transforme terme a terme.
    """
    transform = domain_transform(scene)
    usable = domain_usable_bounds(scene)
    if transform is None or usable is None:
        return None
    origin, size = transform
    lo, hi = usable

    corners_solver = [
        (
            lo[0] if bit_x == 0 else hi[0],
            lo[1] if bit_y == 0 else hi[1],
            lo[2] if bit_z == 0 else hi[2],
        )
        for bit_x in (0, 1)
        for bit_y in (0, 1)
        for bit_z in (0, 1)
    ]
    corners_world = [solver_to_world(c, origin, size) for c in corners_solver]
    xs = [c[0] for c in corners_world]
    ys = [c[1] for c in corners_world]
    zs = [c[2] for c in corners_world]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _draw_box(shader, min_corner, max_corner, color):
    points = _box_line_points(min_corner, max_corner)
    batch = batch_for_shader(shader, "LINES", {"pos": points})
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_lines(shader, points, color):
    batch = batch_for_shader(shader, "LINES", {"pos": points})
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw():
    context = bpy.context
    scene = getattr(context, "scene", None)
    if scene is None:
        return

    elements = iter_elements(scene)
    if not elements:
        return

    shader = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    shader.bind()
    shader.uniform_float("lineWidth", _LINE_WIDTH)
    shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:4])

    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("LESS_EQUAL")

    try:
        for obj in elements:
            if not obj.visible_get():
                continue

            role = obj.bourrasque.role
            if role == "EMITTER":
                if obj.bourrasque.emit_source != "BOUNDS":
                    # La forme reellement emise est celle du maillage,
                    # deja visible dans le viewport comme l'objet
                    # lui-meme : dessiner son AABB suggererait a tort un
                    # volume qui ne sera pas rempli.
                    continue
                min_corner, max_corner = _object_world_bounds(obj)
                r, g, b, _a = (
                    _COLOR_WATER
                    if obj.bourrasque.model == "WATER"
                    else _COLOR_ELASTIC
                )
                alpha = (
                    _ALPHA_INFLOW
                    if obj.bourrasque.emit_mode == "INFLOW"
                    else _ALPHA_BLOCK
                )
                _draw_box(shader, min_corner, max_corner, (r, g, b, alpha))
            elif role == "COLLIDER":
                # Boite ORIENTEE (pas un AABB monde) : un collider est
                # l'obstacle REEL simule (voir sampling.py/ops.py, qui
                # extraient sa geometrie evaluee telle quelle, rotation
                # comprise), donc son contour doit suivre sa rotation --
                # sinon l'artiste voit un contour qui ne correspond pas a
                # son objet des qu'il le tourne. Couleur distincte de celle
                # des emetteurs (voir _COLOR_COLLIDER) pour que l'artiste
                # distingue au premier coup d'oeil obstacle et matiere
                # simulee.
                points = _object_world_oriented_box_points(obj)
                _draw_lines(shader, points, _COLOR_COLLIDER)
            elif role == "DOMAIN":
                # Dessine la zone UTILE REELLEMENT simulee (pave solveur
                # moins la marge de stencil, voir `_domain_usable_world_bounds`
                # et docs/plan-milestone-5.md D4), pas la bbox brute de
                # l'objet : depuis M5, l'arrondi par axe de la resolution
                # (`res[i] = round(...) + 2*SOLVER_STENCIL_BOUND`) introduit
                # un ecart d'au plus `dx` par axe entre les deux, qui doit
                # rester visible pour l'artiste plutot que d'etre masque en
                # dessinant systematiquement la boite qu'il a placee.
                bounds = _domain_usable_world_bounds(scene)
                if bounds is None:
                    continue
                min_corner, max_corner = bounds
                _draw_box(shader, min_corner, max_corner, _COLOR_DOMAIN)
    finally:
        gpu.state.blend_set("NONE")
        gpu.state.depth_test_set("NONE")


def _draw_safe():
    # Une exception dans un draw_handler se repete a chaque rafraichissement
    # du viewport et rend Blender inutilisable : on l'avale toujours, quitte
    # a ne rien dessiner pour cette frame.
    try:
        _draw()
    except Exception:
        import traceback

        print("[bourrasque] echec de overlay._draw() :")
        traceback.print_exc()


def register():
    global _draw_handler
    # Retire un handler eventuellement laisse par un enregistrement
    # precedent (rechargement de l'extension) avant d'en ajouter un nouveau,
    # pour ne jamais le dupliquer.
    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        except (ValueError, RuntimeError):
            pass
        _draw_handler = None

    _draw_handler = bpy.types.SpaceView3D.draw_handler_add(
        _draw_safe, (), "WINDOW", "POST_VIEW"
    )


def unregister():
    global _draw_handler
    if _draw_handler is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handler, "WINDOW")
        except (ValueError, RuntimeError):
            pass
        _draw_handler = None
