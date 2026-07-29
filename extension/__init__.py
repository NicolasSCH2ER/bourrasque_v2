"""Point d'entree de l'extension Bourrasque.

Ordre d'enregistrement : `props` -> `overlay` -> `display` -> `ops` -> `ui`.
`props` en premier car les autres modules dependent des PropertyGroups
qu'il declare (`BqObjectProps`, `BqSceneProps`). `overlay` juste apres car
il ne depend que de `props` (`iter_elements`, `domain_transform`). `ui` en
dernier car ses panneaux referencent les idnames des operateurs de `ops`.
`unregister()` fait l'inverse exact.

Imports relatifs uniquement (`from . import props`) : les extensions
Blender 5.x sont chargees comme des packages ; un import absolu casserait
des que l'extension est installee sous un nom de package different du
notre en developpement.
"""

import importlib
import sys

_MODULE_NAMES = ("props", "overlay", "display", "ops", "ui")

# Un module deja present dans sys.modules avant notre `from . import ...`
# ci-dessous signifie que l'extension a deja ete chargee une fois dans ce
# processus Blender (rechargement pendant le developpement) : il faut le
# recharger explicitement, sinon le `from . import` suivant se contente de
# renvoyer l'objet module deja en cache et l'ancien code reste actif en
# silence.
_needs_reload = {
    name: f"{__name__}.{name}" in sys.modules for name in _MODULE_NAMES
}

from . import display, ops, overlay, props, ui

_MODULES = (props, overlay, display, ops, ui)

# Recharge dans l'ordre de dependance (props d'abord) pour que chaque
# module qui fait `from .props import ...` etc. capture bien les objets
# fraichement recharges des modules dont il depend.
for _mod in _MODULES:
    if _needs_reload[_mod.__name__.rsplit(".", 1)[-1]]:
        importlib.reload(_mod)


def register():
    for mod in _MODULES:
        mod.register()


def unregister():
    # unregister() doit etre robuste : si un module echoue a se
    # desenregistrer, les suivants doivent quand meme etre traites, sinon un
    # unregister partiel laisse Blender dans un etat ou l'extension ne peut
    # plus etre reactivee sans redemarrage.
    for mod in reversed(_MODULES):
        try:
            mod.unregister()
        except Exception:
            import traceback

            print(f"[bourrasque] echec de unregister() pour {mod.__name__} :")
            traceback.print_exc()
