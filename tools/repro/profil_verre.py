"""Profil geometrique du verre reel : epaisseur du fond, epaisseur de la paroi
selon la hauteur. Une fuite « sous le fond » peut traverser le fond ou longer la
paroi ; sans ce profil on ne sait pas ce qui est sous-resolu.

Methode : lancer de rayons verticaux et horizontaux contre les 1020 triangles,
et mesurer les longueurs des segments interieurs a la matiere.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\nicol\AppData\Local\Temp\claude"
                   r"\C--Users-nicol-Code-bourrasque-v2"
                   r"\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad")
import glb

tri = dict(glb.objects(r"C:\Users\nicol\Code\bourrasque_v2"
                       r"\Bourrasque_test_verre.glb"))["verre"]
V = tri.reshape(-1, 3)
CX, CZ = V[:, 0].mean(), V[:, 2].mean()
Y0, Y1 = V[:, 1].min(), V[:, 1].max()
A, B, C = tri[:, 0], tri[:, 1], tri[:, 2]


def hits(orig, dirv):
    """Parametres t des intersections rayon/triangles (Moller-Trumbore)."""
    e1, e2 = B - A, C - A
    p = np.cross(dirv, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = orig - A
    u = np.einsum("ij,ij->i", tv, p) * inv
    q = np.cross(tv, e1)
    v = np.einsum("ij,ij->i", np.broadcast_to(dirv, tv.shape), q) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    good = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-7)
    return np.sort(t[good])


def segments(orig, dirv):
    """Longueurs des portions de matiere traversees."""
    t = hits(np.asarray(orig, float), np.asarray(dirv, float))
    if len(t) < 2:
        return []
    return [(t[i + 1] - t[i]) for i in range(0, len(t) - 1, 2)]


print("--- Epaisseur du FOND (rayon vertical montant, sous le verre)")
for r in (0.0, 0.02, 0.04, 0.06, 0.08, 0.10):
    seg = segments((CX + r, Y0 - 0.05, CZ), (0.0, 1.0, 0.0))
    if seg:
        print(f"  r={r * 100:5.1f} cm : segments (mm) {np.round(np.array(seg) * 1000, 2)}")
    else:
        print(f"  r={r * 100:5.1f} cm : aucun segment (hors du verre)")

print("\n--- Epaisseur de la PAROI (rayon horizontal, depuis l'axe)")
for frac in (0.05, 0.15, 0.3, 0.5, 0.7, 0.9, 0.98):
    y = Y0 + frac * (Y1 - Y0)
    seg = segments((CX, y, CZ), (1.0, 0.0, 0.0))
    h = (y - Y0) * 100
    if seg:
        print(f"  y={h:6.2f} cm ({frac:4.2f}h) : segments (mm) "
              f"{np.round(np.array(seg) * 1000, 2)}")
    else:
        print(f"  y={h:6.2f} cm ({frac:4.2f}h) : aucun segment")

print("\n--- Rappel des resolutions")
for R in (48, 64, 96):
    print(f"  R={R:>3} : dx = {0.3998 / R * 1000:5.2f} mm")
