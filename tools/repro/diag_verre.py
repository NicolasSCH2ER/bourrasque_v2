"""Diagnostic geometrique du verre reel : a-t-il une epaisseur, est-il etanche ?

Si la paroi est une simple coque (surface d'epaisseur nulle), aucune approche par
volume ne peut la voir -- et la question devient tout autre.
"""
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\nicol\AppData\Local\Temp\claude"
                   r"\C--Users-nicol-Code-bourrasque-v2"
                   r"\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad")
import glb

GLB = r"C:\Users\nicol\Code\bourrasque_v2\Bourrasque_test_verre.glb"
objs = dict(glb.objects(GLB))
tri = objs["verre"]
V = tri.reshape(-1, 3)
cx, cz = V[:, 0].mean(), V[:, 2].mean()
print(f"verre : {len(tri)} triangles, centre ({cx:.4f}, {cz:.4f})")

# --- Etancheite : chaque arete doit etre partagee par exactement 2 triangles.
# Sommets fusionnes a 1e-6 pres (l'export duplique les sommets par face).
key = np.round(V, 6)
uniq, inv = np.unique(key, axis=0, return_inverse=True)
faces = inv.reshape(-1, 3)
edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
edges = np.sort(edges, axis=1)
_, cnt = np.unique(edges, axis=0, return_counts=True)
print(f"  sommets uniques {len(uniq)}   aretes {len(cnt)}")
print(f"  aretes vues 1 fois (bord libre) : {int((cnt == 1).sum())}")
print(f"  aretes vues 2 fois (etanche)    : {int((cnt == 2).sum())}")
print(f"  aretes vues >2 fois (non variete): {int((cnt > 2).sum())}")

# --- Epaisseur : distribution des rayons a mi-hauteur.
y_lo, y_hi = V[:, 1].min(), V[:, 1].max()
mid = 0.5 * (y_lo + y_hi)
band = np.abs(V[:, 1] - mid) < 0.02
rr = np.hypot(V[band, 0] - cx, V[band, 2] - cz)
print(f"\n  coupe a mi-hauteur y={mid:.4f} ({band.sum()} sommets)")
if len(rr):
    ru = np.unique(np.round(rr, 4))
    print(f"  rayons distincts (mm) : {np.round(ru * 1000, 2)}")
    if len(ru) >= 2:
        print(f"  -> epaisseur de paroi = {(ru[-1] - ru[0]) * 1000:.2f} mm")
    else:
        print("  -> UN SEUL rayon : la paroi est une COQUE SANS EPAISSEUR")

# --- Volume signe : nul (au bruit pres) pour une coque, non nul pour un solide.
a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
aire = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
print(f"\n  aire totale   = {aire * 1e4:.1f} cm2")
print(f"  volume signe  = {vol * 1e6:.2f} cm3")
print(f"  epaisseur equivalente volume/aire = {vol / aire * 1000:.3f} mm")
for R in (48, 64, 96, 128):
    dx = 0.40 / R
    print(f"     R={R:>3} : dx={dx * 1000:5.2f} mm -> paroi = "
          f"{vol / aire / dx:.3f} dx")
