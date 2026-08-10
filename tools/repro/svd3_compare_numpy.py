"""svd3_compare_numpy.py -- verification independante de core/src/svd3.cuh.

Lit svd3_harness_out.csv (produit par svd3_harness.exe) et compare les
valeurs singulieres calculees sur device a numpy.linalg.svd. Les valeurs
singulieres sont uniques (au signe pres de la derniere, absorbe par notre
convention de reflexion) -- c'est donc une verification INDEPENDANTE de
l'implementation, contrairement au test de reconstruction ||USV^T - F||
qui pourrait passer avec une decomposition valide mais des valeurs fausses.

Usage : python svd3_compare_numpy.py [chemin_csv]
"""
import csv
import sys

import numpy as np

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "svd3_harness_out.csv"

max_abs_diff = 0.0
max_rel_diff = 0.0
worst_row = None
n = 0
per_cat_max = {}

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n += 1
        F = np.array([float(row[f"F{i}"]) for i in range(9)], dtype=np.float64).reshape(3, 3)
        S_gpu = np.array([abs(float(row["Sx"])), abs(float(row["Sy"])), abs(float(row["Sz"]))])
        # svd3 trie deja par magnitude decroissante ; on prend |S| pour
        # comparer a numpy qui renvoie des valeurs singulieres >= 0.
        S_np = np.linalg.svd(F, compute_uv=False)  # deja triees decroissant, >= 0

        diff = np.abs(S_gpu - S_np)
        d = float(np.max(diff))
        scale = max(float(np.max(S_np)), 1e-8)
        rel = d / scale

        cat = row["cat"]
        per_cat_max[cat] = max(per_cat_max.get(cat, 0.0), d)

        if d > max_abs_diff:
            max_abs_diff = d
            worst_row = (cat, S_gpu.tolist(), S_np.tolist())
        if rel > max_rel_diff:
            max_rel_diff = rel

print(f"{n} lignes comparees a numpy.linalg.svd")
print(f"ecart max (absolu)   sur les valeurs singulieres : {max_abs_diff:.6e}")
print(f"ecart max (relatif)  sur les valeurs singulieres : {max_rel_diff:.6e}")
if worst_row is not None:
    print(f"pire cas : categorie={worst_row[0]}  S_gpu={worst_row[1]}  S_numpy={worst_row[2]}")

print("\n--- ecart max par categorie ---")
for cat, d in sorted(per_cat_max.items(), key=lambda kv: -kv[1]):
    print(f"  {cat:<28} : {d:.6e}")
