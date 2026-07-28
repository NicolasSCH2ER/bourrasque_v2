# Bourrasque 🌊

Solveur **MLS-MPM sur GPU (CUDA)** avec intégration Blender 5.x visée.
Un seul pipeline particules ↔ grille (transferts APIC, Hu et al. 2018),
des matériaux qui s'empilent dessus : élastique et eau aujourd'hui,
sable, neige et déformables demain — sans réécrire les transferts.

> Statut : **M1 (élastique) et M2 (eau) implémentés**, validés
> numériquement par la référence NumPy. Historique : le prototype
> XPBD/PBF vit dans la branche `archive/xpbd` — abandonné pour cause
> de dissipation structurelle (fluides gélatineux).

## Pourquoi MLS-MPM

- Les transferts APIC préservent le moment angulaire (PIC le dissipe,
  FLIP le récupère au prix du bruit) ;
- la fusion de la matrice affine C avec le gradient de vitesse rend
  chaque particule ~2× moins chère que le MPM classique ;
- c'est l'architecture de Taichi et des travaux Disney/NVIDIA : chaque
  nouveau matériau est une fonction de contrainte, pas un solveur.

## Architecture

```
extension/   Extension Blender (reconnexion prévue une fois le core stable)
core/
  include/bourrasque.h   API C plate (ctypes-ready, zéro dépendance)
  src/mlsmpm.cu          Kernels CUDA : clear / P2G / grid / G2P
  headless/main.cpp      Scènes de test, n'utilise QUE l'API publique
scripts/
  ref_mlsmpm.py          ⚠ SPÉCIFICATION EXÉCUTABLE du solveur (NumPy)
  view_dump.py           Viewer matplotlib des dumps .bqd (+ sidecar .mat)
```

**Règle de développement** : `ref_mlsmpm.py` fait foi. Toute évolution
de l'algorithme se prototype et se valide d'abord dans la référence
NumPy, puis se transcrit dans le .cu. Une divergence DLL/référence est
un bug de la DLL.

## Algorithme (par substep)

1. **P2G** — mise à jour de F par `F ← (I + dt·C)F` (élastique),
   contrainte de Cauchy selon le matériau, scatter atomique de la masse
   et de la quantité de mouvement avec le terme affine
   `(−dt·vol·4/dx²)·σ + m·C`.
2. **Grille** — `v = mv/m`, gravité, conditions aux limites séparantes
   (composante normale annulée vers la paroi) sur une marge de 3 nœuds.
3. **G2P** — gather de v et de `C = 4/dx²·Σ w·v⊗dpos` (APIC),
   advection, mise à jour de `J ← J(1 + dt·tr C)` pour l'eau.

Matériaux :

| Modèle | Contrainte | Paramètres | Notes |
|---|---|---|---|
| Élastique | Corotationnel fixe `2μ(F−R)Fᵀ + λJ(J−1)I` | ρ, E, ν | R par itération de Higham (SVD complète en M3) |
| Eau | EOS de Tait `p = (k/γ)(J^−γ − 1)`, `σ = −pI` | ρ, k, γ | Faiblement compressible ; J clampé [0.5, 1.5] |

Le pas de temps est fixé par la CFL acoustique du matériau le plus
raide : `dt = cfl·dx/c` avec `c = √(raideur/ρ)`. E trop bas = gelée
qui s'effondre sous son propre poids (ρgh ≈ E), k trop bas = eau qui
rebondit — c'est le compromis assumé du M2, la projection de pression
incompressible est l'option M3+.

## Prérequis

- **RTX 50xx (Blackwell)** : CUDA Toolkit **≥ 12.8** (sm_120) ;
  autre GPU : `-DCMAKE_CUDA_ARCHITECTURES=native`
- Visual Studio 2026 (MSVC v145, ≥ 14.50) ou Visual Studio 2022 (MSVC v143),
  CMake ≥ 3.24
- CUDA ≥ 13.0 est requis avec MSVC v145 : CUDA 12.8 ne reconnaît pas ce host
  compiler. VS2022 + CUDA 12.8 reste une combinaison valide.
- Python 3 + numpy + matplotlib pour la référence et le viewer

## Build (Windows)

```bat
cmake -B build -G "Visual Studio 18 2026" -A x64
cmake --build build --config Release
```

Avec Visual Studio 2022, remplacer par `-G "Visual Studio 17 2022"`.

Produit `bourrasque.dll` (copiée dans `extension/bin/`) et
`bourrasque_headless.exe`.

## Valider les jalons

```bat
:: référence NumPy (lente mais faisant foi)
python scripts\ref_mlsmpm.py jelly 12
python scripts\ref_mlsmpm.py dam 10

:: core CUDA
build\Release\bourrasque_headless.exe jelly 120 jelly.bqd
build\Release\bourrasque_headless.exe dam 240 dam.bqd
build\Release\bourrasque_headless.exe splash 240 splash.bqd
python scripts\view_dump.py splash.bqd --stride 4
```

Comportements attendus :
- **jelly** : chute, écrasement à l'impact, rebond en gelée, repos.
- **dam** : effondrement de la colonne, vague qui remonte le mur opposé,
  clapotis, mise à niveau.
- **splash** : le cube élastique plonge dans l'eau — les deux matériaux
  dans le même pas de simulation, c'est la démo du pipeline unifié.

Si ça explose : baisser `cfl` (0.3 → 0.2). Si l'eau tremble : monter
`bulk` (et accepter le dt plus petit qui va avec).

## Roadmap

| # | Livrable | Done quand |
|---|---|---|
| ✅ 1 | Pipeline MLS-MPM + élastique corotationnel | cube de gelée : chute, rebond, repos |
| ✅ 2 | Eau EOS de Tait (J-based) | dam break stable et vivant |
| 3 | SVD 3×3 (McAdams) + plasticité : sable Drucker-Prager, neige | château de sable qui s'écoule |
| 4 | Colliders SDF (plans, sphères, mesh) + projection pression optionnelle | l'eau remplit un verre |
| 5 | Extension Blender live viewport (bridge ctypes conservé) | l'eau coule dans Blender |
| 6 | Reconstruction de surface (SDF + marching cubes GPU) | surface rendable Cycles |
| 7 | Particules secondaires (mousse, embruns) | whitewater sur le dam break |

## Références

- Hu et al., *A Moving Least Squares Material Point Method with
  Displacement Discontinuity and Two-Way Rigid Body Coupling*
  (MLS-MPM), SIGGRAPH 2018
- Jiang et al., *The Affine Particle-In-Cell Method* (APIC), SIGGRAPH 2015
- Stomakhin et al., *A Material Point Method for Snow Simulation*,
  SIGGRAPH 2013 (M3)
- Jiang et al., *The Material Point Method for Simulating Continuum
  Materials*, SIGGRAPH Course Notes 2016 (la bible)
- taichi mpm128 / NVIDIA Warp comme implémentations de contrôle

## Licence

GPL-3.0-or-later (obligatoire pour une extension Blender de toute façon).
