#!/bin/bash
densities="2000 1000 500 200 100 50 20"
alphas="0 0.25 0.5 1"
echo "density,alpha,diverged,diverge_frame,exited_water,exited_frame,eq_frac,archimede_frac,amp_frac,amp_trend" > sweep_results.csv
for d in $densities; do
  for a in $alphas; do
    echo "running density=$d alpha=$a ..." >&2
    line=$(./test_floatbody_a1b.exe $d $a 90 2>>sweep_diag.log)
    echo "$line" >> sweep_results.csv
  done
done
