"""Verifie le pattern reel utilise pour valider un jalon : ecrire un script
dans le scratchpad, puis l'executer, en chargeant la DLL du solveur."""
import sys

sys.path.insert(0, r"C:\Users\nicol\Code\bourrasque_v2\extension")
import lib

cfg = lib.default_config()
print(f"DLL chargee, garde-fou ABI franchi")
print(f"  grid_res  = {list(cfg.grid_res)}")
print(f"  cell_size = {cfg.cell_size}")
with lib.Sim(cfg) as sim:
    m = sim.add_material(lib.BQ_MODEL_WATER, rho=1000.0, bulk=4.0e4, gamma=3.0)
    n = sim.emit_box(m, (0.3, 0.3, 0.3), (0.5, 0.5, 0.5))
    sim.step(1.0 / 24.0)
    print(f"  sim OK : {n} particules, 1 step, GPU joignable")
