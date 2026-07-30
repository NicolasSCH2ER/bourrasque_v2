"""
TEST A1 — Référence API directe

Script Python hors Blender qui utilise extension/lib.py directement.
Config spécifique, un seul matériau eau, émission dans une boîte,
10 frames en enregistrant les positions.
"""
import sys
import os

# Ajouter le répertoire extension au chemin
extension_dir = r"C:\Users\nicol\Code\bourrasque_v2\extension"
sys.path.insert(0, extension_dir)

# Importer directement les modules sans passer par __init__.py
import importlib.util

# Charger lib.py
spec = importlib.util.spec_from_file_location("lib", os.path.join(extension_dir, "lib.py"))
lib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib)

# Charger cache.py
spec = importlib.util.spec_from_file_location("cache", os.path.join(extension_dir, "cache.py"))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)

def main():
    print("TEST A1 — Référence API directe")
    print("=" * 60)
    
    # Configuration
    cfg = lib.default_config()
    cfg.grid_res = 64
    cfg.ppc_axis = 2
    cfg.domain = 1.0
    cfg.cfl = 0.3
    cfg.gravity_y = -9.8
    cfg.max_particles = 2000000
    
    print(f"Config: grid_res={cfg.grid_res} domain={cfg.domain} cfl={cfg.cfl}")
    print(f"        gravity_y={cfg.gravity_y} ppc_axis={cfg.ppc_axis}")
    
    # Créer la simulation et le matériau eau
    with lib.Sim(cfg) as sim:
        mat_id = sim.add_material(
            lib.BQ_MODEL_WATER,
            rho=1000.0,
            bulk=4e4,
            gamma=3.0
        )
        print(f"Matériau eau créé, id={mat_id}")
        
        # Émettre les particules
        lo = (0.10, 0.10, 0.10)
        hi = (0.35, 0.60, 0.90)
        vel = (0.0, 0.0, 0.0)
        n_emit = sim.emit_box(mat_id, lo, hi, vel=vel)
        print(f"Particules émises: {n_emit}")
        print(f"Particule count: {sim.particle_count}")
        
        # Préparation du cache
        scratchpad_dir = r"C:\Users\nicol\AppData\Local\Temp\claude\C--Users-nicol-Code-bourrasque-v2\e0f7a97b-4c70-4b8a-91d6-a31aa8642d46\scratchpad"
        os.makedirs(scratchpad_dir, exist_ok=True)
        bqd_path = os.path.join(scratchpad_dir, "ref.bqd")
        
        with cache.CacheWriter(bqd_path, sim.particle_count) as writer:
            writer.write_materials(sim.read_materials())
            
            # 10 steps de 1/24
            frame_dt = 1.0 / 24.0
            for frame_idx in range(10):
                substeps = sim.step(frame_dt)
                pos = sim.read_positions()
                writer.append_frame(pos)
                print(f"Frame {frame_idx+1}: {substeps} substeps, {pos.shape}")
    
    print(f"\nRéférence écrite: {bqd_path}")
    
    # Vérifier le fichier produit
    with cache.CacheReader(bqd_path) as reader:
        print(f"\nVérification du cache de référence:")
        print(f"  n_particles: {reader.n_particles}")
        print(f"  frame_count: {reader.frame_count}")
        frame0 = reader.read_frame(0)
        print(f"  Frame 0 shape: {frame0.shape}")
        print(f"  Frame 0 min: {frame0.min(axis=0)}")
        print(f"  Frame 0 max: {frame0.max(axis=0)}")
    
    print("\nTEST A1 OK")

if __name__ == "__main__":
    main()
