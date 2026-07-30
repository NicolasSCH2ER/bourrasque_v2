import numpy as np


def load(path):
    with open(path, 'rb') as f:
        hdr = np.fromfile(f, dtype=np.int32, count=2)
        n, frames = int(hdr[0]), int(hdr[1])
        data = np.fromfile(f, dtype=np.float32, count=frames * n * 3)
        data = data.reshape(frames, n, 3)
    return n, frames, data


def compare(a_path, b_path, label):
    n1, f1, d1 = load(a_path)
    n2, f2, d2 = load(b_path)
    if n1 != n2:
        print(f"[{label}] MISMATCH particle count {n1} vs {n2}")
        return None, n1, n2
    if f1 != f2:
        print(f"[{label}] MISMATCH frame count {f1} vs {f2}")
        return None, n1, n2
    diff = np.abs(d1 - d2)
    print(f"[{label}] n={n1} frames={f1}  max|diff| all frames = {diff.max():.6e}")
    return diff.max(), n1, n2


for scene in ("jelly", "dam", "splash"):
    print(f"\n=== {scene} ===")
    noise, n_b, _ = compare(f"before_{scene}_1.bqd", f"before_{scene}_2.bqd",
                             f"{scene} bruit before (run1 vs run2)")
    noise2, n_a, _ = compare(f"after_{scene}_1.bqd", f"after_{scene}_2.bqd",
                              f"{scene} bruit after (run1 vs run2)")
    d, _, _ = compare(f"before_{scene}_1.bqd", f"after_{scene}_1.bqd",
                       f"{scene} avant vs apres (run1 vs run1)")
    print(f"n_particules before={n_b}  after={n_a}")
