import numpy as np
import sys

def load(path):
    with open(path, 'rb') as f:
        hdr = np.fromfile(f, dtype=np.int32, count=2)
        n, frames = int(hdr[0]), int(hdr[1])
        data = np.fromfile(f, dtype=np.float32, count=frames*n*3)
        data = data.reshape(frames, n, 3)
    return n, frames, data

def compare(a_path, b_path, label):
    n1, f1, d1 = load(a_path)
    n2, f2, d2 = load(b_path)
    assert n1 == n2, f"particle count mismatch {n1} vs {n2}"
    assert f1 == f2
    diff = np.abs(d1 - d2)
    last_diff = diff[-1]
    print(f"[{label}] n={n1} frames={f1}  max|diff| all frames = {diff.max():.6e}  "
          f"max|diff| last frame = {last_diff.max():.6e}  mean|diff| last frame = {last_diff.mean():.6e}")
    return diff.max()

if __name__ == '__main__':
    noise = compare("before_run1.bqd", "before_run2.bqd", "bruit before (run1 vs run2)")
    noise2 = compare("after_run1.bqd", "after_run2.bqd", "bruit after (run1 vs run2)")
    invariant = compare("before_run1.bqd", "after_run1.bqd", "avant vs apres (run1 vs run1)")
