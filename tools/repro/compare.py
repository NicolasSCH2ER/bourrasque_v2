import struct
import numpy as np

def load(path):
    with open(path, 'rb') as f:
        n, frames = struct.unpack('<ii', f.read(8))
        data = np.fromfile(f, dtype=np.float32)
    data = data.reshape(frames, n, 3)
    return n, frames, data

def compare(a, b, label):
    na, fa, da = load(a)
    nb, fb, db = load(b)
    assert na == nb and fa == fb, f"MISMATCH counts: {na},{fa} vs {nb},{fb}"
    diff = np.abs(da[-1] - db[-1])
    print(f"{label}: n={na} frames={fa} max|diff|last={diff.max():.6g} mean|diff|last={diff.mean():.6g}")
    return diff.max()

if __name__ == '__main__':
    base = "C:/tmp/bqtest"
    print("--- bruit (meme binaire, deux runs) ---")
    compare(f"{base}/dam_base1.bqd", f"{base}/dam_base2.bqd", "dam noise (pre-change binary)")
    compare(f"{base}/dam_a.bqd", f"{base}/dam_b.bqd", "dam noise (post-change binary)")
    print("--- ecart avant/apres (n_tri=0) ---")
    compare(f"{base}/dam_base1.bqd", f"{base}/dam_new1.bqd", "dam drift (pre vs post-change)")
    compare(f"{base}/jelly_base1.bqd", f"{base}/jelly_new1.bqd", "jelly drift (pre vs post-change)")
    compare(f"{base}/splash_base1.bqd", f"{base}/splash_new1.bqd", "splash drift (pre vs post-change)")
