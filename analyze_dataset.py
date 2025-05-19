import os
import math
import numpy as np
from scipy.spatial import cKDTree

def read_points(path):
    with open(path, 'r') as f:
        n, d = map(int, f.readline().split())
        data = np.loadtxt(f)
    assert data.shape == (n, d)
    return data

def compute_stats(data):
    n, d = data.shape
    dims_min = data.min(axis=0)
    dims_max = data.max(axis=0)
    dims_mean = data.mean(axis=0)
    dims_var = data.var(axis=0)
    # 平均最近邻距离
    tree = cKDTree(data)
    dists, _ = tree.query(data, k=2)  # k=2: 第一个是自己，第二个是最近邻
    nn_mean = dists[:,1].mean()
    nn_std  = dists[:,1].std()
    return {
        'n': n,
        'd': d,
        'min': dims_min,
        'max': dims_max,
        'mean': dims_mean,
        'var': dims_var,
        'nn_mean': nn_mean,
        'nn_std': nn_std
    }

def print_stats(name, stats):
    print(f"Dataset {name}:")
    print(f"  points = {stats['n']}, dimension = {stats['d']}")
    print(f"  per-dim range:")
    for i, (mn, mx) in enumerate(zip(stats['min'], stats['max'])):
        print(f"    dim {i}: min={mn:.4f}, max={mx:.4f}, mean={stats['mean'][i]:.4f}, var={stats['var'][i]:.4e}")
    print(f"  nearest‐neighbor distance: mean={stats['nn_mean']:.4e}, std={stats['nn_std']:.4e}")
    print()

if __name__ == "__main__":
    base = "6980-main/dataset"
    sets = {
        'Island': os.path.join(base, 'origin/island.txt'),
        'Car':    os.path.join(base, 'origin/car.txt'),
        'NBA':    os.path.join(base, 'origin/nba.txt'),
    }

    for name, path in sets.items():
        data = read_points(path)
        stats = compute_stats(data)
        print_stats(name, stats)
