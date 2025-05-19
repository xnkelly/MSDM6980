import os
import math

def read_nd_data(path):
    """
    Read a file where:
      - First line: "<num_points> <dim>"
      - Each subsequent line: dim+1 floats (one timestamp + dim coordinates)
    Returns a list of rows, each a list of floats length dim+1.
    """
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip().split()
        if len(header) != 2:
            raise ValueError("Header must be: <num_points> <dimension>")
        n_points, dim = map(int, header)
        expected_cols = dim + 1
        rows = []
        for i in range(n_points):
            parts = f.readline().strip().split()
            if len(parts) != expected_cols:
                raise ValueError(
                    f"Line {i+2}: expected {expected_cols} values, got {len(parts)}"
                )
            rows.append([float(x) for x in parts])
    return rows

def evaluate_sed_with_time(data_orig, data_simp, time_index):
    """
    Compute normalized RMSE and MaxError using time-synchronized Euclidean distance (SED),
    given full rows including timestamp and coordinates.

    - data_orig: list of lists, each has [t, x1, x2, ...] if time_index=0,
                 or [x1, t, x2,...] accordingly.
    - data_simp: same format as data_orig but with fewer points.
    - time_index: integer index of timestamp column in each row.
    """
    # Extract times and spatial points
    orig_times = []
    orig_pts = []
    for row in data_orig:
        t = row[time_index]
        coords = [row[i] for i in range(len(row)) if i != time_index]
        orig_times.append(t)
        orig_pts.append(coords)
    simp_times = []
    simp_pts = []
    for row in data_simp:
        t = row[time_index]
        coords = [row[i] for i in range(len(row)) if i != time_index]
        simp_times.append(t)
        simp_pts.append(coords)

    # Sort simplified points by time
    idx_sorted = sorted(range(len(simp_times)), key=lambda j: simp_times[j])
    simp_times = [simp_times[j] for j in idx_sorted]
    simp_pts = [simp_pts[j] for j in idx_sorted]

    # Normalize spatial coordinates by range
    m = len(orig_pts)
    dim = len(orig_pts[0])
    mins = [min(p[i] for p in orig_pts) for i in range(dim)]
    maxs = [max(p[i] for p in orig_pts) for i in range(dim)]
    ranges = [(maxs[i] - mins[i]) or 1.0 for i in range(dim)]
    def normalize(pts):
        return [[(p[i] - mins[i]) / ranges[i] for i in range(dim)] for p in pts]

    orig_n = normalize(orig_pts)
    simp_n = normalize(simp_pts)

    # Evaluate SED: interpolate between simp_n at simp_times
    sum_sq = 0.0
    max_err = 0.0
    j = 0
    k = len(simp_times)
    for t, p in zip(orig_times, orig_n):
        # find segment [j, j+1] such that simp_times[j] <= t < simp_times[j+1]
        while j+1 < k and simp_times[j+1] <= t:
            j += 1
        t0 = simp_times[j]
        p0 = simp_n[j]
        if j+1 < k:
            t1 = simp_times[j+1]
            p1 = simp_n[j+1]
            alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
            interp = [p0_i + alpha*(p1_i - p0_i) for p0_i, p1_i in zip(p0, p1)]
        else:
            interp = p0
        # compute distance
        dist = math.dist(p, interp)
        sum_sq += dist*dist
        if dist > max_err:
            max_err = dist

    rmse = math.sqrt(sum_sq / m)
    return rmse, max_err

# Example usage
if __name__ == "__main__":   
    ti = input("请输入 timestamp 所在列（0-based，默认最后一列）：").strip()
    if ti:
        time_index = int(ti) 
    else:
        time_index = 1

    #file_prefix = '6980-main/'
    file_prefix = './'
    ori_dir = os.path.join(file_prefix, 'dataset/origin/')
    sim_dir = os.path.join(file_prefix, 'dataset/result/')

    ori_files = sorted(f for f in os.listdir(ori_dir)
                       if os.path.isfile(os.path.join(ori_dir, f)))
    sim_files = sorted(f for f in os.listdir(sim_dir)
                       if os.path.isfile(os.path.join(sim_dir, f)))

    if not ori_files or not sim_files:
        print("Origin or result directory is empty.")

    ori_path = os.path.join(ori_dir, ori_files[0])
    sim_path = os.path.join(sim_dir, sim_files[0])
    print(f"Evaluating:\n  Original:   {ori_files[0]}\n  Simplified: {sim_files[0]}")

    data_orig = read_nd_data(ori_path)
    data_simp = read_nd_data(sim_path)
    rmse, maxe = evaluate_sed_with_time(data_orig, data_simp, time_index)
    print(f"Time-aware normalized RMSE: {rmse:.6f}")
    print(f"Time-aware normalized MaxError: {maxe:.6f}")
