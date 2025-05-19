#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并 n 个维度的简化结果文件，恢复 n 维简化轨迹。
步骤：
1. 从原始完整数据文件读取 full_data[seq] = [v0,...,v{n-1}]。
2. 从 sim_dir 下 0...n-1 文件读取每个维度的 {seq: value, weight}。
3. 统计每个 seq 在多少维出现（频次 freq），以及权重总和 (sum of normalized weights)。
4. 令 m = 原始总点数， K = floor(r * m)：
   a) Phase1：选出所有 freq 较高 的 seq，记为 selected1；如果 len(selected1) > K，则取前 K。
   b) Phase2：否则从剩余 seq 中按权重总和降序选出 K - len(selected1) 个，记为 selected2。
5. 最终 selected = selected1 ∪ selected2，输出前 K 条，按 seq 升序。
输出文件第一行：“K n”，后面每行是 “v0 v1 ... v{n-1}”。
"""

import os
import math
import argparse
from collections import defaultdict


import os
import math
from collections import defaultdict

def merge_dim_results(sim_dir: str,
                      full_file: str,
                      output_file: str,
                      ratio: float,
                      time_index: int):
    """
    合并各维度简化结果，使用 time_index 列作为时间戳 key。

    参数
    ----
    sim_dir : str
        存放各维度简化文件的目录，文件名形如 sim_<dim>，每行：value timestamp weight
    full_file : str
        原始完整数据文件路径，每行若干列数值，time_index 列是 timestamp
    output_file : str
        合并后输出文件路径，格式：
            K n
            <val_dim0> <val_dim1> ... <val_dim{n-1}>
            ...
    ratio : float
        保留点数目占原始 m 的比例，K = floor(ratio * m)
    time_index : int
        在 full_file 中，哪一列（从 0 开始计数）是时间戳

    """
    # 1. 读取原始完整数据，用 timestamp 当 key
    full_data = {}  # ts -> list of str (vals)
    with open(full_file, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            ts = float(parts[time_index])
            # 整行都作为 vals（不排除有多余列）
            full_data[ts] = parts

    # 原始数据点数
    m = len(full_data)
    if m == 0:
        raise ValueError("原始数据为空")
    K = math.floor(ratio * m)

    # 2. 读取每个维度的简化文件
    # 假设文件名形如 sim_<dim>，<dim> 是数字
    dim_files = sorted(
        [fn for fn in os.listdir(sim_dir)
         if fn.startswith('sim_') and fn.split('_')[-1].isdigit()],
        key=lambda fn: int(fn.split('_')[-1])
    )
    n = len(dim_files)

    dim_values = []   # list of dict: ts -> val_str
    dim_weights = []  # list of dict: ts -> weight (float)
    for fn in dim_files:
        vals = {}
        wts  = {}
        with open(os.path.join(sim_dir, fn), 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # 期望每行至少三列：value, timestamp, weight
                if len(parts) < 3:
                    continue
                val_s, time_s, wt_s = parts
                ts = float(time_s)
                vals[ts] = val_s
                wts[ts]  = float(wt_s)
        dim_values.append(vals)
        dim_weights.append(wts)

    # 3. 统计每个 timestamp 上出现的频次和权重总和
    ts_set     = set(full_data.keys())
    freq       = {ts: 0   for ts in ts_set}
    weight_sum = {ts: 0.0 for ts in ts_set}

    for d in range(n):
        for ts, w in dim_weights[d].items():
            if ts in freq:
                freq[ts]       += 1
                weight_sum[ts] += w

    # 4a. Phase1：按频次从高到低，累积选点，直到再加一整层会超过 K 时停止
    groups = defaultdict(list)   # freq -> list of ts
    for ts, f in freq.items():
        groups[f].append(ts)

    freq_levels = sorted(groups.keys(), reverse=True)
    selected1 = []
    for f in freq_levels:
        candidates = sorted(groups[f])
        if len(selected1) + len(candidates) <= K:
            selected1.extend(candidates)
        else:
            break
    k1 = len(selected1)

    # 4b. Phase2：如果 k1 < K，从剩余点按权重降序再选
    selected = list(selected1)
    if k1 < K:
        remain = [ts for ts in ts_set if ts not in selected1]
        remain.sort(key=lambda ts: (-weight_sum[ts], ts))
        selected2 = remain[: (K - k1)]
        selected.extend(selected2)

    # 5. 最终结果按时间戳升序输出
    selected = sorted(selected)

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as fout:
        fout.write(f"{len(selected)} {n}\n")
        for ts in selected:
                    # 直接输出 full_data 中整行原始数据
                    row = full_data[ts]
                    fout.write(" ".join(row) + "\n")

    print(f"Merged {len(selected)}/{m} pts; target ≤ {K} (r={ratio})")

if __name__ == "__main__":

    ratio = input("请输入比率 ratio（浮点数，默认 0.1）：").strip()
    ratio = float(ratio) if ratio else 0.1

    file_prefix = input("请输入文件路径前缀（默认 '6980-main/'）：").strip()
    if not file_prefix:
        # file_prefix = '6980-main/'
        file_prefix = './'
    
    # 取第一个非隐藏文件
    orig_path = 'dataset/origin/'
    dir_path = file_prefix + orig_path
    orig_file = sorted(f for f in os.listdir(dir_path) if not f.startswith('.'))[0]
    print(f"文件名：{orig_file}")
    file_name = dir_path + orig_file

    traj_path = 'dataset/origin/'
    traj_file = file_prefix + traj_path + orig_file

    sim_dir1 ='dataset/dims/sim/'
    sim_dir = file_prefix + sim_dir1
    
    output_path = 'dataset/result/'
    output_file = file_prefix + output_path + orig_file

    input_file = os.path.join(traj_file)

        # <<< MOD: 这里新增 time_index 输入
    ti = input("请输入 timestamp 所在列（0-based，默认最后一列）：").strip()
    if ti:
        time_index = int(ti)
    else:
        # 列数 = n_dims + 1, header 中 n_dims 已在 split 里读取，但此处暂用行长度推断
        sample = open(input_file, 'r', encoding='utf-8').readline().split()
        time_index = len(sample) - 1
        print(time_index)

    merge_dim_results(sim_dir,traj_file, output_file,ratio,time_index)
    
'''
python merge_dim_results.py \
    /path/to/sim_dir \
    /path/to/full_data.txt \
    /path/to/output.txt \
    0.2
'''
