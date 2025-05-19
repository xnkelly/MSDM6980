#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拆分包含 n 维数据 + 时间戳 的文件，
将每个维度与时间戳组合，分别写入单独文件。
文件按维度顺序命名为 1、2…dims
"""

import os
import argparse

def split_and_norm_dims(in_path: str, out_dir: str, norm_dir: str, time_index: int) -> None:
    # 1. 读入所有非空行
    with open(in_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError("输入文件为空！")

    # 2. 处理 header（第一行）, 格式：<data_count> <dims>
    header = lines[0].split()
    if len(header) < 2:
        raise ValueError("第一行格式错误，需 包含 数据量 和 维度数！")
    n_dims = int(header[1])  # <<< MOD: 从 header 读取维度数
    # 3. 数据行
    rows = [line.split() for line in lines[1:]]  # <<< MOD: 跳过首行

    # 4. 计算每个维度的 min/max
    mins = [float('inf')] * n_dims
    maxs = [float('-inf')] * n_dims
    for toks in rows:
        for i in range(n_dims):
            # 如果当前要处理的维度序号 i ≥ time_index，就跳过 time_index 列
            col = i if i < time_index else i+1
            val = float(toks[col])
            mins[i] = min(mins[i], val)
            maxs[i] = max(maxs[i], val)
    
    # for i in range(n_dims):
    #     print(mins[i], maxs[i])

    # 5. 准备输出目录和文件句柄
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)
    raw_fhs  = []
    norm_fhs = []
    for i in range(n_dims):
        raw_fhs.append(open(os.path.join(out_dir,  str(i)), 'w', encoding='utf-8'))
        norm_fhs.append(open(os.path.join(norm_dir, str(i)), 'w', encoding='utf-8'))

    # 6. 写入：原始 + 归一化，使用给定的 time_index
    for toks in rows:
        seq = toks[time_index]  # <<< MOD: 用输入的 time_index 而非固定最后一列
        for i in range(n_dims):
            # 跳过 time_index
            col = i if i < time_index else i+1
            raw_val = toks[col]
            raw_fhs[i].write(f"{raw_val} {seq}\n")
            lo, hi = mins[i], maxs[i]
            x = float(raw_val)
            x_norm = (x - lo)/(hi - lo) if hi != lo else 0.0
            norm_fhs[i].write(f"{x_norm:.6f} {seq}\n")

    # 7. 关闭句柄
    for fh in raw_fhs + norm_fhs:
        fh.close()

def main():
    #file_prefix = input("请输入文件路径前缀（默认 '6980-main'）：").strip() or '6980-main/'
    file_prefix = input("请输入文件路径前缀（默认 './'）：").strip() or './'
    input_file1 = input("请输入轨迹数据文件路径（默认 'dataset/origin/'）：").strip() or 'dataset/origin/'

    dir_path    = file_prefix + input_file1
    output_dir  = file_prefix + 'dataset/dims/'
    norm_dir    = file_prefix + 'dataset/norm_dims/'

    # 取第一个非隐藏文件
    file_name = sorted(f for f in os.listdir(dir_path) if not f.startswith('.'))[0]
    input_file = os.path.join(dir_path, file_name)
    print(f"文件名：{file_name}")

    # <<< MOD: 这里新增 time_index 输入
    ti = input("请输入 timestamp 所在列（0-based，默认最后一列）：").strip()
    if ti:
        time_index = int(ti)
    else:
        # 列数 = n_dims + 1, header 中 n_dims 已在 split 里读取，但此处暂用行长度推断
        sample = open(input_file, 'r', encoding='utf-8').readline().split()
        time_index = len(sample) - 1

    split_and_norm_dims(input_file, output_dir, norm_dir, time_index)
    print(f"已处理：{input_file}")

if __name__ == "__main__":
    main()
