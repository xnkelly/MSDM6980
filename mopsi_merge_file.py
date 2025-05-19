import os
from pathlib import Path

def is_text_file(fp: Path, blocksize: int = 512) -> bool:
    """简单探测前 blocksize 字节有没有 NUL 字节，判断是否文本文件。"""
    try:
        chunk = fp.open('rb').read(blocksize)
        return b'\x00' not in chunk
    except:
        return False

def merge_files_in_dir(src_dir: Path, out_file: Path, time_index: int):
    """
    将 src_dir 目录下所有文本文件合并到 out_file 中。
    第一行写 <data_count> <data_dim>，之后按 time_index 列升序写数据行。
    """
    files = sorted([p for p in src_dir.rglob('*') if p.is_file() and is_text_file(p)])
    if not files:
        print(f"[WARN] {src_dir} 下无可合并的文本文件，跳过。")
        return

    data_rows = []
    num_cols = None

    for fp in files:
        with fp.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if num_cols is None:
                    num_cols = len(parts)
                elif len(parts) != num_cols:
                    # 列数不匹配的行直接跳过
                    continue
                data_rows.append(parts)

    if not data_rows:
        print(f"[WARN] {src_dir} 下无合法数据行，跳过。")
        return

    data_dim = num_cols - 1
    # 排序
    try:
        data_rows.sort(key=lambda row: int(row[time_index]))
    except Exception as e:
        raise ValueError(f"在 {src_dir} 按列 {time_index} 排序失败: {e}")

    total = len(data_rows)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open('w', encoding='utf-8') as fout:
        fout.write(f"{total} {data_dim}\n")
        for row in data_rows:
            #print(row[time_index])
            fout.write(" ".join(row) + "\n")
    print(f"Merged {src_dir} → {out_file} ({total} rows, dim={data_dim})")

def main(input_dir: str, output_dir: str, time_index: int):
    inp = Path(input_dir)
    outp = Path(output_dir)

    if not inp.is_dir():
        raise ValueError(f"{input_dir!r} 不是目录")

    for sub in sorted(inp.iterdir()):
        if not sub.is_dir():
            continue
        out_file = outp / f"{sub.name}.txt"
        merge_files_in_dir(sub, out_file, time_index)

if __name__ == "__main__":
    file_prefix1 = "./"
    # file_prefix1 = "6980-main/"
    file_path = "dataset1/MopsiRoutes"
    file_prefix = input("请输入文件路径：").strip() or file_prefix1
    input_dir = file_prefix1+ file_path
    output_file = file_prefix1 + "dataset/origin/"

    ti = input("请输入 timestamp 所在列（0-based，默认最后一列）：").strip()
    if ti:
        time_index = int(ti) 
    else:
        time_index = 1
    main(input_dir, output_file, time_index)
