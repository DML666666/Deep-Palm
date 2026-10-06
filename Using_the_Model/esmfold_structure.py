#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as p
import sys
import csv
import time
import math
import multiprocessing as mp

import torch
from esm import pretrained

# ================== 这里改路径 ==================
# 输入 CSV/TSV：第一列 ID，第二列 seq，其它列忽略
INPUT_CSV = r"/input.tsv"

# 输出目录：所有 PDB 都写在这里（是“目录”不是 .h5 文件）
OUTPUT_DIR = r"/esmfold.pdb"

# 使用多少个进程并行（建议 2~4，根据显存调整）
NUM_WORKERS = 4
# ============================================


# ---- 工具：把列表平均拆成 n 份 ----
def chunk_list(lst, n):
    n = max(1, min(n, len(lst)))
    k = int(math.ceil(len(lst) / float(n)))
    return [lst[i * k:(i + 1) * k] for i in range(n) if lst[i * k:(i + 1) * k]]


def detect_delimiter(path):
    """自动识别 CSV(逗号) / TSV(tab) 分隔符"""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
    return "\t" if sample.count("\t") > sample.count(",") else ","


# ---- 子进程执行函数 ----
def worker_proc(worker_id, rows_chunk, id_col, seq_col, total_rows):
    """
    rows_chunk: 当前进程需要处理的那一部分 row（是列表，每个元素是 dict）
    """
    torch.set_grad_enabled(False)

    print(f"[Worker {worker_id}] start, {len(rows_chunk)} rows")

    # 每个进程各自加载一份模型
    model = pretrained.esmfold_v1()
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # 只有一张卡的话固定到 0
        model = model.cuda()
    model.eval()

    for idx_local, row in enumerate(rows_chunk, start=1):
        uid = (row.get(id_col) or "").strip()
        seq = (row.get(seq_col) or "").strip()

        if not uid or not seq:
            print(f"[Worker {worker_id}] skip empty id/seq")
            continue

        out_path = p.join(OUTPUT_DIR, f"{uid}.pdb")
        if p.exists(out_path):
            continue

        print(
            f"[Worker {worker_id}] Folding {uid} "
            f"(len={len(seq)})  [{idx_local}/{len(rows_chunk)}]"
        )
        t0 = time.time()
        try:
            pdb_str = model.infer_pdb(seq)
            with open(out_path, "w") as f:
                f.write(pdb_str)
            dt = time.time() - t0
            print(
                f"[Worker {worker_id}] saved {uid}.pdb "
                f"({dt:.2f} s)  total_progress≈{idx_local * 1.0 / total_rows:.2%}"
            )
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR {uid}: {e}")

    print(f"[Worker {worker_id}] done.")


def main():
    print("=== ESMFold CSV/TSV batch run (multi-process) ===")
    print("Python:", sys.version.split()[0])
    print("Torch : ", torch.__version__)
    print("CUDA  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU   :", torch.cuda.get_device_name(0))
    print("-" * 50)
    print("Input file :", INPUT_CSV)
    print("Output dir :", OUTPUT_DIR)

    # ---------- 检查输入 / 输出 ----------
    if not p.exists(INPUT_CSV):
        print("[ERROR] 找不到输入文件:", INPUT_CSV)
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 读 CSV/TSV，自动识别列名 ----------
    print("[1/3] Loading CSV/TSV ...")

    delimiter = detect_delimiter(INPUT_CSV)
    print(f"    Detected delimiter: {repr(delimiter)}")

    # 用 utf-8-sig 自动去掉 BOM
    with open(INPUT_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []
        print("    Columns:", fieldnames)

        ID_CANDIDATES = ["uniprotid-pos", "uniproid-pos",
                         "uniprotid-pos_seq", "ID", "uniprotid"]
        SEQ_CANDIDATES = ["sequence", "seq", "Window"]

        id_col = None
        seq_col = None
        for name in fieldnames:
            if id_col is None and name in ID_CANDIDATES:
                id_col = name
            if seq_col is None and name in SEQ_CANDIDATES:
                seq_col = name

        if id_col is None or seq_col is None:
            print("[ERROR] 找不到 ID 或序列列名。")
            print("       期望 ID 列之一 :", ID_CANDIDATES)
            print("       期望 序列列之一:", SEQ_CANDIDATES)
            print("       实际列名      :", fieldnames)
            sys.exit(1)

        print(f"    使用 ID 列   : {id_col}")
        print(f"    使用 序列列 : {seq_col}")

        rows = list(reader)

    n_rows = len(rows)
    print("    Total rows:", n_rows)
    if n_rows == 0:
        print("[WARN] 文件没有数据行。")
        return

    # 过滤掉已经存在 PDB 的行（避免重复计算）
    todo_rows = []
    for r in rows:
        uid = (r.get(id_col) or "").strip()
        seq = (r.get(seq_col) or "").strip()
        if not uid or not seq:
            continue
        out_path = p.join(OUTPUT_DIR, f"{uid}.pdb")
        if not p.exists(out_path):
            todo_rows.append(r)

    if not todo_rows:
        print("[INFO] 所有条目都已经有 PDB，直接退出。")
        return

    print(f"[INFO] 需要实际折叠的条目数: {len(todo_rows)}")

    # ---------- 多进程切分 ----------
    chunks = chunk_list(todo_rows, NUM_WORKERS)
    print(f"[2/3] Using {len(chunks)} workers, "
          f"each ~{len(todo_rows) // max(1, len(chunks))} rows")

    procs = []
    total_rows = len(todo_rows)
    for w_id, rows_chunk in enumerate(chunks, start=1):
        p_proc = mp.Process(
            target=worker_proc,
            args=(w_id, rows_chunk, id_col, seq_col, total_rows)
        )
        p_proc.start()
        procs.append(p_proc)

    # 等所有进程结束
    for p_proc in procs:
        p_proc.join()

    print("=== All done ===")


if __name__ == "__main__":
    # 对于多进程 + CUDA，推荐 spawn，避免奇怪的 fork 问题
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()