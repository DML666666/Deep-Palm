#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import torch
import pandas as pd
import argparse
from tqdm import tqdm

CONFIG = {
    "CSV": "/input.csv",
    "OUT": "/embedding.h5",

   
    "ID_COL": "ID",   
    "SEQ_COL": "Window",      
    "COL3_COL": "lab",       

    "COL3_PLACEHOLDER": "",  

    "MODEL_PATH": "/esm2_t36_3B_UR50D.pt",
    "MODEL_ID": "esm2_t36_3B_UR50D",
    "DEVICE": "cuda",
    "BATCH_SIZE": 8,
    "LAYERS": "last4",       
    "CENTER_INDEX": 15,      
    "STORE_FP16": True,
    "SAVE_CLS": True,
    "SAVE_MASK_STATS": True,
    "PRINT_CONFIG": True,
    "CSV_ENCODING": "utf-8-sig",
}

def load_model_offline(model_id: str, model_path: str, device: str):
    """
    离线路径加载 ESM2 模型（兼容 PyTorch 2.6 weights_only=True 默认行为）
    """
    import esm

    print(f"[信息] 离线加载 ESM 模型: {model_id}")
    print(f"[信息] 权重文件: {model_path}")

    if getattr(torch, "serialization", None) is not None and hasattr(torch.serialization, "safe_globals"):
        try:
            with torch.serialization.safe_globals([argparse.Namespace]):
                _ = torch.load(model_path, map_location="cpu", weights_only=True)
        except Exception:
            _ = torch.load(model_path, map_location="cpu")
    else:
        _ = torch.load(model_path, map_location="cpu")

    if hasattr(esm.pretrained, "load_model_and_alphabet_local"):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        core_fn = esm.pretrained.load_model_and_alphabet_core
        try:
            model, alphabet = core_fn(model_name, _, None)
        except TypeError:
            model, alphabet = core_fn(model_name, _)

    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model = model.to(device)
    return model, alphabet, batch_converter

def clean_seq(s: str) -> str:
    s = (s or "").strip().upper()
    allowed = set(list("ACDEFGHIKLMNPQRSTVWY") + ["B", "Z", "X", "U", "O"])
    return "".join(ch if ch in allowed else "X" for ch in s)

def ensure_len_31(s: str) -> str:
    if len(s) == 31:
        return s
    if len(s) > 31:
        mid = len(s) // 2
        start = max(0, mid - 15)
        s = s[start:start + 31]
        if len(s) != 31:
            s = s[:31]
        return s
    pad_left = (31 - len(s)) // 2
    pad_right = 31 - len(s) - pad_left
    return "X" * pad_left + s + "X" * pad_right

def get_layer_ids(model_id: str, mode: str):
    if model_id.startswith("esm2_t36"):
        last = 36
    elif "t33_" in model_id:
        last = 33
    elif "t30_" in model_id:
        last = 30
    elif "t12_" in model_id:
        last = 12
    else:
        last = 36
    return [last] if mode == "last" else [last - 3, last - 2, last - 1, last]

def _safe_str_series(s: pd.Series, placeholder: str = ""):
    # 保留“原始信息”的最佳近似：NaN -> placeholder，其余用 str()
    return [placeholder if pd.isna(x) else str(x) for x in s.tolist()]

def main():
    cfg = CONFIG
    if cfg["PRINT_CONFIG"]:
        print("========== 配置 ==========")
        for k, v in cfg.items():
            if k != "PRINT_CONFIG":
                print(f"{k}: {v}")
        print("==========================")

    # 1) 读取 CSV（自动推断分隔符）
    df = pd.read_csv(cfg["CSV"], sep=None, engine="python", encoding=cfg["CSV_ENCODING"])
    raw_cols = list(df.columns)
    df.columns = [str(c).strip() for c in df.columns]
    print("CSV 列名（原始）:", raw_cols)
    print("CSV 列名（清洗后）:", list(df.columns))

    cols_lower = [c.lower() for c in df.columns]

    def get_col_required(colname: str, fallback_idx: int):
        if colname and colname.lower() in cols_lower:
            return df.columns[cols_lower.index(colname.lower())]
        if fallback_idx < len(df.columns):
            fallback_col = df.columns[fallback_idx]
            print(f"[警告] 未找到列名 '{colname}'，使用第 {fallback_idx+1} 列 '{fallback_col}'")
            return fallback_col
        raise RuntimeError(f"CSV 列数不足，无法获取第 {fallback_idx+1} 列。")

    def get_col_optional(colname: str, fallback_idx: int):
        if colname and colname.lower() in cols_lower:
            return df.columns[cols_lower.index(colname.lower())]
        if fallback_idx < len(df.columns):
            return df.columns[fallback_idx]
        return None

    # 按你的要求：第1列保留，第2列 embedding，第3列保留（可缺失）
    id_col = get_col_required(cfg.get("ID_COL", ""), 0)
    seq_col = get_col_required(cfg.get("SEQ_COL", ""), 1)
    col3_col = get_col_optional(cfg.get("COL3_COL", ""), 2)

    ids = _safe_str_series(df[id_col], placeholder="")
    seq_raw = _safe_str_series(df[seq_col], placeholder="")
    seq_31 = [ensure_len_31(clean_seq(s)) for s in seq_raw]

    if col3_col is None:
        col3_raw = [cfg["COL3_PLACEHOLDER"]] * len(df)
        print("[信息] CSV 不存在第3列，已用占位符填充 col3_raw。")
    else:
        col3_raw = _safe_str_series(df[col3_col], placeholder=cfg["COL3_PLACEHOLDER"])

    # 2) 加载模型
    model, alphabet, batch_converter = load_model_offline(cfg["MODEL_ID"], cfg["MODEL_PATH"], cfg["DEVICE"])
    layer_ids = get_layer_ids(cfg["MODEL_ID"], cfg["LAYERS"])

    # 3) 推断 embedding 维度 D
    with torch.no_grad():
        _, _, tokens = batch_converter([("tmp", "A" * 31)])
        tokens = tokens.to(cfg["DEVICE"])
        out = model(tokens, repr_layers=[layer_ids[-1]], return_contacts=False)
        D = out["representations"][layer_ids[-1]].shape[-1]

    # 4) 写 H5
    os.makedirs(os.path.dirname(cfg["OUT"]), exist_ok=True)
    use_fp16 = bool(cfg["STORE_FP16"])
    h5_dtype = "f2" if use_fp16 else "f4"
    N = len(seq_31)

    with h5py.File(cfg["OUT"], "w") as h5:
        # 只对第二列（seq_31）产生 embedding
        d_win = h5.create_dataset("window_emb", shape=(N, 31, D), dtype=h5_dtype)
        d_cls = h5.create_dataset("cls", shape=(N, D), dtype=h5_dtype) if cfg["SAVE_CLS"] else None
        d_logpC = h5.create_dataset("mask_logp_C", shape=(N,), dtype="f4") if cfg["SAVE_MASK_STATS"] else None
        d_entropy = h5.create_dataset("mask_entropy", shape=(N,), dtype="f4") if cfg["SAVE_MASK_STATS"] else None

        str_dt = h5py.string_dtype(encoding="utf-8")

        # 第1列：原样保留（用于索引/对齐）
        h5.create_dataset("uniprotid", data=ids, dtype=str_dt)

        # 第2列：原始与 31aa 版本都保留；embedding 基于 seq_31
        h5.create_dataset("seq_raw", data=seq_raw, dtype=str_dt)
        h5.create_dataset("seq", data=seq_31, dtype=str_dt)

        # 第3列：原样保留（可缺失）
        h5.create_dataset("col3_raw", data=[str(x) for x in col3_raw], dtype=str_dt)
        h5.attrs["col3_src_colname"] = (str(col3_col) if col3_col is not None else "")

        # 元信息
        h5.attrs["model_id"] = cfg["MODEL_ID"]
        h5.attrs["model_path"] = cfg["MODEL_PATH"] or ""
        h5.attrs["layers"] = cfg["LAYERS"]
        h5.attrs["center_index"] = int(cfg["CENTER_INDEX"])
        h5.attrs["store_fp16"] = int(use_fp16)
        h5.attrs["src_id_colname"] = str(id_col)
        h5.attrs["src_seq_colname"] = str(seq_col)

        B = int(cfg["BATCH_SIZE"])
        for i in tqdm(range(0, N, B), desc="提取 ESM2 特征（只 embedding 第2列）"):
            batch_pairs = [(ids[j], seq_31[j]) for j in range(i, min(i + B, N))]
            _, _, tokens = batch_converter(batch_pairs)
            tokens = tokens.to(cfg["DEVICE"], non_blocking=True)

            with torch.no_grad():
                out = model(tokens, repr_layers=layer_ids, return_contacts=False)
                if len(layer_ids) == 1:
                    rep = out["representations"][layer_ids[0]]
                else:
                    reps = [out["representations"][lid] for lid in layer_ids]
                    rep = torch.stack(reps, dim=0).mean(0)

                rep_core = rep[:, 1:1 + 31, :]  # 只取 31aa 窗口
                if use_fp16:
                    rep_core = rep_core.to(torch.float16)
                d_win[i:i + len(batch_pairs)] = rep_core.cpu().numpy()

                if d_cls is not None:
                    cls_vec = rep[:, 0, :]
                    if use_fp16:
                        cls_vec = cls_vec.to(torch.float16)
                    d_cls[i:i + len(batch_pairs)] = cls_vec.cpu().numpy()

                if d_logpC is not None and d_entropy is not None:
                    center_tok = 1 + int(cfg["CENTER_INDEX"])
                    tokens_mask = tokens.clone()
                    tokens_mask[:, center_tok] = alphabet.mask_idx

                    out_mask = model(tokens_mask, return_contacts=False)
                    logits = out_mask["logits"]
                    log_probs = torch.log_softmax(logits, dim=-1)

                    idx_C = alphabet.tok_to_idx.get("C", None)
                    if idx_C is None:
                        raise RuntimeError("词表中未找到 'C' 的 token。")

                    logpC = log_probs[:, center_tok, idx_C]
                    probs = torch.exp(log_probs[:, center_tok, :])
                    entropy = -(probs * log_probs[:, center_tok, :]).sum(dim=-1)

                    d_logpC[i:i + len(batch_pairs)] = logpC.detach().float().cpu().numpy()
                    d_entropy[i:i + len(batch_pairs)] = entropy.detach().float().cpu().numpy()

    print(f"[完成] 已保存到：{cfg['OUT']}")

if __name__ == "__main__":
    main()
