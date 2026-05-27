#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deep-Palm prediction script.

This script is a prediction-only entry point derived from the evaluation script.
It loads the trained Deep-Palm checkpoint and precomputed features for candidate
cysteine-centered windows, and outputs Deep-Palm prediction scores.

Expected default repository layout:
    Deep-Palm.pth
    kmer_vocab.json
    predict.py
    input/
        input.csv
        embedding.h5
        aaindex1.txt
        esmfold.pdb/

The ID values in input/input.csv must match the entries in embedding.h5 and the
PDB filenames in input/esmfold.pdb. For new samples, first generate the
corresponding ESM embeddings and predicted structures using the feature scripts.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBParser, Polypeptide

BASE_DIR = Path(__file__).resolve().parent

CONFIG = {
    # Default input / output paths relative to this script.
    "INPUT_CSV": str(BASE_DIR / "input" / "input.csv"),
    "OUTPUT_CSV": str(BASE_DIR / "prediction_result.csv"),
    "FUSED_CKPT": str(BASE_DIR / "Deep-Palm.pth"),

    # Precomputed feature sources matching INPUT_CSV.
    "ESM_H5": str(BASE_DIR / "input" / "embedding.h5"),
    "PDB_DIR": str(BASE_DIR / "input" / "esmfold.pdb"),
    "AAINDEX1_PATH": str(BASE_DIR / "input" / "aaindex1.txt"),
    "KMER_VOCAB_JSON": str(BASE_DIR / "kmer_vocab.json"),

    # Retained as a fallback only when no k-mer vocabulary JSON is provided.
    "TRAIN_CSV_FOR_VOCAB": "",
    "KMERS": [2, 3, 4],
    "TOP_KMER_K5": 200000,
    "TOP_KMER_K4": None,

    # Runtime settings.
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 0,  # Windows-safe default; increase if needed on Linux.
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "STRICT": True,
    "THRESHOLD": 0.5,
}

# ================
# 1) Small utils
# ================
MAX_LEN = 31
CONTACT_THRESH = 8.0

AA_INDEX_ORDER = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]
AA_3_TO_1 = Polypeptide.protein_letters_3to1

AAINDEX_IDS = [
    "KYTJ820101",
    "GRAR740102",
    "GRAR740103",
    "KARP850101",
    "CHOP780201",
    "CHOP780202",
    "ZIMJ680104",
    "HOPT810101",
    "CHOC760101",
    "VINM940101",
    "PUNT030101",
    "CHOP780203",
    "RACS770102",
    "MIYS990101",
]
SEQ_FEAT_DIM = len(AAINDEX_IDS)


def clean_seq(s: str) -> str:
    s = (s or "").strip().upper()
    allow = set("ACDEFGHIKLMNPQRSTVWY*")
    return "".join(ch if ch in allow else "*" for ch in s)


def ensure_len_31(s: str) -> str:
    s = clean_seq(s)
    if len(s) == 31:
        return s
    if len(s) > 31:
        mid = len(s) // 2
        st = max(0, mid - 15)
        s = s[st:st + 31]
        return s[:31]
    lpad = (31 - len(s)) // 2
    rpad = 31 - len(s) - lpad
    return "*" * lpad + s + "*" * rpad


def _base_uid(uid: str) -> str:
    return str(uid).strip().split("-")[0]


def _strip_uid_suffix(uid: str) -> str:
    return _base_uid(uid)


def detect_sep(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(8192)
    return "\t" if sample.count("\t") > sample.count(",") else ","


def pick_columns(df: pd.DataFrame):
    """提取 ID, seq, label 以及所有的 WT 列"""
    cols = [c.strip() for c in df.columns]
    low = [c.lower() for c in cols]

    def pick(target_names, default_idx):
        for t in target_names:
            if t in low:
                return cols[low.index(t)]
        if default_idx < len(cols):
            return cols[default_idx]
        return None

    idc = pick(["id", "uniprotid", "uid"], 0)
    seqc = pick(["window", "seq", "sequence"], 1)
    labc = pick(["lab", "label", "target", "y"], 2)  # optional
    
    # 提取所有WT列
    wt_cols = [c for c in cols if c.upper().startswith("WT")]
    
    return idc, seqc, labc, wt_cols


def move_to_device(obj: Any, device: str):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(x, device) for x in obj)
    return obj


# ==========================
# 2) H5 uid2idx for ESM
# ==========================
def build_uid2idx(h5_path: str) -> dict:
    import h5py
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"ESM_H5 not found: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        if "uniprotid" not in f:
            raise RuntimeError("H5 missing dataset 'uniprotid'")
        uids = f["uniprotid"][...]
    uids = [u.decode() if isinstance(u, (bytes, bytearray)) else str(u) for u in uids]
    return {u: i for i, u in enumerate(uids)}


def esm_channel_stats_for_uids(h5_path: str, uid2idx: dict, uids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    import h5py
    idxes = []
    for u in uids:
        if u in uid2idx:
            idxes.append(uid2idx[u])
        else:
            ub = _base_uid(u)
            if ub in uid2idx:
                idxes.append(uid2idx[ub])
    if not idxes:
        raise RuntimeError("No uids matched in H5 for ESM normalization.")
    idxes = np.array(idxes, dtype=np.int64)
    uniq_idx, counts = np.unique(idxes, return_counts=True)

    with h5py.File(h5_path, "r") as f:
        emb = f["window_emb"]
        D = emb.shape[-1]
        S = np.zeros(D, dtype=np.float64)
        SS = np.zeros(D, dtype=np.float64)
        n = 0
        for i, c in zip(uniq_idx, counts):
            x = emb[int(i)].astype(np.float32)  # (31, D)
            S += x.sum(axis=0) * c
            SS += (x ** 2).sum(axis=0) * c
            n += x.shape[0] * c
        mean = S / max(n, 1)
        var = SS / max(n, 1) - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


# ==========================
# 3) k-mer vocab
# ==========================
def kmer_tokens(seq: str, k: int) -> List[str]:
    seq = ensure_len_31(seq)
    return [seq[i:i + k] for i in range(0, len(seq) - k + 1)]


class KmerVocab:
    def __init__(self, k: int, max_size: Optional[int] = None):
        self.k = k
        self.max_size = max_size
        self.counts: Dict[str, int] = {}
        self.stoi: Dict[str, int] = {"<OOV>": 0}
        self.itos: List[str] = ["<OOV>"]

    def add_seq(self, seq: str):
        for tok in kmer_tokens(seq, self.k):
            self.counts[tok] = self.counts.get(tok, 0) + 1

    def finalize(self):
        items = sorted(self.counts.items(), key=lambda x: (-x[1], x[0]))
        if self.max_size is not None:
            items = items[: self.max_size]
        for tok, _ in items:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def encode(self, seq: str) -> List[int]:
        toks = kmer_tokens(seq, self.k)
        return [self.stoi.get(t, 0) for t in toks]

    @property
    def size(self):
        return len(self.itos)


def save_kmer_vocabs_json(path: str, kvoc: Dict[int, KmerVocab]):
    obj = {}
    for k, v in kvoc.items():
        obj[str(k)] = {"itos": v.itos}
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"[Save] kmer vocab json -> {path}", flush=True)


def load_kmer_vocabs_json(path: str) -> Dict[int, KmerVocab]:
    with open(path, "r") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "vocabs" in obj and isinstance(obj["vocabs"], dict):
        obj = obj["vocabs"]

    kvoc: Dict[int, KmerVocab] = {}
    for ks, data in obj.items():
        try:
            k = int(ks)
        except Exception:
            continue

        v = KmerVocab(k, max_size=None)

        if isinstance(data, dict) and "itos" in data:
            itos = data["itos"]
        elif isinstance(data, list):
            itos = data
        else:
            raise RuntimeError(
                f"Unrecognized vocab format for k={k}: type={type(data)} keys={getattr(data, 'keys', lambda: [])()}"
            )

        v.itos = itos
        v.stoi = {tok: i for i, tok in enumerate(v.itos)}
        kvoc[k] = v

    if not kvoc:
        raise RuntimeError(f"No valid k-mer vocabs loaded from {path}. Check json keys/format.")
    return kvoc


def build_kmer_vocabs_from_csv(train_csv: str) -> Dict[int, KmerVocab]:
    df = pd.read_csv(train_csv)
    idc, seqc, _, _ = pick_columns(df)
    if idc is None or seqc is None:
        raise RuntimeError("Cannot infer id/seq columns from TRAIN_CSV_FOR_VOCAB")

    seqs = [ensure_len_31(str(x)) for x in df[seqc].tolist()]
    kvoc: Dict[int, KmerVocab] = {}

    for k in CONFIG["KMERS"]:
        max_size = None
        if k == 5:
            max_size = CONFIG["TOP_KMER_K5"]
        if k == 4 and CONFIG["TOP_KMER_K4"] is not None:
            max_size = CONFIG["TOP_KMER_K4"]
        voc = KmerVocab(k, max_size=max_size)
        for s in tqdm(seqs, desc=f"Build k-mer vocab k={k}", total=len(seqs)):
            voc.add_seq(s)
        voc.finalize()
        kvoc[k] = voc
        print(f"[kmer] k={k} vocab_size={voc.size}", flush=True)

    return kvoc


# ==========================
# 4) AAindex features
# ==========================
def load_aaindex_entry(filepath: str, entry_id: str):
    vals = []
    in_entry = False
    reading_I_block = False
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("H "):
                in_entry = line[2:].strip().startswith(entry_id)
                reading_I_block = False
            elif in_entry and line.startswith("I"):
                reading_I_block = True
                continue
            elif in_entry and reading_I_block:
                if line.startswith("//") or line.startswith("H "):
                    break
                s = line.strip()
                if not s:
                    continue
                first_token = s.split()[0]
                if any(c.isalpha() or c == "/" for c in first_token):
                    continue
                for tok in s.split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
                if len(vals) >= 20:
                    break
    if len(vals) != 20:
        raise ValueError(f"{entry_id}: parsed {len(vals)} values, expected 20")
    return dict(zip(AA_INDEX_ORDER, vals))


def build_physico_chemical_features(aaindex_path: str):
    num_idx = len(AAINDEX_IDS)
    num_aa = len(AA_INDEX_ORDER)
    raw_matrix = np.zeros((num_idx, num_aa), dtype=np.float32)
    for i, entry_id in enumerate(AAINDEX_IDS):
        entry_dict = load_aaindex_entry(aaindex_path, entry_id)
        raw_matrix[i] = np.array([entry_dict[aa] for aa in AA_INDEX_ORDER], dtype=np.float32)

    means = raw_matrix.mean(axis=1, keepdims=True)
    stds = raw_matrix.std(axis=1, ddof=0, keepdims=True)
    stds[stds == 0] = 1.0
    z_matrix = (raw_matrix - means) / stds

    feat_dict = {aa: z_matrix[:, j].astype(np.float32).tolist() for j, aa in enumerate(AA_INDEX_ORDER)}
    feat_dict["*"] = [0.0] * num_idx
    feat_dict["X"] = feat_dict["*"][:]
    return feat_dict


# ==========================
# 5) PDB features for physchem/esmfold
# ==========================
parser = PDBParser(QUIET=True)
BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def load_residues_from_pdb(pdb_path: str):
    structure = parser.get_structure(os.path.basename(pdb_path), pdb_path)
    model = structure[0]
    residues = []
    for chain in model:
        for res in chain:
            if "CA" in res:
                residues.append(res)
    return residues


def get_ca_coord(residue) -> np.ndarray:
    return residue["CA"].get_coord().astype(np.float32)


def get_heavy_atom_coords(residue) -> np.ndarray:
    coords = []
    for atom in residue:
        name = atom.get_name()
        if not name.startswith("H"):
            coords.append(atom.get_coord())
    if not coords:
        coords.append(residue["CA"].get_coord())
    return np.array(coords, dtype=np.float32)


def get_sidechain_center(residue) -> np.ndarray:
    if "CB" in residue:
        return residue["CB"].get_coord().astype(np.float32)
    coords = []
    for atom in residue:
        name = atom.get_name()
        if not name.startswith("H") and name not in BACKBONE_ATOMS:
            coords.append(atom.get_coord())
    if coords:
        coords = np.stack(coords, axis=0)
        return coords.mean(axis=0).astype(np.float32)
    return get_ca_coord(residue)


def pairwise_dist_matrix(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return dist.astype(np.float32)


def compute_min_heavy_atom_dist(residues) -> np.ndarray:
    L = len(residues)
    heavy_coords = [get_heavy_atom_coords(r) for r in residues]
    mat = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        ci = heavy_coords[i]
        for j in range(L):
            cj = heavy_coords[j]
            diff = ci[:, None, :] - cj[None, :, :]
            d = np.linalg.norm(diff, axis=-1)
            mat[i, j] = d.min()
    return mat


def build_pair_feature_from_residues(residues, max_len: int = MAX_LEN) -> Tuple[np.ndarray, int]:
    residues = residues[:max_len]
    L = len(residues)
    if L == 0:
        raise ValueError("No residues with CA in this PDB")
    ca = np.stack([get_ca_coord(r) for r in residues], axis=0)
    sc = np.stack([get_sidechain_center(r) for r in residues], axis=0)
    dist_ca = pairwise_dist_matrix(ca)
    dist_sc = pairwise_dist_matrix(sc)
    dist_min = compute_min_heavy_atom_dist(residues)
    contact = (dist_ca < CONTACT_THRESH).astype(np.float32)
    idx = np.arange(L)
    seq_sep = np.abs(idx[:, None] - idx[None, :]).astype(np.float32)
    long_range = ((seq_sep >= 6) & (dist_ca < CONTACT_THRESH)).astype(np.float32)
    seq_sep_norm = seq_sep / (L - 1) if L > 1 else seq_sep

    feat = np.stack([dist_ca, dist_min, dist_sc, contact, long_range, seq_sep_norm], axis=0)  # (6,L,L)
    pair_feat = np.zeros((6, max_len, max_len), dtype=np.float32)
    pair_feat[:, :L, :L] = feat
    return pair_feat, L


def get_seq_feat(residues, feat_dict, max_len: int = MAX_LEN) -> Tuple[np.ndarray, int]:
    seq_feat = []
    residues = residues[:max_len]
    L = len(residues)
    for res in residues:
        res_name = res.get_resname().upper()
        aa_code = AA_3_TO_1.get(res_name, "*")
        seq_feat.append(feat_dict.get(aa_code, feat_dict["*"]))
    if L < max_len:
        seq_feat.extend([feat_dict["*"]] * (max_len - L))
    return np.array(seq_feat, dtype=np.float32)[:max_len], L


_PDB_UID_MAP_CACHE: Dict[str, Dict[str, str]] = {}


def build_pdb_uid_map(pdb_dir: str) -> Dict[str, str]:
    global _PDB_UID_MAP_CACHE
    if pdb_dir in _PDB_UID_MAP_CACHE:
        return _PDB_UID_MAP_CACHE[pdb_dir]
    if not os.path.exists(pdb_dir):
        raise FileNotFoundError(f"PDB_DIR not found: {pdb_dir}")

    mp: Dict[str, str] = {}
    if os.path.isdir(pdb_dir):
        for fn in os.listdir(pdb_dir):
            if not fn.lower().endswith(".pdb"):
                continue
            stem = fn[:-4]
            path = os.path.join(pdb_dir, fn)
            mp.setdefault(stem, path)
            mp.setdefault(_strip_uid_suffix(stem), path)
        print(f"[PDB] found mappings: {len(mp)}", flush=True)
    else:
        if not pdb_dir.lower().endswith(".pdb"):
            raise FileNotFoundError(f"PDB_DIR is neither dir nor .pdb: {pdb_dir}")
        stem = os.path.basename(pdb_dir)[:-4]
        mp.setdefault(stem, pdb_dir)
        mp.setdefault(_strip_uid_suffix(stem), pdb_dir)
        print(f"[PDB] single pdb -> mappings: {len(mp)}", flush=True)

    _PDB_UID_MAP_CACHE[pdb_dir] = mp
    return mp


# ==========================
# 6) Model definitions
# ==========================
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj_norm):
        h_prime = torch.bmm(adj_norm, x)
        h_prime = self.linear(h_prime)
        return F.relu(h_prime)


class AttentionReadout(nn.Module):
    def __init__(self, node_dim: int, attn_dim: int = 128):
        super().__init__()
        self.attn_layer = nn.Linear(node_dim, attn_dim)
        self.output_proj = nn.Linear(attn_dim, 1)
        nn.init.xavier_uniform_(self.output_proj.weight)
        self.output_proj.bias.data.fill_(0)

    def forward(self, h, mask):
        scores = torch.tanh(self.attn_layer(h))
        logits = self.output_proj(scores).squeeze(-1)
        mask_inf = (1.0 - mask) * -1e9
        masked_logits = logits + mask_inf
        attn_weights = F.softmax(masked_logits, dim=-1)
        g_attn = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)
        return g_attn


class BranchESMFold(nn.Module):
    def __init__(self, in_channels: int = 6, pair_hidden: int = 128, node_hidden: int = 256, gnn_layers: int = 2):
        super().__init__()
        self.contact_channel_index = 3
        self.pair_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, pair_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(pair_hidden), nn.ReLU(inplace=True),
        )
        self.node_proj = nn.Linear(pair_hidden * 2, node_hidden)
        self.gnn_layers = nn.ModuleList([GCNLayer(node_hidden, node_hidden) for _ in range(gnn_layers)])
        self.struct_attn_readout = AttentionReadout(node_dim=node_hidden)
        self.readout = nn.Sequential(
            nn.Linear(node_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def build_adj(self, pair_feat):
        device = pair_feat.device
        L = pair_feat.shape[-1]
        contact = pair_feat[:, self.contact_channel_index, :, :]
        adj = (contact > 0.5).float()
        eye = torch.eye(L, device=device).unsqueeze(0)
        A_tilde = adj + eye
        D_tilde = A_tilde.sum(dim=-1, keepdim=True).pow(-0.5)
        adj_norm = D_tilde * A_tilde * D_tilde.transpose(-2, -1)
        return (adj_norm + adj_norm.transpose(-2, -1)) / 2

    def forward(self, x_dict):
        pair_feat = x_dict["pair"]  # (B,6,L,L)
        valid_lengths = x_dict["length"]
        device = pair_feat.device
        L = pair_feat.shape[-1]
        idx = torch.arange(L, device=device).unsqueeze(0)
        mask = (idx < valid_lengths.unsqueeze(1)).float()

        h_pair = self.pair_cnn(pair_feat)  # (B,C',L,L)
        node = torch.cat([h_pair.mean(dim=3), h_pair.mean(dim=2)], dim=1)  # (B,2C',L)
        node = self.node_proj(node.permute(0, 2, 1))  # (B,L,node_hidden)

        adj_norm = self.build_adj(pair_feat)
        h_gnn = node
        for layer in self.gnn_layers:
            h_gnn = layer(h_gnn, adj_norm)
        g_struct = self.struct_attn_readout(h_gnn, mask)
        logits = self.readout(g_struct).squeeze(1).unsqueeze(-1)
        return logits


class BranchPhyschem(nn.Module):
    def __init__(self, seq_feat_dim: int = SEQ_FEAT_DIM, seq_hidden: int = 128):
        super().__init__()
        self.seq_proj = nn.Linear(seq_feat_dim, seq_hidden)
        self.lstm = nn.LSTM(seq_hidden, seq_hidden, batch_first=True, bidirectional=True)
        self.seq_attn_readout = AttentionReadout(node_dim=seq_hidden * 2)
        self.readout = nn.Sequential(
            nn.Linear(seq_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x_dict):
        seq_feat = x_dict["seq"]  # (B,L,D)
        valid_lengths = x_dict["length"]
        device = seq_feat.device
        L = seq_feat.shape[1]
        idx = torch.arange(L, device=device).unsqueeze(0)
        mask = (idx < valid_lengths.unsqueeze(1)).float()

        h_seq = F.relu(self.seq_proj(seq_feat))
        packed_input = nn.utils.rnn.pack_padded_sequence(
            h_seq, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        h_lstm, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=L)
        g_seq = self.seq_attn_readout(h_lstm, mask)
        logits = self.readout(g_seq).squeeze(1).unsqueeze(-1)
        return logits


class VConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        K = kernel_size
        L0 = torch.tensor(0.0)
        R0 = torch.tensor(float(K - 1))
        self.boundary = nn.Parameter(torch.stack([L0.repeat(out_channels), R0.repeat(out_channels)], dim=1))

    def _mask(self) -> torch.Tensor:
        w = self.conv.weight
        C_out, _, K = w.shape
        device = w.device
        idx = torch.arange(K, device=device).float()[None, None, :]
        L = self.boundary[:, 0].view(C_out, 1, 1)
        R = self.boundary[:, 1].view(C_out, 1, 1)
        s1 = torch.sigmoid(idx - L)
        s2 = torch.sigmoid(R - idx)
        m = torch.clamp(s1 + s2 - 1.0, 0.0, 1.0)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self._mask()
        w = self.conv.weight * m
        return F.conv1d(
            x, w, bias=self.conv.bias,
            stride=self.conv.stride, padding=self.conv.padding,
            dilation=self.conv.dilation, groups=self.conv.groups
        )


class ConvBlock(nn.Module):
    def __init__(self, C_in, C_out, K, dropout=0.3):
        super().__init__()
        pad = K // 2
        self.seq = nn.Sequential(
            VConv1d(C_in, C_out, K, padding=pad),
            nn.BatchNorm1d(C_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.seq(x)


class BranchESM(nn.Module):
    def __init__(self, D_in, hidden=256, ksize=7, n_layers=2, dropout=0.5, vconv_kernels=128):
        super().__init__()
        layers = []
        C = D_in
        for _ in range(n_layers):
            layers.append(ConvBlock(C, vconv_kernels, ksize, dropout))
            C = vconv_kernels
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(C, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h)


class KmerHead(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, ksize=5, n_layers=1, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers = []
        C = embed_dim
        for _ in range(n_layers):
            layers.append(ConvBlock(C, 64, ksize, dropout))
            C = 64
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(C, 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_ids):
        e = self.emb(x_ids).transpose(1, 2)  # (B,embed,L)
        h = self.backbone(e)
        return self.proj(h)


class BranchKmer(nn.Module):
    def __init__(self, vocab_sizes: Dict[int, int], embed_dim=32, dropout=0.4):
        super().__init__()
        self.ks = sorted(vocab_sizes.keys())
        self.heads = nn.ModuleDict({
            str(k): KmerHead(vocab_sizes[k], embed_dim, ksize=max(3, k), n_layers=1, dropout=dropout)
            for k in self.ks
        })
        in_dim = 128 * len(self.ks)
        self.out = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x_dict):
        feats = [self.heads[str(k)](x_dict[k]) for k in self.ks]
        h = torch.cat(feats, dim=1)
        return self.out(h)


class FourWayBlend(nn.Module):
    def __init__(self, branch_models: Dict[str, nn.Module], weights: np.ndarray):
        super().__init__()
        self.branch_names = list(branch_models.keys())
        self.branches = nn.ModuleDict(branch_models)

        w_np = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert w_np.shape[0] == len(self.branch_names), "weights length mismatch branches"
        w = torch.from_numpy(w_np).view(1, -1, 1)  # (1,K,1)
        self.register_buffer("w", w)

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        probs = []
        for name in self.branch_names:
            m = self.branches[name]
            p = torch.sigmoid(m(batch[name]))
            probs.append(p)
        P = torch.stack(probs, dim=1)  # (B,K,1)
        out = (P * self.w).sum(dim=1)  # (B,1)
        return out


class FiveWayStack(nn.Module):
    def __init__(self, branch_models: Dict[str, nn.Module], meta_state_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.branch_names = list(branch_models.keys())
        self.branches = nn.ModuleDict(branch_models)
        self.meta = nn.Linear(len(self.branch_names), 1)
        if meta_state_dict is not None:
            self.meta.load_state_dict(meta_state_dict)

    @torch.no_grad()
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        probs = []
        for name in self.branch_names:
            m = self.branches[name]
            p = torch.sigmoid(m(batch[name]))
            probs.append(p)
        feats = torch.cat(probs, dim=1)  # (B,K)
        logit = self.meta(feats)
        return torch.sigmoid(logit)


# ==========================
# 7) Dataset
# ==========================
@dataclass
class InferSample:
    uid: str
    seq: str


class MultiBranchInferDataset(Dataset):
    def __init__(
        self,
        samples: List[InferSample],
        branches: List[str],
        esm_h5: str,
        pdb_dir: str,
        aaindex_path: str,
        kmer_vocabs: Optional[Dict[int, KmerVocab]],
        esm_norm: Optional[Tuple[np.ndarray, np.ndarray]],
        strict: bool = True,
    ):
        self.samples = samples
        self.branches = branches
        self.strict = strict

        self.esm_h5_path = esm_h5
        self.uid2idx = build_uid2idx(esm_h5) if "esm" in branches else None
        self._h5 = None

        self.pdb_dir = pdb_dir
        self.uid2pdb = build_pdb_uid_map(pdb_dir) if (("physchem" in branches) or ("esmfold" in branches)) else None

        self.kvoc = kmer_vocabs
        self.esm_mean = None
        self.esm_std = None
        if esm_norm is not None:
            self.esm_mean, self.esm_std = esm_norm

        self.feat_dict = None
        if "physchem" in branches:
            if not os.path.exists(aaindex_path):
                raise FileNotFoundError(f"AAINDEX1_PATH not found: {aaindex_path}")
            self.feat_dict = build_physico_chemical_features(aaindex_path)

        self._validate_and_report()

    def _open_h5(self):
        import h5py
        if self._h5 is None:
            self._h5 = h5py.File(self.esm_h5_path, "r")
        return self._h5

    def _find_esm_index(self, uid: str) -> Optional[int]:
        if self.uid2idx is None:
            return None
        if uid in self.uid2idx:
            return self.uid2idx[uid]
        ub = _base_uid(uid)
        return self.uid2idx.get(ub)

    def _find_pdb_path(self, uid: str) -> Optional[str]:
        if self.uid2pdb is None:
            return None
        p = self.uid2pdb.get(uid)
        if p is None:
            p = self.uid2pdb.get(_strip_uid_suffix(uid))
        return p

    def _validate_and_report(self):
        n = len(self.samples)
        miss_esm = 0
        miss_pdb = 0
        for s in self.samples:
            if "esm" in self.branches:
                if self._find_esm_index(s.uid) is None:
                    miss_esm += 1
            if ("physchem" in self.branches) or ("esmfold" in self.branches):
                if self._find_pdb_path(s.uid) is None:
                    miss_pdb += 1

        print(f"[Infer] samples={n}", flush=True)
        if "esm" in self.branches:
            print(f"[Infer] missing ESM embeddings: {miss_esm}/{n}", flush=True)
        if ("physchem" in self.branches) or ("esmfold" in self.branches):
            print(f"[Infer] missing PDB files: {miss_pdb}/{n}", flush=True)

        if self.strict:
            if ("esm" in self.branches) and miss_esm > 0:
                raise RuntimeError("STRICT=True but some samples have no ESM embedding in H5. Fix IDs or set STRICT=False.")
            if (("physchem" in self.branches) or ("esmfold" in self.branches)) and miss_pdb > 0:
                raise RuntimeError("STRICT=True but some samples have no PDB file. Fix PDB names or set STRICT=False.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        out: Dict[str, Any] = {}

        # ---- ESM ----
        if "esm" in self.branches:
            j = self._find_esm_index(s.uid)
            if j is None:
                if self.strict:
                    raise RuntimeError(f"[esm] uid not found in H5: {s.uid}")
                return {"uid": s.uid, "seq": s.seq, "x": None}

            h5 = self._open_h5()
            x = h5["window_emb"][int(j)].astype(np.float32)  # (31, D)
            x = x.T  # (D, 31)

            if self.esm_mean is not None and self.esm_std is not None:
                x = (x - self.esm_mean[:, None]) / (self.esm_std[:, None] + 1e-8)

            out["esm"] = torch.from_numpy(x)

        # ---- kmer ----
        if "kmer" in self.branches:
            if self.kvoc is None:
                raise RuntimeError("kmer branch enabled but kmer_vocabs is None")
            xk = {}
            for k, vocab in self.kvoc.items():
                ids = vocab.encode(s.seq)
                xk[k] = torch.tensor(ids, dtype=torch.long)
            out["kmer"] = xk

        # ---- PDB-based branches ----
        if ("physchem" in self.branches) or ("esmfold" in self.branches):
            p = self._find_pdb_path(s.uid)
            if p is None:
                if self.strict:
                    raise RuntimeError(f"[pdb] uid not found: {s.uid}")
                return {"uid": s.uid, "seq": s.seq, "x": None}
            residues = load_residues_from_pdb(p)

            if "physchem" in self.branches:
                assert self.feat_dict is not None
                seq_feat, L = get_seq_feat(residues, self.feat_dict, MAX_LEN)
                out["physchem"] = {
                    "seq": torch.from_numpy(seq_feat),
                    "length": torch.tensor(L, dtype=torch.long),
                }

            if "esmfold" in self.branches:
                pair_feat, L = build_pair_feature_from_residues(residues, MAX_LEN)
                out["esmfold"] = {
                    "pair": torch.from_numpy(pair_feat),
                    "length": torch.tensor(L, dtype=torch.long),
                }

        return {"uid": s.uid, "seq": s.seq, "x": out}


def collate_skip_none(batch):
    batch2 = [b for b in batch if b["x"] is not None]
    if not batch2:
        return None
    return torch.utils.data.default_collate(batch2)


# ==========================
# 8) Load model
# ==========================
def infer_model_type_from_state_dict(sd: dict) -> str:
    for k in sd.keys():
        if k.startswith("meta."):
            return "stack"
    return "blend"


def build_fused_model(ckpt: dict, esm_h5: str, kmer_vocabs: Optional[Dict[int, KmerVocab]]):
    cfg = ckpt.get("config", {}) or {}
    branches = ckpt.get("branches", None)
    if branches is None:
        raise RuntimeError("Checkpoint missing 'branches' list")

    branch_models: Dict[str, nn.Module] = {}

    D_in = None
    if "esm" in branches:
        import h5py
        with h5py.File(esm_h5, "r") as f:
            D_in = int(f["window_emb"].shape[-1])

    HIDDEN = int(cfg.get("HIDDEN", 256))
    DROPOUT = float(cfg.get("DROPOUT", 0.5))
    VCONV_KSIZE = int(cfg.get("VCONV_KSIZE", 7))
    VCONV_LAYERS = int(cfg.get("VCONV_LAYERS", 2))
    VCONV_KERNELS = int(cfg.get("VCONV_KERNELS", 128))
    KMER_EMBED = int(cfg.get("KMER_EMBED", 32))
    DROPOUT_KMER = float(cfg.get("DROPOUT_KMER", 0.4))

    for b in branches:
        if b == "physchem":
            branch_models[b] = BranchPhyschem(seq_feat_dim=SEQ_FEAT_DIM, seq_hidden=128)
        elif b == "esm":
            if D_in is None:
                raise RuntimeError("Cannot determine ESM embedding dim from H5")
            branch_models[b] = BranchESM(
                D_in=D_in,
                hidden=HIDDEN,
                ksize=VCONV_KSIZE,
                n_layers=VCONV_LAYERS,
                dropout=DROPOUT,
                vconv_kernels=VCONV_KERNELS,
            )
        elif b == "kmer":
            if kmer_vocabs is None:
                raise RuntimeError("kmer branch enabled but kmer_vocabs is None")
            vocab_sizes = {k: v.size for k, v in kmer_vocabs.items()}
            branch_models[b] = BranchKmer(vocab_sizes, embed_dim=KMER_EMBED, dropout=DROPOUT_KMER)
        elif b == "esmfold":
            branch_models[b] = BranchESMFold(in_channels=6, pair_hidden=128, node_hidden=256, gnn_layers=2)
        else:
            raise ValueError(f"Unknown branch name in ckpt: {b}")

    model_type = infer_model_type_from_state_dict(ckpt["state_dict"])
    if model_type == "blend":
        weights = np.array(ckpt.get("weights", [1.0 / len(branches)] * len(branches)), dtype=np.float32)
        fused = FourWayBlend(branch_models, weights=weights)
    else:
        fused = FiveWayStack(branch_models, meta_state_dict=None)

    fused.load_state_dict(ckpt["state_dict"], strict=True)
    return fused, branches, model_type



# ==========================
# 9) Prediction entry point
# ==========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict S-palmitoylation scores using the trained Deep-Palm model."
    )
    parser.add_argument("--input-csv", default=CONFIG["INPUT_CSV"], help="Input CSV/TSV with ID and seq columns.")
    parser.add_argument("--output-csv", default=CONFIG["OUTPUT_CSV"], help="Output CSV path.")
    parser.add_argument("--checkpoint", default=CONFIG["FUSED_CKPT"], help="Path to Deep-Palm checkpoint.")
    parser.add_argument("--embedding-h5", default=CONFIG["ESM_H5"], help="Path to precomputed ESM embedding H5 file.")
    parser.add_argument("--pdb-dir", default=CONFIG["PDB_DIR"], help="Directory containing matching predicted PDB files.")
    parser.add_argument("--aaindex", default=CONFIG["AAINDEX1_PATH"], help="Path to aaindex1.txt.")
    parser.add_argument("--kmer-vocab", default=CONFIG["KMER_VOCAB_JSON"], help="Path to kmer_vocab.json.")
    parser.add_argument("--batch-size", type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument("--num-workers", type=int, default=CONFIG["NUM_WORKERS"])
    parser.add_argument("--device", default=CONFIG["DEVICE"], help="cpu, cuda, or cuda:0, for example.")
    parser.add_argument("--threshold", type=float, default=CONFIG["THRESHOLD"], help="Threshold for binary calls.")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip samples with missing precomputed features rather than raising an error.",
    )
    return parser.parse_args()


def require_path(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)

    require_path(args.input_csv, "Input file")
    require_path(args.checkpoint, "Deep-Palm checkpoint")
    require_path(args.embedding_h5, "ESM embedding H5 file")
    require_path(args.pdb_dir, "PDB directory")
    require_path(args.aaindex, "AAindex file")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    branches = ckpt.get("branches", None)
    if branches is None:
        raise RuntimeError("Checkpoint does not contain a 'branches' field required by the prediction script.")

    print(f"[Load] checkpoint={args.checkpoint}", flush=True)
    print(f"[Load] branches={branches}", flush=True)

    kvoc = None
    if "kmer" in branches:
        require_path(args.kmer_vocab, "K-mer vocabulary JSON")
        kvoc = load_kmer_vocabs_json(args.kmer_vocab)
        print(f"[Load] kmer vocab={args.kmer_vocab}", flush=True)
        print("[kmer] loaded vocabs:", {k: v.size for k, v in kvoc.items()}, flush=True)

    sep = detect_sep(args.input_csv)
    df = pd.read_csv(args.input_csv, sep=sep, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    idc, seqc, _, _ = pick_columns(df)
    if idc is None or seqc is None:
        raise RuntimeError(f"Cannot infer ID/SEQ columns from input file. Columns={df.columns.tolist()}")

    samples: List[InferSample] = []
    for _, row in df.iterrows():
        uid = str(row[idc]).strip()
        seq = ensure_len_31(str(row[seqc]))
        if uid:
            samples.append(InferSample(uid=uid, seq=seq))
    if not samples:
        raise RuntimeError("No valid samples were found in the input file.")

    print(f"[Data] input={args.input_csv}", flush=True)
    print(f"[Data] samples={len(samples)}, id_column={idc}, seq_column={seqc}", flush=True)

    esm_norm = None
    if "esm" in branches:
        uid2idx = build_uid2idx(args.embedding_h5)
        mean, std = esm_channel_stats_for_uids(args.embedding_h5, uid2idx, [s.uid for s in samples])
        esm_norm = (mean, std)
        print("[ESM] normalization computed from the supplied embedding set to preserve the original pipeline.", flush=True)

    strict = not args.allow_missing
    ds = MultiBranchInferDataset(
        samples=samples,
        branches=branches,
        esm_h5=args.embedding_h5,
        pdb_dir=args.pdb_dir,
        aaindex_path=args.aaindex,
        kmer_vocabs=kvoc,
        esm_norm=esm_norm,
        strict=strict,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(str(args.device).startswith("cuda")),
        collate_fn=collate_skip_none if args.allow_missing else None,
    )

    model, _, model_type = build_fused_model(ckpt, args.embedding_h5, kvoc)
    device = args.device
    model = model.to(device).eval()
    print(f"[Model] type={model_type}, device={device}", flush=True)

    all_uids: List[str] = []
    all_seqs: List[str] = []
    all_probs: List[float] = []
    for batch in tqdm(dl, desc="Predict", dynamic_ncols=True):
        if batch is None:
            continue
        x = move_to_device(batch["x"], device)
        prob = model(x).detach().cpu().numpy().reshape(-1)
        all_uids.extend(list(batch["uid"]))
        all_seqs.extend(list(batch["seq"]))
        all_probs.extend(prob.tolist())

    if not all_probs:
        raise RuntimeError("No predictions were generated. Check whether the requested input IDs have matching features.")

    scores = np.asarray(all_probs, dtype=float)
    out = pd.DataFrame({
        "ID": all_uids,
        "seq": all_seqs,
        "Deep-Palm_score": scores,
        "prediction": (scores >= float(args.threshold)).astype(int),
    })
    output_parent = os.path.dirname(os.path.abspath(args.output_csv))
    os.makedirs(output_parent, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"[Save] predictions={args.output_csv}", flush=True)
    print(f"[Save] threshold={args.threshold}; predicted_positive={(out['prediction'] == 1).sum()}/{len(out)}", flush=True)


if __name__ == "__main__":
    main()
