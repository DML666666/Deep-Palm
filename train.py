#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import copy
import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
                    
try:
    from sklearn.metrics import roc_auc_score, accuracy_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    BIOPY_OK = True
except Exception:
    BIOPY_OK = False


def build_uid2idx(h5_path: str) -> dict:
    import h5py, os
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"ESM_H5 不存在: {h5_path}")
    if not h5py.is_hdf5(h5_path):
        raise RuntimeError(f"文件不是合法的 HDF5: {h5_path} ；请检查生成脚本是否成功写出 .h5")
    try:
        with h5py.File(h5_path, "r", libver="latest") as f:
            if "uniprotid" not in f:
                raise RuntimeError("H5 里没有 'uniprotid' 数据集，请确认是前面脚本生成的 h5")
            uids = f["uniprotid"][...]
    except OSError as e:
        raise OSError(
            f"无法打开 HDF5 文件（可能是损坏或 HDF5 版本不兼容）：{h5_path}\n"
            f"{e}\n建议：1) 重新生成该 .h5；或 2) 升级当前环境的 h5py/libhdf5 版本。"
        )
    uids = [u.decode() if isinstance(u, (bytes, bytearray)) else str(u) for u in uids]
    return {u: i for i, u in enumerate(uids)}


                               
                     
                               
CONFIG: Dict = {
                    
    'CSV_PATH': 'esm_input.csv',
    'ESM_H5':   'embedding.h5',
    'OUT_DIR':  'output',
    'MODEL_DIR': 'models',
    'SPECIES_CSV':'uniprotid-物种.csv',
    'SPECIES_MIN_SAMPLES': 100,
                                    
    'PDB_DIR': 'esmfold.pdb',

                      
                                        
                                                 
                                     
    'DIST2D_MODE': 'inv',
    'DIST2D_MAX_DIST': 30.0,
    'DIST2D_CONTACT_THRESH': 8.0,
    'CADIST_IN_CHANNELS': 6,
           
    'FINAL_REPEAT_ID': 666,

                      
    'RANDOM_SEED': 3407,
    'SPLIT_SEED':  2025,
    'CV_SEED':     1314,
    'N_REPEATS':   1,
    'N_FOLDS':     5,
    'TEST_RATIO':  0.10,
    'EPOCHS':      128,
    'BATCH_SIZE':  1024,
    'LR':          5e-4,
    'WEIGHT_DECAY':5e-4,
    'DEVICE':      'cuda' if torch.cuda.is_available() else 'cpu',
                                                           
                                    
    'AMP':         False,
                                                      
    'DETERMINISTIC': True,
"""复现优先：AMP=False, DETERMINISTIC=True, ALLOW_TF32=False, CUDNN_BENCHMARK=False, NUM_WORKERS=0
   速度优先：AMP=True, DETERMINISTIC=False, ALLOW_TF32=True, CUDNN_BENCHMARK=True"""
                     
    'EARLY_STOP_PATIENCE': 5,
    'PRIMARY_METRIC': 'auc',
    'METRIC_THRESHOLD': 0.5,
    'BALANCE_POS_NEG': True,                       

                       
    'DROPOUT': 0.5,
    'HIDDEN':  256,

                           
    'VCONV_KERNELS': 128,
    'VCONV_KSIZE':  7,
    'VCONV_LAYERS': 2,
                           
    'AUG_PER_SAMPLE': 2,
    'AUG_MAX_MUTS':   2,

                            
    'KMERS': [2, 3, 4],
    'KMER_EMBED': 32,
    'TOP_KMER_K5': 200000,
    'TOP_KMER_K4': None,

                             
    'LABEL_SMOOTH': 0.05,
    'GRAD_CLIP_NORM': 1.0,
    'USE_PLATEAU_SCHED': True,
    'MIN_LR': 1e-5,
    'DROPOUT_KMER': 0.4,
    'WEIGHT_DECAY_KMER': 1e-3,

                    
    'BLEND_GRID_STEP': 0.1,
    'BLEND_REFINE': True,
    'BLEND_MIN_WEIGHT': 0.1,

                    
    'SCORE_RIGHT_ALPHA': 1.0,
    'SCORE_WRONG_BETA': 1.0,

                           
                                  
    'ENABLED_BRANCHES': {
        'physchem':        True,  
        'esm':             True,
        'kmer':            True,
        'esmfold':         True,   
    },

}

                                                           
                                            
                        
                                                           

import copy
from typing import Tuple
from Bio.PDB import PDBParser, Polypeptide
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

                                                 
MAX_LEN = 31
CONTACT_THRESH = 8.0

                                                         
AAINDEX1_PATH = "aaindex1.txt"  

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

AA_INDEX_ORDER = [
    "A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"
]

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
        raise ValueError(
            f"{entry_id}: 解析到 {len(vals)} 个数 (期望 20)，请检查 aaindex1 路径或文件格式。"
        )
    return dict(zip(AA_INDEX_ORDER, vals))
           
def build_physico_chemical_features(aaindex_path: str):
    num_idx = len(AAINDEX_IDS)
    num_aa = len(AA_INDEX_ORDER)
    raw_matrix = np.zeros((num_idx, num_aa), dtype=np.float32)
    for i, entry_id in enumerate(AAINDEX_IDS):
        entry_dict = load_aaindex_entry(aaindex_path, entry_id)
        raw_matrix[i] = np.array(
            [entry_dict[aa] for aa in AA_INDEX_ORDER],
            dtype=np.float32
        )
    means = raw_matrix.mean(axis=1, keepdims=True)
    stds = raw_matrix.std(axis=1, ddof=0, keepdims=True)
    stds[stds == 0] = 1.0
    z_matrix = (raw_matrix - means) / stds
    feat_dict = {
        aa: z_matrix[:, j].astype(np.float32).tolist()
        for j, aa in enumerate(AA_INDEX_ORDER)
    }
    feat_dict["X"] = [0.0] * num_idx
    return feat_dict

PHYSICO_CHEMICAL_FEATURES = build_physico_chemical_features(AAINDEX1_PATH)
SEQ_FEAT_DIM = len(AAINDEX_IDS)
AA_3_TO_1 = Polypeptide.protein_letters_3to1

                                                            
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

def build_pair_feature_from_residues(
    residues,
    max_len: int = MAX_LEN
) -> Tuple[np.ndarray, int]:
    residues = residues[:max_len]
    L = len(residues)
    if L == 0:
        raise ValueError("该 PDB 中没有带 CA 的残基")
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
    feat = np.stack(
        [dist_ca, dist_min, dist_sc, contact, long_range, seq_sep_norm],
        axis=0
    )             
    pair_feat = np.zeros((6, max_len, max_len), dtype=np.float32)
    pair_feat[:, :L, :L] = feat
    return pair_feat, L

                                                 
def get_seq_feat(residues, max_len: int = MAX_LEN) -> Tuple[np.ndarray, int]:
    seq_feat = []
    residues = residues[:max_len]
    L = len(residues)
    for res in residues:
        res_name = res.get_resname().upper()
        aa_code = AA_3_TO_1.get(res_name, 'X')
        feat = PHYSICO_CHEMICAL_FEATURES.get(
            aa_code,
            PHYSICO_CHEMICAL_FEATURES['X']
        )
        seq_feat.append(feat)
    if L < max_len:
        padding_feat = PHYSICO_CHEMICAL_FEATURES['X']
        seq_feat.extend([padding_feat] * (max_len - L))
    return np.array(seq_feat, dtype=np.float32)[:max_len], L

                                                    
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
       
                                                       
       
                                      
                              
       
                     
       
    def __init__(self,
                 in_channels: int = 6,
                 pair_hidden: int = 128,
                 node_hidden: int = 256,
                 gnn_layers: int = 2):
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
        self.gnn_layers = nn.ModuleList(
            [GCNLayer(node_hidden, node_hidden) for _ in range(gnn_layers)]
        )
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
        pair_feat = x_dict['pair']                 
        valid_lengths = x_dict['length']
        device = pair_feat.device
        L = pair_feat.shape[-1]

        idx = torch.arange(L, device=device).unsqueeze(0)
        mask = (idx < valid_lengths.unsqueeze(1)).float()

              
        h_pair = self.pair_cnn(pair_feat)                             
        node = torch.cat(
            [h_pair.mean(dim=3), h_pair.mean(dim=2)],
            dim=1
        )                                                           
        node = self.node_proj(node.permute(0, 2, 1))                        

        adj_norm = self.build_adj(pair_feat)
        h_gnn = node
        for layer in self.gnn_layers:
            h_gnn = layer(h_gnn, adj_norm)
        g_struct = self.struct_attn_readout(h_gnn, mask)

        logits = self.readout(g_struct).squeeze(1).unsqueeze(-1)          
        return logits

class BranchPhyschem(nn.Module):
       
                                                       
                                  
       
                                              
                              
       
                     
       
    def __init__(self,
                 seq_feat_dim: int = SEQ_FEAT_DIM,
                 seq_hidden: int = 128):
        super().__init__()
        self.seq_proj = nn.Linear(seq_feat_dim, seq_hidden)
        self.lstm = nn.LSTM(
            seq_hidden, seq_hidden,
            batch_first=True,
            bidirectional=True
        )
        self.seq_attn_readout = AttentionReadout(node_dim=seq_hidden * 2)
        self.readout = nn.Sequential(
            nn.Linear(seq_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x_dict):
        seq_feat = x_dict['seq']                 
        valid_lengths = x_dict['length']
        device = seq_feat.device
        L = seq_feat.shape[1]

        idx = torch.arange(L, device=device).unsqueeze(0)
        mask = (idx < valid_lengths.unsqueeze(1)).float()

        h_seq = F.relu(self.seq_proj(seq_feat))                          
        packed_input = nn.utils.rnn.pack_padded_sequence(
            h_seq,
            valid_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed_input)
        h_lstm, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=L,
        )                                                                  
        g_seq = self.seq_attn_readout(h_lstm, mask)                      

        logits = self.readout(g_seq).squeeze(1).unsqueeze(-1)          
        return logits


os.makedirs(CONFIG['OUT_DIR'], exist_ok=True)
os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)

        
def set_global_seed(seed: int):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(CONFIG['RANDOM_SEED'])

                                                               
CONFIG.update({
                                       
    'NUM_WORKERS': max(32, (os.cpu_count() or 8)//2),
    'PREFETCH_FACTOR': 4,
    'PERSISTENT_WORKERS': True,
    'PIN_MEMORY': True,
    'PIN_MEMORY_DEVICE': 'cuda',
    'NON_BLOCKING': True,
                                      
                                                
    'ALLOW_TF32': False,
    'CUDNN_BENCHMARK': False,
    'TORCH_COMPILE': False,
    'COMPILE_MODE': 'reduce-overhead',
})

def enable_speedups():
    deterministic = bool(CONFIG.get('DETERMINISTIC', False))
    allow_tf32 = bool(CONFIG.get('ALLOW_TF32', False)) and not deterministic
    cudnn_benchmark = bool(CONFIG.get('CUDNN_BENCHMARK', False)) and not deterministic

    try:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_tf32:
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass
    except Exception:
        pass

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except Exception:
        pass


def amp_is_enabled() -> bool:
    return bool(CONFIG.get('AMP', False) and str(CONFIG.get('DEVICE', 'cpu')).startswith('cuda') and torch.cuda.is_available())


class _NoOpGradScaler:
    def __init__(self):
        self.enabled = False

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def unscale_(self, optimizer):
        return None


def build_grad_scaler():
    enabled = amp_is_enabled()
    if not enabled:
        return _NoOpGradScaler()

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        try:
            return torch.amp.GradScaler('cuda', enabled=True)
        except TypeError:
            try:
                return torch.amp.GradScaler(enabled=True)
            except TypeError:
                pass
        except AttributeError:
            pass

    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
        try:
            return torch.cuda.amp.GradScaler(enabled=True)
        except TypeError:
            try:
                return torch.cuda.amp.GradScaler()
            except Exception:
                pass

    print('[WARN] AMP=True，但当前 PyTorch 不支持 GradScaler；自动回退到 FP32。')
    return _NoOpGradScaler()


def get_autocast_context():
    if not amp_is_enabled():
        return nullcontext()

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        try:
            return torch.amp.autocast(device_type='cuda', enabled=True)
        except TypeError:
            try:
                return torch.amp.autocast('cuda', enabled=True)
            except TypeError:
                pass
        except AttributeError:
            pass

    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
        try:
            return torch.cuda.amp.autocast(enabled=True)
        except TypeError:
            return torch.cuda.amp.autocast()

    print('[WARN] AMP=True，但当前 PyTorch 不支持 autocast；自动回退到 FP32。')
    return nullcontext()


def backward_with_optional_amp(loss, optimizer, scaler):
    if getattr(scaler, 'enabled', False):
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()


BRANCH_ORDER = ['physchem', 'esm', 'kmer', 'esmfold']

def get_enabled_branches() -> List[str]:
       
                                                  
                     
       
    cfg = CONFIG.get('ENABLED_BRANCHES', {})
    enabled = [b for b in BRANCH_ORDER if cfg.get(b, False)]
    if not enabled:
        enabled = BRANCH_ORDER[:]             
    return enabled


def loader_kwargs():
    nw = int(CONFIG.get('NUM_WORKERS', 0) or 0)
    kw = dict(
        num_workers=nw,
        pin_memory=bool(CONFIG.get('PIN_MEMORY', True)),
    )
    if nw > 0:
        kw['persistent_workers'] = bool(CONFIG.get('PERSISTENT_WORKERS', True))
        kw['prefetch_factor']    = int(CONFIG.get('PREFETCH_FACTOR', 2))
        if 'PIN_MEMORY_DEVICE' in CONFIG:
            kw['pin_memory_device'] = CONFIG['PIN_MEMORY_DEVICE']
    return kw

enable_speedups()
         
             
                      
                    
                         
                 
         
            
                       
                   
                        

                               
                    
                               
def _safe_div(a, b):
    return float(a) / float(b) if (b is not None and b != 0) else 0.0

def bin_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= thr).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    P = TP + FN
    N = TN + FP
    sen = _safe_div(TP, P)
    spe = _safe_div(TN, N)
    acc = _safe_div(TP + TN, P + N)
    auc = float(roc_auc_score(y_true, y_prob)) if SKLEARN_OK and len(np.unique(y_true)) > 1 else float('nan')
    return {'tp': TP, 'tn': TN, 'fp': FP, 'fn': FN, 'sen': sen, 'spe': spe, 'acc': acc, 'auc': auc}

def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    m = bin_metrics(y_true, y_prob, thr=CONFIG['METRIC_THRESHOLD'])
    return {'auc': m['auc'], 'acc': m['acc'], 'sen': m['sen'], 'spe': m['spe']}

def make_scores(uids, labels, probs, thr: float = 0.5) -> pd.DataFrame:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= thr).astype(int)
    conf = np.maximum(probs, 1 - probs)
    correct = (preds == labels).astype(float)
    score = np.where(
        correct > 0,
        CONFIG['SCORE_RIGHT_ALPHA'] * conf,
        -CONFIG['SCORE_WRONG_BETA'] * conf
    )
    return pd.DataFrame({
        'uniprotid': list(uids),
        'label': labels,
        'prob': probs,
        'pred': preds,
        'conf_score': score,
    })


def plot_roc_multi(save_path: str,
                   curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   title: str = 'ROC curves (multi)'):
    if not SKLEARN_OK:
        print("[WARN] sklearn 未安装，跳过 ROC 绘图:", save_path)
        return

    from sklearn.metrics import roc_curve

    plt.figure(figsize=(6.4, 5.0))
    for name, (yt, yp) in curves.items():
        yt = np.asarray(yt).astype(int)
        yp = np.asarray(yp).astype(float)
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_auc_errorbars_by_method(save_path: str,
                                 metrics_df: pd.DataFrame,
                                 n_repeats: int):
    if metrics_df.empty:
        print("[WARN] metrics_df 为空，跳过误差棒图:", save_path)
        return

    df = metrics_df[metrics_df['set'] == 'test'].copy()
    if df.empty:
        print("[WARN] metrics_df 中没有 set=='test' 的记录，跳过误差棒图。")
        return

    order = sorted(df['method'].unique().tolist())
    agg = df.groupby('method')['auc'].agg(['mean', 'std']).reindex(order)

    x = np.arange(len(order))
    plt.figure(figsize=(8, 4))
    yerr = agg['std'].values if n_repeats > 1 else None
    plt.errorbar(x, agg['mean'].values,
                 yerr=yerr,
                 fmt='o', capsize=5 if n_repeats > 1 else 0)

    for i, m in enumerate(order):
        plt.text(x[i],
                 agg['mean'].iloc[i] + 0.01,
                 f"{agg['mean'].iloc[i]:.3f}",
                 ha='center', va='bottom', fontsize=9)

    plt.xticks(x, order, rotation=0)
    plt.ylim(0, 1.02)
    plt.ylabel('AUC')
    plt.title('Test AUC by method (mean±std over repeats)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_metrics_bars_by_method(save_path: str,
                                metrics_by_method: Dict[str, Dict[str, float]],
                                methods: List[str],
                                title: str = 'Metrics comparison'):
    metric_names = ['sen', 'spe', 'acc', 'auc']
    x = np.arange(len(methods))
    width = 0.18

    plt.figure(figsize=(max(8, 1.5 * len(methods)), 4.5))
    for i, mname in enumerate(metric_names):
        vals = [float(metrics_by_method[m][mname]) for m in methods]
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=mname.upper())
        for j, v in enumerate(vals):
            plt.text(x[j] + (i - 1.5) * width,
                     v + 0.01,
                     f"{v:.2f}",
                     ha='center', va='bottom', fontsize=8)

    plt.xticks(x, methods, rotation=0)
    plt.ylim(0, 1.05)
    plt.ylabel('Value')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_species_metrics(save_path: str,
                         df_sp: pd.DataFrame,
                         title: str = 'Per-species metrics'):
       
                                         
                                                         
       
    metric_names = ['sen', 'spe', 'acc', 'auc']
    species = df_sp['species'].tolist()
    x = np.arange(len(species))
    width = 0.18

    plt.figure(figsize=(max(8, 1.2 * len(species)), 4.5))
    for i, m in enumerate(metric_names):
        vals = df_sp[m].values
        plt.bar(x + (i - 1.5) * width, vals, width=width, label=m.upper())
        for j, v in enumerate(vals):
            plt.text(
                x[j] + (i - 1.5) * width,
                v + 0.005,
                f"{v:.2f}",
                ha='center',
                va='bottom',
                fontsize=7
            )

    plt.xticks(x, species, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.ylabel('Metric value')
    plt.title(title)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _base_uid(uid: str) -> str:
    return str(uid).strip().split('-')[0]


def _load_species_map_for_final(path: str) -> Dict[str, str]:
    df = pd.read_csv(path, header=None, usecols=[0, 1],
                     names=['uidpos', 'species'])
    df['uidpos'] = df['uidpos'].astype(str).str.strip()
    df['species'] = df['species'].astype(str).str.strip()
    mp = {}
    for _, r in df.iterrows():
        uid_full = r['uidpos']
        sp = r['species']
        mp[uid_full] = sp
        mp[_base_uid(uid_full)] = sp
    return mp


def species_stats_and_roc(uids: List[str],
                          labels: np.ndarray,
                          probs: np.ndarray,
                          species_csv: str,
                          out_dir: str,
                          min_n: int = 100):
    if (not species_csv) or (not os.path.exists(species_csv)):
        print("[Info] SPECIES_CSV 不存在，跳过物种统计。")
        return
    if not SKLEARN_OK:
        print("[Info] 未安装 sklearn，跳过物种统计。")
        return

    from sklearn.metrics import roc_curve                       

    sp_map = _load_species_map_for_final(species_csv)
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)

           
    bucket: Dict[str, Dict[str, List[float]]] = {}
    for uid, y, p in zip(uids, labels, probs):
        sp = sp_map.get(uid) or sp_map.get(_base_uid(uid))
        if sp is None:
            continue
        bucket.setdefault(sp, {'y': [], 'p': []})
        bucket[sp]['y'].append(int(y))
        bucket[sp]['p'].append(float(p))

    rows = []
    curves = {}
    for sp, d in bucket.items():
        y = np.array(d['y'], dtype=int)
        p = np.array(d['p'], dtype=float)
        if len(y) <= min_n:
            continue
        if len(np.unique(y)) < 2:
            continue

                                       
        mets = calc_metrics(y, p)
        rows.append({
            'species': sp,
            'n': len(y),
            'sen': mets['sen'],
            'spe': mets['spe'],
            'acc': mets['acc'],
            'auc': mets['auc'],
        })
        curves[sp] = (y, p)

    if not curves:
        print(f"[Info] 没有物种满足 N>{min_n} 的条件，跳过物种统计。")
        return

               
    df_sp = pd.DataFrame(rows).sort_values('auc', ascending=False)
    df_sp.to_csv(os.path.join(out_dir, 'species_metrics_FINAL.csv'), index=False)

                                         
    plot_species_metrics(
        os.path.join(out_dir, 'species_metrics_FINAL.svg'),
        df_sp,
        title=f"Per-species metrics (N>{min_n})"
    )

                                         
    plot_roc_multi(
        os.path.join(out_dir, 'species_roc_FINAL.svg'),
        curves,
        title=f"Species ROC (N>{min_n})"
    )
    print("[Save] 物种统计已输出到:", out_dir)


def summarize_and_dedup(samples: List['Sample']) -> List['Sample']:
    total = len(samples)
    pos = sum(1 for s in samples if int(s.label) == 1)
    neg = total - pos
    print(f"[数据概况] 原始总数={total}  阳性={pos}  阴性={neg}  正例占比={pos/max(1,total):.2%}")

    seen: Dict[str, 'Sample'] = {}
    dup, conflicts = 0, 0
    for s in samples:
        if s.uid not in seen:
            seen[s.uid] = s
        else:
            dup += 1
            s0 = seen[s.uid]
            if (s.seq != s0.seq) or (int(s.label) != int(s0.label)):
                conflicts += 1
    if dup > 0:
        print(f"[去重] 发现重复 uid 数量={dup}，其中冲突={conflicts}；保留首次出现。")
    samples_new = list(seen.values())
    total2 = len(samples_new)
    pos2 = sum(1 for s in samples_new if int(s.label) == 1)
    neg2 = total2 - pos2
    print(f"[去重后] 总数={total2}  阳性={pos2}  阴性={neg2}  正例占比={pos2/max(1,total2):.2%}")
    return samples_new

def _project_simplex_nonneg(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0)
    s = w.sum()
    if s == 0:
        w[0] = 1.0
        return w
    return w / s

def _project_simplex_floor(w: np.ndarray, floor: float) -> np.ndarray:
    n = w.shape[0]
    if floor <= 0:
        return _project_simplex_nonneg(w.copy())
    v = np.maximum(w - floor, 0.0)
    S = v.sum()
    target = max(1.0 - n * floor, 1e-12)
    if S <= 1e-12:
        v = np.zeros_like(v); v[0] = target
    else:
        v = v * (target / S)
    return v + floor

def grid_search_blend_weights(oof_preds: np.ndarray,
                              y_true: np.ndarray,
                              step: float = 0.1,
                              refine: bool = True,
                              min_w: float = 0.1) -> Tuple[np.ndarray, float]:
       
                                  
                                    
       
    from itertools import product

    n_models = oof_preds.shape[1]
    if n_models < 1 or n_models > 5:
        raise ValueError(f"grid_search_blend_weights 目前只支持 1~5 个分支, got {n_models}")
    if min_w * n_models > 1 + 1e-12:
        raise ValueError(f"min_w={min_w} 过大 (n_models={n_models})")

                           
    if n_models == 1:
        w = np.array([1.0], dtype=np.float32)
        yb = oof_preds[:, 0]
        auc = roc_auc_score(y_true, yb) if SKLEARN_OK else calc_metrics(y_true, yb)['auc']
        return w, float(auc)

    ws = np.arange(min_w, 1.0 + 1e-12, step)
    best_auc, best_w = -1.0, None

                                                  
    for prefix in product(ws, repeat=n_models - 1):
        s = float(sum(prefix))
        w_last = 1.0 - s
        if w_last < min_w - 1e-12 or w_last > 1.0 + 1e-12:
            continue
        w = np.array(list(prefix) + [w_last], dtype=np.float64)
        yb = (oof_preds * w[None, :]).sum(axis=1)
        auc = roc_auc_score(y_true, yb) if SKLEARN_OK else calc_metrics(y_true, yb)['auc']
        if auc > best_auc:
            best_auc, best_w = float(auc), w.copy()

                              
    if refine and best_w is not None:
        local = max(step / 2.0, 0.02)
        cand = []
        for deltas in product((-local, 0.0, local), repeat=n_models):
            w = best_w.copy() + np.array(deltas, dtype=np.float64)
            w = _project_simplex_floor(w, floor=min_w)
            cand.append(w)
        uniq = np.unique(np.round(np.stack(cand, 0), 6), axis=0)
        for w in uniq:
            yb = (oof_preds * w[None, :]).sum(axis=1)
            auc = roc_auc_score(y_true, yb) if SKLEARN_OK else calc_metrics(y_true, yb)['auc']
            if auc > best_auc:
                best_auc, best_w = float(auc), w.copy()

    return best_w.astype(np.float32), best_auc



                               
         
                               
def clean_seq(s: str) -> str:
    s = (s or '').strip().upper()
    allow = set(list('ACDEFGHIKLMNPQRSTVWY') + ['X'])
    return ''.join(ch if ch in allow else 'X' for ch in s)

def ensure_len_31(s: str) -> str:
    if len(s) == 31:
        return s
    if len(s) > 31:
        mid = len(s)//2
        st = max(0, mid-15)
        s = s[st:st+31]
        if len(s) != 31:
            s = s[:31]
        return s
    lpad = (31 - len(s))//2
    rpad = 31 - len(s) - lpad
    return 'X'*lpad + s + 'X'*rpad

                  
CONS_GROUPS = [list('ST'), list('NQ'), list('DE'), list('KR'), list('FWY'), list('ILVM'), list('AG'), list('PH')]
CONS_MAP = {c:g for g in CONS_GROUPS for c in g}

def aug_mutate_seq(seq: str, max_muts: int = 2, keep_center_c=True) -> str:
    seq = list(ensure_len_31(clean_seq(seq)))
    idxs = list(range(31)); center = 15
    if keep_center_c and center in idxs:
        idxs.remove(center)
    n = random.randint(1, max(1, max_muts))
    choose = random.sample(idxs, k=min(n, len(idxs)))
    for i in choose:
        aa = seq[i]
        cand = CONS_MAP.get(aa, None)
        if cand:
            rep = random.choice([x for x in cand if x!=aa])
        else:
            pool = list('ACDEFGHIKLMNPQRSTVWY')
            if keep_center_c and i==center:
                pool = [x for x in pool if x!='C']
            rep = random.choice(pool)
        seq[i] = rep
    return ''.join(seq)

def physchem_from_seq(seq: str) -> List[float]:
    if not BIOPY_OK:
        return [0.0]*7
    pa = ProteinAnalysis(seq.replace('X','A'))
    mw   = pa.molecular_weight()
    pI   = pa.isoelectric_point()
    aro  = pa.aromaticity()
    gravy= pa.gravy()
    helix, turn, sheet = pa.secondary_structure_fraction()
    return [float(mw), float(pI), float(aro), float(gravy), float(helix), float(turn), float(sheet)]

                                    

_PDB_UID_MAP_CACHE: Dict[str, Dict[str, str]] = {}

def _strip_uid_suffix(uid: str) -> str:
    u = str(uid).strip()
    return u.split('-')[0]

def build_pdb_uid_map(pdb_dir: str) -> Dict[str, str]:
       
                                            
       
    global _PDB_UID_MAP_CACHE
    if pdb_dir in _PDB_UID_MAP_CACHE:
        return _PDB_UID_MAP_CACHE[pdb_dir]

    if not os.path.exists(pdb_dir):
        raise FileNotFoundError(f"PDB_DIR 不存在: {pdb_dir}")

    mp: Dict[str, str] = {}

    if os.path.isdir(pdb_dir):
        for fn in os.listdir(pdb_dir):
            if not fn.lower().endswith('.pdb'):
                continue
            stem = fn[:-4]
            path = os.path.join(pdb_dir, fn)
            if stem not in mp:
                mp[stem] = path
            base = _strip_uid_suffix(stem)
            if base not in mp:
                mp[base] = path
        print(f"[PDB] 目录 {pdb_dir} 中共发现 {len(mp)} 个 uid->pdb 映射")
    else:
                    
        if not pdb_dir.lower().endswith('.pdb'):
            raise FileNotFoundError(f"PDB_DIR 既不是目录也不是 .pdb 文件: {pdb_dir}")
        fn = os.path.basename(pdb_dir)
        stem = fn[:-4]
        path = pdb_dir
        if stem not in mp:
            mp[stem] = path
        base = _strip_uid_suffix(stem)
        if base not in mp:
            mp[base] = path
        print(f"[PDB] 单一 pdb 文件: {pdb_dir} -> 映射数={len(mp)}")

    _PDB_UID_MAP_CACHE[pdb_dir] = mp
    return mp

                       
def kmer_tokens(seq: str, k: int) -> List[str]:
    seq = ensure_len_31(clean_seq(seq))
    return [seq[i:i+k] for i in range(0, len(seq)-k+1)]

class KmerVocab:
    def __init__(self, k: int, max_size: Optional[int]=None):
        self.k = k
        self.max_size = max_size
        self.counts: Dict[str,int] = {}
        self.stoi: Dict[str,int] = {'<OOV>':0}
        self.itos: List[str] = ['<OOV>']
    def add_seq(self, seq: str):
        for tok in kmer_tokens(seq, self.k):
            self.counts[tok] = self.counts.get(tok, 0) + 1
    def finalize(self):
        items = sorted(self.counts.items(), key=lambda x:(-x[1], x[0]))
        if self.max_size is not None:
            items = items[:self.max_size]
        for tok,_ in items:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
    def encode(self, seq: str) -> List[int]:
        toks = kmer_tokens(seq, self.k)
        return [self.stoi.get(t, 0) for t in toks]
    @property
    def size(self):
        return len(self.itos)


                               
             
                               
class VConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1,
                 bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        K = kernel_size
        L0 = torch.tensor(0.0)
        R0 = torch.tensor(float(K-1))
        self.boundary = nn.Parameter(torch.stack([
            L0.repeat(out_channels), R0.repeat(out_channels)
        ], dim=1))

    def _mask(self) -> torch.Tensor:
        w = self.conv.weight
        C_out, C_in, K = w.shape
        device = w.device
        idx = torch.arange(K, device=device).float()[None, None, :]
        L = self.boundary[:,0].view(C_out,1,1)
        R = self.boundary[:,1].view(C_out,1,1)
        s1 = torch.sigmoid(idx - L)
        s2 = torch.sigmoid(R - idx)
        m = torch.clamp(s1 + s2 - 1.0, 0.0, 1.0)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self._mask()
        w = self.conv.weight * m
        return F.conv1d(x, w, bias=self.conv.bias,
                         stride=self.conv.stride, padding=self.conv.padding,
                         dilation=self.conv.dilation, groups=self.conv.groups)


                               
                  
                               
class ConvBlock(nn.Module):
    def __init__(self, C_in, C_out, K, dropout=0.3):
        super().__init__()
        pad = K//2
        self.seq = nn.Sequential(
            VConv1d(C_in, C_out, K, padding=pad),
            nn.BatchNorm1d(C_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.seq(x)

class BranchESM(nn.Module):
    def __init__(self, D_in, hidden=256, ksize=7, n_layers=2, dropout=0.3):
        super().__init__()
        layers = []; C = D_in
        for _ in range(n_layers):
            layers.append(ConvBlock(C, CONFIG['VCONV_KERNELS'], ksize, dropout)); C = CONFIG['VCONV_KERNELS']
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                                  nn.Linear(C, hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, 1))
    def forward(self, x):
        h = self.backbone(x); return self.head(h)

class KmerHead(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, ksize=5, n_layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        layers = []; C = embed_dim
        for _ in range(n_layers):
            layers.append(ConvBlock(C, 64, ksize, dropout)); C = 64
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                                  nn.Linear(C, 128), nn.GELU(), nn.Dropout(dropout))
    def forward(self, x_ids):
        e = self.emb(x_ids).transpose(1,2); h = self.backbone(e); return self.proj(h)

class BranchKmer(nn.Module):
    def __init__(self, vocab_sizes: Dict[int,int], embed_dim=32, ksize=5, dropout=0.3):
        super().__init__()
        self.ks = sorted(vocab_sizes.keys())
        self.heads = nn.ModuleDict({
            str(k): KmerHead(vocab_sizes[k], embed_dim, ksize=max(3, k), n_layers=1, dropout=dropout)
            for k in self.ks
        })
        in_dim = 128 * len(self.ks)
        self.out = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(256, 1))
    def forward(self, x_dict):
        feats = [self.heads[str(k)](x_dict[k]) for k in self.ks]
        h = torch.cat(feats, dim=1); return self.out(h)


class FourWayBlend(nn.Module):
       
                 
                                                                                
                                      
       
    def __init__(self,
                 branch_models: Dict[str, nn.Module],
                 weights: np.ndarray):
        super().__init__()
        self.branch_names = list(branch_models.keys())
        self.branches = nn.ModuleDict(branch_models)

        w_np = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert w_np.shape[0] == len(self.branch_names), \
            f"FourWayBlend: 权重个数 {w_np.shape[0]} 和分支数 {len(self.branch_names)} 不一致"

        w = torch.from_numpy(w_np).view(1, -1, 1)           
        self.register_buffer('w', w)

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        probs = []
        for name in self.branch_names:
            m = self.branches[name]
            if name in ('kmer', 'esmfold', 'physchem'):
                p = torch.sigmoid(m(batch[name]))
            else:
                p = torch.sigmoid(m(batch[name]))

            probs.append(p)
        P = torch.stack(probs, dim=1)               
        out = (P * self.w).sum(dim=1)             
        return out


class FiveWayStack(nn.Module):
       
                       
                                  
       
    def __init__(self,
                 branch_models: Dict[str, nn.Module],
                 meta_state_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.branch_names = list(branch_models.keys())
        self.branches = nn.ModuleDict(branch_models)

        self.meta = nn.Linear(len(self.branch_names), 1)
        if meta_state_dict is not None:
            self.meta.load_state_dict(meta_state_dict)

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        probs = []
        for name in self.branch_names:
            m = self.branches[name]
            if name == 'kmer':
                p = torch.sigmoid(m(batch['kmer']))
            else:
                p = torch.sigmoid(m(batch[name]))
            probs.append(p)

        feats = torch.cat(probs, dim=1)           
        logit = self.meta(feats)                  
        return torch.sigmoid(logit)


                               
             
                               
@dataclass
class Sample:
    uid: str
    seq: str
    label: int
    physchem: Optional[np.ndarray] = None

class PalmDataset(Dataset):
    def __init__(self, samples,
                 branch: str,
                 esm_h5: Optional[str] = None,
                 kmer_vocabs: Optional[Dict[int, KmerVocab]] = None,
                 norm_stats: Optional[Dict[str, np.ndarray]] = None,
                 uid2idx: Optional[dict] = None,
                 pdb_dir: Optional[str] = None) -> None:
        self.samples = samples
        self.branch = branch
        self.esm_h5_path = esm_h5
        self.kvoc = kmer_vocabs
        self.stats = norm_stats or {}
        self.uid2idx = uid2idx

                   
        self.h5 = None
        if self.branch == 'esm' and self.esm_h5_path is not None:
            import h5py
            self.h5 = h5py.File(self.esm_h5_path, "r")

                    
        self.pdb_dir = pdb_dir
        self.uid2pdb: Optional[Dict[str, str]] = None
        self._dist_cache: Dict[str, np.ndarray] = {}
        if self.branch in ('physchem', 'esmfold'):
            if not self.pdb_dir:
                raise RuntimeError(f"{self.branch} 分支需要提供 pdb_dir")
            self.uid2pdb = build_pdb_uid_map(self.pdb_dir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        y = np.float32(s.label)

        if self.branch == 'physchem':
                             
            if self.uid2pdb is None:
                if self.pdb_dir is None:
                    raise RuntimeError("physchem 分支需要传入 pdb_dir")
                self.uid2pdb = build_pdb_uid_map(self.pdb_dir)

            path = self.uid2pdb.get(s.uid)
            if path is None:
                path = self.uid2pdb.get(_strip_uid_suffix(s.uid))
            if path is None:
                raise RuntimeError(f"[physchem] 找不到 {s.uid} 对应的 PDB 文件")

            residues = load_residues_from_pdb(path)
            seq_feat, L = get_seq_feat(residues, MAX_LEN)                     

            x = {
                'seq':    torch.from_numpy(seq_feat),                              
                'length': torch.tensor(L, dtype=torch.long),
            }
            return {'x': x, 'y': torch.tensor(y)}

        if self.branch == 'esmfold':
                             
            if self.uid2pdb is None:
                if self.pdb_dir is None:
                    raise RuntimeError("esmfold 分支需要传入 pdb_dir")
                self.uid2pdb = build_pdb_uid_map(self.pdb_dir)

            path = self.uid2pdb.get(s.uid)
            if path is None:
                path = self.uid2pdb.get(_strip_uid_suffix(s.uid))
            if path is None:
                raise RuntimeError(f"[esmfold] 找不到 {s.uid} 对应的 PDB 文件")

            residues = load_residues_from_pdb(path)
            pair_feat, L = build_pair_feature_from_residues(residues, MAX_LEN)             

            x = {
                'pair':   torch.from_numpy(pair_feat),                    
                'length': torch.tensor(L, dtype=torch.long),
            }
            return {'x': x, 'y': torch.tensor(y)}


        if self.branch == 'esm':
            if self.h5 is None:
                raise RuntimeError("ESM 分支未打开 h5")
            if self.uid2idx is None or s.uid not in self.uid2idx:
                raise RuntimeError(f"uid={s.uid} 不在 uid2idx 中")
            j = self.uid2idx[s.uid]
            x = self.h5['window_emb'][j].astype(np.float32).T
            if 'esm_mean' in self.stats:
                x = (x - self.stats['esm_mean'][:, None]) / (self.stats['esm_std'][:, None] + 1e-8)
            return {'x': torch.from_numpy(x), 'y': torch.tensor(y)}

        if self.branch == 'kmer':
            x_dict = {}
            for k, vocab in self.kvoc.items():
                ids = vocab.encode(s.seq)
                x_dict[k] = torch.tensor(ids, dtype=torch.long)
            return {'x': x_dict, 'y': torch.tensor(y)}
        raise ValueError('Unknown branch')

                             
def load_csv(csv_path: str) -> List[Sample]:
    df = pd.read_csv(csv_path)
    
                    
    cols = [c.strip() for c in df.columns]
    low = [c.lower() for c in cols]
    
                   
    def pick(target_names, default_idx):
        for t in target_names:
            if t in low:
                return cols[low.index(t)]
        if default_idx < len(cols):
            return cols[default_idx]
        return None

                                                   
    idc = pick(['id', 'uniprotid', 'uid'], 0)
    seqc = pick(['window', 'seq', 'sequence'], 1)
    labc = pick(['lab', 'label', 'target'], 2)
    
    if not all([idc, seqc, labc]):
        print(f"[Warn] 无法完美匹配列名，尝试使用默认索引 0,1,2。检测到的列: {cols}")
                          
        idc, seqc, labc = cols[0], cols[1], cols[2]

    print(f"[Data] 使用列名: ID={idc}, Seq={seqc}, Label={labc}")

    samples = []
    for _, row in df.iterrows():
        uid = str(row[idc])
        seq = ensure_len_31(clean_seq(str(row[seqc])))
        lab = int(row[labc])
        
                      
                                          
                                         
        pc = np.zeros(7, dtype=np.float32)
        
        samples.append(Sample(uid, seq, lab, pc))
    return samples


                      
def augment_samples_for_physchem(samples: List[Sample], per_sample=2, max_muts=2) -> List[Sample]:
    if per_sample <= 0 or not BIOPY_OK:
        return samples
    auged: List[Sample] = []
    for s in samples:
        auged.append(s)
        for i_aug in range(per_sample):
            new_seq = aug_mutate_seq(s.seq, max_muts=max_muts, keep_center_c=True)
            new_pc = np.array(physchem_from_seq(new_seq), dtype=np.float32)
            auged.append(Sample(s.uid+f'#aug{i_aug}', new_seq, s.label, new_pc))
    return auged


                      
from sklearn.model_selection import StratifiedKFold, train_test_split

def split_train_test(samples: List[Sample], test_ratio=0.1, split_seed=2025):
    y = np.array([s.label for s in samples])
    idx = np.arange(len(samples))
    tr_idx, te_idx = train_test_split(idx, test_size=test_ratio, random_state=split_seed, stratify=y)
    train_samples = [samples[i] for i in tr_idx]
    test_samples  = [samples[i] for i in te_idx]
    return train_samples, test_samples

def build_folds(train_samples: List[Sample], n_folds=5, cv_seed=1314):
    y_tr = np.array([s.label for s in train_samples])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)
    folds = []
    for tr_sub, va_sub in skf.split(np.arange(len(train_samples)), y_tr):
        tr_list = [train_samples[i] for i in tr_sub]
        va_list = [train_samples[i] for i in va_sub]
        folds.append((tr_list, va_list))
    return folds


                 
def esm_channel_stats(esm_h5: str, samples: List[Sample], uid2idx: dict) -> Tuple[np.ndarray, np.ndarray]:
    import h5py
    idxes = np.array([uid2idx[s.uid] for s in samples], dtype=np.int64)
    uniq_idx, counts = np.unique(idxes, return_counts=True)
    with h5py.File(esm_h5, 'r') as f:
        D = f['window_emb'].shape[-1]
        S  = np.zeros(D, dtype=np.float64)
        SS = np.zeros(D, dtype=np.float64)
        n  = 0
        for i, c in zip(uniq_idx, counts):
            x = f['window_emb'][int(i)].astype(np.float32)
            S  += x.sum(axis=0)      * c
            SS += (x**2).sum(axis=0) * c
            n  += x.shape[0] * c
        mean = S / n
        var  = SS / n - mean**2
        std  = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)



def physchem_stats(samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.stack([s.physchem for s in samples], axis=0).astype(np.float32)
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return mean, std


                               
                 
                               
def make_balanced_by_pos(samples: List[Sample], seed: int = 0) -> List[Sample]:
    rng = random.Random(seed)
    pos = [s for s in samples if int(s.label) == 1]
    neg = [s for s in samples if int(s.label) == 0]
    if len(pos) == 0 or len(neg) == 0:
        return samples[:]
    if len(neg) >= len(pos):
        neg_sel = rng.sample(neg, len(pos))
        balanced = pos + neg_sel
    else:
        pos_sel = rng.sample(pos, len(neg))
        balanced = pos_sel + neg
    rng.shuffle(balanced)
    print(f"[Balance] train subset: pos={len(pos)}, neg={len(neg)} -> balanced={len(balanced)} (1:1 by pos)")
    return balanced


                               
                 
                               
@dataclass
class TrainResult:
    oof_prob: np.ndarray
    test_prob: np.ndarray
    best_states: List[Dict]
    fold_metrics: List[Dict]

CURRENT_REPEAT = 0

def train_one_branch(branch_name: str,
                     train_samples: List[Sample],
                     test_samples: List[Sample],
                     folds: List[Tuple[List[Sample], List[Sample]]],
                     kmer_vocabs: Optional[Dict[int, KmerVocab]]=None) -> TrainResult:
    device = CONFIG['DEVICE']
    B = CONFIG['BATCH_SIZE']
    E = CONFIG['EPOCHS']
    lr = CONFIG['LR']
    wd = CONFIG['WEIGHT_DECAY']

    uid2idx = None
    if branch_name == 'esm':
        uid2idx = build_uid2idx(CONFIG['ESM_H5'])

    N_train = len(train_samples)
    N_test  = len(test_samples)
    oof_prob = np.zeros(N_train, dtype=np.float32)
    test_prob_folds: List[np.ndarray] = []
    best_states: List[Dict] = []
    fold_metrics: List[Dict] = []

    all_train_idx = {id(s): i for i, s in enumerate(train_samples)}
    overall_best_metric = -1.0
    overall_best_state  = None
    metric_key = 'best_val_' + CONFIG['PRIMARY_METRIC']

    for fold_id, (tr_list, va_list) in enumerate(folds):
        print(f"\n==== [{branch_name}] Fold {fold_id+1}/{len(folds)} ====")

        tr_core = tr_list
        if CONFIG.get('BALANCE_POS_NEG', False):
            seed_bal = (CONFIG['CV_SEED'] + CURRENT_REPEAT*10007 + fold_id*97)
            tr_core = make_balanced_by_pos(tr_list, seed=seed_bal)

        stats: Dict[str, np.ndarray] = {}
        if branch_name == 'esm':
            mean, std = esm_channel_stats(CONFIG['ESM_H5'], tr_core, uid2idx)
            stats['esm_mean'] = mean
            stats['esm_std']  = std

        if branch_name == 'physchem':
            ds_tr = PalmDataset(tr_core,      'physchem', pdb_dir=CONFIG['PDB_DIR'])
            ds_va = PalmDataset(va_list,      'physchem', pdb_dir=CONFIG['PDB_DIR'])
            ds_te = PalmDataset(test_samples, 'physchem', pdb_dir=CONFIG['PDB_DIR'])

        elif branch_name == 'esm':
            ds_tr = PalmDataset(tr_core,      'esm', esm_h5=CONFIG['ESM_H5'],
                                 norm_stats=stats, uid2idx=uid2idx)
            ds_va = PalmDataset(va_list,      'esm', esm_h5=CONFIG['ESM_H5'],
                                 norm_stats=stats, uid2idx=uid2idx)
            ds_te = PalmDataset(test_samples, 'esm', esm_h5=CONFIG['ESM_H5'],
                                 norm_stats=stats, uid2idx=uid2idx)

        elif branch_name == 'kmer':
            ds_tr = PalmDataset(tr_core,      'kmer', kmer_vocabs=kmer_vocabs)
            ds_va = PalmDataset(va_list,      'kmer', kmer_vocabs=kmer_vocabs)
            ds_te = PalmDataset(test_samples, 'kmer', kmer_vocabs=kmer_vocabs)

        elif branch_name == 'esmfold':
            ds_tr = PalmDataset(tr_core,      'esmfold', pdb_dir=CONFIG['PDB_DIR'])
            ds_va = PalmDataset(va_list,      'esmfold', pdb_dir=CONFIG['PDB_DIR'])
            ds_te = PalmDataset(test_samples, 'esmfold', pdb_dir=CONFIG['PDB_DIR'])

        else:
            raise ValueError('Unknown branch')

        kw = loader_kwargs()
        kw_tr = {**kw, 'shuffle': True}
        kw_ev = {**kw, 'shuffle': False}
        dl_tr = DataLoader(ds_tr, batch_size=B, **kw_tr)
        dl_va = DataLoader(ds_va, batch_size=B, **kw_ev)
        dl_te = DataLoader(ds_te, batch_size=B, **kw_ev)

        if branch_name == 'physchem':
            model = BranchPhyschem(
                seq_feat_dim=SEQ_FEAT_DIM,
                seq_hidden=128,
            ).to(device)

        elif branch_name == 'esm':
            with h5py.File(CONFIG['ESM_H5'], 'r') as f:
                D_in = f['window_emb'].shape[-1]
            model = BranchESM(
                D_in=D_in, hidden=CONFIG['HIDDEN'],
                ksize=CONFIG['VCONV_KSIZE'],
                n_layers=CONFIG['VCONV_LAYERS'],
                dropout=CONFIG['DROPOUT'],
            ).to(device)

        elif branch_name == 'esmfold':
            model = BranchESMFold(
                in_channels=6,
                pair_hidden=128,
                node_hidden=256,
                gnn_layers=2,
            ).to(device)
        else:          
            if kmer_vocabs is None:
                raise RuntimeError("branch='kmer' 需要传入 kmer_vocabs")
            vocab_sizes = {k: v.size for k, v in kmer_vocabs.items()}
            model = BranchKmer(vocab_sizes, embed_dim=CONFIG['KMER_EMBED'], ksize=5,
                               dropout=CONFIG['DROPOUT_KMER']).to(device)


        wd_eff = CONFIG['WEIGHT_DECAY_KMER'] if branch_name == 'kmer' else wd
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd_eff)
        criterion = nn.BCEWithLogitsLoss()
        scaler = build_grad_scaler()

        best_metric = -1.0
        best_state = None
        patience = int(CONFIG.get('EARLY_STOP_PATIENCE', 5))
        no_improve = 0

        for epoch in range(1, E + 1):
            model.train()
            for batch in dl_tr:
                optimizer.zero_grad(set_to_none=True)
                y = batch['y'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float().unsqueeze(-1)
                if branch_name in ('kmer', 'esmfold', 'physchem'):
                    x = {k: v.to(device, non_blocking=CONFIG['NON_BLOCKING'])
                         for k, v in batch['x'].items()}
                    with get_autocast_context():
                        logit = model(x)
                        loss = criterion(logit, y)
                else:
                    x = batch['x'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float()
                    with get_autocast_context():
                        logit = model(x)
                        loss = criterion(logit, y)
                backward_with_optional_amp(loss, optimizer, scaler)

            model.eval()
            with torch.no_grad():
                probs, ys = [], []
                for batch in dl_va:
                    y = batch['y'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float().unsqueeze(-1)
                    if branch_name in ('kmer', 'esmfold', 'physchem'):
                        x = {k: v.to(device, non_blocking=CONFIG['NON_BLOCKING'])
                             for k, v in batch['x'].items()}
                        logit = model(x)
                    else:
                        x = batch['x'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float()
                        logit = model(x)
                    probs.append(torch.sigmoid(logit).cpu().numpy().ravel())
                    ys.append(y.cpu().numpy().ravel())
                yv = np.concatenate(ys); pv = np.concatenate(probs)
                met = calc_metrics(yv, pv)
                print(f"Epoch {epoch:02d}: val_auc={met['auc']:.4f}")

                cur = met[CONFIG['PRIMARY_METRIC']]
                if cur > best_metric:
                    best_metric = cur
                    best_state = copy.deepcopy(model.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print("Early stopping triggered."); break

        assert best_state is not None, "best_state 不应为 None"
        save_dir = os.path.join(CONFIG['MODEL_DIR'], f"repeat{CURRENT_REPEAT}", branch_name)
        os.makedirs(save_dir, exist_ok=True)
        fold_path = os.path.join(save_dir, f"{branch_name}_rep{CURRENT_REPEAT}_fold{fold_id+1}_best.pth")
        torch.save({'state_dict': best_state,
                    'branch': branch_name,
                    'fold': fold_id+1,
                    'repeat': CURRENT_REPEAT,
                    'metric': best_metric,
                    'config': CONFIG}, fold_path)

        if best_metric > overall_best_metric:
            overall_best_metric = best_metric
            overall_best_state  = best_state

        best_states.append(best_state)

        model.load_state_dict(best_state); model.eval()

        with torch.no_grad():
            probs = []
            for batch in dl_va:
                if branch_name in ('kmer', 'esmfold', 'physchem'):
                    x = {k: v.to(device, non_blocking=CONFIG['NON_BLOCKING'])
                         for k, v in batch['x'].items()}
                    logit = model(x)
                else:
                    x = batch['x'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float()
                    logit = model(x)
                probs.append(torch.sigmoid(logit).cpu().numpy().ravel())
            pv = np.concatenate(probs)
        va_global_idx = [all_train_idx[id(s)] for s in va_list]
        oof_prob[va_global_idx] = pv

        with torch.no_grad():
            probs = []
            for batch in dl_te:
                if branch_name in ('kmer', 'esmfold', 'physchem'):
                    x = {k: v.to(device, non_blocking=CONFIG['NON_BLOCKING'])
                         for k, v in batch['x'].items()}
                    logit = model(x)
                else:
                    x = batch['x'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float()
                    logit = model(x)

                probs.append(torch.sigmoid(logit).cpu().numpy().ravel())
            pt = np.concatenate(probs)
        test_prob_folds.append(pt)

        fold_metrics.append({'fold': fold_id, metric_key: float(best_metric)})

    if overall_best_state is not None:
        save_dir = os.path.join(CONFIG['MODEL_DIR'], f"repeat{CURRENT_REPEAT}", branch_name)
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, f"{branch_name}_rep{CURRENT_REPEAT}_BESTOVERALL.pth")
        torch.save({'state_dict': overall_best_state,
                    'branch': branch_name,
                    'repeat': CURRENT_REPEAT,
                    'metric': overall_best_metric,
                    'config': CONFIG}, best_path)

    test_prob = np.mean(np.stack(test_prob_folds, axis=0), axis=0)

    with open(os.path.join(CONFIG['OUT_DIR'], f'{branch_name}_fold_metrics.json'), 'w') as f:
        json.dump(fold_metrics, f, indent=2)

    return TrainResult(oof_prob, test_prob, best_states, fold_metrics)


def train_branch_full_on_all(branch_name: str,
                             train_all: List[Sample],
                             kmer_vocabs: Optional[Dict[int, KmerVocab]] = None) -> Dict:
       
                                                   
                                                     
       
    device = CONFIG['DEVICE']
    B = CONFIG['BATCH_SIZE']
    E = CONFIG['EPOCHS']
    lr = CONFIG['LR']
    wd = CONFIG['WEIGHT_DECAY']

                           
    y = np.array([s.label for s in train_all])
    idx = np.arange(len(train_all))
    tr_idx, va_idx = train_test_split(
        idx,
        test_size=0.1,
        random_state=CONFIG['SPLIT_SEED'],
        stratify=y
    )
    tr_list = [train_all[i] for i in tr_idx]
    va_list = [train_all[i] for i in va_idx]

                       
    tr_core = tr_list
    if CONFIG.get('BALANCE_POS_NEG', False):
        seed_bal = CONFIG['CV_SEED'] + 777
        tr_core = make_balanced_by_pos(tr_list, seed=seed_bal)

                
    stats: Dict[str, np.ndarray] = {}
    uid2idx = None
    if branch_name == 'esm':
        uid2idx = build_uid2idx(CONFIG['ESM_H5'])
        mean, std = esm_channel_stats(CONFIG['ESM_H5'], tr_core, uid2idx)
        stats['esm_mean'] = mean
        stats['esm_std'] = std

                                     
    if branch_name == 'physchem':
        ds_tr = PalmDataset(tr_core, 'physchem', pdb_dir=CONFIG['PDB_DIR'])
        ds_va = PalmDataset(va_list, 'physchem', pdb_dir=CONFIG['PDB_DIR'])

    elif branch_name == 'esm':
        ds_tr = PalmDataset(
            tr_core,
            'esm',
            esm_h5=CONFIG['ESM_H5'],
            norm_stats=stats,
            uid2idx=uid2idx,
        )
        ds_va = PalmDataset(
            va_list,
            'esm',
            esm_h5=CONFIG['ESM_H5'],
            norm_stats=stats,
            uid2idx=uid2idx,
        )

    elif branch_name == 'esmfold':
        ds_tr = PalmDataset(tr_core, 'esmfold', pdb_dir=CONFIG['PDB_DIR'])
        ds_va = PalmDataset(va_list, 'esmfold', pdb_dir=CONFIG['PDB_DIR'])

    elif branch_name == 'kmer':
        if kmer_vocabs is None:
            raise RuntimeError("branch='kmer' 需要传入 kmer_vocabs")
        ds_tr = PalmDataset(tr_core, 'kmer', kmer_vocabs=kmer_vocabs)
        ds_va = PalmDataset(va_list, 'kmer', kmer_vocabs=kmer_vocabs)

    else:
        raise ValueError(f'Unknown branch_name={branch_name}')

    kw = loader_kwargs()
    dl_tr = DataLoader(ds_tr, batch_size=B, **{**kw, 'shuffle': True})
    dl_va = DataLoader(ds_va, batch_size=B, **{**kw, 'shuffle': False})
                  
    if branch_name == 'physchem':
        model = BranchPhyschem(
            seq_feat_dim=SEQ_FEAT_DIM,
            seq_hidden=128,
        ).to(device)

    elif branch_name == 'esm':
        with h5py.File(CONFIG['ESM_H5'], 'r') as f:
            D_in = f['window_emb'].shape[-1]
        model = BranchESM(
            D_in=D_in,
            hidden=CONFIG['HIDDEN'],
            ksize=CONFIG['VCONV_KSIZE'],
            n_layers=CONFIG['VCONV_LAYERS'],
            dropout=CONFIG['DROPOUT'],
        ).to(device)

    elif branch_name == 'esmfold':
        model = BranchESMFold(
            in_channels=6,                                                     
            pair_hidden=128,
            node_hidden=256,
            gnn_layers=2,
        ).to(device)

    else:          
        if kmer_vocabs is None:
            raise RuntimeError("branch='kmer' 需要传入 kmer_vocabs")
        vocab_sizes = {k: v.size for k, v in kmer_vocabs.items()}
        model = BranchKmer(
            vocab_sizes,
            embed_dim=CONFIG['KMER_EMBED'],
            ksize=5,
            dropout=CONFIG['DROPOUT_KMER'],
        ).to(device)



    wd_eff = CONFIG['WEIGHT_DECAY_KMER'] if branch_name == 'kmer' else wd
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd_eff)
    criterion = nn.BCEWithLogitsLoss()
    scaler = build_grad_scaler()

    best_metric = -1.0
    best_state = None
    patience = int(CONFIG.get('EARLY_STOP_PATIENCE', 5))
    no_improve = 0

    for epoch in range(1, E + 1):
                      
        model.train()
        for batch in dl_tr:
            optimizer.zero_grad(set_to_none=True)
            yb = batch['y'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float().unsqueeze(-1)

            if branch_name in ('kmer', 'esmfold', 'physchem'):
                xb = {k: v.to(device, non_blocking=CONFIG['NON_BLOCKING'])
                      for k, v in batch['x'].items()}
                with get_autocast_context():
                    logit = model(xb)
                    loss = criterion(logit, yb)
            else:
                xb = batch['x'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float()
                with get_autocast_context():
                    logit = model(xb)
                    loss = criterion(logit, yb)

            backward_with_optional_amp(loss, optimizer, scaler)

                      
        model.eval()
                      
        model.eval()
        with torch.no_grad():
            probs, ys = [], []
            for batch in dl_va:
                yb = batch['y'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float().unsqueeze(-1)
                if branch_name in ('kmer', 'esmfold', 'physchem'):
                    xb = {k: v.to(device, non_blocking=CONFIG['NON_BLOCKING'])
                          for k, v in batch['x'].items()}
                    logit = model(xb)
                else:
                    xb = batch['x'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float()
                    logit = model(xb)

                probs.append(torch.sigmoid(logit).cpu().numpy().ravel())
                ys.append(yb.cpu().numpy().ravel())

            yv = np.concatenate(ys)
            pv = np.concatenate(probs)
            met = calc_metrics(yv, pv)
            print(f"[FINAL-{branch_name}] Epoch {epoch:02d}: val_auc={met['auc']:.4f}")

            cur = met[CONFIG['PRIMARY_METRIC']]
            if cur > best_metric:
                best_metric = cur
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"[FINAL-{branch_name}] Early stopping.")
                break

    assert best_state is not None, "FINAL 阶段 best_state 不应为 None"

                                                     
    save_dir = os.path.join(CONFIG['MODEL_DIR'], f"repeat{CONFIG['FINAL_REPEAT_ID']}", branch_name)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(
        save_dir,
        f"{branch_name}_rep{CONFIG['FINAL_REPEAT_ID']}_BESTOVERALL.pth"
    )
    ckpt = {
        'state_dict': best_state,
        'branch': branch_name,
        'repeat': CONFIG['FINAL_REPEAT_ID'],
        'metric': float(best_metric),
        'config': CONFIG,
    }
    torch.save(ckpt, best_path)
    print(f"[Save][FINAL] {branch_name} -> {best_path}")
    return ckpt


                               
        
                               
def main():
    print('==> 加载数据…')
    samples = load_csv(CONFIG['CSV_PATH'])
    samples = summarize_and_dedup(samples)

    train_all, test_samples = split_train_test(samples, CONFIG['TEST_RATIO'], CONFIG['SPLIT_SEED'])
    print(f"[划分] 训练={len(train_all)}  测试={len(test_samples)}  (SPLIT_SEED={CONFIG['SPLIT_SEED']})")

    enabled_branches = get_enabled_branches()
    print(f"[Config] 启用的分支 = {enabled_branches}")

    kvoc: Dict[int, KmerVocab] = {}
    if 'kmer' in enabled_branches:
        print('==> 构建 k-mer 词表…')
        for k in CONFIG['KMERS']:
            max_size = None
            if k == 5:
                max_size = CONFIG['TOP_KMER_K5']
            elif k == 4 and CONFIG['TOP_KMER_K4'] is not None:
                max_size = CONFIG['TOP_KMER_K4']
            voc = KmerVocab(k, max_size=max_size)
            for s in train_all:
                voc.add_seq(s.seq)
            voc.finalize()
            kvoc[k] = voc
            print(f'k={k}  词表大小: {voc.size}')
    else:
        print('[Config] kmer 分支未启用，跳过 k-mer 词表构建。')


    y_train = np.array([s.label for s in train_all]).astype(float)
    y_test  = np.array([s.label for s in test_samples]).astype(float)
    uids_tr = [s.uid for s in train_all]
    uids_te = [s.uid for s in test_samples]

    rows_metrics = []
    last_pred_cache = {}

    global CURRENT_REPEAT
    for rep in range(CONFIG['N_REPEATS']):
        CURRENT_REPEAT = rep + 1
        cv_seed = CONFIG['CV_SEED'] + rep
        print(f"\n========== 重复 {rep+1}/{CONFIG['N_REPEATS']}（CV_SEED={cv_seed}） ==========")
        folds = build_folds(train_all, CONFIG['N_FOLDS'], cv_seed=cv_seed)

                              
        branch_results: Dict[str, TrainResult] = {}
        for b in enabled_branches:
            print(f"\n[Repeat {rep+1}] 训练分支: {b}")
            if b == 'kmer':
                branch_results[b] = train_one_branch(b, train_all, test_samples, folds, kmer_vocabs=kvoc)
            else:
                branch_results[b] = train_one_branch(b, train_all, test_samples, folds)

                                    
        branch_order = [b for b in BRANCH_ORDER if b in enabled_branches]
        oof_mat = np.stack([branch_results[b].oof_prob for b in branch_order], axis=1)
        test_mat = np.stack([branch_results[b].test_prob for b in branch_order], axis=1)

                                 
        w_best, _ = grid_search_blend_weights(
            oof_mat, y_train,
            step=CONFIG['BLEND_GRID_STEP'],
            refine=CONFIG['BLEND_REFINE'],
            min_w=CONFIG['BLEND_MIN_WEIGHT']
        )

        fused_save_dir = os.path.join(CONFIG['MODEL_DIR'], f"repeat{CURRENT_REPEAT}", "fused")
        os.makedirs(fused_save_dir, exist_ok=True)

                                              
        def _load_best(branch: str):
            path = os.path.join(
                CONFIG['MODEL_DIR'], f"repeat{CURRENT_REPEAT}", branch,
                f"{branch}_rep{CURRENT_REPEAT}_BESTOVERALL.pth"
            )
            ckpt = torch.load(path, map_location='cpu')
            return ckpt['state_dict']

        branch_models_for_fused: Dict[str, nn.Module] = {}
        for b in branch_order:
            if b == 'physchem':
                                                    
                m = BranchPhyschem(
                    seq_feat_dim=SEQ_FEAT_DIM,
                    seq_hidden=128,
                )
            elif b == 'esm':
                with h5py.File(CONFIG['ESM_H5'], 'r') as f_h5:
                    D_in = f_h5['window_emb'].shape[-1]
                m = BranchESM(
                    D_in=D_in, hidden=CONFIG['HIDDEN'],
                    ksize=CONFIG['VCONV_KSIZE'],
                    n_layers=CONFIG['VCONV_LAYERS'],
                    dropout=CONFIG['DROPOUT'],
                )
            elif b == 'kmer':
                vocab_sizes = {k: v.size for k, v in kvoc.items()}
                m = BranchKmer(
                    vocab_sizes, embed_dim=CONFIG['KMER_EMBED'],
                    ksize=5, dropout=CONFIG['DROPOUT_KMER'],
                )
            elif b == 'esmfold':
                                             
                m = BranchESMFold(
                    in_channels=6,
                    pair_hidden=128,
                    node_hidden=256,
                    gnn_layers=2,
                )
            else:
                raise ValueError(f"未知分支: {b}")

            state_dict = _load_best(b)
            m.load_state_dict(state_dict)
            branch_models_for_fused[b] = m



                 
        fused_model = FourWayBlend(branch_models_for_fused, weights=w_best)
        fused_path = os.path.join(fused_save_dir, f"fused_rep{CURRENT_REPEAT}_blend.pth")
        torch.save({
            'state_dict': fused_model.state_dict(),
            'weights': w_best.tolist(),
            'branches': branch_order,
            'repeat': CURRENT_REPEAT,
            'config': CONFIG,
            'note': f"Prob-level weighted sum of branches: {branch_order}",
        }, fused_path)
        print(f"[Save] Fused model saved: {fused_path}")

                                             
        Xtr = torch.tensor(oof_mat, dtype=torch.float32)                               
        ytr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)                 

        meta = nn.Linear(Xtr.shape[1], 1)
        optm = torch.optim.LBFGS(meta.parameters(), lr=1.0, max_iter=100)

        def closure():
            optm.zero_grad()
            p = torch.sigmoid(meta(Xtr))
            loss = F.binary_cross_entropy(p, ytr)
            loss.backward()
            return loss

        optm.step(closure)

        meta_state = copy.deepcopy(meta.state_dict())

        with torch.no_grad():
            oof_prob_stack = torch.sigmoid(
                meta(torch.tensor(oof_mat, dtype=torch.float32))
            ).numpy().ravel()
            test_prob_stack = torch.sigmoid(
                meta(torch.tensor(test_mat, dtype=torch.float32))
            ).numpy().ravel()

        fused_stack_model = FiveWayStack(branch_models_for_fused, meta_state_dict=meta_state)
        fused_stack_path = os.path.join(fused_save_dir, f"fused_rep{CURRENT_REPEAT}_stack.pth")
        torch.save({
            'state_dict': fused_stack_model.state_dict(),
            'branches': branch_order,
            'repeat': CURRENT_REPEAT,
            'config': CONFIG,
            'note': f"Prob-level stacking of branches: {branch_order}",
        }, fused_stack_path)
        print(f"[Save] Fused STACK model saved: {fused_stack_path}")

                                     
        test_prob_blend = (test_mat * w_best[None, :]).sum(axis=1)
        oof_prob_blend  = (oof_mat  * w_best[None, :]).sum(axis=1)

        auc_blend_te = roc_auc_score(y_test, test_prob_blend) if SKLEARN_OK else 0.0
        auc_stack_te = roc_auc_score(y_test, test_prob_stack) if SKLEARN_OK else 0.0
        print(f"[Repeat {rep+1}] blend AUC={auc_blend_te:.4f}, stack AUC={auc_stack_te:.4f}")

        method_probs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
            b: (branch_results[b].oof_prob, branch_results[b].test_prob)
            for b in branch_order
        }
        method_probs['blend'] = (oof_prob_blend, test_prob_blend)
        method_probs['stack'] = (oof_prob_stack, test_prob_stack)

        repeat_out_dir = os.path.join(CONFIG['OUT_DIR'], 'repeat-result')
        os.makedirs(repeat_out_dir, exist_ok=True)

        print(f"\n[Repeat {rep+1}] Test metrics:")
        for m, (_, p_te) in method_probs.items():
            mt = calc_metrics(y_test, p_te)
            print(f"  - {m:8s}: AUC={mt['auc']:.4f}, ACC={mt['acc']:.4f}, "
                  f"SEN={mt['sen']:.4f}, SPE={mt['spe']:.4f}")

        df_scores_blend = make_scores(
            uids_te,
            y_test,
            test_prob_blend,
            thr=CONFIG['METRIC_THRESHOLD']
        ).assign(method='blend')
        df_scores_blend.to_csv(
            os.path.join(repeat_out_dir, f'test_scores_rep{rep+1}_blend.csv'),
            index=False
        )

        df_scores_stack = make_scores(
            uids_te,
            y_test,
            test_prob_stack,
            thr=CONFIG['METRIC_THRESHOLD']
        ).assign(method='stack')
        df_scores_stack.to_csv(
            os.path.join(repeat_out_dir, f'test_scores_rep{rep+1}_stack.csv'),
            index=False
        )

        curves_rep = {m: (y_test, p_te) for m, (_, p_te) in method_probs.items()}
        plot_roc_multi(
            os.path.join(repeat_out_dir, f'roc_rep{rep+1}.svg'),
            curves_rep,
            title=f'ROC curves (Repeat {rep+1})'
        )

        for m, (p_oof, p_te) in method_probs.items():
            auc_oof = roc_auc_score(y_train, p_oof) if SKLEARN_OK and len(np.unique(y_train))>1 else 0.0
            auc_te  = roc_auc_score(y_test,  p_te)  if SKLEARN_OK and len(np.unique(y_test))>1 else 0.0
            rows_metrics.append({'repeat': rep+1, 'set': 'oof',  'method': m, 'auc': auc_oof})
            rows_metrics.append({'repeat': rep+1, 'set': 'test', 'method': m, 'auc': auc_te})

        if rep == CONFIG['N_REPEATS'] - 1:
            last_pred_cache = {
                'branches': branch_order,
                'preds_train': {m: p[0] for m, p in method_probs.items()},
                'preds_test':  {m: p[1] for m, p in method_probs.items()},
                'w_best': w_best.tolist(),
                'meta_state_dict': meta_state,
            }



    repeat_out_dir = os.path.join(CONFIG['OUT_DIR'], 'repeat-result')
    os.makedirs(repeat_out_dir, exist_ok=True)

    metrics_df = pd.DataFrame(rows_metrics)
    metrics_df.to_csv(
        os.path.join(repeat_out_dir, 'metrics_by_repeat.csv'),
        index=False
    )

    plot_auc_errorbars_by_method(
        os.path.join(repeat_out_dir, 'auc_errorbars_test.svg'),
        metrics_df,
        CONFIG['N_REPEATS']
    )

                                  
    print("\n==> 训练 FINAL 分支 (使用 train_all)...")
    ckpts: Dict[str, Dict] = {}
    for b in enabled_branches:
        if b == 'kmer':
            ckpts[b] = train_branch_full_on_all('kmer', train_all, kmer_vocabs=kvoc)
        else:
            ckpts[b] = train_branch_full_on_all(b, train_all)

    preds_train = last_pred_cache['preds_train']
    preds_test  = last_pred_cache['preds_test']
    w_best      = np.array(last_pred_cache['w_best'], dtype=np.float32)
    meta_state_final = last_pred_cache['meta_state_dict']

                                                                       
    final_oof_prob  = preds_train['stack']
    final_test_prob = preds_test['stack']



    FINA_DIR = os.path.join(CONFIG['OUT_DIR'], 'FINA')
    os.makedirs(FINA_DIR, exist_ok=True)

              
    tr_aug_tmp = train_all
    if CONFIG.get('AUG_PER_SAMPLE', 0) and BIOPY_OK:
        tr_aug_tmp = augment_samples_for_physchem(
            train_all,
            per_sample=CONFIG['AUG_PER_SAMPLE'],
            max_muts=CONFIG['AUG_MAX_MUTS']
        )
    pc_mean, pc_std = physchem_stats(tr_aug_tmp)
    stats_pc = {'pc_mean': pc_mean, 'pc_std': pc_std}

    stats_esm = {}
    if 'esm' in enabled_branches:
        uid2idx_final = build_uid2idx(CONFIG['ESM_H5'])
        esm_mean, esm_std = esm_channel_stats(CONFIG['ESM_H5'], train_all, uid2idx_final)
        stats_esm = {'esm_mean': esm_mean, 'esm_std': esm_std}

                                        
    kw = loader_kwargs()
    dl_tr_by_branch: Dict[str, DataLoader] = {}
    dl_te_by_branch: Dict[str, DataLoader] = {}

    if 'physchem' in enabled_branches:
                                                       
        ds_tr = PalmDataset(train_all, 'physchem',
                            norm_stats=stats_pc,
                            pdb_dir=CONFIG['PDB_DIR'])
        ds_te = PalmDataset(test_samples, 'physchem',
                            norm_stats=stats_pc,
                            pdb_dir=CONFIG['PDB_DIR'])
        dl_tr_by_branch['physchem'] = DataLoader(
            ds_tr, batch_size=CONFIG['BATCH_SIZE'], **{**kw, 'shuffle': False}
        )
        dl_te_by_branch['physchem'] = DataLoader(
            ds_te, batch_size=CONFIG['BATCH_SIZE'], **{**kw, 'shuffle': False}
        )

    if 'esm' in enabled_branches:
        ds_tr = PalmDataset(train_all, 'esm', esm_h5=CONFIG['ESM_H5'],
                            norm_stats=stats_esm, uid2idx=uid2idx_final)
        ds_te = PalmDataset(test_samples, 'esm', esm_h5=CONFIG['ESM_H5'],
                            norm_stats=stats_esm, uid2idx=uid2idx_final)
        dl_tr_by_branch['esm'] = DataLoader(
            ds_tr, batch_size=CONFIG['BATCH_SIZE'], **{**kw, 'shuffle': False}
        )
        dl_te_by_branch['esm'] = DataLoader(
            ds_te, batch_size=CONFIG['BATCH_SIZE'], **{**kw, 'shuffle': False}
        )

    if 'kmer' in enabled_branches:
        ds_tr = PalmDataset(train_all, 'kmer', kmer_vocabs=kvoc)
        ds_te = PalmDataset(test_samples, 'kmer', kmer_vocabs=kvoc)
        dl_tr_by_branch['kmer'] = DataLoader(
            ds_tr, batch_size=CONFIG['BATCH_SIZE'], **{**kw, 'shuffle': False}
        )
        dl_te_by_branch['kmer'] = DataLoader(
            ds_te, batch_size=CONFIG['BATCH_SIZE'], **{**kw, 'shuffle': False}
        )

    if 'esmfold' in enabled_branches:
                      
        ds_tr = PalmDataset(train_all, 'esmfold', pdb_dir=CONFIG['PDB_DIR'])
        ds_te = PalmDataset(test_samples, 'esmfold', pdb_dir=CONFIG['PDB_DIR'])
        dl_tr_by_branch['esmfold'] = DataLoader(
            ds_tr, batch_size=CONFIG['BATCH_SIZE'], **{**kw, 'shuffle': False}
        )
        dl_te_by_branch['esmfold'] = DataLoader(
            ds_te, batch_size=CONFIG['BATCH_SIZE'], **{**kw, 'shuffle': False}
        )



                                                 
    device = CONFIG['DEVICE']

    def _infer_branch(dl, model, is_dict_input=False):
        preds = []
        with torch.no_grad():
            for batch in dl:
                if is_dict_input:
                    xb = {k: v.to(device, non_blocking=CONFIG['NON_BLOCKING'])
                          for k, v in batch['x'].items()}
                    logit = model(xb)
                else:
                    xb = batch['x'].to(device, non_blocking=CONFIG['NON_BLOCKING']).float()
                    logit = model(xb)
                preds.append(torch.sigmoid(logit).cpu().numpy().ravel())
        return np.concatenate(preds)

    preds_tr_by_branch: Dict[str, np.ndarray] = {}
    preds_te_by_branch: Dict[str, np.ndarray] = {}
    branch_order_final = [b for b in BRANCH_ORDER if b in enabled_branches]

    for b in branch_order_final:
        if b == 'physchem':
            m = BranchPhyschem(
                seq_feat_dim=SEQ_FEAT_DIM,
                seq_hidden=128,
            ).to(device)
        elif b == 'esm':
            with h5py.File(CONFIG['ESM_H5'], 'r') as f_h5:
                D_in_fin = f_h5['window_emb'].shape[-1]
            m = BranchESM(
                D_in=D_in_fin, hidden=CONFIG['HIDDEN'],
                ksize=CONFIG['VCONV_KSIZE'],
                n_layers=CONFIG['VCONV_LAYERS'],
                dropout=CONFIG['DROPOUT'],
            ).to(device)
        elif b == 'kmer':
            vocab_sizes_fin = {k: v.size for k, v in kvoc.items()}
            m = BranchKmer(
                vocab_sizes_fin, embed_dim=CONFIG['KMER_EMBED'],
                ksize=5, dropout=CONFIG['DROPOUT_KMER'],
            ).to(device)
        elif b == 'esmfold':
            m = BranchESMFold(
                in_channels=6,
                pair_hidden=128,
                node_hidden=256,
                gnn_layers=2,
            ).to(device)
        else:
            raise ValueError(f"未知分支: {b}")




                        
        m.load_state_dict(ckpts[b]['state_dict'])
        m.eval()

        preds_tr_by_branch[b] = _infer_branch(
            dl_tr_by_branch[b], m,
            is_dict_input=(b in ('kmer', 'esmfold', 'physchem'))
        )
        preds_te_by_branch[b] = _infer_branch(
            dl_te_by_branch[b], m,
            is_dict_input=(b in ('kmer', 'esmfold', 'physchem'))
        )


                                                 
    w_best = np.array(last_pred_cache['w_best'], dtype=np.float32)
    mat_tr = np.stack([preds_tr_by_branch[b] for b in branch_order_final], axis=1)
    mat_te = np.stack([preds_te_by_branch[b] for b in branch_order_final], axis=1)

    prob_tr_blend = mat_tr @ w_best
    prob_te_blend = mat_te @ w_best

    meta_final = nn.Linear(len(branch_order_final), 1)
    meta_final.load_state_dict(meta_state_final)
    with torch.no_grad():
        prob_tr_stack = torch.sigmoid(
            meta_final(torch.from_numpy(mat_tr.astype(np.float32)))
        ).numpy().ravel()
        prob_te_stack = torch.sigmoid(
            meta_final(torch.from_numpy(mat_te.astype(np.float32)))
        ).numpy().ravel()

    final_oof_prob  = prob_tr_stack
    final_test_prob = prob_te_stack
    p_stack_te = prob_te_stack

                          
    methods_final = branch_order_final + ['blend', 'stack']
    preds_final_test: Dict[str, np.ndarray] = {
        **{b: preds_te_by_branch[b] for b in branch_order_final},
        'blend': prob_te_blend,
        'stack': p_stack_te,
    }

    metrics_final = {m: calc_metrics(y_test, preds_final_test[m])
                     for m in methods_final}

    plot_metrics_bars_by_method(
        os.path.join(FINA_DIR, 'metrics_bar_FINAL_test.svg'),
        metrics_final,
        methods_final,
        title='FINAL model metrics on test'
    )

    curves_final = {m: (y_test, preds_final_test[m]) for m in methods_final}
    plot_roc_multi(
        os.path.join(FINA_DIR, 'roc_FINAL_test.svg'),
        curves_final,
        title='FINAL model ROC (test)'
    )

    df_train_final = pd.DataFrame({
        'uniprotid': uids_tr,
        'label': y_train.astype(int),
    })

               
    for b in branch_order_final:
        df_train_final[f'prob_{b}'] = preds_tr_by_branch[b]

                 
    df_train_final['prob_blend'] = prob_tr_blend
    df_train_final['prob_stack'] = prob_tr_stack

    df_train_final.to_csv(
        os.path.join(FINA_DIR, 'train_predictions_FINAL.csv'),
        index=False
    )
    print(f"[Save][FINAL] 训练集各分支与融合方式预测已保存 ->",
          os.path.join(FINA_DIR, 'train_predictions_FINAL.csv'))


              
              

                                  
    df_test_final = pd.DataFrame({
        'uniprotid': uids_te,
        'label': y_test.astype(int),
        'prob_blend': prob_te_blend
    })
    df_test_final.to_csv(
        os.path.join(FINA_DIR, 'test_predictions_FINAL.csv'),
        index=False
    )

                                  
    df_test_full = pd.DataFrame({
        'uniprotid': uids_te,
        'label': y_test.astype(int),
    })

                    
    for b in branch_order_final:                                         
        df_test_full[f'prob_{b}'] = preds_te_by_branch[b]

                 
    df_test_full['prob_blend'] = prob_te_blend
    df_test_full['prob_stack'] = prob_te_stack

    df_test_full.to_csv(
        os.path.join(FINA_DIR, 'test_predictions_FULL_FINAL.csv'),
        index=False
    )
    print(f"[Save][FINAL] 测试集各分支与融合方式预测已保存 ->",
          os.path.join(FINA_DIR, 'test_predictions_FULL_FINAL.csv'))


    species_stats_and_roc(
        uids_te,
        y_test,
        prob_te_blend,
        CONFIG.get('SPECIES_CSV', ''),
        FINA_DIR,
        min_n=CONFIG.get('SPECIES_MIN_SAMPLES', 100)
    )


                                                                  
    final_rep = CONFIG.get('FINAL_REPEAT_ID', 1000)
    fused_dir = os.path.join(CONFIG['MODEL_DIR'], f"repeat{final_rep}", "fused")
    os.makedirs(fused_dir, exist_ok=True)

    branch_models_final: Dict[str, nn.Module] = {}
    for b in branch_order_final:
        if b == 'physchem':
            m = BranchPhyschem(
                seq_feat_dim=SEQ_FEAT_DIM,
                seq_hidden=128,
            )
        elif b == 'esm':
            with h5py.File(CONFIG['ESM_H5'], 'r') as f_h5:
                D_in_final = f_h5['window_emb'].shape[-1]
            m = BranchESM(
                D_in=D_in_final, hidden=CONFIG['HIDDEN'],
                ksize=CONFIG['VCONV_KSIZE'],
                n_layers=CONFIG['VCONV_LAYERS'],
                dropout=CONFIG['DROPOUT'],
            )
        elif b == 'kmer':
            vocab_sizes_final = {k: v.size for k, v in kvoc.items()}
            m = BranchKmer(
                vocab_sizes_final, embed_dim=CONFIG['KMER_EMBED'],
                ksize=5, dropout=CONFIG['DROPOUT_KMER'],
            )
        elif b == 'esmfold':
            m = BranchESMFold(
                in_channels=6,
                pair_hidden=128,
                node_hidden=256,
                gnn_layers=2,
            )
        else:
            raise ValueError(f"未知分支: {b}")

        m.load_state_dict(ckpts[b]['state_dict'])
        branch_models_final[b] = m



    fused_final = FourWayBlend(branch_models_final, weights=w_best)
    fused_path_final = os.path.join(fused_dir, f"fused_rep{final_rep}_blend.pth")
    torch.save({
        'state_dict': fused_final.state_dict(),
        'weights': w_best.tolist(),
        'branches': branch_order_final,
        'repeat': final_rep,
        'config': CONFIG,
        'note': f'FINAL prob-level blend of branches {branch_order_final} trained on train_all',
    }, fused_path_final)
    print(f"[Save][FINAL] Fused -> {fused_path_final}")

                           
    stack_final = FiveWayStack(branch_models_final, meta_state_dict=meta_state_final)
    fused_path_final_stack = os.path.join(fused_dir, f"fused_rep{final_rep}_stack.pth")
    torch.save({
        'state_dict': stack_final.state_dict(),
        'branches': branch_order_final,
        'repeat': final_rep,
        'config': CONFIG,
        'note': f'FINAL prob-level stacking of branches {branch_order_final} trained on train_all',
    }, fused_path_final_stack)
    print(f"[Save][FINAL] Fused STACK -> {fused_path_final_stack}")

                       
    mets_oof_final = calc_metrics(y_train, final_oof_prob)
    mets_te_final  = calc_metrics(y_test,  final_test_prob)
    report = {
        'config': CONFIG,
        'final_stack_oof': {'auc': mets_oof_final['auc']},
        'final_stack_test': {'auc': mets_te_final['auc']},
        'blend_weight_lastrep': w_best.tolist(),
        'branches': branch_order_final,
        'notes': 'models/ 目录包含每个分支每折最优权重、分支整体最佳权重，每个 repeat 的融合模型，以及 “FINAL” 基于 train_all 的 N+1 模型。'
    }
    with open(os.path.join(CONFIG['OUT_DIR'], 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    kmer_vocab_path = os.path.join(CONFIG['MODEL_DIR'], "kmer_vocab.json")
    obj = {
     "meta": {
        "csv_path": CONFIG['CSV_PATH'],
        "test_ratio": CONFIG['TEST_RATIO'],
        "split_seed": CONFIG['SPLIT_SEED'],
        "kmers": CONFIG['KMERS'],
     },
     "vocabs": {
        str(k): {"k": k, "max_size": kvoc[k].max_size, "itos": kvoc[k].itos}
        for k in sorted(kvoc.keys())
     }
    }
    with open(kmer_vocab_path, "w", encoding="utf-8") as f:
     json.dump(obj, f, ensure_ascii=False)
    print("[Save] kmer vocab ->", kmer_vocab_path)

    



def species_roc_by_method(uids: List[str],
                          labels: np.ndarray,
                          probs_by_method: Dict[str, np.ndarray],
                          species_csv: str,
                          out_dir: str,
                          min_n: int = 100):
       
                           
       
    if (not species_csv) or (not os.path.exists(species_csv)):
        print("[Info] SPECIES_CSV 不存在，跳过按物种多方法 ROC。")
        return
    if not SKLEARN_OK:
        print("[Info] 未安装 sklearn，跳过按物种多方法 ROC。")
        return

    sp_map = _load_species_map_for_final(species_csv)

    uids = list(uids)
    labels = np.asarray(labels).astype(int)
    n = len(uids)

             
    for m, p in probs_by_method.items():
        if len(p) != n:
            raise ValueError(f"[species_roc_by_method] method={m} 的预测长度={len(p)} "
                             f"和样本数 n={n} 不一致")

                       
    bucket_idx: Dict[str, List[int]] = {}
    for i, uid in enumerate(uids):
        sp = sp_map.get(uid) or sp_map.get(_base_uid(uid))
        if sp is None:
            continue
        bucket_idx.setdefault(sp, []).append(i)

    rows = []
    for sp, idxs in bucket_idx.items():
        if len(idxs) <= min_n:
            continue

        y_sp = labels[idxs]
        if len(np.unique(y_sp)) < 2:
            continue

        curves = {}
        for m, p_all in probs_by_method.items():
            p_sp = np.asarray(p_all)[idxs]
                                          
            if len(np.unique(y_sp)) < 2:
                continue
            auc = roc_auc_score(y_sp, p_sp)
            curves[m] = (y_sp, p_sp)
            rows.append({'species': sp, 'method': m,
                         'n': len(y_sp), 'auc': float(auc)})

        if len(curves) < 2:
            continue

        safe_sp = sp.replace('/', '_').replace(' ', '_')
        fig_name = f"species_{safe_sp}_roc_methods_FINAL.svg"
        plot_roc_multi(
            os.path.join(out_dir, fig_name),
            curves,
            title=f"{sp} ROC by method (test)"
        )
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(out_dir, 'species_auc_by_method_FINAL.csv'),
                  index=False)
        print("[Save] 物种-方法 AUC 已输出到:", out_dir)


if __name__ == '__main__':
    main()
