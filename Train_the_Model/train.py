import os
import sys
import platform
import math
import json
import copy
import random
import re
import hashlib
import warnings
import io
import zipfile
import tempfile
import shutil


os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
import sklearn
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _ensure_python_hash_seed(seed: int = 3407):

    desired = str(int(seed))
    if os.environ.get('PYTHONHASHSEED') == desired:
        return
    if os.environ.get('DEEPPALM_HASHSEED_REEXEC') == '1':
        raise RuntimeError('Unable to enforce PYTHONHASHSEED before startup.')
    os.environ['PYTHONHASHSEED'] = desired
    os.environ['DEEPPALM_HASHSEED_REEXEC'] = '1'
    os.execv(sys.executable, [sys.executable] + sys.argv)


if __name__ == '__main__':
    _ensure_python_hash_seed(3407)
try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
    )
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    from Bio.PDB import PDBParser
    BIOPDB_OK = True
except Exception:
    PDBParser = None
    BIOPDB_OK = False

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


CSV_PATH = (
    'input.csv'
)

ESM_H5 = (
    'embedding.h5'
)

PDB_DIR = (
    'esmfold.pdb'
)

AAINDEX1_PATH = (
    'aaindex1.txt'
)

AAINDEX_PCA_PATH = (
    'AAindex_PCA.csv'
)





RUN_ROOT = (
    'output'
)


OUT_DIR = os.path.join(
    RUN_ROOT,
    "output",
)

MODEL_DIR = os.path.join(
    RUN_ROOT,
    "models",
)

CACHE_DIR = os.path.join(
    RUN_ROOT,
    "cache",
)

STRUCTURE_CACHE_DIR = os.path.join(
    CACHE_DIR,
    "structure",
)


FINAL_MODEL_PATH = os.path.join(
    RUN_ROOT,
    "DeepPalm.pth",
)







DEPLOY_BUNDLE_PATH = (
    'DeepPalm.dpalm'
)



RUN_MODE = 'auto_resume'


EXISTING_FINAL_MODEL_PATH = (
    'DeepPalm_V14_FINAL.pth'
)
EXISTING_REFERENCE_PREDICTIONS = (
    'single_model_test_predictions_FINAL.csv'
)


DEPLOY_VERIFY_MAX_ABS = 5e-7
DEPLOY_VERIFY_MEAN_ABS = 5e-8
DEPLOY_VERIFY_THRESHOLDS = (0.5, 0.9)
DEPLOY_TRACE_BATCH_SIZE = 8


DEPLOY_RESUME = True
DEPLOY_EXPORT_VERSION = 2
DEPLOY_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(DEPLOY_BUNDLE_PATH)),
    '_deploy_resume_cache',
)









CONFIG: Dict = {

    'CSV_PATH': CSV_PATH,
    'ESM_H5': ESM_H5,
    'PDB_DIR': PDB_DIR,
    'AAINDEX1_PATH': AAINDEX1_PATH,
    'AAINDEX_PCA_PATH': AAINDEX_PCA_PATH,
    'OUT_DIR': OUT_DIR,
    'MODEL_DIR': MODEL_DIR,
    'CACHE_DIR': CACHE_DIR,
    'STRUCTURE_CACHE_DIR': STRUCTURE_CACHE_DIR,
    'FINAL_MODEL_PATH': FINAL_MODEL_PATH,

    'RANDOM_SEED': 3407,
    'SPLIT_SEED': 2025,
    'CV_SEED': 1314,
    'N_REPEATS': 3,
    'N_FOLDS': 5,
    'TEST_RATIO': 0.2,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'AMP': False,
    'DETERMINISTIC': True,
    'PRIMARY_METRIC': 'auc',
    'METRIC_THRESHOLD': 0.5,
    'GROUP_SPLIT_BY_PROTEIN': True,

    'RESUME': True,
    'RESUME_VERSION': 17,
    'SAVE_RESUME_EVERY_EPOCH': True,
    'KEEP_LAST_RESUME_AFTER_DONE': True,
    'ESM_CACHE_IN_RAM': True,
    'ESM_CACHE_MAX_GB': 6.0,
    'ESM_NORMALIZE_VALID_ONLY': True,
    'SUPPRESS_DETERMINISTIC_WARNINGS': True,

    'NEG_POS_RATIO': 1,
    'NEG_POS_RATIO_KMER': 2,
    'NEG_SAMPLE_SEED': 20260817,
    'V12_HARD_PAIR_RANK': True,
    'V12_HARD_PAIR_FRACTION': 0.5,

    'PHYSCHEM_PCA_DIM': 14,

    'ESM_CQT_DIM': 256,
    'ESM_CQT_HEADS': 8,
    'ESM_CQT_LAYERS': 2,
    'ESM_CQT_FFN': 512,
    'ESM_CQT_DROPOUT': 0.15,
    'ESM_CQT_LOCAL_RADIUS': 7,
    'ESM_CQT_SIDE_RADIUS': 10,
    'ESM_SC_DIM': 192,
    'ESM_SC_REL_DIM': 192,
    'ESM_SC_RADII': [1, 2, 3, 5, 7, 10, 15],
    'ESM_SC_DROPOUT': 0.18,
    'ESM_SC_USE_POSITION_EMB': True,
    'WEIGHT_DECAY_ESM_SITECONTRAST': 0.0005,
    'ESM_SC2_DIM': 192,
    'ESM_SC2_REL_DIM': 192,
    'ESM_SC2_RADII': [1, 2, 3, 5, 7, 10, 15],
    'ESM_SC2_DROPOUT': 0.16,
    'ESM_SC2_RANK_LOSS_ALPHA': 0.06,
    'ESM_SC2_RANK_WARMUP_EPOCHS': 6,
    'ESM_SC2_RANK_MAX_PAIRS': 4096,

    'KMERS': [1, 2, 3],
    'KMER_EMBED': 64,
    'KMER_CONV_CHANNELS': 64,
    'KMER_MULTI_KERNELS': [3, 5, 7],
    'KMER_HEAD_DIM': 192,
    'KMER_LOCAL_RADIUS': 4,
    'KMER_MIN_COUNT': {
        1: 1,
        2: 2,
        3: 2,
    },
    'KMER_FOLD_MIN_COUNT': 1,
    'DROPOUT_KMER': 0.25,
    'WEIGHT_DECAY_KMER': 0.0003,
    'KMER_RANK_LOSS_ALPHA': 0.03,
    'KMER_RANK_WARMUP_EPOCHS': 6,
    'KMER_RANK_MAX_PAIRS': 4096,

    'STRUCTURE_PRELOAD': True,
    'STRUCTURE_ALLOW_BASE_FALLBACK': True,
    'STRUCTURE_REQUIRE_CENTER_CYS': False,
    'STRUCTURE_CONTACT_THRESH': 8.0,
    'STRUCTURE_EDGE_DIM': 192,
    'STRUCTURE_EDGE_LAYERS': 3,
    'STRUCTURE_DROPOUT': 0.2,
    'STRUCTURE_CHEM_HIDDEN': 96,
    'STRUCTURE_NODE_SCALAR_DIM': 8,
    'STRUCTURE_EDGE_AUG_DIM': 22,
    'STRUCTURE_SHELL_RADII': [4.0, 6.0, 8.0, 10.0],
    'STRUCTURE_SHELL_ATTN_DIM': 96,
    'STRUCTURE_HEAD_HIDDEN': 384,
    'STRUCTURE_DISTANCE_SCALE': 20.0,
    'STRUCTURE_QUALITY_DIM': 3,
    'STRUCTURE_RANK_LOSS_ALPHA': 0.1,
    'STRUCTURE_RANK_WARMUP_EPOCHS': 8,
    'STRUCTURE_RANK_MAX_PAIRS': 4096,
    'BATCH_SIZE_STRUCTURE': 128,
    'LR_STRUCTURE': 0.0003,
    'EARLY_STOP_PATIENCE_STRUCTURE': 16,
    'EPOCHS_STRUCTURE': 180,
    'WEIGHT_DECAY_STRUCTURE': 0.0003,

    'EPOCHS': 128,
    'BATCH_SIZE': 1024,
    'BATCH_SIZE_PHYSCHEM': 256,
    'BATCH_SIZE_ESM': 512,
    'BATCH_SIZE_ESM_CQT': 256,
    'BATCH_SIZE_ESM_SITECONTRAST': 256,
    'BATCH_SIZE_KMER': 256,
    'LR': 0.0005,
    'LR_PHYSCHEM': 0.0005,
    'LR_ESM': 0.0004,
    'LR_ESM_CQT': 0.0003,
    'LR_ESM_SITECONTRAST': 0.0003,
    'LR_KMER': 0.0008,
    'WEIGHT_DECAY': 0.0005,
    'EARLY_STOP_PATIENCE': 10,
    'EARLY_STOP_PATIENCE_KMER': 12,
    'HIDDEN': 256,
    'BASE_RANK_LOSS_ALPHA': 0.0,
    'BASE_RANK_WARMUP_EPOCHS': 5,
    'BASE_RANK_MAX_PAIRS': 2048,
    'LABEL_SMOOTH': 0.0,
    'GRAD_CLIP_NORM': 1.0,
    'USE_PLATEAU_SCHED': True,
    'MIN_LR': 1e-05,

    'META_N_FOLDS': 5,
    'META_SEED': 9527,
    'META_SHARED_SEED': 91001,
    'META_C_GRID': [
        0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0,
    ],
    'META_LOGIT_CLIP': 0.0001,





    'GROUP_FUSION_C': 0.1,
    'GROUP_FUSION_LOGIT_CLIP': 1e-5,


    'FUSION_CANDIDATES': [
        'four_modality_interaction_stack_cf',
        'four_modality_linear_stack_cf',
    ],
    'FUSION_SELECTION_PRIMARY': 'auc',
    'FUSION_SELECTION_SECONDARY': 'auprc',

    'TRAIN_SINGLE_FINAL_MODEL': True,

    'FINAL_EPOCH_SCALE': 1.0,
    'FINAL_MIN_EPOCHS': 4,
    'FINAL_SEED': 20260828,


    'FINAL_SCORE_ALIGNMENT': True,

    'EXPERT_ORDER': ['physchem', 'physchem_pca', 'esm_cqt', 'esm_sitecontrast', 'esm_sitecontrast_v2', 'kmer', 'structure'],
    'FOLD_SCORE_ALIGNMENT': True,
    'FOLD_ALIGN_EPS': 1e-06,
    'USE_ALIGNED_FOR_FUSION': True,


    'STRICT_REPRODUCIBILITY': True,
    'WRITE_REPRODUCIBILITY_MANIFEST': True,


    'HASH_INPUT_CONTENTS': True,
}


for _d in (
    CONFIG['OUT_DIR'],
    CONFIG['MODEL_DIR'],
    CONFIG['CACHE_DIR'],
    CONFIG['STRUCTURE_CACHE_DIR'],
):
    os.makedirs(_d, exist_ok=True)


def _sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _fingerprint_file(path: str, hash_contents: bool) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Reproducibility input missing: {path}')
    row = {
        'path': os.path.abspath(path),
        'size_bytes': int(os.path.getsize(path)),
    }
    row['sha256'] = _sha256_file(path) if hash_contents else None
    return row


def _fingerprint_directory(path: str, hash_contents: bool) -> Dict:
    if not os.path.isdir(path):
        raise NotADirectoryError(f'Reproducibility input directory missing: {path}')
    root = os.path.abspath(path)
    tree = hashlib.sha256()
    count = 0
    total_bytes = 0
    for current, dirs, files in os.walk(root):
        dirs.sort()
        files.sort()
        for name in files:
            full = os.path.join(current, name)
            if not os.path.isfile(full):
                continue
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            size = int(os.path.getsize(full))
            count += 1
            total_bytes += size
            tree.update(rel.encode('utf-8'))
            tree.update(b'\0')
            tree.update(str(size).encode('ascii'))
            tree.update(b'\0')
            if hash_contents:
                tree.update(_sha256_file(full).encode('ascii'))
            tree.update(b'\n')
    return {
        'path': root,
        'file_count': int(count),
        'total_bytes': int(total_bytes),
        'tree_sha256': tree.hexdigest(),
        'content_hashed': bool(hash_contents),
    }


def write_reproducibility_manifest() -> str:

    if not bool(CONFIG.get('WRITE_REPRODUCIBILITY_MANIFEST', True)):
        return ''
    hash_contents = bool(CONFIG.get('HASH_INPUT_CONTENTS', True))
    input_files = {
        'training_csv': CONFIG['CSV_PATH'],
        'esm_h5': CONFIG['ESM_H5'],
        'aaindex1': CONFIG['AAINDEX1_PATH'],
        'aaindex_pca': CONFIG['AAINDEX_PCA_PATH'],
    }
    inputs = {
        name: _fingerprint_file(path, hash_contents)
        for name, path in input_files.items()
    }
    inputs['pdb_dir'] = _fingerprint_directory(
        CONFIG['PDB_DIR'], hash_contents=hash_contents,
    )

    script_path = os.path.abspath(__file__)
    manifest = {
        'pipeline': 'DeepPalm V14 train-and-auto-fusion',
        'script': _fingerprint_file(script_path, hash_contents=True),
        'inputs': inputs,
        'seeds': {
            key: CONFIG[key]
            for key in (
                'RANDOM_SEED', 'SPLIT_SEED', 'CV_SEED', 'NEG_SAMPLE_SEED',
                'META_SEED', 'META_SHARED_SEED', 'FINAL_SEED',
            )
        },
        'environment': {
            'python': sys.version,
            'platform': platform.platform(),
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'h5py': h5py.__version__,
            'sklearn': sklearn.__version__,
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'cuda_available': bool(torch.cuda.is_available()),
            'gpu_names': [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ] if torch.cuda.is_available() else [],
            'PYTHONHASHSEED': os.environ.get('PYTHONHASHSEED'),
        },
        'determinism': {
            'strict': bool(CONFIG.get('STRICT_REPRODUCIBILITY', True)),
            'amp': bool(CONFIG.get('AMP', False)),
            'allow_tf32': bool(CONFIG.get('ALLOW_TF32', False)),
            'cudnn_benchmark': bool(CONFIG.get('CUDNN_BENCHMARK', False)),
            'num_workers_esm': int(CONFIG.get('NUM_WORKERS_ESM', 0)),
            'num_workers_other': int(CONFIG.get('NUM_WORKERS_OTHER', 0)),
        },
        'config': copy.deepcopy(CONFIG),
    }
    path = os.path.join(CONFIG['OUT_DIR'], 'reproducibility_manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    print('[Reproducibility] manifest ->', path)
    return path


def write_result_checksums(fina_dir: str, selected_method: str,
                           include_model: bool) -> str:

    candidates = [
        os.path.join(fina_dir, 'train_OOF_predictions_FINAL.csv'),
        os.path.join(fina_dir, 'test_predictions_FULL_FINAL.csv'),
        os.path.join(fina_dir, 'test_predictions_FINAL.csv'),
        os.path.join(fina_dir, 'fusion_candidates_FINAL.csv'),
        os.path.join(fina_dir, 'fusion_params_FINAL.json'),
        os.path.join(fina_dir, 'v14_final_summary.json'),
    ]
    if include_model:
        candidates.insert(0, CONFIG['FINAL_MODEL_PATH'])
        candidates.append(
            os.path.join(fina_dir, 'single_model_test_predictions_FINAL.csv')
        )
    files = {}
    for path in candidates:
        if os.path.isfile(path):
            files[os.path.basename(path)] = {
                'size_bytes': int(os.path.getsize(path)),
                'sha256': _sha256_file(path),
            }
    obj = {
        'selected_fusion_method': str(selected_method),
        'files': files,
        'comparison_rule': (
            'With identical reproducibility_manifest inputs/environment, '
            'matching hashes confirm an identical run artifact.'
        ),
    }
    path = os.path.join(fina_dir, 'result_checksums_FINAL.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print('[Reproducibility] result checksums ->', path)
    return path

FINAL_EXPERTS = [
    'physchem',
    'physchem_pca',
    'esm_cqt',
    'esm_sitecontrast',
    'esm_sitecontrast_v2',
    'kmer',
    'structure',
]


FUSED_MODALITIES = [
    'physchem_fused',
    'esm_fused',
    'kmer',
    'structure',
]
PHYS_EXPERTS = ['physchem', 'physchem_pca']
ESM_EXPERTS = ['esm_cqt', 'esm_sitecontrast', 'esm_sitecontrast_v2']



MAX_LEN = 31

def resolve_aaindex1_path() -> str:
    path = str(CONFIG.get('AAINDEX1_PATH', '')).strip()
    if path and os.path.exists(path):
        print(f"[AAindex] 使用: {path}")
        return path
    raise FileNotFoundError(
        "找不到 aaindex1.txt。请只修改文件顶部路径区的 AAINDEX1_PATH。"
    )

AAINDEX1_PATH = resolve_aaindex1_path()

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
    feat_dict["*"] = [0.0] * num_idx
    return feat_dict

PHYSICO_CHEMICAL_FEATURES = build_physico_chemical_features(AAINDEX1_PATH)
SEQ_FEAT_DIM = len(AAINDEX_IDS)

def get_seq_feat_from_sequence(seq: str, max_len: int = MAX_LEN) -> Tuple[np.ndarray, np.ndarray]:
    seq = ensure_len_31(clean_seq(seq))
    seq = seq[:max_len]

    feat = [
        PHYSICO_CHEMICAL_FEATURES.get(aa, PHYSICO_CHEMICAL_FEATURES["X"])
        for aa in seq
    ]
    mask = [0.0 if aa == '*' else 1.0 for aa in seq]

    if len(feat) < max_len:
        n_pad = max_len - len(feat)
        feat.extend([PHYSICO_CHEMICAL_FEATURES["*"]] * n_pad)
        mask.extend([0.0] * n_pad)

    return (
        np.asarray(feat, dtype=np.float32),
        np.asarray(mask, dtype=np.float32),
    )



def _v9_sort_pc_cols(cols):
    def key(c):
        m = re.search(r"(\d+)", str(c))
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=key)

_PHYSCHEM_PCA_LOOKUP = None

def load_physchem_pca_lookup(path: str, n_components: int = 14):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'PCA lookup不存在: {path}')
    df = pd.read_csv(path)
    aa_col = None
    for c in df.columns:
        if str(c).strip().lower() in {'aa','amino_acid','aminoacid','residue','amino acid'}:
            aa_col = c; break
    if aa_col is None:
        c0 = df.columns[0]
        vals = df[c0].astype(str).str.strip().str.upper().tolist()
        if sum(v in set(AA_INDEX_ORDER) for v in vals) >= 18:
            aa_col = c0
    if aa_col is None and len(df) == 20:
        df = df.copy(); df.insert(0, 'AA', AA_INDEX_ORDER); aa_col = 'AA'
    if aa_col is None:
        raise RuntimeError(f'无法识别PCA氨基酸列: columns={list(df.columns)}')
    pc_cols = [c for c in df.columns if re.match(r'(?i)^pc\s*\d+$', str(c).strip())]
    if not pc_cols:
        pc_cols = [c for c in df.columns if c != aa_col and pd.api.types.is_numeric_dtype(df[c])]
    pc_cols = _v9_sort_pc_cols(pc_cols)
    if len(pc_cols) < n_components:
        raise RuntimeError(f'PCA列不足: have={len(pc_cols)} need={n_components}')
    use = pc_cols[:n_components]
    out = {}
    for _, row in df.iterrows():
        aa = str(row[aa_col]).strip().upper()
        if aa in AA_INDEX_ORDER:
            out[aa] = np.asarray([row[c] for c in use], dtype=np.float32)
    miss = [aa for aa in AA_INDEX_ORDER if aa not in out]
    if miss:
        raise RuntimeError(f'PCA lookup缺少AA: {miss}')
    out['X'] = np.zeros(n_components, dtype=np.float32)
    out['*'] = np.zeros(n_components, dtype=np.float32)
    print(f'[Physchem PCA] {path} -> dim={n_components}, last={use[-1]}')
    return out

def get_physchem_pca_lookup():
    global _PHYSCHEM_PCA_LOOKUP
    if _PHYSCHEM_PCA_LOOKUP is None:
        _PHYSCHEM_PCA_LOOKUP = load_physchem_pca_lookup(
            CONFIG['AAINDEX_PCA_PATH'], int(CONFIG.get('PHYSCHEM_PCA_DIM', 14))
        )
    return _PHYSCHEM_PCA_LOOKUP

def get_pca_feat_from_sequence(seq: str, max_len: int = MAX_LEN):
    lookup = get_physchem_pca_lookup()
    dim = int(CONFIG.get('PHYSCHEM_PCA_DIM', 14))
    seq = ensure_len_31(clean_seq(seq))[:max_len]
    feat, mask = [], []
    for aa in seq:
        feat.append(lookup.get(aa, lookup['X']))
        mask.append(0.0 if aa == '*' else 1.0)
    while len(feat) < max_len:
        feat.append(lookup['*']); mask.append(0.0)
    arr = np.asarray(feat, dtype=np.float32)
    if arr.shape != (max_len, dim):
        raise RuntimeError(f'PCA feature shape={arr.shape}, expected={(max_len,dim)}')
    return arr, np.asarray(mask, dtype=np.float32)

class AttentionReadout(nn.Module):
    def __init__(self, node_dim: int, attn_dim: int = 128):
        super().__init__()
        self.attn_layer = nn.Linear(node_dim, attn_dim)
        self.output_proj = nn.Linear(attn_dim, 1)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, h, mask):
        scores = torch.tanh(self.attn_layer(h))
        logits = self.output_proj(scores).squeeze(-1)
        masked_logits = logits + (1.0 - mask) * -1e9
        attn_weights = F.softmax(masked_logits, dim=-1)
        return torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)

class BranchPhyschem(nn.Module):
    def __init__(self, seq_feat_dim: int = SEQ_FEAT_DIM, seq_hidden: int = 128):
        super().__init__()
        self.seq_proj = nn.Linear(seq_feat_dim, seq_hidden)
        self.seq_norm = nn.LayerNorm(seq_hidden)
        self.lstm = nn.LSTM(
            seq_hidden, seq_hidden,
            batch_first=True,
            bidirectional=True
        )
        out_dim = seq_hidden * 2
        self.out_norm = nn.LayerNorm(out_dim)
        self.seq_attn_readout = AttentionReadout(node_dim=out_dim)
        self.readout = nn.Sequential(
            nn.Linear(out_dim * 4, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, x_dict):
        seq_feat = x_dict['seq']
        mask = x_dict['mask'].float()

        h_seq = self.seq_norm(F.gelu(self.seq_proj(seq_feat)))
        h_seq = h_seq * mask.unsqueeze(-1)
        h_lstm, _ = self.lstm(h_seq)
        h_lstm = self.out_norm(h_lstm)

        g_attn = self.seq_attn_readout(h_lstm, mask)
        den = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        g_mean = (h_lstm * mask.unsqueeze(-1)).sum(dim=1) / den

        h_for_max = h_lstm.masked_fill(mask.unsqueeze(-1) <= 0, -1e4)
        g_max = h_for_max.max(dim=1).values

        center_idx = min(15, h_lstm.shape[1] - 1)
        g_center = h_lstm[:, center_idx, :]

        g = torch.cat([g_attn, g_mean, g_max, g_center], dim=-1)
        return self.readout(g)

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
    'NUM_WORKERS_ESM': 0,


    'NUM_WORKERS_OTHER': 0,
    'PREFETCH_FACTOR': 2,
    'PERSISTENT_WORKERS': True,
    'PIN_MEMORY': str(CONFIG.get('DEVICE', 'cpu')).startswith('cuda'),
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
        torch.use_deterministic_algorithms(
            deterministic,
            warn_only=not bool(CONFIG.get('STRICT_REPRODUCIBILITY', True)),
        )
    except TypeError:

        torch.use_deterministic_algorithms(deterministic)

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

def smooth_binary_targets(y: torch.Tensor) -> torch.Tensor:
    eps = float(CONFIG.get('LABEL_SMOOTH', 0.0) or 0.0)
    if eps <= 0:
        return y
    return y * (1.0 - eps) + 0.5 * eps

def backward_with_optional_amp(loss, optimizer, scaler, model=None):
    clip = float(CONFIG.get('GRAD_CLIP_NORM', 0.0) or 0.0)
    if getattr(scaler, 'enabled', False):
        scaler.scale(loss).backward()
        if clip > 0 and model is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if clip > 0 and model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

def is_esm_branch(branch_name: str) -> bool:
    return str(branch_name).lower() in {
        'esm_cqt',
        'esm_sitecontrast',
        'esm_sitecontrast_v2',
    }

def branch_batch_size(branch_name: str) -> int:
    if branch_name == 'structure':
        return int(CONFIG['BATCH_SIZE_STRUCTURE'])
    if branch_name in {'esm_sitecontrast', 'esm_sitecontrast_v2'}:
        return int(CONFIG['BATCH_SIZE_ESM_SITECONTRAST'])
    if branch_name == 'esm_cqt':
        return int(CONFIG['BATCH_SIZE_ESM_CQT'])
    if branch_name in {'physchem', 'physchem_pca'}:
        return int(CONFIG['BATCH_SIZE_PHYSCHEM'])
    if branch_name == 'kmer':
        return int(CONFIG['BATCH_SIZE_KMER'])
    raise ValueError(branch_name)

def branch_lr(branch_name: str) -> float:
    if branch_name == 'structure':
        return float(CONFIG['LR_STRUCTURE'])
    if branch_name in {'esm_sitecontrast', 'esm_sitecontrast_v2'}:
        return float(CONFIG['LR_ESM_SITECONTRAST'])
    if branch_name == 'esm_cqt':
        return float(CONFIG['LR_ESM_CQT'])
    if branch_name in {'physchem', 'physchem_pca'}:
        return float(CONFIG['LR_PHYSCHEM'])
    if branch_name == 'kmer':
        return float(CONFIG['LR_KMER'])
    raise ValueError(branch_name)

def branch_neg_ratio(branch_name: str):
    return CONFIG['NEG_POS_RATIO_KMER'] if branch_name == 'kmer' else CONFIG['NEG_POS_RATIO']

def branch_patience(branch_name: str) -> int:
    if branch_name == 'structure':
        return int(CONFIG['EARLY_STOP_PATIENCE_STRUCTURE'])
    if branch_name == 'kmer':
        return int(CONFIG['EARLY_STOP_PATIENCE_KMER'])
    return int(CONFIG['EARLY_STOP_PATIENCE'])

def branch_weight_decay(branch_name: str) -> float:
    if branch_name == 'structure':
        return float(CONFIG['WEIGHT_DECAY_STRUCTURE'])
    if branch_name in {'esm_sitecontrast', 'esm_sitecontrast_v2'}:
        return float(CONFIG['WEIGHT_DECAY_ESM_SITECONTRAST'])
    if branch_name == 'kmer':
        return float(CONFIG['WEIGHT_DECAY_KMER'])
    return float(CONFIG['WEIGHT_DECAY'])

def branch_rank_loss_cfg(branch_name: str) -> Tuple[float, int, int]:
    if branch_name == 'structure':
        return (
            float(CONFIG['STRUCTURE_RANK_LOSS_ALPHA']),
            int(CONFIG['STRUCTURE_RANK_WARMUP_EPOCHS']),
            int(CONFIG['STRUCTURE_RANK_MAX_PAIRS']),
        )
    if branch_name == 'esm_sitecontrast_v2':
        return (
            float(CONFIG['ESM_SC2_RANK_LOSS_ALPHA']),
            int(CONFIG['ESM_SC2_RANK_WARMUP_EPOCHS']),
            int(CONFIG['ESM_SC2_RANK_MAX_PAIRS']),
        )
    if branch_name == 'kmer':
        return (
            float(CONFIG['KMER_RANK_LOSS_ALPHA']),
            int(CONFIG['KMER_RANK_WARMUP_EPOCHS']),
            int(CONFIG['KMER_RANK_MAX_PAIRS']),
        )
    return (0.0, 0, 0)

def loader_kwargs(branch_name: Optional[str] = None):
    if str(branch_name).lower() == 'structure':
        nw = 0
    elif is_esm_branch(branch_name) and CONFIG.get('ESM_CACHE_IN_RAM', True):
        nw = int(CONFIG.get('NUM_WORKERS_ESM', 0) or 0)
    else:
        nw = int(CONFIG.get('NUM_WORKERS_OTHER', 0) or 0)

    kw = dict(
        num_workers=nw,
        pin_memory=bool(CONFIG.get('PIN_MEMORY', True)),
    )
    if nw > 0:
        kw['persistent_workers'] = bool(CONFIG.get('PERSISTENT_WORKERS', True))
        kw['prefetch_factor'] = int(CONFIG.get('PREFETCH_FACTOR', 2))
    return kw

enable_speedups()

if (
    CONFIG.get('SUPPRESS_DETERMINISTIC_WARNINGS', False)
    and not CONFIG.get('STRICT_REPRODUCIBILITY', True)
):
    warnings.filterwarnings(
        "ignore",
        message=r".*does not have a deterministic implementation.*",
        category=UserWarning,
    )



_ESM_RAM_STORE = {}

def get_esm_ram_store(h5_path: str):
    if not CONFIG.get('ESM_CACHE_IN_RAM', True):
        return None

    if h5_path in _ESM_RAM_STORE:
        return _ESM_RAM_STORE[h5_path]

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"ESM_H5 不存在: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        if 'window_emb' not in f or 'uniprotid' not in f:
            raise RuntimeError("embedding.h5 缺少 window_emb 或 uniprotid")

        ds = f['window_emb']
        est_gb = float(np.prod(ds.shape) * ds.dtype.itemsize) / (1024 ** 3)
        print(
            f"[ESM cache] window_emb={ds.shape}, dtype={ds.dtype}, "
            f"compression={ds.compression}, raw≈{est_gb:.2f} GB"
        )

        max_gb = float(CONFIG.get('ESM_CACHE_MAX_GB', 6.0))
        if est_gb > max_gb:
            print(
                f"[ESM cache][WARN] {est_gb:.2f} GB > ESM_CACHE_MAX_GB={max_gb:.2f}; "
                "不载入RAM，回退HDF5读取。"
            )
            return None

        print("[ESM cache] 正在一次性读取/解压到 CPU RAM；这一步只发生一次……")
        emb = f['window_emb'][...]
        valid_mask = (
            f['valid_mask'][...].astype(np.uint8, copy=False)
            if 'valid_mask' in f else None
        )
        raw_uids = f['uniprotid'][...]

    uids = [
        u.decode() if isinstance(u, (bytes, bytearray)) else str(u)
        for u in raw_uids
    ]
    uid2idx = {u: i for i, u in enumerate(uids)}

    store = {
        'window_emb': emb,
        'valid_mask': valid_mask,
        'uids': uids,
        'uid2idx': uid2idx,
    }
    _ESM_RAM_STORE[h5_path] = store

    total_gb = emb.nbytes / (1024 ** 3)
    if valid_mask is not None:
        total_gb += valid_mask.nbytes / (1024 ** 3)
    print(f"[ESM cache] 完成，RAM占用约 {total_gb:.2f} GB。之后 epoch 不再解压 gzip。")
    return store

_build_uid2idx_h5 = build_uid2idx

def build_uid2idx(h5_path: str) -> dict:
    store = get_esm_ram_store(h5_path)
    if store is not None:
        return store['uid2idx']
    return _build_uid2idx_h5(h5_path)



def _resume_signature(branch: str, repeat: int, fold: Optional[int], final: bool = False) -> str:
    keys = [
        'CSV_PATH', 'ESM_H5', 'AAINDEX1_PATH', 'AAINDEX_PCA_PATH',
        'PDB_DIR',
        'RANDOM_SEED', 'SPLIT_SEED', 'CV_SEED', 'N_REPEATS', 'N_FOLDS', 'TEST_RATIO',
        'EPOCHS', 'EPOCHS_STRUCTURE',
        'BATCH_SIZE_PHYSCHEM', 'BATCH_SIZE_ESM_CQT', 'BATCH_SIZE_ESM_SITECONTRAST',
        'BATCH_SIZE_KMER', 'BATCH_SIZE_STRUCTURE',
        'LR_PHYSCHEM', 'LR_ESM_CQT', 'LR_ESM_SITECONTRAST', 'LR_KMER', 'LR_STRUCTURE',
        'WEIGHT_DECAY', 'WEIGHT_DECAY_ESM_SITECONTRAST', 'WEIGHT_DECAY_KMER', 'WEIGHT_DECAY_STRUCTURE',
        'NEG_POS_RATIO', 'NEG_POS_RATIO_KMER', 'NEG_SAMPLE_SEED',
        'PHYSCHEM_PCA_DIM',
        'ESM_CQT_DIM', 'ESM_CQT_HEADS', 'ESM_CQT_LAYERS', 'ESM_CQT_FFN', 'ESM_CQT_DROPOUT',
        'ESM_CQT_LOCAL_RADIUS', 'ESM_CQT_SIDE_RADIUS',
        'ESM_SC_DIM', 'ESM_SC_REL_DIM', 'ESM_SC_RADII', 'ESM_SC_DROPOUT', 'ESM_SC_USE_POSITION_EMB',
        'ESM_SC2_DIM', 'ESM_SC2_REL_DIM', 'ESM_SC2_RADII', 'ESM_SC2_DROPOUT',
        'ESM_SC2_RANK_LOSS_ALPHA', 'ESM_SC2_RANK_WARMUP_EPOCHS', 'ESM_SC2_RANK_MAX_PAIRS',
        'KMERS', 'KMER_EMBED', 'KMER_CONV_CHANNELS', 'KMER_MULTI_KERNELS', 'KMER_HEAD_DIM',
        'KMER_LOCAL_RADIUS', 'KMER_MIN_COUNT', 'KMER_FOLD_MIN_COUNT', 'DROPOUT_KMER',
        'KMER_RANK_LOSS_ALPHA', 'KMER_RANK_WARMUP_EPOCHS', 'KMER_RANK_MAX_PAIRS',
        'STRUCTURE_CONTACT_THRESH', 'STRUCTURE_EDGE_DIM', 'STRUCTURE_EDGE_LAYERS',
        'STRUCTURE_DROPOUT', 'STRUCTURE_CHEM_HIDDEN', 'STRUCTURE_NODE_SCALAR_DIM',
        'STRUCTURE_EDGE_AUG_DIM', 'STRUCTURE_SHELL_RADII', 'STRUCTURE_SHELL_ATTN_DIM',
        'STRUCTURE_HEAD_HIDDEN', 'STRUCTURE_DISTANCE_SCALE', 'STRUCTURE_QUALITY_DIM',
        'STRUCTURE_RANK_LOSS_ALPHA', 'STRUCTURE_RANK_WARMUP_EPOCHS', 'STRUCTURE_RANK_MAX_PAIRS',
        'FOLD_SCORE_ALIGNMENT', 'USE_ALIGNED_FOR_FUSION',
        'META_N_FOLDS', 'META_SHARED_SEED', 'META_C_GRID', 'META_LOGIT_CLIP',
        'FUSION_CANDIDATES', 'GROUP_FUSION_C', 'GROUP_FUSION_LOGIT_CLIP',
        'FUSION_SELECTION_PRIMARY', 'FUSION_SELECTION_SECONDARY',
        'STRICT_REPRODUCIBILITY', 'HASH_INPUT_CONTENTS',
        'RESUME_VERSION',
    ]
    payload = {
        'recipe': 'DeepPalm_V14_train_and_auto_fusion_reproducible',
        'branch': branch,
        'repeat': int(repeat),
        'fold': None if fold is None else int(fold),
        'final': bool(final),
        'config': {k: CONFIG.get(k) for k in keys},
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def _atomic_torch_save(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)

def safe_torch_load(path: str, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def _atomic_npz_save(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)

def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state

def _restore_rng_state(state):
    if not state:
        return
    try:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'])
        if torch.cuda.is_available() and 'torch_cuda' in state:
            torch.cuda.set_rng_state_all(state['torch_cuda'])
    except Exception as e:
        print(f"[Resume][WARN] RNG state 恢复失败，继续训练: {e}")

def _scaler_state_dict(scaler):
    if hasattr(scaler, 'state_dict'):
        try:
            return scaler.state_dict()
        except Exception:
            return None
    return None

def _load_scaler_state(scaler, state):
    if state is None or not hasattr(scaler, 'load_state_dict'):
        return
    try:
        scaler.load_state_dict(state)
    except Exception as e:
        print(f"[Resume][WARN] GradScaler state 恢复失败: {e}")

def _fold_paths(branch: str, repeat: int, fold_num: int):
    save_dir = os.path.join(CONFIG['MODEL_DIR'], f"repeat{repeat}", branch)
    os.makedirs(save_dir, exist_ok=True)
    stem = f"{branch}_rep{repeat}_fold{fold_num}"
    return {
        'best': os.path.join(save_dir, stem + "_best.pth"),
        'last': os.path.join(save_dir, stem + "_last_resume.pth"),
        'done': os.path.join(save_dir, stem + "_DONE.npz"),
    }

def _save_epoch_resume(path, signature, epoch, model, optimizer, scheduler,
                       scaler, best_metric, no_improve):
    if not CONFIG.get('SAVE_RESUME_EVERY_EPOCH', True):
        return
    obj = {
        'signature': signature,
        'epoch': int(epoch),
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state_dict': _scaler_state_dict(scaler),
        'best_metric': float(best_metric),
        'no_improve': int(no_improve),
        'rng_state': _capture_rng_state(),
    }
    _atomic_torch_save(obj, path)

def _try_load_epoch_resume(path, signature, model, optimizer, scheduler, scaler, device):
    if not CONFIG.get('RESUME', True) or not os.path.exists(path):
        return None
    try:
        ckpt = safe_torch_load(path, map_location=device)
    except Exception as e:
        print(f"[Resume][WARN] 无法读取 {path}: {e}")
        return None

    if ckpt.get('signature') != signature:
        print(f"[Resume] 找到旧checkpoint但参数signature不同，忽略: {path}")
        return None

    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    _load_scaler_state(scaler, ckpt.get('scaler_state_dict'))
    _restore_rng_state(ckpt.get('rng_state'))

    print(f"[Resume] 从 epoch {int(ckpt['epoch']) + 1} 继续: {path}")
    return ckpt



def _safe_div(a, b):
    return float(a) / float(b) if (b is not None and b != 0) else 0.0

def bin_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
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
    precision = _safe_div(TP, TP + FP)
    f1 = _safe_div(2 * precision * sen, precision + sen)
    fpr = _safe_div(FP, N)
    bacc = 0.5 * (sen + spe)

    den = math.sqrt(
        max(1, TP + FP) *
        max(1, TP + FN) *
        max(1, TN + FP) *
        max(1, TN + FN)
    )
    mcc = ((TP * TN - FP * FN) / den) if den > 0 else 0.0

    auc = float('nan')
    auprc = float('nan')
    if SKLEARN_OK and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))

    return {
        'tp': TP, 'tn': TN, 'fp': FP, 'fn': FN,
        'sen': sen, 'recall': sen, 'spe': spe, 'acc': acc,
        'precision': precision, 'f1': f1, 'mcc': mcc,
        'fpr': fpr, 'bacc': bacc,
        'auc': auc, 'auprc': auprc,
    }

def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return bin_metrics(y_true, y_prob, thr=CONFIG['METRIC_THRESHOLD'])

def _base_uid(uid: str) -> str:
    return str(uid).strip().split('-')[0]

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



def clean_seq(s: str) -> str:
    s = (s or '').strip().upper()
    allow = set(list('ACDEFGHIKLMNPQRSTVWY') + ['X', '*'])
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
    return '*'*lpad + s + '*'*rpad


CONS_GROUPS = [list('ST'), list('NQ'), list('DE'), list('KR'), list('FWY'), list('ILVM'), list('AG'), list('PH')]
CONS_MAP = {c:g for g in CONS_GROUPS for c in g}


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
        self.allowed_tokens = None
        self.min_count = 1

    def add_seq(self, seq: str):
        for tok in kmer_tokens(seq, self.k):
            self.counts[tok] = self.counts.get(tok, 0) + 1

    def finalize(self, min_count: int = 1):
        self.min_count = int(max(1, min_count))
        items = [(t, c) for t, c in self.counts.items() if c >= self.min_count]
        items = sorted(items, key=lambda x:(-x[1], x[0]))
        if self.max_size is not None:
            items = items[:self.max_size]
        for tok,_ in items:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def encode(self, seq: str) -> List[int]:
        toks = kmer_tokens(seq, self.k)
        if self.allowed_tokens is None:
            return [self.stoi.get(t, 0) for t in toks]
        return [self.stoi.get(t, 0) if t in self.allowed_tokens else 0 for t in toks]

    @property
    def size(self):
        return len(self.itos)

def build_active_kmer_views(kmer_vocabs: Dict[int, KmerVocab],
                            samples: List['Sample'],
                            min_count: int = 1) -> Dict[int, KmerVocab]:
    out: Dict[int, KmerVocab] = {}
    min_count = int(max(1, min_count))
    for k, base in kmer_vocabs.items():
        cnt: Dict[str, int] = {}
        for smp in samples:
            for tok in kmer_tokens(smp.seq, k):
                cnt[tok] = cnt.get(tok, 0) + 1
        allowed = {tok for tok, c in cnt.items()
                   if c >= min_count and tok in base.stoi}
        view = copy.copy(base)
        view.allowed_tokens = allowed
        out[k] = view
        print(
            f"[kmer fold-view] k={k}: active={len(allowed)}/{max(1, base.size-1)} "
            f"(min_count={min_count})"
        )
    return out



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

class SafeQueryPool(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(d_model) ** -0.5

    def forward(self, h: torch.Tensor, query: torch.Tensor,
                valid: torch.Tensor, region: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = valid.bool()
        if region is not None:
            mask = mask & region.bool()
        empty = mask.sum(dim=1) == 0
        if empty.any():
            mask = mask.clone()
            mask[empty] = valid.bool()[empty]

        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(h)
        v = self.v_proj(h)
        score = (q * k).sum(dim=-1) * self.scale
        score = score.masked_fill(~mask, -1e4)
        a = F.softmax(score, dim=-1)
        a = a * mask.float()
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        a = self.dropout(a)
        return self.out((v * a.unsqueeze(-1)).sum(dim=1))

class BranchESMCenterQueryTransformer(nn.Module):
    def __init__(self, D_in: int, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        d = int(CONFIG.get('ESM_CQT_DIM', 256))
        heads = int(CONFIG.get('ESM_CQT_HEADS', 8))
        layers = int(CONFIG.get('ESM_CQT_LAYERS', 2))
        ffn = int(CONFIG.get('ESM_CQT_FFN', 512))
        drop = float(CONFIG.get('ESM_CQT_DROPOUT', dropout))
        self.local_radius = int(CONFIG.get('ESM_CQT_LOCAL_RADIUS', 7))
        self.side_radius = int(CONFIG.get('ESM_CQT_SIDE_RADIUS', 10))

        self.input_norm = nn.LayerNorm(D_in)
        self.input_proj = nn.Sequential(
            nn.Linear(D_in, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(drop),
        )
        self.pos_emb = nn.Embedding(MAX_LEN, d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=ffn,
            dropout=drop, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers, norm=nn.LayerNorm(d))

        self.q_global = nn.Parameter(torch.zeros(d))
        self.q_local = nn.Parameter(torch.zeros(d))
        self.q_left = nn.Parameter(torch.zeros(d))
        self.q_right = nn.Parameter(torch.zeros(d))
        for p in (self.q_global, self.q_local, self.q_left, self.q_right):
            nn.init.normal_(p, mean=0.0, std=0.02)

        self.pool_global = SafeQueryPool(d, dropout=drop * 0.5)
        self.pool_local = SafeQueryPool(d, dropout=drop * 0.5)
        self.pool_left = SafeQueryPool(d, dropout=drop * 0.5)
        self.pool_right = SafeQueryPool(d, dropout=drop * 0.5)

        self.head = nn.Sequential(
            nn.Linear(d * 6, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _region_mask(valid: torch.Tensor, center: int, radius: int, kind: str) -> torch.Tensor:
        B, L = valid.shape
        r = torch.zeros_like(valid)
        if kind == 'local':
            r[:, max(0, center-radius):min(L, center+radius+1)] = 1.0
        elif kind == 'left':
            r[:, max(0, center-radius):center] = 1.0
        elif kind == 'right':
            r[:, center+1:min(L, center+radius+1)] = 1.0
        else:
            r[:] = 1.0
        return r * valid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = (x.abs().sum(dim=1) > 0).float()
        xt = x.transpose(1, 2)
        B, L, _ = xt.shape
        c = min(15, L - 1)

        h = self.input_proj(self.input_norm(xt))
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = (h + self.pos_emb(pos)) * valid.unsqueeze(-1)
        h = self.encoder(h, src_key_padding_mask=~valid.bool())
        h = h * valid.unsqueeze(-1)

        center = h[:, c]
        global_mean = (h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1.0)

        local_mask = self._region_mask(valid, c, self.local_radius, 'local')
        left_mask = self._region_mask(valid, c, self.side_radius, 'left')
        right_mask = self._region_mask(valid, c, self.side_radius, 'right')

        g_global = self.pool_global(h, center + self.q_global, valid)
        g_local = self.pool_local(h, center + self.q_local, valid, local_mask)
        g_left = self.pool_left(h, center + self.q_left, valid, left_mask)
        g_right = self.pool_right(h, center + self.q_right, valid, right_mask)
        feat = torch.cat([center, global_mean, g_global, g_local, g_left, g_right], dim=1)
        return self.head(feat)

class BranchESMSiteContrast(nn.Module):
    def __init__(self, D_in: int, hidden: int = 256, dropout: float = 0.18):
        super().__init__()
        d = int(CONFIG.get('ESM_SC_DIM', 192))
        rel_dim = int(CONFIG.get('ESM_SC_REL_DIM', d))
        radii = [int(r) for r in CONFIG.get('ESM_SC_RADII', [1, 2, 3, 5, 7, 10, 15])]
        if not radii:
            raise ValueError('ESM_SC_RADII cannot be empty')
        if any(r <= 0 for r in radii):
            raise ValueError(f'ESM_SC_RADII must be positive, got {radii}')

        self.d = d
        self.rel_dim = rel_dim
        self.radii = tuple(radii)
        self.drop = float(CONFIG.get('ESM_SC_DROPOUT', dropout))
        self.use_position_emb = bool(CONFIG.get('ESM_SC_USE_POSITION_EMB', True))

        self.input_norm = nn.LayerNorm(D_in)
        self.input_proj = nn.Sequential(
            nn.Linear(D_in, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Dropout(self.drop),
        )
        self.pos_emb = nn.Embedding(MAX_LEN, d) if self.use_position_emb else None

        self.radius_relation = nn.Sequential(
            nn.Linear(d * 6, rel_dim),
            nn.LayerNorm(rel_dim),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(rel_dim, rel_dim),
            nn.GELU(),
        )
        self.radius_emb = nn.Embedding(len(self.radii), rel_dim)
        self.radius_gate = nn.Sequential(
            nn.LayerNorm(rel_dim),
            nn.Linear(rel_dim, max(32, rel_dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, rel_dim // 2), 1),
        )

        self.site_relation = nn.Sequential(
            nn.Linear(d * 3, rel_dim),
            nn.LayerNorm(rel_dim),
            nn.GELU(),
            nn.Dropout(self.drop * 0.75),
            nn.Linear(rel_dim, rel_dim),
            nn.GELU(),
        )
        self.distance_bias = nn.Embedding(MAX_LEN, 1)
        nn.init.zeros_(self.distance_bias.weight)
        self.site_score = nn.Linear(rel_dim, 1)
        self.site_value = nn.Linear(rel_dim, rel_dim)

        self.center_proj = nn.Sequential(nn.Linear(d, rel_dim), nn.GELU())
        self.global_proj = nn.Sequential(nn.Linear(d, rel_dim), nn.GELU())
        self.side_proj = nn.Sequential(nn.Linear(d, rel_dim), nn.GELU())

        fusion_dim = rel_dim * 6
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(self.drop),
            nn.Linear(hidden, 128),
            nn.GELU(),
            nn.Dropout(self.drop * 0.5),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(-1)
        den = m.sum(dim=1).clamp_min(1.0)
        return (h * m).sum(dim=1) / den

    @staticmethod
    def _has_any(mask: torch.Tensor) -> torch.Tensor:
        return mask.float().sum(dim=1) > 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        valid = (x.abs().sum(dim=1) > 0)
        xt = x.transpose(1, 2)
        B, L, _ = xt.shape
        c = min(15, L - 1)

        h = self.input_proj(self.input_norm(xt))
        if self.pos_emb is not None:
            pos = torch.arange(L, device=x.device)
            h = h + self.pos_emb(pos).unsqueeze(0)
        h = h * valid.unsqueeze(-1).float()

        center = h[:, c, :]
        other_valid = valid.clone()
        other_valid[:, c] = False

        global_context = self._masked_mean(h, other_valid)

        pos_idx = torch.arange(L, device=x.device).unsqueeze(0)
        left_all_mask = other_valid & (pos_idx < c)
        right_all_mask = other_valid & (pos_idx > c)
        left_all = self._masked_mean(h, left_all_mask)
        right_all = self._masked_mean(h, right_all_mask)
        global_side_delta = left_all - right_all


        radius_tokens = []
        radius_valid = []
        dist = (pos_idx - c).abs()
        for r in self.radii:
            local_mask = other_valid & (dist <= r)
            left_mask = local_mask & (pos_idx < c)
            right_mask = local_mask & (pos_idx > c)

            local = self._masked_mean(h, local_mask)
            left = self._masked_mean(h, left_mask)
            right = self._masked_mean(h, right_mask)

            delta = center - local
            rel = torch.cat([
                center,
                local,
                delta,
                delta.abs(),
                center * local,
                left - right,
            ], dim=-1)
            radius_tokens.append(self.radius_relation(rel))
            radius_valid.append(self._has_any(local_mask))

        rt = torch.stack(radius_tokens, dim=1)
        rv = torch.stack(radius_valid, dim=1)
        radius_ids = torch.arange(len(self.radii), device=x.device)
        rt = rt + self.radius_emb(radius_ids).unsqueeze(0)

        empty = ~rv.any(dim=1)
        if empty.any():
            rv = rv.clone()
            rv[empty, 0] = True

        gate_logits = self.radius_gate(rt).squeeze(-1)
        gate_logits = gate_logits.masked_fill(~rv, -1e4)
        gate = F.softmax(gate_logits, dim=1)
        radius_fused = (rt * gate.unsqueeze(-1)).sum(dim=1)

        rt_for_max = rt.masked_fill(~rv.unsqueeze(-1), -1e4)
        radius_max = rt_for_max.max(dim=1).values


        center_expand = center.unsqueeze(1).expand(-1, L, -1)
        site_delta = h - center_expand
        site_rel = torch.cat([
            site_delta,
            site_delta.abs(),
            h * center_expand,
        ], dim=-1)
        site_h = self.site_relation(site_rel)

        dist_idx = dist.squeeze(0).clamp(max=MAX_LEN - 1).long()
        site_logits = self.site_score(site_h).squeeze(-1)
        site_logits = site_logits + self.distance_bias(dist_idx).view(1, L)
        site_logits = site_logits.masked_fill(~other_valid, -1e4)

        has_other = other_valid.any(dim=1)
        site_attn = F.softmax(site_logits, dim=1)
        site_attn = site_attn * other_valid.float()
        site_attn = site_attn / site_attn.sum(dim=1, keepdim=True).clamp_min(1e-8)
        site_pool = (self.site_value(site_h) * site_attn.unsqueeze(-1)).sum(dim=1)
        site_pool = site_pool * has_other.float().unsqueeze(-1)

        feat = torch.cat([
            self.center_proj(center),
            self.global_proj(global_context),
            radius_fused,
            radius_max,
            site_pool,
            self.side_proj(global_side_delta),
        ], dim=-1)
        return self.head(feat)

class BranchESMSiteContrastV2(nn.Module):
    def __init__(self, D_in: int, hidden: int = 256, dropout: float = 0.16):
        super().__init__()
        d = int(CONFIG.get('ESM_SC2_DIM', 192))
        rel_dim = int(CONFIG.get('ESM_SC2_REL_DIM', 192))
        self.radii = tuple(int(r) for r in CONFIG.get('ESM_SC2_RADII', [1,2,3,5,7,10,15]))
        self.drop = float(CONFIG.get('ESM_SC2_DROPOUT', dropout))
        self.input_norm = nn.LayerNorm(D_in)
        self.input_proj = nn.Sequential(nn.Linear(D_in,d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(self.drop))
        self.pos_emb = nn.Embedding(MAX_LEN, d)

        self.radius_relation = nn.Sequential(
            nn.Linear(d*9, rel_dim), nn.LayerNorm(rel_dim), nn.GELU(), nn.Dropout(self.drop),
            nn.Linear(rel_dim, rel_dim), nn.GELU()
        )
        self.radius_emb = nn.Embedding(len(self.radii), rel_dim)
        self.radius_gate = nn.Sequential(nn.LayerNorm(rel_dim), nn.Linear(rel_dim, max(32,rel_dim//2)), nn.GELU(), nn.Linear(max(32,rel_dim//2),1))

        self.site_relation = nn.Sequential(
            nn.Linear(d*4, rel_dim), nn.LayerNorm(rel_dim), nn.GELU(), nn.Dropout(self.drop*0.75),
            nn.Linear(rel_dim, rel_dim), nn.GELU()
        )
        self.signed_distance_bias = nn.Embedding(2*MAX_LEN-1, 1)
        nn.init.zeros_(self.signed_distance_bias.weight)
        self.side_emb = nn.Embedding(3, rel_dim)
        self.site_score = nn.Linear(rel_dim,1)
        self.site_value = nn.Linear(rel_dim,rel_dim)

        self.proj = nn.ModuleDict({k: nn.Sequential(nn.Linear(d,rel_dim), nn.GELU()) for k in ['center','global','left','right','side']})
        self.head = nn.Sequential(
            nn.Linear(rel_dim*8, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(self.drop),
            nn.Linear(hidden,128), nn.GELU(), nn.Dropout(self.drop*0.5), nn.Linear(128,1)
        )

    @staticmethod
    def _mean(h, mask):
        m=mask.float().unsqueeze(-1); den=m.sum(1).clamp_min(1.0); return (h*m).sum(1)/den

    def forward(self, x):
        valid=(x.abs().sum(dim=1)>0)
        xt=x.transpose(1,2); B,L,_=xt.shape; c=min(15,L-1)
        h=self.input_proj(self.input_norm(xt))
        pos=torch.arange(L,device=x.device)
        h=h+self.pos_emb(pos).unsqueeze(0)
        h=h*valid.unsqueeze(-1).float()
        center=h[:,c]
        other=valid.clone(); other[:,c]=False
        idx=pos.unsqueeze(0)
        left_all_m=other & (idx<c); right_all_m=other & (idx>c)
        left_all=self._mean(h,left_all_m); right_all=self._mean(h,right_all_m)
        glob=self._mean(h,other); side=left_all-right_all
        dist=(idx-c).abs()
        rts=[]; rvalid=[]
        for r in self.radii:
            lm=other & (dist<=r); lmask=lm&(idx<c); rmask=lm&(idx>c)
            local=self._mean(h,lm); left=self._mean(h,lmask); right=self._mean(h,rmask)
            rel=torch.cat([center,local,left,right,center-local,center-left,center-right,left-right,(left-right).abs()],dim=-1)
            rts.append(self.radius_relation(rel)); rvalid.append(lm.any(dim=1))
        rt=torch.stack(rts,1); rv=torch.stack(rvalid,1)
        rid=torch.arange(len(self.radii),device=x.device); rt=rt+self.radius_emb(rid).unsqueeze(0)
        empty=~rv.any(1)
        if empty.any(): rv=rv.clone(); rv[empty,0]=True
        gl=self.radius_gate(rt).squeeze(-1).masked_fill(~rv,-1e4)
        gw=F.softmax(gl,dim=1); radius_fused=(rt*gw.unsqueeze(-1)).sum(1)
        radius_max=rt.masked_fill(~rv.unsqueeze(-1),-1e4).max(1).values

        ce=center.unsqueeze(1).expand(-1,L,-1); delta=h-ce
        signed=((idx-c)+(MAX_LEN-1)).clamp(0,2*MAX_LEN-2).long().squeeze(0)
        side_id=torch.zeros(L,device=x.device,dtype=torch.long); side_id[pos<c]=1; side_id[pos>c]=2
        sh=self.site_relation(torch.cat([delta,delta.abs(),h*ce,h],dim=-1)) + self.side_emb(side_id).unsqueeze(0)
        slog=self.site_score(sh).squeeze(-1)+self.signed_distance_bias(signed).view(1,L)
        slog=slog.masked_fill(~other,-1e4)
        att=F.softmax(slog,dim=1)*other.float(); att=att/att.sum(1,keepdim=True).clamp_min(1e-8)
        sp=(self.site_value(sh)*att.unsqueeze(-1)).sum(1)*other.any(1).float().unsqueeze(-1)
        feat=torch.cat([
            self.proj['center'](center), self.proj['global'](glob), radius_fused, radius_max, sp,
            self.proj['side'](side), self.proj['left'](left_all), self.proj['right'](right_all)
        ],dim=-1)
        return self.head(feat)

AA_SEQ_ORDER = list('ACDEFGHIKLMNPQRSTVWY')
AA_SEQ_STOI = {'<PAD>': 0, **{aa: i + 1 for i, aa in enumerate(AA_SEQ_ORDER)}, 'X': 21}
AA_SEQ_VOCAB_SIZE = max(AA_SEQ_STOI.values()) + 1

class KmerHead(nn.Module):
    def __init__(self, vocab_size, k: int, embed_dim=64,
                 conv_channels=64, kernels=(3,5,7), head_dim=192,
                 local_radius=4, dropout=0.25):
        super().__init__()
        self.k = int(k)
        self.local_radius = int(local_radius)
        self.max_tokens = MAX_LEN - self.k + 1

        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_tokens, embed_dim)
        self.in_norm = nn.LayerNorm(embed_dim)

        kernels = [int(x) for x in kernels]
        self.multi = nn.ModuleList([
            ConvBlock(embed_dim, conv_channels, K, dropout)
            for K in kernels
        ])
        fused_c = conv_channels * len(kernels)
        self.fuse = nn.Sequential(
            nn.Conv1d(fused_c, conv_channels * 2, kernel_size=1),
            nn.BatchNorm1d(conv_channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_channels = conv_channels * 2
        self.attn_score = nn.Conv1d(self.out_channels, 1, kernel_size=1)

        self.proj = nn.Sequential(
            nn.Linear(self.out_channels * 5, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_ids):
        B, L = x_ids.shape
        pos = torch.arange(L, device=x_ids.device).unsqueeze(0).expand(B, -1)
        e = self.emb(x_ids) + self.pos_emb(pos)
        e = self.in_norm(e).transpose(1, 2)

        hs = [blk(e) for blk in self.multi]
        h = self.fuse(torch.cat(hs, dim=1))

        g_max = h.max(dim=-1).values
        g_mean = h.mean(dim=-1)

        a = F.softmax(self.attn_score(h).squeeze(1), dim=-1)
        g_attn = (h * a.unsqueeze(1)).sum(dim=-1)

        c_lo = max(0, 15 - self.k + 1)
        c_hi = min(L, 16)
        if c_hi <= c_lo:
            g_center = h[:, :, L // 2]
        else:
            g_center = h[:, :, c_lo:c_hi].mean(dim=-1)

        aa_lo = 15 - self.local_radius
        aa_hi = 15 + self.local_radius
        l_lo = max(0, aa_lo - self.k + 1)
        l_hi = min(L, aa_hi + 1)
        if l_hi <= l_lo:
            g_local = h[:, :, L // 2]
        else:
            g_local = h[:, :, l_lo:l_hi].max(dim=-1).values

        g = torch.cat([g_max, g_mean, g_attn, g_center, g_local], dim=1)
        return self.proj(g)

class BranchKmer(nn.Module):
    def __init__(self, vocab_sizes: Dict[int,int], embed_dim=64, ksize=5, dropout=0.25):
        super().__init__()
        self.ks = sorted(vocab_sizes.keys())
        self.heads = nn.ModuleDict({
            str(k): KmerHead(
                vocab_size=vocab_sizes[k],
                k=k,
                embed_dim=embed_dim,
                conv_channels=int(CONFIG.get('KMER_CONV_CHANNELS', 64)),
                kernels=tuple(CONFIG.get('KMER_MULTI_KERNELS', [3,5,7])),
                head_dim=int(CONFIG.get('KMER_HEAD_DIM', 192)),
                local_radius=int(CONFIG.get('KMER_LOCAL_RADIUS', 4)),
                dropout=dropout,
            )
            for k in self.ks
        })
        in_dim = int(CONFIG.get('KMER_HEAD_DIM', 192)) * len(self.ks)
        self.out = nn.Sequential(
            nn.Linear(in_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, 1),
        )

    def forward(self, x_dict):
        feats = [self.heads[str(k)](x_dict[k]) for k in self.ks]
        h = torch.cat(feats, dim=1)
        return self.out(h)



@dataclass
class Sample:
    uid: str
    seq: str
    label: int
    physchem: Optional[np.ndarray] = None
    feature_uid: Optional[str] = None

    def feat_uid(self) -> str:
        return str(self.feature_uid if self.feature_uid is not None else self.uid)

class PalmDataset(Dataset):
    def __init__(self, samples,
                 branch: str,
                 esm_h5: Optional[str] = None,
                 kmer_vocabs: Optional[Dict[int, KmerVocab]] = None,
                 norm_stats: Optional[Dict[str, np.ndarray]] = None,
                 uid2idx: Optional[dict] = None) -> None:
        self.samples = samples
        self.branch = branch
        self.esm_h5_path = esm_h5
        self.kvoc = kmer_vocabs
        self.stats = norm_stats or {}
        self.uid2idx = uid2idx
        self.h5 = {}

    def __len__(self):
        return len(self.samples)

    def _get_h5(self, source: str = 'main'):
        if 'main' not in self.h5:
            if not self.esm_h5_path:
                raise RuntimeError('未提供 main ESM H5')
            self.h5['main'] = h5py.File(self.esm_h5_path, 'r')
        return self.h5['main']

    def __getstate__(self):
        state=self.__dict__.copy()
        state["h5"]={}
        return state

    def __getitem__(self, idx):
        s = self.samples[idx]
        y = np.float32(s.label)

        if self.branch == 'physchem':
            seq_feat, residue_mask = get_seq_feat_from_sequence(s.seq, MAX_LEN)
            x = {
                'seq': torch.from_numpy(seq_feat),
                'mask': torch.from_numpy(residue_mask),
            }
            return {'x': x, 'y': torch.tensor(y)}

        if self.branch == 'physchem_pca':
            seq_feat, residue_mask = get_pca_feat_from_sequence(s.seq, MAX_LEN)
            x = {
                'seq': torch.from_numpy(seq_feat),
                'mask': torch.from_numpy(residue_mask),
            }
            return {'x': x, 'y': torch.tensor(y)}

        if self.branch == 'esm':
            fuid = s.feat_uid() if hasattr(s, 'feat_uid') else str(s.uid)
            if self.uid2idx is None or fuid not in self.uid2idx:
                raise RuntimeError(f"uid={fuid} 不在 main ESM H5 中")
            j = self.uid2idx[fuid]

            store = get_esm_ram_store(self.esm_h5_path)
            if store is not None:
                x_raw = store['window_emb'][j]
                mask = (
                    store['valid_mask'][j]
                    if store.get('valid_mask') is not None
                    else np.ones((x_raw.shape[0],), dtype=np.uint8)
                )
            else:
                f = self._get_h5('main')
                x_raw = f['window_emb'][j]
                mask = (
                    f['valid_mask'][j].astype(np.uint8)
                    if 'valid_mask' in f
                    else np.ones((x_raw.shape[0],), dtype=np.uint8)
                )

            x = np.asarray(x_raw, dtype=np.float32).T

            if 'esm_mean' in self.stats:
                x = (x - self.stats['esm_mean'][:, None]) / (
                    self.stats['esm_std'][:, None] + 1e-8
                )

            if mask is not None:
                x[:, np.asarray(mask) == 0] = 0.0

            return {'x': torch.from_numpy(x), 'y': torch.tensor(y)}

        if self.branch == 'structure':
            store = _STRUCTURE_STORES.get('main')
            if store is None:
                raise RuntimeError(
                    "主 StructureFeatureStore 尚未初始化；"
                    "请先调用 prepare_structure_store(..., source='main')"
                )
            fuid = s.feat_uid() if hasattr(s, 'feat_uid') else str(s.uid)
            feat = store.get(fuid)
            chem = feat.chem.astype(np.float32, copy=True)
            if 'struct_chem_mean' in self.stats:
                chem = (chem - self.stats['struct_chem_mean']) / (
                    self.stats['struct_chem_std'] + 1e-8
                )
            x = {
              'pair': torch.from_numpy(feat.pair13.astype(np.float32, copy=False)),
              'length': torch.tensor(int(feat.length), dtype=torch.long),
              'valid_mask': torch.from_numpy(feat.valid_mask.astype(np.float32, copy=False)),
              'chem': torch.from_numpy(chem),
              'quality': torch.tensor([
                  float(feat.center_conf),
                  float(feat.mean_conf),
                  float(feat.sequence_match_fraction),
              ], dtype=torch.float32),
            }
            return {'x': x, 'y': torch.tensor(y)}

        if self.branch == 'kmer':
            if self.kvoc is None:
                raise RuntimeError("kmer 分支未提供 kmer_vocabs")
            x_dict = {}
            for k, vocab in self.kvoc.items():
                ids = vocab.encode(s.seq)
                x_dict[k] = torch.tensor(ids, dtype=torch.long)
            return {'x': x_dict, 'y': torch.tensor(y)}

        raise ValueError(f'Unknown branch={self.branch}')




_V12_AA20 = "ACDEFGHIKLMNPQRSTVWY"
_V12_AAIDX = {a:i for i,a in enumerate(_V12_AA20)}






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



    total_rows = len(df)

    labels_raw = df[labc].astype(int)
    raw_pos = int((labels_raw == 1).sum())
    raw_neg = int((labels_raw == 0).sum())

    samples = []
    removed_bad_pos_rows = []

    for row_idx, row in df.iterrows():
        uid = str(row[idc]).strip()
        raw_seq = str(row[seqc]).strip().upper()
        lab = int(row[labc])

        seq = ensure_len_31(clean_seq(raw_seq))

        center_aa = seq[15] if len(seq) > 15 else ""

        if lab == 1 and center_aa != "C":
            removed_bad_pos_rows.append({
                "row_index": row_idx,
                "uid": uid,
                "raw_seq": raw_seq,
                "processed_seq31": seq,
                "center_aa": center_aa,
                "label": lab,
            })
            continue

        pc = np.zeros(7, dtype=np.float32)

        samples.append(Sample(uid, seq, lab, pc))



    kept_total = len(samples)
    kept_pos = sum(1 for s in samples if int(s.label) == 1)
    kept_neg = sum(1 for s in samples if int(s.label) == 0)
    removed_bad_pos = len(removed_bad_pos_rows)

    print("\n[Center-C QC] 阳性中心位点检查")
    print(f"[Center-C QC] 原始总行数: {total_rows}")
    print(f"[Center-C QC] 原始阳性 lab=1: {raw_pos}")
    print(f"[Center-C QC] 原始阴性 lab=0: {raw_neg}")
    print(f"[Center-C QC] 删除的阳性中心非 C 行数: {removed_bad_pos}")
    print(f"[Center-C QC] 保留总数: {kept_total}")
    print(f"[Center-C QC] 保留阳性 lab=1: {kept_pos}")
    print(f"[Center-C QC] 保留阴性 lab=0: {kept_neg}")

    if samples:
        star_counts = np.array([s.seq.count('*') for s in samples], dtype=int)
        n_with_star = int((star_counts > 0).sum())
        print(f"[Star-padding QC] 含'*'窗口: {n_with_star}/{len(samples)} "
              f"({n_with_star/max(1,len(samples)):.2%})")
        print(f"[Star-padding QC] '*'总数: {int(star_counts.sum())}; "
              f"单窗口最大'*'数: {int(star_counts.max())}")

    if removed_bad_pos > 0:
        try:
            out_dir = CONFIG.get("OUT_DIR", os.path.dirname(csv_path))
            os.makedirs(out_dir, exist_ok=True)

            bad_path = os.path.join(out_dir, "removed_positive_center_not_C.csv")
            pd.DataFrame(removed_bad_pos_rows).to_csv(bad_path, index=False)

            print(f"[Center-C QC] 异常阳性样本明细已保存: {bad_path}")

            print("[Center-C QC] 前 10 条异常样本:")
            print(
                pd.DataFrame(removed_bad_pos_rows)
                [["row_index", "uid", "center_aa", "processed_seq31"]]
                .head(10)
                .to_string(index=False)
            )
        except Exception as e:
            print(f"[Center-C QC][WARN] 异常样本明细保存失败: {e}")

    print()

    return samples




from sklearn.model_selection import StratifiedKFold, train_test_split
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:
    StratifiedGroupKFold = None

def split_train_test(samples: List[Sample], test_ratio=0.1, split_seed=2025):
    if not bool(CONFIG.get('GROUP_SPLIT_BY_PROTEIN', True)):
        raise RuntimeError(
            "v3 要求 GROUP_SPLIT_BY_PROTEIN=True；同一蛋白的不同位点不能跨 train/test。"
        )
    if StratifiedGroupKFold is None:
        raise RuntimeError(
            "当前 sklearn 没有 StratifiedGroupKFold。请升级 scikit-learn；"
            "本版本不会回退到样本级 split，以免产生蛋白泄漏。"
        )

    y = np.array([int(s.label) for s in samples], dtype=int)
    idx = np.arange(len(samples))
    groups = np.array([_base_uid(s.uid) for s in samples], dtype=object)

    n_splits = max(2, int(round(1.0 / float(test_ratio))))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=split_seed,
    )

    global_rate = float(y.mean())
    best = None
    for cand_id, (tr_idx, te_idx) in enumerate(splitter.split(idx, y, groups), start=1):
        actual_ratio = len(te_idx) / max(1, len(idx))
        te_rate = float(y[te_idx].mean()) if len(te_idx) else 0.0
        objective = abs(actual_ratio - float(test_ratio)) + 0.5 * abs(te_rate - global_rate)
        item = (objective, cand_id, tr_idx, te_idx, actual_ratio, te_rate)
        if best is None or item[0] < best[0]:
            best = item

    _, cand_id, tr_idx, te_idx, actual_ratio, te_rate = best
    tr_groups = set(groups[tr_idx].tolist())
    te_groups = set(groups[te_idx].tolist())
    overlap = tr_groups & te_groups
    if overlap:
        raise RuntimeError(f"Protein group leakage detected: {len(overlap)} groups overlap")

    print(
        f"[Split] protein-group split enabled; candidate={cand_id}/{n_splits}; "
        f"target test={test_ratio:.3f}, actual={actual_ratio:.3f}; "
        f"global pos={global_rate:.3f}, test pos={te_rate:.3f}; "
        f"train proteins={len(tr_groups)}, test proteins={len(te_groups)}, overlap=0"
    )

    train_samples = [samples[i] for i in tr_idx]
    test_samples = [samples[i] for i in te_idx]
    return train_samples, test_samples

def build_folds(train_samples: List[Sample], n_folds=5, cv_seed=1314):
    if StratifiedGroupKFold is None:
        raise RuntimeError("需要 StratifiedGroupKFold；拒绝回退到样本级 CV。")

    y_tr = np.array([int(s.label) for s in train_samples], dtype=int)
    idx = np.arange(len(train_samples))
    groups = np.array([_base_uid(s.uid) for s in train_samples], dtype=object)

    splitter = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=cv_seed,
    )
    folds = []
    seen_val_groups = set()
    for fold_id, (tr_sub, va_sub) in enumerate(splitter.split(idx, y_tr, groups), start=1):
        tr_g = set(groups[tr_sub].tolist())
        va_g = set(groups[va_sub].tolist())
        overlap = tr_g & va_g
        if overlap:
            raise RuntimeError(f"Fold {fold_id}: protein leakage, overlap={len(overlap)}")
        seen_val_groups.update(va_g)
        tr_list = [train_samples[i] for i in tr_sub]
        va_list = [train_samples[i] for i in va_sub]
        print(
            f"[CV] fold={fold_id}: train={len(tr_list)} val={len(va_list)}; "
            f"train proteins={len(tr_g)} val proteins={len(va_g)}; "
            f"val pos={np.mean([s.label for s in va_list]):.3f}; overlap=0"
        )
        folds.append((tr_list, va_list))

    if len(folds) != n_folds:
        raise RuntimeError(f"Expected {n_folds} folds, got {len(folds)}")
    return folds


def esm_channel_stats(esm_h5: str,
                      samples: List[Sample],
                      uid2idx: dict) -> Tuple[np.ndarray, np.ndarray]:
    idxes = np.array([uid2idx[s.uid] for s in samples], dtype=np.int64)
    uniq_idx, counts = np.unique(idxes, return_counts=True)

    store = get_esm_ram_store(esm_h5)

    if store is not None:
        emb = store['window_emb']
        valid_all = store.get('valid_mask')
        D = emb.shape[-1]

        S = np.zeros(D, dtype=np.float64)
        SS = np.zeros(D, dtype=np.float64)
        n = 0

        for i, c in zip(uniq_idx, counts):
            x = np.asarray(emb[int(i)], dtype=np.float32)
            if CONFIG.get('ESM_NORMALIZE_VALID_ONLY', True) and valid_all is not None:
                m = np.asarray(valid_all[int(i)]).astype(bool)
                x_use = x[m]
            else:
                x_use = x

            if x_use.shape[0] == 0:
                continue

            S += x_use.sum(axis=0, dtype=np.float64) * int(c)
            SS += np.square(x_use, dtype=np.float32).sum(axis=0, dtype=np.float64) * int(c)
            n += x_use.shape[0] * int(c)

    else:
        with h5py.File(esm_h5, 'r') as f:
            D = f['window_emb'].shape[-1]
            has_mask = 'valid_mask' in f
            S = np.zeros(D, dtype=np.float64)
            SS = np.zeros(D, dtype=np.float64)
            n = 0

            for i, c in zip(uniq_idx, counts):
                x = f['window_emb'][int(i)].astype(np.float32)
                if CONFIG.get('ESM_NORMALIZE_VALID_ONLY', True) and has_mask:
                    m = f['valid_mask'][int(i)].astype(bool)
                    x_use = x[m]
                else:
                    x_use = x

                if x_use.shape[0] == 0:
                    continue

                S += x_use.sum(axis=0, dtype=np.float64) * int(c)
                SS += np.square(x_use, dtype=np.float32).sum(axis=0, dtype=np.float64) * int(c)
                n += x_use.shape[0] * int(c)

    if n <= 0:
        raise RuntimeError("ESM mean/std 统计没有有效残基。")

    mean = S / n
    var = SS / n - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)







def make_fixed_ratio_by_pos(samples: List[Sample],
                            neg_pos_ratio: Optional[float],
                            seed: int = 0,
                            branch_name: Optional[str] = None,
                            repeat_id: Optional[int] = None,
                            fold_id: Optional[int] = None) -> List[Sample]:







    pos = [s for s in samples if int(s.label) == 1]
    neg = [s for s in samples if int(s.label) == 0]

    if not pos or not neg or neg_pos_ratio is None:
        out = list(samples)
        random.Random(seed).shuffle(out)
        return out

    ratio = float(neg_pos_ratio)
    if ratio <= 0:
        raise ValueError(f"NEG_POS_RATIO must be > 0 or None, got {neg_pos_ratio}")

    n_neg = min(int(round(len(pos) * ratio)), len(neg))
    rng = random.Random(seed)
    neg_pool = sorted(neg, key=lambda s: (str(s.uid), str(s.seq)))
    rng.shuffle(neg_pool)
    out = pos + neg_pool[:n_neg]
    rng.shuffle(out)

    print(
        f"[Ratio] branch={branch_name} rep={repeat_id} fold={fold_id}: "
        f"pool pos={len(pos)} neg={len(neg)} -> used pos={len(pos)} neg={n_neg}"
    )
    return out




def other_cys_count(seq: str) -> int:
    seq = ensure_len_31(clean_seq(seq))
    total = int(seq.count('C'))
    if len(seq) > 15 and seq[15] == 'C':
        total -= 1
    return max(0, total)

def _np_logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0-eps)
    return np.log(p) - np.log1p(-p)

def _np_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out.astype(np.float32)

def fit_robust_score_alignment(val_prob: np.ndarray) -> Dict[str, float]:
    z = _np_logit(val_prob)
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < float(CONFIG.get('FOLD_ALIGN_EPS', 1e-6)):
        scale = float(np.std(z))
    if not np.isfinite(scale) or scale < float(CONFIG.get('FOLD_ALIGN_EPS', 1e-6)):
        scale = 1.0
    return {'center': med, 'scale': scale}

def apply_score_alignment(prob: np.ndarray, pars: Dict[str, float]) -> np.ndarray:
    z = (_np_logit(prob) - float(pars['center'])) / max(float(pars['scale']), 1e-6)
    return _np_sigmoid(z)

def align_fold_predictions(raw_oof: np.ndarray,
                           test_prob_folds: List[np.ndarray],
                           train_samples: List['Sample'],
                           folds: List[Tuple[List['Sample'], List['Sample']]]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    if not bool(CONFIG.get('FOLD_SCORE_ALIGNMENT', True)):
        return np.asarray(raw_oof, dtype=np.float32), np.mean(np.stack(test_prob_folds), axis=0).astype(np.float32), []
    idx_map = {id(s): i for i, s in enumerate(train_samples)}
    oof_aligned = np.zeros(len(train_samples), dtype=np.float32)
    test_aligned_folds = []
    params = []
    for fold_id, (_, va) in enumerate(folds):
        ids = np.asarray([idx_map[id(s)] for s in va], dtype=int)
        pv = np.asarray(raw_oof)[ids]
        pars = fit_robust_score_alignment(pv)
        pars['fold'] = int(fold_id + 1)
        oof_aligned[ids] = apply_score_alignment(pv, pars)
        test_aligned_folds.append(apply_score_alignment(test_prob_folds[fold_id], pars))
        params.append(pars)
    return oof_aligned, np.mean(np.stack(test_aligned_folds), axis=0).astype(np.float32), params




STRUCT_CENTER = 15
STRUCT_CYS_NAMES = {"CYS", "CYX", "CYM"}
STRUCT_BACKBONE = {"N", "CA", "C", "O", "OXT"}
STRUCT_HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
STRUCT_POLAR       = {"SER", "THR", "ASN", "GLN", "TYR", "HIS"}
STRUCT_POSITIVE    = {"LYS", "ARG", "HIS"}
STRUCT_NEGATIVE    = {"ASP", "GLU"}
STRUCT_AROMATIC    = {"PHE", "TRP", "TYR", "HIS"}

STRUCT_AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','CYX':'C','CYM':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',
    'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y',
    'VAL':'V','SEC':'U','PYL':'O',
}

STRUCT_CHEM_NAMES = [
    "sg_present",
    "other_cys_count",
    "other_cys_seq_within3",
    "other_cys_seq_within5",
    "other_cys_seq_within10",
    "nearest_other_cys_seq_sep",
    "min_other_cys_sg_dist",
    "disulfide_like_lt3A",
    "other_cys_sg_within4A",
    "other_cys_sg_within6A",
    "other_cys_sg_within8A",
    "neighbor_res_within4A",
    "neighbor_res_within6A",
    "neighbor_res_within8A",
    "neighbor_res_within10A",
    "min_NO_atom_dist",
    "NO_atom_count_within4A",
    "NO_atom_count_within6A",
    "neighbor8_hydrophobic_frac",
    "neighbor8_polar_frac",
    "neighbor8_positive_frac",
    "neighbor8_negative_frac",
    "neighbor8_aromatic_frac",
    "neighbor8_cys_frac",
    "sg_to_CA_centroid_dist",
    "mean_3_nearest_residue_dist",
]
STRUCT_CHEM_DIM = len(STRUCT_CHEM_NAMES)

_STRUCT_PARSER = PDBParser(QUIET=True) if BIOPDB_OK else None

def _struct_residues_from_pdb(path: str):
    if _STRUCT_PARSER is None:
        raise RuntimeError("structure branch 需要 Biopython Bio.PDB；当前环境无法 import PDBParser")
    st = _STRUCT_PARSER.get_structure(os.path.basename(path), path)
    residues = []
    for chain in st[0]:
        for r in chain:
            if "CA" in r:
                residues.append(r)
    return residues[:MAX_LEN]

def _struct_resname(r) -> str:
    return r.get_resname().strip().upper()

def _struct_coord(r, name: str) -> Optional[np.ndarray]:
    if name in r:
        return r[name].get_coord().astype(np.float32)
    return None

def _struct_ca(r) -> np.ndarray:
    return r["CA"].get_coord().astype(np.float32)

def _struct_sidechain_proxy(r) -> np.ndarray:
    if "CB" in r:
        return r["CB"].get_coord().astype(np.float32)
    arr = []
    for atom in r:
        name = atom.get_name().strip()
        if not name.startswith("H") and name not in STRUCT_BACKBONE:
            arr.append(atom.get_coord())
    if arr:
        return np.mean(np.stack(arr), axis=0).astype(np.float32)
    return _struct_ca(r)

def _struct_heavy_atoms(r) -> np.ndarray:
    arr = []
    for atom in r:
        name = atom.get_name().strip()
        if not name.startswith("H"):
            arr.append(atom.get_coord())
    if not arr:
        arr = [r["CA"].get_coord()]
    return np.asarray(arr, dtype=np.float32)

def _struct_normalize_vec(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)

def _struct_local_frame(r) -> np.ndarray:
    CA = _struct_ca(r)
    C = _struct_coord(r, "C")
    N = _struct_coord(r, "N")
    if C is None or N is None:
        return np.eye(3, dtype=np.float32)
    e1 = _struct_normalize_vec(C - CA)
    v2 = N - CA
    v2 = v2 - np.dot(v2, e1) * e1
    e2 = _struct_normalize_vec(v2)
    e3 = _struct_normalize_vec(np.cross(e1, e2))
    return np.stack([e1, e2, e3], axis=1).astype(np.float32)

def _struct_atom_is_NO(atom) -> bool:
    name = atom.get_name().strip().upper()
    return bool(name) and name[0] in {"N", "O"}

def _struct_min_dist_point_to_residue(point: np.ndarray, r) -> float:
    H = _struct_heavy_atoms(r)
    return float(np.linalg.norm(H - point[None, :], axis=1).min())

def _struct_extract_center_chem(residues) -> np.ndarray:
    L = len(residues)

    if L <= STRUCT_CENTER or residues[STRUCT_CENTER] is None:
        return np.zeros(STRUCT_CHEM_DIM, np.float32)

    center = residues[STRUCT_CENTER]
    sg = _struct_coord(center, "SG")
    sg_present = float(sg is not None)

    max_dist = 30.0
    nearest_seq_sep = float(MAX_LEN)
    min_cys_sg_dist = max_dist
    min_no_dist = max_dist
    sg_to_centroid = max_dist
    mean3 = max_dist

    valid_indices = [
        j for j, r in enumerate(residues)
        if r is not None
    ]

    other_cys_idx = [
        j for j in valid_indices
        if j != STRUCT_CENTER and _struct_resname(residues[j]) in STRUCT_CYS_NAMES
    ]

    other_cys_count = len(other_cys_idx)

    if other_cys_idx:
        nearest_seq_sep = float(min(abs(j - STRUCT_CENTER) for j in other_cys_idx))

    cys_seq3 = sum(abs(j - STRUCT_CENTER) <= 3 for j in other_cys_idx)
    cys_seq5 = sum(abs(j - STRUCT_CENTER) <= 5 for j in other_cys_idx)
    cys_seq10 = sum(abs(j - STRUCT_CENTER) <= 10 for j in other_cys_idx)

    cys_sg_dists: List[float] = []
    neighbor_dists: List[Tuple[int, float]] = []
    no_dists: List[float] = []

    if sg is not None:
        for j in valid_indices:
            if j == STRUCT_CENTER:
                continue

            r = residues[j]

            dres = _struct_min_dist_point_to_residue(sg, r)
            neighbor_dists.append((j, dres))

            if _struct_resname(r) in STRUCT_CYS_NAMES and "SG" in r:
                dss = float(
                    np.linalg.norm(
                        r["SG"].get_coord().astype(np.float32) - sg
                    )
                )
                cys_sg_dists.append(dss)

            for atom in r:
                name = atom.get_name().strip().upper()
                if name.startswith("H"):
                    continue
                if _struct_atom_is_NO(atom):
                    no_dists.append(
                        float(
                            np.linalg.norm(
                                atom.get_coord().astype(np.float32) - sg
                            )
                        )
                    )

        if cys_sg_dists:
            min_cys_sg_dist = float(min(cys_sg_dists))

        if no_dists:
            min_no_dist = float(min(no_dists))

        ca_xyz = np.stack([
            _struct_ca(residues[j])
            for j in valid_indices
        ])

        sg_to_centroid = float(np.linalg.norm(sg - ca_xyz.mean(axis=0)))

        if neighbor_dists:
            ds = sorted(d for _, d in neighbor_dists)
            mean3 = float(np.mean(ds[:min(3, len(ds))]))

    cys4 = sum(d <= 4.0 for d in cys_sg_dists)
    cys6 = sum(d <= 6.0 for d in cys_sg_dists)
    cys8 = sum(d <= 8.0 for d in cys_sg_dists)

    n4 = sum(d <= 4.0 for _, d in neighbor_dists)
    n6 = sum(d <= 6.0 for _, d in neighbor_dists)
    n8 = sum(d <= 8.0 for _, d in neighbor_dists)
    n10 = sum(d <= 10.0 for _, d in neighbor_dists)

    no4 = sum(d <= 4.0 for d in no_dists)
    no6 = sum(d <= 6.0 for d in no_dists)

    neigh8_idx = [j for j, d in neighbor_dists if d <= 8.0]
    n_neigh = len(neigh8_idx)

    def frac(group):
        if n_neigh == 0:
            return 0.0
        return float(
            sum(_struct_resname(residues[j]) in group for j in neigh8_idx)
            / n_neigh
        )

    x = np.asarray([
        sg_present,
        float(other_cys_count),
        float(cys_seq3),
        float(cys_seq5),
        float(cys_seq10),
        nearest_seq_sep,
        min_cys_sg_dist,
        float(min_cys_sg_dist < 3.0),
        float(cys4),
        float(cys6),
        float(cys8),
        float(n4),
        float(n6),
        float(n8),
        float(n10),
        min_no_dist,
        float(no4),
        float(no6),
        frac(STRUCT_HYDROPHOBIC),
        frac(STRUCT_POLAR),
        frac(STRUCT_POSITIVE),
        frac(STRUCT_NEGATIVE),
        frac(STRUCT_AROMATIC),
        frac(STRUCT_CYS_NAMES),
        sg_to_centroid,
        mean3,
    ], dtype=np.float32)

    if x.shape[0] != STRUCT_CHEM_DIM:
        raise RuntimeError(
            f"structure chemistry dim mismatch: {x.shape[0]} vs {STRUCT_CHEM_DIM}"
        )

    return x

@dataclass
class StructureFeat:
    pair13: np.ndarray
    chem: np.ndarray
    length: int
    valid_mask: np.ndarray

    center_is_cys: bool
    sg_present: bool
    center_conf: float
    mean_conf: float

    pdb_sequence: str
    aligned_sequence: str
    align_shift: int
    sequence_match_fraction: float
    center_pdb_index: int
def _struct_pdb_sequence(residues) -> str:
    return ''.join(
        STRUCT_AA3_TO_1.get(_struct_resname(r), 'X')
        for r in residues
    )

def _align_seq31_to_pdb(seq31: str, pdb_seq: str) -> Dict:
    seq31 = ensure_len_31(clean_seq(seq31))
    pdb_seq = str(pdb_seq).strip().upper()

    real_positions = [i for i, aa in enumerate(seq31) if aa != '*']
    total_real = len(real_positions)
    if total_real <= 0:
        raise RuntimeError(f"seq31 没有真实残基，无法对齐: {seq31}")

    best = None

    for shift in range(-MAX_LEN, MAX_LEN + 1):
        compared = 0
        matched = 0
        mismatch = 0

        for i in real_positions:
            aa_csv = seq31[i]
            j = i + shift
            if j < 0 or j >= len(pdb_seq):
                mismatch += 1
                continue

            compared += 1
            aa_pdb = pdb_seq[j]
            if aa_csv == aa_pdb or aa_pdb == 'X':
                matched += 1
            else:
                mismatch += 1

        frac = matched / total_real

        item = {
            'shift': int(shift),
            'compared': int(compared),
            'matched': int(matched),
            'mismatch': int(mismatch),
            'total_real': int(total_real),
            'match_fraction': float(frac),
        }

        if best is None:
            best = item
        else:
            key = (item['match_fraction'], item['matched'], item['compared'])
            best_key = (best['match_fraction'], best['matched'], best['compared'])
            if key > best_key:
                best = item

    if best is None:
        raise RuntimeError("无法对齐 seq31 和 PDB sequence")

    return best

def _make_aligned_residue_slots(seq31: str, residues) -> Tuple[List, Dict]:
    seq31 = ensure_len_31(clean_seq(seq31))
    pdb_seq = _struct_pdb_sequence(residues)

    align = _align_seq31_to_pdb(seq31, pdb_seq)
    shift = int(align['shift'])

    slot_residues = [None] * MAX_LEN

    for i, aa_csv in enumerate(seq31):
        if aa_csv == '*':
            continue

        j = i + shift
        if 0 <= j < len(residues):
            slot_residues[i] = residues[j]

    center_pdb_index = STRUCT_CENTER + shift

    align['center_pdb_index'] = int(center_pdb_index)
    align['pdb_seq'] = pdb_seq

    return slot_residues, align

def _struct_extract_feat(path: str, seq31: Optional[str] = None) -> StructureFeat:
    rs_raw = _struct_residues_from_pdb(path)

    if seq31 is None:
        seq31 = _struct_pdb_sequence(rs_raw)
        seq31 = ensure_len_31(seq31)

    seq31 = ensure_len_31(clean_seq(seq31))

    slot_residues, align = _make_aligned_residue_slots(seq31, rs_raw)

    valid_mask = np.asarray(
        [r is not None for r in slot_residues],
        dtype=np.bool_
    )

    valid_indices = np.where(valid_mask)[0].tolist()

    if len(valid_indices) == 0:
        raise RuntimeError(f"No valid residues after alignment: {path}")

    center_residue = slot_residues[STRUCT_CENTER]

    if center_residue is None:
        raise RuntimeError(
            f"Aligned center is padding/None: {path}, seq31={seq31}, align={align}"
        )

    center_name = _struct_resname(center_residue)
    center_is_cys = center_name in STRUCT_CYS_NAMES

    if bool(CONFIG.get('STRUCTURE_REQUIRE_CENTER_CYS', True)) and not center_is_cys:
        raise RuntimeError(
            f"Aligned center residue at CSV index15 is not Cys: "
            f"{path}, center={center_name}, seq31={seq31}, align={align}"
        )




    pair13 = np.zeros((13, MAX_LEN, MAX_LEN), np.float32)

    ca = {}
    sc = {}
    heavy = {}
    frames = {}

    for i in valid_indices:
        r = slot_residues[i]
        ca[i] = _struct_ca(r)
        sc[i] = _struct_sidechain_proxy(r)
        heavy[i] = _struct_heavy_atoms(r)
        frames[i] = _struct_local_frame(r)

    contact_thresh = float(CONFIG.get('STRUCTURE_CONTACT_THRESH', 8.0))

    for i in valid_indices:
        for j in valid_indices:
            d_ca = float(np.linalg.norm(ca[i] - ca[j]))
            d_sc = float(np.linalg.norm(sc[i] - sc[j]))

            d_min = float(
                np.linalg.norm(
                    heavy[i][:, None, :] - heavy[j][None, :, :],
                    axis=-1
                ).min()
            )

            sep = float(abs(i - j))
            sep_norm = sep / max(MAX_LEN - 1, 1)

            contact = float(d_ca < contact_thresh)
            long_range = float((sep >= 6) and (d_ca < contact_thresh))

            pair13[0, i, j] = d_ca
            pair13[1, i, j] = d_min
            pair13[2, i, j] = d_sc
            pair13[3, i, j] = contact
            pair13[4, i, j] = long_range
            pair13[5, i, j] = sep_norm

            if i == STRUCT_CENTER or j == STRUCT_CENTER:
                pair13[6, i, j] = 1.0
            if i == STRUCT_CENTER and j == STRUCT_CENTER:
                pair13[6, i, j] = 2.0

            Ri = frames[i]
            Rj = frames[j]

            if i != j:
                u = _struct_normalize_vec(ca[j] - ca[i])
                pair13[7:10, i, j] = Ri.T @ u

            rel = Ri.T @ Rj
            pair13[10, i, j] = rel[0, 0]
            pair13[11, i, j] = rel[1, 1]
            pair13[12, i, j] = rel[2, 2]

    chem = _struct_extract_center_chem(slot_residues)

    conf_values = []
    for i in valid_indices:
        try:
            conf_values.append(float(slot_residues[i]["CA"].get_bfactor()))
        except Exception:
            pass

    conf = np.asarray(conf_values, dtype=np.float32)

    if len(conf) and np.nanmax(conf) > 1.5:
        conf = conf / 100.0

    conf = np.clip(conf, 0.0, 1.0)

    try:
        center_conf = float(center_residue["CA"].get_bfactor())
        if center_conf > 1.5:
            center_conf = center_conf / 100.0
        center_conf = float(np.clip(center_conf, 0.0, 1.0))
    except Exception:
        center_conf = 0.0

    mean_conf = float(np.mean(conf)) if len(conf) else 0.0

    aligned_seq = ''.join(
        STRUCT_AA3_TO_1.get(_struct_resname(r), 'X') if r is not None else '*'
        for r in slot_residues
    )

    return StructureFeat(
        pair13=pair13,
        chem=chem,
        length=int(valid_mask.sum()),
        valid_mask=valid_mask.astype(np.float32),

        center_is_cys=center_is_cys,
        sg_present=bool("SG" in center_residue),
        center_conf=center_conf,
        mean_conf=mean_conf,

        pdb_sequence=_struct_pdb_sequence(rs_raw),
        aligned_sequence=aligned_seq,
        align_shift=int(align['shift']),
        sequence_match_fraction=float(align['match_fraction']),
        center_pdb_index=int(align['center_pdb_index']),
    )

class StructureFeatureStore:
    def __init__(self, samples: List[Sample], pdb_dir: str):
        if not BIOPDB_OK:
            raise RuntimeError("structure branch 需要 biopython (Bio.PDB)")
        if not pdb_dir or not os.path.isdir(pdb_dir):
            raise FileNotFoundError(f"STRUCTURE PDB_DIR 不存在: {pdb_dir}")

        self.samples = samples
        self.sample_by_uid = {str(s.uid): s for s in samples}
        self.pdb_dir = pdb_dir
        self.cache: Dict[str, StructureFeat] = {}
        self.path: Dict[str, str] = {}
        self.match_mode: Dict[str, str] = {}

        exact: Dict[str, str] = {}
        base_paths: Dict[str, List[str]] = {}
        for fn in sorted(os.listdir(pdb_dir)):
            if not fn.lower().endswith('.pdb'):
                continue
            stem = fn[:-4]
            path = os.path.join(pdb_dir, fn)
            exact[stem] = path
            base_paths.setdefault(_base_uid(stem), []).append(path)

        sample_base_count: Dict[str, int] = {}
        for s in samples:
            b = _base_uid(s.uid)
            sample_base_count[b] = sample_base_count.get(b, 0) + 1

        unresolved = []
        for s in samples:
            uid = str(s.uid)
            base = _base_uid(uid)
            if uid in exact:
                self.path[uid] = exact[uid]
                self.match_mode[uid] = 'exact_uid'
                continue

            candidate = exact.get(base)
            unique_base = base_paths.get(base, [])
            if candidate is None and len(unique_base) == 1:
                candidate = unique_base[0]

            if (
                candidate is not None
                and bool(CONFIG.get('STRUCTURE_ALLOW_BASE_FALLBACK', True))
                and sample_base_count.get(base, 0) == 1
            ):
                self.path[uid] = candidate
                self.match_mode[uid] = 'unique_base_fallback'
            else:
                reason = 'missing'
                if candidate is not None and sample_base_count.get(base, 0) > 1:
                    reason = 'ambiguous_base_fallback_multiple_sites'
                elif len(unique_base) > 1:
                    reason = 'ambiguous_multiple_pdbs'
                unresolved.append({
                    'uniprotid': uid,
                    'base_uniprot': base,
                    'reason': reason,
                    'n_dataset_sites_for_base': sample_base_count.get(base, 0),
                    'n_pdbs_for_base': len(unique_base),
                })

        if unresolved:
            os.makedirs(CONFIG['OUT_DIR'], exist_ok=True)
            path = os.path.join(CONFIG['OUT_DIR'], 'structure_unresolved_pdb.csv')
            pd.DataFrame(unresolved).to_csv(path, index=False)
            raise RuntimeError(
                f"Structure branch: {len(unresolved)} samples cannot be mapped safely to PDBs. "
                f"See {path}. 不会静默复用错误的 base-level PDB。"
            )

        if bool(CONFIG.get('STRUCTURE_PRELOAD', True)):
            self.preload_and_qc()

    def get(self, uid: str) -> StructureFeat:
        uid = str(uid)
        if uid not in self.cache:
            s = self.sample_by_uid[uid]
            seq31 = ensure_len_31(clean_seq(s.seq))
            self.cache[uid] = _struct_extract_feat(self.path[uid], seq31=seq31)
        return self.cache[uid]

    def preload_and_qc(self):
        print(f"[Structure] preload/QC {len(self.samples)} PDB-derived samples ...")
        qc_rows = []
        chem_rows = []
        fatal = []

        for i, s in enumerate(self.samples, 1):
            try:
                f = self.get(s.uid)
            except Exception as e:
                fatal.append({'uniprotid': s.uid, 'reason': f'parse_error: {e}'})
                continue

            seq31 = ensure_len_31(clean_seq(s.seq))
            center_seq_is_c = bool(len(seq31) > STRUCT_CENTER and seq31[STRUCT_CENTER] == 'C')

            aligned_sequence = str(getattr(f, 'aligned_sequence', ''))
            if len(aligned_sequence) == MAX_LEN:
                compared = 0
                matched = 0
                for j in range(MAX_LEN):
                    if seq31[j] == '*':
                        continue
                    compared += 1
                    if seq31[j] == aligned_sequence[j] or aligned_sequence[j] == 'X':
                        matched += 1
                seq_match_fraction = float(matched / compared) if compared else np.nan
            else:
                seq_match_fraction = float(getattr(f, 'sequence_match_fraction', np.nan))

            row = {
                'uniprotid': s.uid,
                'base_uniprot': _base_uid(s.uid),
                'label': int(s.label),
                'pdb_path': self.path[s.uid],
                'match_mode': self.match_mode[s.uid],
                'pdb_length': int(f.length),
                'sequence_match_fraction': seq_match_fraction,
                'csv_center_is_cys': int(center_seq_is_c),
                'pdb_center_is_cys': int(f.center_is_cys),
                'sg_present': int(f.sg_present),
                'center_conf': float(f.center_conf),
                'mean_conf': float(f.mean_conf),
                'align_shift': int(getattr(f, 'align_shift', 0)),
                'center_pdb_index': int(getattr(f, 'center_pdb_index', -1)),
                'pdb_sequence': str(getattr(f, 'pdb_sequence', '')),
                'aligned_sequence': aligned_sequence,
            }
            qc_rows.append(row)

            cr = {'uniprotid': s.uid, 'base_uniprot': _base_uid(s.uid), 'label': int(s.label)}
            cr.update({name: float(v) for name, v in zip(STRUCT_CHEM_NAMES, f.chem)})
            chem_rows.append(cr)

            if bool(CONFIG.get('STRUCTURE_REQUIRE_CENTER_CYS', True)) and not f.center_is_cys:
                fatal.append({'uniprotid': s.uid, 'reason': 'Aligned center residue at CSV index15 is not Cys'})

            if i % 500 == 0 or i == len(self.samples):
                print(f"  [Structure] {i}/{len(self.samples)}")

        os.makedirs(CONFIG['OUT_DIR'], exist_ok=True)
        pd.DataFrame(qc_rows).to_csv(os.path.join(CONFIG['OUT_DIR'], 'structure_pdb_qc.csv'), index=False)
        pd.DataFrame(chem_rows).to_csv(os.path.join(CONFIG['OUT_DIR'], 'structure_chemistry_raw.csv'), index=False)

        if fatal:
            path = os.path.join(CONFIG['OUT_DIR'], 'structure_fatal_qc.csv')
            pd.DataFrame(fatal).to_csv(path, index=False)
            raise RuntimeError(
                f"Structure QC failed for {len(fatal)} samples; see {path}. "
                f"最常见原因是 PDB 与 seq31 无法正确对齐，或 aligned center 不是 Cys。"
            )

        if qc_rows:
            q = pd.DataFrame(qc_rows)
            print(
                f"[Structure QC] exact={int((q.match_mode=='exact_uid').sum())}, "
                f"base-fallback={int((q.match_mode=='unique_base_fallback').sum())}, "
                f"center-Cys={int(q.pdb_center_is_cys.sum())}/{len(q)}, "
                f"SG-present={int(q.sg_present.sum())}/{len(q)}"
            )
            finite_match = q['sequence_match_fraction'].dropna()
            if len(finite_match):
                print(
                    f"[Structure QC] median sequence match={finite_match.median():.3f}; "
                    f"<0.80 count={int((finite_match < 0.80).sum())}"
                )

_STRUCTURE_STORES: Dict[str, StructureFeatureStore] = {}


def prepare_structure_store(
    samples: List[Sample],
    source: str = 'main',
    pdb_dir: Optional[str] = None,
) -> StructureFeatureStore:
    source = str(source or 'main')

    if source in _STRUCTURE_STORES:
        return _STRUCTURE_STORES[source]

    use_dir = pdb_dir if pdb_dir is not None else CONFIG['PDB_DIR']

    if not use_dir:
        raise RuntimeError(
            f"StructureFeatureStore source={source}: PDB_DIR 为空"
        )

    if not os.path.isdir(use_dir):
        raise FileNotFoundError(
            f"StructureFeatureStore source={source}: PDB_DIR 不存在: {use_dir}"
        )

    store = StructureFeatureStore(
        samples=samples,
        pdb_dir=use_dir,
    )

    _STRUCTURE_STORES[source] = store
    return store


def get_structure_store(source: str = 'main') -> StructureFeatureStore:
    source = str(source or 'main')

    store = _STRUCTURE_STORES.get(source)

    if store is None:
        raise RuntimeError(
            f"StructureFeatureStore source={source} 尚未初始化；"
            f"请先调用 prepare_structure_store(..., source={source!r})"
        )

    return store


def structure_chemistry_stats(
    samples: List[Sample],
) -> Tuple[np.ndarray, np.ndarray]:

    store = _STRUCTURE_STORES.get('main')

    if store is None:
        raise RuntimeError(
            "structure_chemistry_stats: main StructureFeatureStore 尚未初始化"
        )

    chem_rows = []

    for s in samples:
        fuid = s.feat_uid() if hasattr(s, 'feat_uid') else str(s.uid)
        chem_rows.append(
            store.get(fuid).chem
        )

    if not chem_rows:
        raise RuntimeError(
            "structure_chemistry_stats: samples 为空"
        )

    X = np.stack(
        chem_rows,
        axis=0,
    ).astype(np.float32)

    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.where(
        std < 1e-6,
        1.0,
        std,
    ).astype(np.float32)

    return mean, std

class StructChemMLP(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        hidden = int(CONFIG.get('STRUCTURE_CHEM_HIDDEN', 64))
        drop = float(CONFIG.get('STRUCTURE_DROPOUT', 0.25))
        self.net = nn.Sequential(
            nn.LayerNorm(STRUCT_CHEM_DIM),
            nn.Linear(STRUCT_CHEM_DIM, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class StructSafeAttentionReadout(nn.Module):
    def __init__(self, dim: int, attn_dim: int = 96):
        super().__init__()
        self.a = nn.Linear(dim, attn_dim)
        self.o = nn.Linear(attn_dim, 1)
        nn.init.xavier_uniform_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.bool()
        score = self.o(torch.tanh(self.a(h))).squeeze(-1)
        score = score.masked_fill(~m, -1e9)
        w = F.softmax(score, dim=-1)
        w = w * m.float()
        w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        out = torch.bmm(w.unsqueeze(1), h).squeeze(1)
        has_any = m.any(dim=1, keepdim=True).float()
        return out * has_any

class StructCenterAwareEdgeLayer(nn.Module):
    def __init__(self, dim: int, edge_dim: int = 22):
        super().__init__()
        drop = float(CONFIG.get('STRUCTURE_DROPOUT', 0.20))
        gate_hidden = max(64, dim // 2)
        self.edge_gate = nn.Sequential(
            nn.LayerNorm(edge_dim),
            nn.Linear(edge_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(gate_hidden, 1),
        )
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 2, dim),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, h, edge_aug, valid_mask):
        valid_mask = valid_mask.bool()
        logits = self.edge_gate(edge_aug).squeeze(-1)
        pair_valid = valid_mask[:, :, None] & valid_mask[:, None, :]
        logits = logits.masked_fill(~pair_valid, -1e9)

        w = F.softmax(logits, dim=-1)
        w = w * pair_valid.float()
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        msg = torch.bmm(w, self.value(h))
        h = self.norm1(h + self.drop(self.out(msg)))
        h = self.norm2(h + self.drop(self.ff(h)))
        h = h * valid_mask.unsqueeze(-1).float()
        return h

class BranchStructureCenterShellGNNChem(nn.Module):
    def __init__(self):
        super().__init__()
        d = int(CONFIG.get('STRUCTURE_EDGE_DIM', 192))
        n_layers = int(CONFIG.get('STRUCTURE_EDGE_LAYERS', 3))
        drop = float(CONFIG.get('STRUCTURE_DROPOUT', 0.20))
        node_scalar_dim = int(CONFIG.get('STRUCTURE_NODE_SCALAR_DIM', 8))
        edge_aug_dim = int(CONFIG.get('STRUCTURE_EDGE_AUG_DIM', 22))
        shell_radii = [float(x) for x in CONFIG.get('STRUCTURE_SHELL_RADII', [4., 6., 8., 10.])]
        shell_attn_dim = int(CONFIG.get('STRUCTURE_SHELL_ATTN_DIM', 96))
        head_hidden = int(CONFIG.get('STRUCTURE_HEAD_HIDDEN', 384))
        quality_dim = int(CONFIG.get('STRUCTURE_QUALITY_DIM', 3))

        self.shell_radii = shell_radii
        self.distance_scale = float(CONFIG.get('STRUCTURE_DISTANCE_SCALE', 20.0))

        self.row_proj = nn.Sequential(
            nn.Linear(13 * MAX_LEN, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )
        self.node_proj = nn.Sequential(
            nn.LayerNorm(node_scalar_dim),
            nn.Linear(node_scalar_dim, d),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
        )
        self.input_norm = nn.LayerNorm(d)

        self.layers = nn.ModuleList([
            StructCenterAwareEdgeLayer(d, edge_aug_dim) for _ in range(n_layers)
        ])

        self.global_attn = StructSafeAttentionReadout(d, attn_dim=shell_attn_dim)
        self.shell_attn = nn.ModuleList([
            StructSafeAttentionReadout(d, attn_dim=shell_attn_dim)
            for _ in self.shell_radii
        ])

        self.chem = StructChemMLP(64)
        self.quality = nn.Sequential(
            nn.LayerNorm(quality_dim),
            nn.Linear(quality_dim, 24),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
        )

        fused_dim = d * (2 + len(self.shell_radii)) + 64 + 24
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(head_hidden, 128),
            nn.GELU(),
            nn.Dropout(drop * 0.5),
            nn.Linear(128, 1),
        )

    def _center_relative_tensors(self, pair: torch.Tensor, valid: torch.Tensor):
        B, C, L, _ = pair.shape
        device = pair.device
        scale = max(self.distance_scale, 1e-6)

        dca = pair[:, 0, STRUCT_CENTER, :] / scale
        dheavy = pair[:, 1, STRUCT_CENTER, :] / scale
        dsc = pair[:, 2, STRUCT_CENTER, :] / scale
        center_contact = pair[:, 3, STRUCT_CENTER, :]
        center_long = pair[:, 4, STRUCT_CENTER, :]

        pos = torch.arange(L, device=device, dtype=pair.dtype)
        rel = (pos - float(STRUCT_CENTER)) / float(max(STRUCT_CENTER, 1))
        rel = rel.unsqueeze(0).expand(B, -1)
        abs_rel = rel.abs()
        is_center = torch.zeros((B, L), device=device, dtype=pair.dtype)
        is_center[:, STRUCT_CENTER] = 1.0

        node_scalar = torch.stack([
            rel,
            abs_rel,
            is_center,
            dca,
            dheavy,
            dsc,
            center_contact,
            center_long,
        ], dim=-1)
        node_scalar = node_scalar * valid.unsqueeze(-1).float()

        edge = pair.permute(0, 2, 3, 1).contiguous()

        dca_i = dca[:, :, None].expand(-1, -1, L)
        dca_j = dca[:, None, :].expand(-1, L, -1)
        dheavy_i = dheavy[:, :, None].expand(-1, -1, L)
        dheavy_j = dheavy[:, None, :].expand(-1, L, -1)
        rel_i = rel[:, :, None].expand(-1, -1, L)
        rel_j = rel[:, None, :].expand(-1, L, -1)
        ci = is_center[:, :, None].expand(-1, -1, L)
        cj = is_center[:, None, :].expand(-1, L, -1)

        extra = torch.stack([
            dca_i,
            dca_j,
            (dca_i - dca_j).abs(),
            dheavy_i,
            dheavy_j,
            rel_i,
            rel_j,
            ci,
            cj,
        ], dim=-1)
        edge_aug = torch.cat([edge, extra], dim=-1)
        return node_scalar, edge_aug

    def forward(self, x):
        pair = x['pair'].float()
        chem = x['chem'].float()
        B, C, L, _ = pair.shape

        if 'valid_mask' in x:
            valid = x['valid_mask'].to(pair.device).bool()
        else:
            lengths = x['length'].long()
            valid = torch.arange(L, device=pair.device)[None, :] < lengths[:, None]

        node_scalar, edge_aug = self._center_relative_tensors(pair, valid)

        row = pair.permute(0, 2, 1, 3).contiguous().view(B, L, C * L)
        h = self.row_proj(row) + self.node_proj(node_scalar)
        h = self.input_norm(h)
        h = h * valid.unsqueeze(-1).float()

        for layer in self.layers:
            h = layer(h, edge_aug, valid)

        center = h[:, STRUCT_CENTER, :]
        global_repr = self.global_attn(h, valid)

        d_center = pair[:, 1, STRUCT_CENTER, :]
        not_center = torch.ones((1, L), device=pair.device, dtype=torch.bool)
        not_center[:, STRUCT_CENTER] = False

        shell_repr = []
        for radius, readout in zip(self.shell_radii, self.shell_attn):
            smask = valid & not_center & (d_center <= float(radius))
            shell_repr.append(readout(h, smask))

        if 'quality' in x:
            quality = x['quality'].to(pair.device).float()
        else:
            quality = torch.ones((B, int(CONFIG.get('STRUCTURE_QUALITY_DIM', 3))),
                                 device=pair.device, dtype=pair.dtype)
        q = self.quality(quality)

        z = torch.cat([
            center,
            global_repr,
            *shell_repr,
            self.chem(chem),
            q,
        ], dim=1)
        return self.head(z)



_BRANCH_SEED_OFFSET = {
    'physchem': 100_000,
    'esm_v2': 200_000,
    'esm_v3': 210_000,
    'esm_v4': 220_000,
    'esm_cqt': 230_000,
    'esm_sitecontrast': 240_000,
    'kmer': 300_000,
    'structure': 400_000,
    'seq_conformer': 500_000,
}

def branch_fold_seed(branch_name: str, repeat: int, fold_num: int) -> int:
    if branch_name in _BRANCH_SEED_OFFSET:
        off = _BRANCH_SEED_OFFSET[branch_name]
    else:
        off = 600_000 + int(hashlib.sha256(str(branch_name).encode('utf-8')).hexdigest()[:8], 16) % 100_000
    return (
        int(CONFIG.get('RANDOM_SEED', 3407))
        + int(off)
        + int(repeat) * 10_007
        + int(fold_num) * 1_009
    )

def branch_epochs(branch_name: str) -> int:
    if branch_name == 'structure':
        return int(CONFIG.get('EPOCHS_STRUCTURE', CONFIG.get('EPOCHS', 128)))
    return int(CONFIG.get('EPOCHS', 128))



@dataclass
class TrainResult:
    oof_prob: np.ndarray
    test_prob: np.ndarray
    best_states: List[Dict]
    fold_metrics: List[Dict]
    aligned_oof_prob: Optional[np.ndarray] = None
    aligned_test_prob: Optional[np.ndarray] = None

CURRENT_REPEAT = 0

def build_branch_model(branch_name: str,
                       kmer_vocabs: Optional[Dict[int, KmerVocab]] = None) -> nn.Module:
    device = CONFIG['DEVICE']

    if branch_name == 'physchem':
        return BranchPhyschem(seq_feat_dim=SEQ_FEAT_DIM, seq_hidden=128).to(device)

    if branch_name == 'physchem_pca':
        get_physchem_pca_lookup()
        return BranchPhyschem(
            seq_feat_dim=int(CONFIG.get('PHYSCHEM_PCA_DIM', 14)),
            seq_hidden=128,
        ).to(device)

    with h5py.File(CONFIG['ESM_H5'], 'r') as f:
        esm_dim = int(f['window_emb'].shape[-1])

    if branch_name == 'esm_cqt':
        return BranchESMCenterQueryTransformer(
            D_in=esm_dim,
            hidden=CONFIG['HIDDEN'],
            dropout=float(CONFIG.get('ESM_CQT_DROPOUT', 0.15)),
        ).to(device)

    if branch_name == 'esm_sitecontrast':
        return BranchESMSiteContrast(
            D_in=esm_dim,
            hidden=CONFIG['HIDDEN'],
            dropout=float(CONFIG.get('ESM_SC_DROPOUT', 0.18)),
        ).to(device)

    if branch_name == 'esm_sitecontrast_v2':
        return BranchESMSiteContrastV2(
            D_in=esm_dim,
            hidden=CONFIG['HIDDEN'],
            dropout=float(CONFIG.get('ESM_SC2_DROPOUT', 0.16)),
        ).to(device)

    if branch_name == 'kmer':
        if kmer_vocabs is None:
            raise RuntimeError("branch='kmer' 需要传入 kmer_vocabs")
        vocab_sizes = {k: v.size for k, v in kmer_vocabs.items()}
        return BranchKmer(
            vocab_sizes,
            embed_dim=CONFIG['KMER_EMBED'],
            ksize=5,
            dropout=CONFIG['DROPOUT_KMER'],
        ).to(device)

    if branch_name == 'structure':
        return BranchStructureCenterShellGNNChem().to(device)

    raise ValueError(f'Final V12 does not contain branch={branch_name}')

def build_branch_dataset(branch_name: str,
                         samples: List[Sample],
                         stats: Optional[Dict[str, np.ndarray]] = None,
                         uid2idx: Optional[dict] = None,
                         kmer_vocabs: Optional[Dict[int, KmerVocab]] = None) -> PalmDataset:
    if branch_name == 'physchem':
        return PalmDataset(samples, 'physchem')
    if branch_name == 'physchem_pca':
        return PalmDataset(samples, 'physchem_pca')
    if is_esm_branch(branch_name):
        main_uidmap = uid2idx if uid2idx is not None else build_uid2idx(CONFIG['ESM_H5'])
        return PalmDataset(
            samples,
            'esm',
            esm_h5=CONFIG['ESM_H5'],
            norm_stats=stats or {},
            uid2idx=main_uidmap,
        )
    if branch_name == 'kmer':
        return PalmDataset(samples, 'kmer', kmer_vocabs=kmer_vocabs)
    if branch_name == 'structure':
        return PalmDataset(samples, 'structure', norm_stats=stats or {})
    raise ValueError(f'Final V12 does not contain branch={branch_name}')

def forward_branch_batch(model: nn.Module,
                         branch_name: str,
                         batch,
                         device: str) -> torch.Tensor:
    if branch_name in {'physchem', 'physchem_pca', 'kmer', 'structure'}:
        xb = {
            k: v.to(device, non_blocking=CONFIG['NON_BLOCKING'])
            for k, v in batch['x'].items()
        }
        return model(xb)

    if is_esm_branch(branch_name):
        xb = batch['x'].to(
            device,
            non_blocking=CONFIG['NON_BLOCKING'],
        ).float()
        return model(xb)

    raise ValueError(f'Final V12 does not contain branch={branch_name}')

def build_scheduler(optimizer):
    if not CONFIG.get('USE_PLATEAU_SCHED', False):
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        min_lr=float(CONFIG.get('MIN_LR', 1e-6)),
    )

def v12_hard_pair_rank_loss(logit: torch.Tensor, y: torch.Tensor, max_pairs: int):
    pos = logit[y.view(-1) > 0.5].view(-1)
    neg = logit[y.view(-1) <= 0.5].view(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    frac = float(CONFIG.get('V12_HARD_PAIR_FRACTION', 0.50))
    frac = min(max(frac, 0.05), 1.0)
    nmax = min(pos.numel(), neg.numel(), int(max_pairs))
    n = max(1, int(math.ceil(nmax * frac)))
    hp = torch.sort(pos, descending=False).values[:n]
    hn = torch.sort(neg, descending=True).values[:n]
    return F.softplus(-(hp - hn)).mean()

def train_one_branch(branch_name: str,
                     train_samples: List[Sample],
                     test_samples: List[Sample],
                     folds: List[Tuple[List[Sample], List[Sample]]],
                     kmer_vocabs: Optional[Dict[int, KmerVocab]]=None) -> TrainResult:
    device = CONFIG['DEVICE']
    B = branch_batch_size(branch_name)
    E = branch_epochs(branch_name)
    lr = branch_lr(branch_name)
    wd = CONFIG['WEIGHT_DECAY']

    uid2idx = build_uid2idx(CONFIG['ESM_H5']) if is_esm_branch(branch_name) else None

    N_train = len(train_samples)
    oof_prob = np.zeros(N_train, dtype=np.float32)
    test_prob_folds: List[np.ndarray] = []
    best_states: List[Dict] = []
    fold_metrics: List[Dict] = []

    all_train_idx = {id(s): i for i, s in enumerate(train_samples)}
    overall_best_metric = -float('inf')
    overall_best_state = None
    metric_name = CONFIG['PRIMARY_METRIC']
    metric_key = 'best_val_' + metric_name

    for fold_id, (tr_list, va_list) in enumerate(folds):
        fold_num = fold_id + 1
        train_seed = branch_fold_seed(branch_name, CURRENT_REPEAT, fold_num)
        set_global_seed(train_seed)
        print(f"\n==== [{branch_name}] Fold {fold_num}/{len(folds)} ==== seed={train_seed}")

        paths = _fold_paths(branch_name, CURRENT_REPEAT, fold_num)
        signature = _resume_signature(
            branch_name, CURRENT_REPEAT, fold_num, final=False
        )

        va_global_idx = [all_train_idx[id(s)] for s in va_list]



        if CONFIG.get('RESUME', True) and os.path.exists(paths['done']) and os.path.exists(paths['best']):
            try:
                done = np.load(paths['done'], allow_pickle=False)
                done_sig = str(done['signature'].item())
                if done_sig == signature:
                    pv = done['val_prob'].astype(np.float32)
                    pt = done['test_prob'].astype(np.float32)
                    best_metric = float(done['best_metric'].item())
                    best_epoch = int(done['best_epoch'].item()) if 'best_epoch' in done.files else -1

                    if len(pv) != len(va_list) or len(pt) != len(test_samples):
                        raise RuntimeError("DONE缓存长度与当前fold不一致")

                    oof_prob[va_global_idx] = pv
                    test_prob_folds.append(pt)

                    best_ckpt = safe_torch_load(paths['best'], map_location='cpu')
                    best_state = best_ckpt['state_dict']
                    if best_epoch <= 0:
                        best_epoch = int(best_ckpt.get('best_epoch', -1))
                    best_states.append(best_state)

                    fold_metrics.append({
                        'fold': fold_num,
                        metric_key: best_metric,
                        'best_epoch': int(best_epoch),
                    })

                    if best_metric > overall_best_metric:
                        overall_best_metric = best_metric
                        overall_best_state = copy.deepcopy(best_state)

                    print(
                        f"[Resume] Fold {fold_num} 已完成，直接跳过训练；"
                        f"{metric_name}={best_metric:.5f}"
                    )
                    continue
            except Exception as e:
                print(f"[Resume][WARN] DONE缓存读取失败，将重新/续训该fold: {e}")



        fold_seed = (
            int(CONFIG.get('NEG_SAMPLE_SEED', CONFIG['CV_SEED']))
            + CURRENT_REPEAT * 10007
            + fold_id * 97
        )
        tr_core = make_fixed_ratio_by_pos(
            tr_list,
            branch_neg_ratio(branch_name),
            seed=fold_seed,
            branch_name=branch_name,
            repeat_id=CURRENT_REPEAT,
            fold_id=fold_num,
        )

        stats: Dict[str, np.ndarray] = {}
        if is_esm_branch(branch_name):
            mean, std = esm_channel_stats(CONFIG['ESM_H5'], tr_list, uid2idx)
            stats['esm_mean'] = mean
            stats['esm_std'] = std

        if branch_name == 'structure':
            cmean, cstd = structure_chemistry_stats(tr_list)
            stats['struct_chem_mean'] = cmean
            stats['struct_chem_std'] = cstd

        fold_kvoc = kmer_vocabs
        if branch_name == 'kmer':
            if kmer_vocabs is None:
                raise RuntimeError('kmer 分支缺少全局词表')
            fold_kvoc = build_active_kmer_views(
                kmer_vocabs,
                tr_core,
                min_count=int(CONFIG.get('KMER_FOLD_MIN_COUNT', 1)),
            )

        ds_tr = build_branch_dataset(
            branch_name, tr_core, stats=stats,
            uid2idx=uid2idx, kmer_vocabs=fold_kvoc
        )
        ds_va = build_branch_dataset(
            branch_name, va_list, stats=stats,
            uid2idx=uid2idx, kmer_vocabs=fold_kvoc
        )
        ds_te = build_branch_dataset(
            branch_name, test_samples, stats=stats,
            uid2idx=uid2idx, kmer_vocabs=fold_kvoc
        )

        kw = loader_kwargs(branch_name)
        sampler_tr = None
        if sampler_tr is not None:
            dl_tr = DataLoader(ds_tr, batch_size=B, sampler=sampler_tr, **kw)
            if fold_id == 0:
                print('[Cys-deconfound] neural sampler enabled: label × neighboring-Cys-bin tempered balancing')
        else:
            dl_tr = DataLoader(ds_tr, batch_size=B, **{**kw, 'shuffle': True})
        dl_va = DataLoader(ds_va, batch_size=B, **{**kw, 'shuffle': False})
        dl_te = DataLoader(ds_te, batch_size=B, **{**kw, 'shuffle': False})

        model = build_branch_model(branch_name, kmer_vocabs)
        wd_eff = branch_weight_decay(branch_name)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd_eff
        )
        scheduler = build_scheduler(optimizer)
        criterion = nn.BCEWithLogitsLoss()
        scaler = build_grad_scaler()

        best_metric = -float('inf')
        best_state = None
        best_epoch = 0
        patience = branch_patience(branch_name)
        no_improve = 0
        start_epoch = 1



        resumed = _try_load_epoch_resume(
            paths['last'], signature,
            model, optimizer, scheduler, scaler, device
        )
        if resumed is not None:
            start_epoch = int(resumed['epoch']) + 1
            best_metric = float(resumed.get('best_metric', -float('inf')))
            no_improve = int(resumed.get('no_improve', 0))

            if os.path.exists(paths['best']):
                best_ckpt = safe_torch_load(paths['best'], map_location='cpu')
                if best_ckpt.get('signature', signature) == signature:
                    best_state = best_ckpt['state_dict']
                    best_epoch = int(best_ckpt.get('best_epoch', best_epoch))

        for epoch in range(start_epoch, E + 1):
            model.train()
            running_loss = 0.0
            n_seen = 0

            for batch in dl_tr:
                optimizer.zero_grad(set_to_none=True)
                y = batch['y'].to(
                    device,
                    non_blocking=CONFIG['NON_BLOCKING']
                ).float().unsqueeze(-1)

                with get_autocast_context():
                    logit = forward_branch_batch(
                        model, branch_name, batch, device
                    )
                    loss = criterion(logit, smooth_binary_targets(y))
                    rank_alpha, rank_warmup, max_pairs = branch_rank_loss_cfg(branch_name)
                    if rank_alpha > 0 and epoch > rank_warmup:
                        if bool(CONFIG.get('V12_HARD_PAIR_RANK', True)):
                            rank_loss = v12_hard_pair_rank_loss(logit, y, max_pairs)
                        else:
                            pos_logits = logit[y.view(-1) > 0.5].view(-1)
                            neg_logits = logit[y.view(-1) <= 0.5].view(-1)
                            rank_loss = None
                            if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                                n_pair = min(max_pairs, max(pos_logits.numel(), neg_logits.numel()))
                                ip = torch.randint(pos_logits.numel(), (n_pair,), device=logit.device)
                                ineg = torch.randint(neg_logits.numel(), (n_pair,), device=logit.device)
                                rank_loss = F.softplus(-(pos_logits[ip] - neg_logits[ineg])).mean()
                        if rank_loss is not None:
                            loss = (1.0 - rank_alpha) * loss + rank_alpha * rank_loss

                backward_with_optional_amp(
                    loss, optimizer, scaler, model=model
                )
                running_loss += float(loss.detach().cpu()) * len(y)
                n_seen += len(y)

            model.eval()
            with torch.no_grad():
                probs, ys = [], []
                for batch in dl_va:
                    y = batch['y'].to(
                        device,
                        non_blocking=CONFIG['NON_BLOCKING']
                    ).float().unsqueeze(-1)
                    logit = forward_branch_batch(
                        model, branch_name, batch, device
                    )
                    probs.append(torch.sigmoid(logit).cpu().numpy().ravel())
                    ys.append(y.cpu().numpy().ravel())

            yv = np.concatenate(ys)
            pv_epoch = np.concatenate(probs)
            met = calc_metrics(yv, pv_epoch)
            cur = float(met[metric_name])

            if scheduler is not None:
                scheduler.step(cur)

            lr_now = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:03d}: "
                f"loss={running_loss/max(1,n_seen):.5f} "
                f"val_AUROC={met['auc']:.4f} "
                f"val_AUPRC={met['auprc']:.4f} "
                f"val_MCC={met['mcc']:.4f} "
                f"lr={lr_now:.2e}"
            )

            if cur > best_metric + 1e-8:
                best_metric = cur
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0

                _atomic_torch_save({
                    'state_dict': best_state,
                    'branch': branch_name,
                    'fold': fold_num,
                    'repeat': CURRENT_REPEAT,
                    'metric_name': metric_name,
                    'metric': float(best_metric),
                    'best_epoch': int(best_epoch),
                    'signature': signature,
                    'config': CONFIG,
                }, paths['best'])
            else:
                no_improve += 1

            _save_epoch_resume(
                paths['last'], signature, epoch,
                model, optimizer, scheduler, scaler,
                best_metric, no_improve
            )

            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        if best_state is None and os.path.exists(paths['best']):
            best_ckpt = safe_torch_load(paths['best'], map_location='cpu')
            best_state = best_ckpt['state_dict']
            best_metric = float(best_ckpt.get('metric', best_metric))
            best_epoch = int(best_ckpt.get('best_epoch', best_epoch))

        if best_state is None:
            raise RuntimeError(
                f"{branch_name} Fold {fold_num}: 没有best_state；"
                "如果你删除过best.pth，请同时删除对应last_resume.pth后重跑。"
            )

        if best_metric > overall_best_metric:
            overall_best_metric = best_metric
            overall_best_state = copy.deepcopy(best_state)

        best_states.append(best_state)



        model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            probs = []
            for batch in dl_va:
                logit = forward_branch_batch(
                    model, branch_name, batch, device
                )
                probs.append(torch.sigmoid(logit).cpu().numpy().ravel())
            pv = np.concatenate(probs).astype(np.float32)

        oof_prob[va_global_idx] = pv

        with torch.no_grad():
            probs = []
            for batch in dl_te:
                logit = forward_branch_batch(
                    model, branch_name, batch, device
                )
                probs.append(torch.sigmoid(logit).cpu().numpy().ravel())
            pt = np.concatenate(probs).astype(np.float32)

        test_prob_folds.append(pt)
        fold_metrics.append({
            'fold': fold_num,
            metric_key: float(best_metric),
            'best_epoch': int(best_epoch),
        })

        _atomic_npz_save(
            paths['done'],
            signature=np.array(signature),
            val_prob=pv,
            test_prob=pt,
            best_metric=np.array(best_metric, dtype=np.float64),
            best_epoch=np.array(best_epoch, dtype=np.int32),
        )
        print(f"[Resume] Fold {fold_num} DONE -> {paths['done']}")

        if not CONFIG.get('KEEP_LAST_RESUME_AFTER_DONE', True):
            try:
                os.remove(paths['last'])
            except FileNotFoundError:
                pass

        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if overall_best_state is not None:
        save_dir = os.path.join(
            CONFIG['MODEL_DIR'],
            f"repeat{CURRENT_REPEAT}",
            branch_name
        )
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(
            save_dir,
            f"{branch_name}_rep{CURRENT_REPEAT}_BESTOVERALL.pth"
        )
        _atomic_torch_save({
            'state_dict': overall_best_state,
            'branch': branch_name,
            'repeat': CURRENT_REPEAT,
            'metric_name': metric_name,
            'metric': overall_best_metric,
            'config': CONFIG,
        }, best_path)

    if len(test_prob_folds) != len(folds):
        raise RuntimeError(
            f"{branch_name}: 只有 {len(test_prob_folds)}/{len(folds)} 个fold test预测。"
        )

    test_prob = np.mean(np.stack(test_prob_folds, axis=0), axis=0)

    with open(
        os.path.join(CONFIG['OUT_DIR'], f'{branch_name}_fold_metrics.json'),
        'w'
    ) as f:
        json.dump(fold_metrics, f, indent=2)

    aligned_oof, aligned_test, align_params = align_fold_predictions(
        oof_prob, test_prob_folds, train_samples, folds
    )
    if align_params:
        align_path = os.path.join(CONFIG['OUT_DIR'], f'{branch_name}_fold_alignment.json')
        with open(align_path, 'w', encoding='utf-8') as f:
            json.dump(align_params, f, indent=2, ensure_ascii=False)

    return TrainResult(
        oof_prob, test_prob, best_states, fold_metrics,
        aligned_oof_prob=aligned_oof,
        aligned_test_prob=aligned_test,
    )



def _prob_to_logit_np(p: np.ndarray,
                      eps: Optional[float] = None) -> np.ndarray:
    eps = float(
        CONFIG.get('META_LOGIT_CLIP', 1e-4)
        if eps is None else eps
    )
    a = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(a / (1.0 - a)).astype(np.float32)

def _make_meta_splits(y: np.ndarray, groups: np.ndarray,
                      n_splits: Optional[int] = None,
                      seed: Optional[int] = None):
    if StratifiedGroupKFold is None:
        raise RuntimeError('Fusion v3 requires StratifiedGroupKFold.')
    n_splits = int(n_splits or CONFIG.get('META_N_FOLDS', 5))
    seed = int(seed if seed is not None else CONFIG.get('META_SEED', 9527))
    idx = np.arange(len(y))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    out = []
    for tr, va in splitter.split(idx, y, groups):
        if set(groups[tr].tolist()) & set(groups[va].tolist()):
            raise RuntimeError('Meta-CV protein leakage detected.')
        out.append((tr, va))
    return out

def make_interaction_features(prob_mat: np.ndarray,
                              interaction: bool = True,
                              logit_clip: Optional[float] = None) -> np.ndarray:
    z = _prob_to_logit_np(prob_mat, eps=logit_clip).astype(np.float64)
    return _append_pairwise_interactions(z, interaction=interaction)


def _append_pairwise_interactions(X: np.ndarray,
                                  interaction: bool = True) -> np.ndarray:

    X = np.asarray(X, dtype=np.float64)
    feats = [X]
    if interaction:
        ints = []
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                ints.append((X[:, i] * X[:, j])[:, None])
        if ints:
            feats.append(np.concatenate(ints, axis=1))
    return np.concatenate(feats, axis=1).astype(np.float32)


def _validate_base_probability_matrix(prob_mat: np.ndarray) -> np.ndarray:
    p = np.asarray(prob_mat, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != len(FINAL_EXPERTS):
        raise ValueError(
            f'Expected base probability matrix [N,{len(FINAL_EXPERTS)}], got {p.shape}'
        )
    return p


def _base_probability_columns(prob_mat: np.ndarray, names: List[str]) -> np.ndarray:
    p = _validate_base_probability_matrix(prob_mat)
    col = {name: i for i, name in enumerate(FINAL_EXPERTS)}
    missing = [name for name in names if name not in col]
    if missing:
        raise KeyError(f'Unknown base experts: {missing}')
    return p[:, [col[name] for name in names]]


def _fit_logistic_numpy(Xtr, ytr, Xva, C: float):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)
    clf = LogisticRegression(
        C=float(C), solver='lbfgs',
        max_iter=5000, random_state=CONFIG['RANDOM_SEED']
    )
    clf.fit(Xtr_s, ytr)
    p = clf.predict_proba(Xva_s)[:, 1].astype(np.float32)
    return p, scaler, clf




def _crossfit_logistic_features(X: np.ndarray, y: np.ndarray,
                                groups: np.ndarray, Xt: np.ndarray,
                                C_grid: List[float], seed: int,
                                method: str, feature_recipe: str):

    splits = _make_meta_splits(y, groups, seed=seed)
    C_grid = [float(c) for c in C_grid]
    if not C_grid:
        raise ValueError(f'{method}: empty C grid')

    best = None
    for C in C_grid:
        p_oof = np.zeros(len(y), dtype=np.float32)
        for tr, va in splits:
            p, _, _ = _fit_logistic_numpy(X[tr], y[tr], X[va], C)
            p_oof[va] = p
        auc = float(roc_auc_score(y, p_oof))
        if best is None or auc > best['auc']:
            best = {'C': C, 'auc': auc, 'oof_prob': p_oof.copy()}

    p_test, scaler, clf = _fit_logistic_numpy(X, y, Xt, best['C'])
    params = {
        'method': method,
        'feature_recipe': feature_recipe,
        'C': float(best['C']),
        'meta_oof_auc': float(best['auc']),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'coef': clf.coef_.ravel().tolist(),
        'intercept': clf.intercept_.ravel().tolist(),
        'n_features': int(X.shape[1]),
    }
    return {'oof_prob': best['oof_prob'], 'test_prob': p_test, 'params': params}


def _crossfit_fixed_group_fusion(prob_oof: np.ndarray, y: np.ndarray,
                                 groups: np.ndarray, prob_test: np.ndarray,
                                 expert_names: List[str], group_name: str,
                                 seed: int) -> Dict:







    clip = float(CONFIG.get('GROUP_FUSION_LOGIT_CLIP', 1e-5))
    C = float(CONFIG.get('GROUP_FUSION_C', 0.1))

    p_oof_group = _base_probability_columns(prob_oof, expert_names)
    p_test_group = _base_probability_columns(prob_test, expert_names)
    X = _prob_to_logit_np(p_oof_group, eps=clip).astype(np.float64)
    Xt = _prob_to_logit_np(p_test_group, eps=clip).astype(np.float64)

    splits = _make_meta_splits(y, groups, seed=seed)
    fused_oof = np.zeros(len(y), dtype=np.float32)
    for tr, va in splits:
        p, _, _ = _fit_logistic_numpy(X[tr], y[tr], X[va], C)
        fused_oof[va] = p

    fused_test, scaler, clf = _fit_logistic_numpy(X, y, Xt, C)
    params = {
        'method': 'fixed_group_logit_lr_cf',
        'group_name': str(group_name),
        'expert_names': list(expert_names),
        'expert_order': list(FINAL_EXPERTS),
        'C': C,
        'logit_clip': clip,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'coef': clf.coef_.ravel().tolist(),
        'intercept': clf.intercept_.ravel().tolist(),
        'n_features': int(X.shape[1]),
        'oof_auc': float(roc_auc_score(y, fused_oof)),
        'oof_auprc': float(average_precision_score(y, fused_oof)),
        'seed': int(seed),
    }
    return {
        'oof_prob': fused_oof,
        'test_prob': fused_test.astype(np.float32),
        'params': params,
    }


def fit_first_stage_modality_fusions(prob_oof: np.ndarray, y: np.ndarray,
                                     groups: np.ndarray, prob_test: np.ndarray,
                                     seed: int) -> Dict:

    _validate_base_probability_matrix(prob_oof)
    _validate_base_probability_matrix(prob_test)

    phys = _crossfit_fixed_group_fusion(
        prob_oof, y, groups, prob_test,
        expert_names=PHYS_EXPERTS,
        group_name='physchem_fused',
        seed=int(seed) + 101,
    )
    esm = _crossfit_fixed_group_fusion(
        prob_oof, y, groups, prob_test,
        expert_names=ESM_EXPERTS,
        group_name='esm_fused',
        seed=int(seed) + 202,
    )

    col = {name: i for i, name in enumerate(FINAL_EXPERTS)}
    modality_oof = np.stack([
        phys['oof_prob'],
        esm['oof_prob'],
        np.asarray(prob_oof[:, col['kmer']], dtype=np.float32),
        np.asarray(prob_oof[:, col['structure']], dtype=np.float32),
    ], axis=1).astype(np.float32)
    modality_test = np.stack([
        phys['test_prob'],
        esm['test_prob'],
        np.asarray(prob_test[:, col['kmer']], dtype=np.float32),
        np.asarray(prob_test[:, col['structure']], dtype=np.float32),
    ], axis=1).astype(np.float32)

    return {
        'oof_mat': modality_oof,
        'test_mat': modality_test,
        'params': {
            'expert_order': list(FINAL_EXPERTS),
            'modality_order': list(FUSED_MODALITIES),
            'physchem_fused': phys['params'],
            'esm_fused': esm['params'],
        },
        'metrics': {
            'physchem_fused': calc_metrics(y, phys['oof_prob']),
            'esm_fused': calc_metrics(y, esm['oof_prob']),
        },
    }


def crossfit_four_modality_meta(modality_oof: np.ndarray, y: np.ndarray,
                                groups: np.ndarray, modality_test: np.ndarray,
                                interaction: bool, method: str,
                                seed: int = 9527) -> Dict:
    if modality_oof.ndim != 2 or modality_oof.shape[1] != len(FUSED_MODALITIES):
        raise ValueError(
            f'Expected modality matrix [N,{len(FUSED_MODALITIES)}], got {modality_oof.shape}'
        )
    if modality_test.ndim != 2 or modality_test.shape[1] != len(FUSED_MODALITIES):
        raise ValueError(
            f'Expected test modality matrix [N,{len(FUSED_MODALITIES)}], got {modality_test.shape}'
        )

    logit_clip = float(CONFIG.get('META_LOGIT_CLIP', 1e-4))
    X = make_interaction_features(
        modality_oof, interaction=interaction, logit_clip=logit_clip,
    )
    Xt = make_interaction_features(
        modality_test, interaction=interaction, logit_clip=logit_clip,
    )
    out = _crossfit_logistic_features(
        X, y, groups, Xt,
        C_grid=CONFIG.get('META_C_GRID', [0.1, 1.0, 10.0]),
        seed=seed,
        method=method,
        feature_recipe=(
            'four_modality_logits_plus_all_pairwise_products'
            if interaction else
            'four_modality_logits_only'
        ),
    )
    out['params']['interaction'] = bool(interaction)
    out['params']['logit_clip'] = logit_clip
    out['params']['modality_order'] = list(FUSED_MODALITIES)
    return out


def fit_and_select_fusion(prob_oof: np.ndarray, y: np.ndarray,
                          groups: np.ndarray, prob_test: np.ndarray,
                          seed: int) -> Dict:












    first_stage = fit_first_stage_modality_fusions(
        prob_oof, y, groups, prob_test, seed=seed,
    )
    modality_oof = first_stage['oof_mat']
    modality_test = first_stage['test_mat']

    runners = {
        'four_modality_interaction_stack_cf': lambda: crossfit_four_modality_meta(
            modality_oof, y, groups, modality_test,
            interaction=True,
            method='four_modality_interaction_stack_cf',
            seed=seed,
        ),
        'four_modality_linear_stack_cf': lambda: crossfit_four_modality_meta(
            modality_oof, y, groups, modality_test,
            interaction=False,
            method='four_modality_linear_stack_cf',
            seed=seed,
        ),
    }
    names = list(CONFIG.get('FUSION_CANDIDATES', runners.keys()))
    if not names:
        raise ValueError('FUSION_CANDIDATES cannot be empty')

    results = {}
    for order, name in enumerate(names):
        if name not in runners:
            raise KeyError(f'Unknown fusion candidate: {name}')
        result = runners[name]()
        result['oof_metrics'] = calc_metrics(y, result['oof_prob'])
        result['selection_order'] = int(order)

        result['params'] = {
            'expert_order': list(FINAL_EXPERTS),
            'modality_order': list(FUSED_MODALITIES),
            'first_stage': copy.deepcopy(first_stage['params']),
            'second_stage': result['params'],
        }
        results[name] = result

    primary = str(CONFIG.get('FUSION_SELECTION_PRIMARY', 'auc'))
    secondary = str(CONFIG.get('FUSION_SELECTION_SECONDARY', 'auprc'))
    if primary not in ('auc', 'auprc') or secondary not in ('auc', 'auprc'):
        raise ValueError('Fusion selection metrics must be auc or auprc.')

    def selection_key(name: str):
        m = results[name]['oof_metrics']
        return (
            float(m[primary]),
            float(m[secondary]),
            -results[name]['selection_order'],
        )

    selected = max(names, key=selection_key)
    return {
        'selected_method': selected,
        'selected': results[selected],
        'candidates': results,
        'modality_oof': modality_oof,
        'modality_test': modality_test,
        'first_stage_metrics': first_stage['metrics'],
        'first_stage_params': first_stage['params'],
        'selection_rule': (
            f'max training grouped-OOF {primary}, then {secondary}, '
            'then declared candidate order'
        ),
        'selection_uses_test': False,
    }

def save_meta_fold_manifest(path: str, uids: List[str], groups: np.ndarray,
                            y: np.ndarray, shared_seed: int):
    fold_id = np.zeros(len(y), dtype=np.int32)
    for fi, (_, va) in enumerate(_make_meta_splits(y, groups, seed=int(shared_seed)), start=1):
        fold_id[va] = fi
    if (fold_id <= 0).any():
        raise RuntimeError('Meta-fold manifest has unassigned samples.')
    pd.DataFrame({
        'uniprotid': list(uids),
        'base_uniprot': np.asarray(groups, dtype=object),
        'label': np.asarray(y, dtype=int),
        'meta_fold': fold_id,
        'meta_seed': int(shared_seed),
    }).to_csv(path, index=False)

def save_split_manifest(path: str, all_samples: List[Sample],
                        train_samples: List[Sample], test_samples: List[Sample],
                        folds: Optional[List[Tuple[List[Sample], List[Sample]]]] = None):
    tr_ids = {id(s) for s in train_samples}
    te_ids = {id(s) for s in test_samples}
    fold_map = {}
    if folds is not None:
        for fi, (_, va) in enumerate(folds, start=1):
            for s in va:
                fold_map[id(s)] = fi
    rows = []
    for s in all_samples:
        sid = id(s)
        split = 'train' if sid in tr_ids else ('test' if sid in te_ids else 'unknown')
        rows.append({
            'uniprotid': s.uid,
            'base_uniprot': _base_uid(s.uid),
            'label': int(s.label),
            'split': split,
            'cv_fold': fold_map.get(sid, np.nan) if split == 'train' else np.nan,
        })
    pd.DataFrame(rows).to_csv(path, index=False)

def save_kmer_vocab_v3(kvoc: Dict[int, KmerVocab]):
    if not kvoc:
        return
    path = os.path.join(CONFIG['MODEL_DIR'], 'kmer_vocab.json')
    obj = {
        'meta': {
            'csv_path': CONFIG['CSV_PATH'],
            'test_ratio': CONFIG['TEST_RATIO'],
            'split_seed': CONFIG['SPLIT_SEED'],
            'kmers': CONFIG['KMERS'],
            'kmer_min_count': CONFIG.get('KMER_MIN_COUNT', {}),
            'architecture': str(CONFIG.get('KMER_VARIANT','v8_position_multiscale')),
        },
        'vocabs': {
            str(k): {
                'k': int(k),
                'max_size': kvoc[k].max_size,
                'min_count': int(getattr(kvoc[k], 'min_count', 1)),
                'itos': kvoc[k].itos,
            }
            for k in sorted(kvoc.keys())
        }
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)
    print('[Save] kmer vocab ->', path)




def aggregate_macro_fold_auc(repeat_results: List[Dict[str, TrainResult]], branch: str) -> Tuple[float, float]:
    vals = []
    for rr in repeat_results:
        for row in rr[branch].fold_metrics:
            if 'best_val_auc' in row:
                vals.append(float(row['best_val_auc']))
    if not vals:
        return float('nan'), float('nan')
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)



_HASH_PROJECTOR_CACHE: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}

_AA20 = list('ACDEFGHIKLMNPQRSTVWY')
_AA20_IDX = {a:i for i,a in enumerate(_AA20)}
_AA22 = _AA20 + ['X','*']
_AA22_IDX = {a:i for i,a in enumerate(_AA22)}





def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def _serialize_kmer_vocabs(kvoc: Dict[int, KmerVocab],
                            active_views: Optional[Dict[int, KmerVocab]] = None) -> Dict:
    out = {}
    for k in sorted(kvoc.keys()):
        base = kvoc[k]
        active = None
        if active_views is not None and k in active_views:
            active = active_views[k].allowed_tokens
        out[str(k)] = {
            'k': int(k),
            'max_size': base.max_size,
            'min_count': int(getattr(base, 'min_count', 1)),
            'itos': list(base.itos),
            'allowed_tokens': sorted(list(active)) if active is not None else None,
        }
    return out


def choose_final_epochs(repeat_results: List[Dict[str, TrainResult]]) -> Dict[str, int]:

    scale = float(CONFIG.get('FINAL_EPOCH_SCALE', 1.0))
    min_epochs = int(CONFIG.get('FINAL_MIN_EPOCHS', 4))
    out = {}

    for branch in FINAL_EXPERTS:
        vals = []
        for rr in repeat_results:
            for row in rr[branch].fold_metrics:
                ep = int(row.get('best_epoch', -1))
                if ep > 0:
                    vals.append(ep)

        if not vals:


            fallback = min(branch_epochs(branch), 32)
            print(
                f"[FINAL][WARN] {branch}: CV best_epoch unavailable; "
                f"fallback={fallback}. Fresh CV is recommended."
            )
            chosen = fallback
        else:
            chosen = int(round(float(np.median(vals)) * scale))

        chosen = max(min_epochs, min(chosen, branch_epochs(branch)))
        out[branch] = int(chosen)
        print(
            f"[FINAL epoch] {branch:20s} -> {chosen} "
            f"(CV best epochs={vals})"
        )

    return out


def _predict_branch_model(model: nn.Module,
                          branch_name: str,
                          samples: List[Sample],
                          stats: Optional[Dict[str, np.ndarray]] = None,
                          uid2idx: Optional[dict] = None,
                          kmer_vocabs: Optional[Dict[int, KmerVocab]] = None) -> np.ndarray:
    ds = build_branch_dataset(
        branch_name,
        samples,
        stats=stats or {},
        uid2idx=uid2idx,
        kmer_vocabs=kmer_vocabs,
    )
    dl = DataLoader(
        ds,
        batch_size=branch_batch_size(branch_name),
        shuffle=False,
        **loader_kwargs(branch_name),
    )

    model.eval()
    probs = []
    with torch.no_grad():
        for batch in dl:
            logit = forward_branch_batch(
                model,
                branch_name,
                batch,
                CONFIG['DEVICE'],
            )
            probs.append(torch.sigmoid(logit).detach().cpu().numpy().ravel())

    if not probs:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(probs).astype(np.float32)


def _apply_saved_logistic_features(X: np.ndarray, params: Dict) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    expected = int(params.get('n_features', X.shape[1]))
    if X.ndim != 2 or X.shape[1] != expected:
        raise RuntimeError(
            f'Fusion feature mismatch: checkpoint={expected}, runtime={X.shape}'
        )

    mean = np.asarray(params['scaler_mean'], dtype=np.float64)
    scale = np.asarray(params['scaler_scale'], dtype=np.float64)
    scale[np.abs(scale) < 1e-12] = 1.0
    coef = np.asarray(params['coef'], dtype=np.float64)
    intercept = float(np.asarray(params['intercept'], dtype=np.float64).ravel()[0])
    Xs = (X - mean[None, :]) / scale[None, :]
    return _np_sigmoid(Xs @ coef + intercept).astype(np.float32)


def _apply_saved_group_fusion(prob_group: np.ndarray, params: Dict) -> np.ndarray:
    p = np.asarray(prob_group, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f'Group probability input must be 2D, got {p.shape}')
    expected = int(params.get('n_features', p.shape[1]))
    if p.shape[1] != expected:
        raise RuntimeError(
            f"{params.get('group_name', 'group')}: expected {expected} experts, got {p.shape[1]}"
        )
    X = _prob_to_logit_np(
        p,
        eps=float(params.get('logit_clip', CONFIG.get('GROUP_FUSION_LOGIT_CLIP', 1e-5))),
    ).astype(np.float64)
    return _apply_saved_logistic_features(X, params)


def apply_saved_first_stage(prob_mat: np.ndarray, params: Dict) -> np.ndarray:

    p = _validate_base_probability_matrix(prob_mat)
    saved_order = params.get('expert_order')
    if saved_order is not None and list(saved_order) != list(FINAL_EXPERTS):
        raise RuntimeError(
            f'Expert order mismatch: checkpoint={saved_order}, runtime={FINAL_EXPERTS}'
        )
    saved_modalities = params.get('modality_order')
    if saved_modalities is not None and list(saved_modalities) != list(FUSED_MODALITIES):
        raise RuntimeError(
            f'Modality order mismatch: checkpoint={saved_modalities}, runtime={FUSED_MODALITIES}'
        )

    col = {name: i for i, name in enumerate(FINAL_EXPERTS)}
    first = params.get('first_stage', params)

    phys_cfg = first['physchem_fused']
    esm_cfg = first['esm_fused']
    phys_names = list(phys_cfg.get('expert_names', PHYS_EXPERTS))
    esm_names = list(esm_cfg.get('expert_names', ESM_EXPERTS))

    phys = _apply_saved_group_fusion(
        p[:, [col[name] for name in phys_names]], phys_cfg,
    )
    esm = _apply_saved_group_fusion(
        p[:, [col[name] for name in esm_names]], esm_cfg,
    )
    return np.stack([
        phys,
        esm,
        p[:, col['kmer']].astype(np.float32),
        p[:, col['structure']].astype(np.float32),
    ], axis=1).astype(np.float32)


def apply_saved_fusion(prob_mat: np.ndarray, method: str,
                       params: Dict) -> np.ndarray:

    if method not in {
        'four_modality_interaction_stack_cf',
        'four_modality_linear_stack_cf',
    }:
        raise KeyError(f'Unsupported saved fusion method: {method}')

    modality_mat = apply_saved_first_stage(prob_mat, params)
    second = params['second_stage']
    expected_order = second.get('modality_order')
    if expected_order is not None and list(expected_order) != list(FUSED_MODALITIES):
        raise RuntimeError(
            f'Second-stage modality order mismatch: checkpoint={expected_order}, '
            f'runtime={FUSED_MODALITIES}'
        )
    X = make_interaction_features(
        modality_mat,
        interaction=bool(second.get('interaction', method == 'four_modality_interaction_stack_cf')),
        logit_clip=second.get('logit_clip'),
    ).astype(np.float64)
    return _apply_saved_logistic_features(X, second)


def apply_saved_interaction_meta(prob_mat: np.ndarray, params: Dict) -> np.ndarray:

    return apply_saved_fusion(
        prob_mat,
        method='four_modality_interaction_stack_cf',
        params=params,
    )


def train_single_final_branch(branch_name: str,
                              train_samples: List[Sample],
                              eval_samples: List[Sample],
                              epochs: int,
                              kmer_vocabs: Optional[Dict[int, KmerVocab]] = None):






    device = CONFIG['DEVICE']
    seed = (
        int(CONFIG.get('FINAL_SEED', 20260828))
        + int(_BRANCH_SEED_OFFSET.get(branch_name, 600000))
    )
    set_global_seed(seed)

    print(
        f"\n========== [FINAL EXPERT] {branch_name} "
        f"epochs={epochs} seed={seed} =========="
    )

    train_core = make_fixed_ratio_by_pos(
        train_samples,
        branch_neg_ratio(branch_name),
        seed=seed + 17,
        branch_name=branch_name,
        repeat_id=0,
        fold_id=0,
    )

    uid2idx = build_uid2idx(CONFIG['ESM_H5']) if is_esm_branch(branch_name) else None

    stats: Dict[str, np.ndarray] = {}
    if is_esm_branch(branch_name):
        mean, std = esm_channel_stats(CONFIG['ESM_H5'], train_samples, uid2idx)
        stats['esm_mean'] = mean
        stats['esm_std'] = std

    if branch_name == 'structure':
        cmean, cstd = structure_chemistry_stats(train_samples)
        stats['struct_chem_mean'] = cmean
        stats['struct_chem_std'] = cstd

    active_kvoc = kmer_vocabs
    if branch_name == 'kmer':
        if kmer_vocabs is None:
            raise RuntimeError('final kmer expert 缺少 kmer vocab')
        active_kvoc = build_active_kmer_views(
            kmer_vocabs,
            train_core,
            min_count=int(CONFIG.get('KMER_FOLD_MIN_COUNT', 1)),
        )

    ds_tr = build_branch_dataset(
        branch_name,
        train_core,
        stats=stats,
        uid2idx=uid2idx,
        kmer_vocabs=active_kvoc,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=branch_batch_size(branch_name),
        shuffle=True,
        **loader_kwargs(branch_name),
    )

    model = build_branch_model(
        branch_name,
        kmer_vocabs=kmer_vocabs if branch_name == 'kmer' else None,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=branch_lr(branch_name),
        weight_decay=branch_weight_decay(branch_name),
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = build_grad_scaler()




    for epoch in range(1, int(epochs) + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for batch in dl_tr:
            optimizer.zero_grad(set_to_none=True)
            y = batch['y'].to(
                device,
                non_blocking=CONFIG['NON_BLOCKING'],
            ).float().unsqueeze(-1)

            with get_autocast_context():
                logit = forward_branch_batch(model, branch_name, batch, device)
                loss = criterion(logit, smooth_binary_targets(y))

                rank_alpha, rank_warmup, max_pairs = branch_rank_loss_cfg(branch_name)
                if rank_alpha > 0 and epoch > rank_warmup:
                    rank_loss = v12_hard_pair_rank_loss(logit, y, max_pairs)
                    if rank_loss is not None:
                        loss = (1.0 - rank_alpha) * loss + rank_alpha * rank_loss

            backward_with_optional_amp(
                loss,
                optimizer,
                scaler,
                model=model,
            )
            running_loss += float(loss.detach().cpu()) * len(y)
            n_seen += len(y)

        print(
            f"[FINAL {branch_name}] epoch {epoch:03d}/{epochs:03d} "
            f"loss={running_loss/max(1,n_seen):.6f}"
        )


    raw_train = _predict_branch_model(
        model,
        branch_name,
        train_samples,
        stats=stats,
        uid2idx=uid2idx,
        kmer_vocabs=active_kvoc,
    )
    raw_eval = _predict_branch_model(
        model,
        branch_name,
        eval_samples,
        stats=stats,
        uid2idx=uid2idx,
        kmer_vocabs=active_kvoc,
    )

    if bool(CONFIG.get('FINAL_SCORE_ALIGNMENT', True)):
        align = fit_robust_score_alignment(raw_train)
        aligned_train = apply_score_alignment(raw_train, align)
        aligned_eval = apply_score_alignment(raw_eval, align)
    else:
        align = {'center': 0.0, 'scale': 1.0}
        aligned_train = raw_train
        aligned_eval = raw_eval

    preprocess = {
        'epochs': int(epochs),
        'seed': int(seed),
        'score_alignment': {
            'center': float(align['center']),
            'scale': float(align['scale']),
        },
        'stats': {
            k: np.asarray(v, dtype=np.float32)
            for k, v in stats.items()
        },
    }

    if branch_name == 'kmer':
        preprocess['kmer_active_tokens'] = {
            str(k): sorted(list(active_kvoc[k].allowed_tokens or []))
            for k in sorted(active_kvoc.keys())
        }

    return {
        'state_dict': _cpu_state_dict(model),
        'preprocess': preprocess,
        'raw_train_prob': raw_train,
        'raw_eval_prob': raw_eval,
        'train_prob': aligned_train,
        'eval_prob': aligned_eval,
    }










def _final_expert_resume_dir() -> str:
    d = os.path.join(CONFIG['MODEL_DIR'], 'final_single_resume')
    os.makedirs(d, exist_ok=True)
    return d


def _final_expert_resume_path(branch_name: str) -> str:
    return os.path.join(_final_expert_resume_dir(), f'{branch_name}_DONE.pth')


def _final_expert_resume_signature(branch_name: str, epochs: int) -> str:
    base = _resume_signature(branch_name, repeat=0, fold=None, final=True)
    payload = {
        'base_signature': base,
        'branch': str(branch_name),
        'epochs': int(epochs),
        'resume_version': int(CONFIG.get('RESUME_VERSION', 0)),
        'final_seed': int(CONFIG.get('FINAL_SEED', 20260828)),
    }
    txt = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(txt.encode('utf-8')).hexdigest()


def _save_final_expert_resume(branch_name: str, epochs: int, result: Dict) -> str:
    path = _final_expert_resume_path(branch_name)
    obj = {
        'signature': _final_expert_resume_signature(branch_name, epochs),
        'branch': str(branch_name),
        'epochs': int(epochs),
        'result': result,
    }
    _atomic_torch_save(obj, path)
    print(f'[FINAL RESUME] saved {branch_name} -> {path}')
    return path


def _try_load_final_expert_resume(branch_name: str, epochs: int):
    if not bool(CONFIG.get('RESUME', True)):
        return None
    path = _final_expert_resume_path(branch_name)
    if not os.path.isfile(path):
        return None
    try:
        obj = safe_torch_load(path, map_location='cpu')
    except Exception as e:
        print(f'[FINAL RESUME][WARN] cannot read {path}: {e}')
        return None
    expected = _final_expert_resume_signature(branch_name, epochs)
    if obj.get('signature') != expected:
        print(f'[FINAL RESUME] stale cache ignored: {branch_name}')
        return None
    result = obj.get('result')
    required = {'state_dict', 'preprocess', 'train_prob', 'eval_prob'}
    if not isinstance(result, dict) or not required.issubset(result):
        print(f'[FINAL RESUME][WARN] incomplete cache ignored: {branch_name}')
        return None
    print(f'[FINAL RESUME] HIT {branch_name}; skip retraining -> {path}')
    return result

def save_single_final_checkpoint(final_expert_results: Dict[str, Dict],
                                 final_epochs: Dict[str, int],
                                 kvoc: Dict[int, KmerVocab],
                                 fusion_method: str,
                                 fusion_params: Dict,
                                 cv_summary: Dict,
                                 deployment_metrics: Dict):
    expert_states = {
        b: final_expert_results[b]['state_dict']
        for b in FINAL_EXPERTS
    }
    expert_preprocess = {
        b: final_expert_results[b]['preprocess']
        for b in FINAL_EXPERTS
    }


    path_keys = {
        'CSV_PATH', 'ESM_H5', 'PDB_DIR', 'AAINDEX1_PATH', 'AAINDEX_PCA_PATH',
        'OUT_DIR', 'MODEL_DIR', 'CACHE_DIR', 'STRUCTURE_CACHE_DIR', 'FINAL_MODEL_PATH',
    }
    deploy_config = {
        k: copy.deepcopy(v)
        for k, v in CONFIG.items()
        if k not in path_keys
    }

    obj = {
        'format': 'DeepPalmV14SingleCheckpoint',
        'format_version': 3,
        'model_name': 'DeepPalm V14 FINAL',
        'description': '7 base experts -> learned physchem/ESM group fusion -> 4 modality inputs -> final stack',
        'expert_order': list(FINAL_EXPERTS),
        'modality_order': list(FUSED_MODALITIES),
        'expert_state_dicts': expert_states,
        'expert_preprocess': expert_preprocess,
        'kmer_vocab': _serialize_kmer_vocabs(
            kvoc,
            active_views=None,
        ),
        'fusion': {
            'method': str(fusion_method),
            'params': copy.deepcopy(fusion_params),
        },
        'final_epochs': {k: int(v) for k, v in final_epochs.items()},
        'config': deploy_config,
        'cv_summary': cv_summary,
        'deployment_test_metrics': deployment_metrics,
    }

    path = CONFIG['FINAL_MODEL_PATH']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _atomic_torch_save(obj, path)
    print(f"[FINAL MODEL] one-file checkpoint -> {path}")
    return path




def print_path_config():
    print("=" * 78)
    print("[Paths] DeepPalm V14 REPRODUCIBLE")
    print("[Paths] CSV_PATH            =", CONFIG['CSV_PATH'])
    print("[Paths] ESM_H5              =", CONFIG['ESM_H5'])
    print("[Paths] PDB_DIR             =", CONFIG['PDB_DIR'])
    print("[Paths] AAINDEX1_PATH       =", CONFIG['AAINDEX1_PATH'])
    print("[Paths] AAINDEX_PCA_PATH    =", CONFIG['AAINDEX_PCA_PATH'])
    print("[Paths] OUT_DIR             =", CONFIG['OUT_DIR'])
    print("[Paths] MODEL_DIR           =", CONFIG['MODEL_DIR'])
    print("[Paths] CACHE_DIR           =", CONFIG['CACHE_DIR'])
    print("[Paths] STRUCTURE_CACHE_DIR =", CONFIG['STRUCTURE_CACHE_DIR'])
    print("[Paths] FINAL_MODEL_PATH     =", CONFIG['FINAL_MODEL_PATH'])
    print("=" * 78)
















def _deploy_sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _restore_kmer_vocabs_from_checkpoint(obj: Dict) -> Dict[int, KmerVocab]:
    out: Dict[int, KmerVocab] = {}
    for k_str, item in obj.items():
        k = int(item.get('k', k_str))
        vocab = KmerVocab(k, max_size=item.get('max_size'))
        vocab.min_count = int(item.get('min_count', 1))
        vocab.itos = list(item['itos'])
        vocab.stoi = {tok: i for i, tok in enumerate(vocab.itos)}
        allowed = item.get('allowed_tokens')
        vocab.allowed_tokens = set(allowed) if allowed is not None else None
        out[k] = vocab
    if not out:
        raise RuntimeError('checkpoint 中 kmer_vocab 为空')
    return out


def _active_kmer_vocabs_for_final(
    base_kvoc: Dict[int, KmerVocab],
    preprocess: Dict,
) -> Dict[int, KmerVocab]:
    active_cfg = preprocess.get('kmer_active_tokens', {}) or {}
    out: Dict[int, KmerVocab] = {}
    for k, base in base_kvoc.items():
        view = copy.copy(base)
        tokens = active_cfg.get(str(k), active_cfg.get(k))
        view.allowed_tokens = set(tokens) if tokens is not None else None
        out[k] = view
    return out


def _deploy_prepare_input(branch_name: str, batch, device: str):
    if branch_name in {'physchem', 'physchem_pca', 'kmer', 'structure'}:
        return {
            k: v.to(device, non_blocking=False)
            for k, v in batch['x'].items()
        }
    if is_esm_branch(branch_name):
        return batch['x'].to(device, non_blocking=False).float()
    raise ValueError(branch_name)


def _predict_script_module(
    scripted,
    branch_name: str,
    samples: List[Sample],
    preprocess: Dict,
    uid2idx: Optional[dict],
    base_kvoc: Optional[Dict[int, KmerVocab]],
) -> np.ndarray:
    stats = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in (preprocess.get('stats', {}) or {}).items()
    }
    active_kvoc = None
    if branch_name == 'kmer':
        if base_kvoc is None:
            raise RuntimeError('kmer deploy validation 缺少 base vocab')
        active_kvoc = _active_kmer_vocabs_for_final(base_kvoc, preprocess)

    ds = build_branch_dataset(
        branch_name,
        samples,
        stats=stats,
        uid2idx=uid2idx,
        kmer_vocabs=active_kvoc,
    )
    dl = DataLoader(
        ds,
        batch_size=branch_batch_size(branch_name),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = CONFIG['DEVICE']
    out = []
    scripted.eval()
    with torch.no_grad():
        for batch in dl:
            x = _deploy_prepare_input(branch_name, batch, device)
            logit = scripted(x)
            out.append(torch.sigmoid(logit).detach().cpu().numpy().ravel())
    if not out:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(out).astype(np.float32)


def _reference_table_for_deploy(
    reference_csv: str,
    samples: List[Sample],
) -> pd.DataFrame:
    if not reference_csv or not os.path.isfile(reference_csv):
        raise FileNotFoundError(
            '部署导出要求已有 single_model_test_predictions_FINAL.csv 用于严格一致性验证。\n'
            f'当前路径: {reference_csv}'
        )
    ref = pd.read_csv(reference_csv)
    if 'uniprotid' not in ref.columns or 'prob_selected_final' not in ref.columns:
        raise RuntimeError(
            'reference predictions 必须至少包含 uniprotid 和 prob_selected_final'
        )
    if ref['uniprotid'].astype(str).duplicated().any():
        raise RuntimeError('reference predictions 中 uniprotid 有重复，拒绝静默对齐')

    wanted = [str(s.uid) for s in samples]
    ref = ref.copy()
    ref['uniprotid'] = ref['uniprotid'].astype(str)
    ref = ref.set_index('uniprotid', drop=False)
    missing = [u for u in wanted if u not in ref.index]
    extra = [u for u in ref.index.tolist() if u not in set(wanted)]
    if missing or extra:
        raise RuntimeError(
            f'reference/test UID 不一致: missing={len(missing)}, extra={len(extra)}; '
            f'missing examples={missing[:10]}, extra examples={extra[:10]}'
        )
    return ref.loc[wanted].reset_index(drop=True)


def _assert_deploy_close(name: str, got: np.ndarray, expected: np.ndarray) -> Dict:
    got = np.asarray(got, dtype=np.float64).ravel()
    expected = np.asarray(expected, dtype=np.float64).ravel()
    if got.shape != expected.shape:
        raise RuntimeError(f'{name}: shape mismatch {got.shape} vs {expected.shape}')
    diff = np.abs(got - expected)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    pearson = (
        float(np.corrcoef(got, expected)[0, 1])
        if len(got) > 1 and np.std(got) > 0 and np.std(expected) > 0
        else 1.0
    )
    rec = {
        'name': name,
        'n': int(len(got)),
        'max_abs_diff': max_abs,
        'mean_abs_diff': mean_abs,
        'pearson': pearson,
    }
    print(
        f"[DEPLOY VERIFY] {name:28s} n={len(got):6d} "
        f"max={max_abs:.3e} mean={mean_abs:.3e} r={pearson:.12f}"
    )
    if max_abs > float(DEPLOY_VERIFY_MAX_ABS):
        raise RuntimeError(
            f'{name}: max_abs_diff={max_abs:.3e} > {DEPLOY_VERIFY_MAX_ABS:.3e}'
        )
    if mean_abs > float(DEPLOY_VERIFY_MEAN_ABS):
        raise RuntimeError(
            f'{name}: mean_abs_diff={mean_abs:.3e} > {DEPLOY_VERIFY_MEAN_ABS:.3e}'
        )
    return rec











class _DeploySafeESMSiteContrastV2(nn.Module):
    def __init__(self, source: nn.Module):
        super().__init__()
        self.radii = tuple(int(r) for r in source.radii)
        self.drop = float(source.drop)
        self.input_norm = copy.deepcopy(source.input_norm)
        self.input_proj = copy.deepcopy(source.input_proj)
        self.pos_emb = copy.deepcopy(source.pos_emb)
        self.radius_relation = copy.deepcopy(source.radius_relation)
        self.radius_emb = copy.deepcopy(source.radius_emb)
        self.radius_gate = copy.deepcopy(source.radius_gate)
        self.site_relation = copy.deepcopy(source.site_relation)
        self.signed_distance_bias = copy.deepcopy(source.signed_distance_bias)
        self.side_emb = copy.deepcopy(source.side_emb)
        self.site_score = copy.deepcopy(source.site_score)
        self.site_value = copy.deepcopy(source.site_value)

        self.proj_center = copy.deepcopy(source.proj['center'])
        self.proj_global_ctx = copy.deepcopy(source.proj['global'])
        self.proj_left = copy.deepcopy(source.proj['left'])
        self.proj_right = copy.deepcopy(source.proj['right'])
        self.proj_side = copy.deepcopy(source.proj['side'])
        self.head = copy.deepcopy(source.head)

    @staticmethod
    def _mean(h, mask):
        m = mask.float().unsqueeze(-1)
        den = m.sum(1).clamp_min(1.0)
        return (h * m).sum(1) / den

    def forward(self, x):
        valid = (x.abs().sum(dim=1) > 0)
        xt = x.transpose(1, 2)
        B, L, _ = xt.shape
        c = min(15, L - 1)
        h = self.input_proj(self.input_norm(xt))
        pos = torch.arange(L, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)
        h = h * valid.unsqueeze(-1).float()
        center = h[:, c]
        other = valid.clone()
        other[:, c] = False
        idx = pos.unsqueeze(0)
        left_all_m = other & (idx < c)
        right_all_m = other & (idx > c)
        left_all = self._mean(h, left_all_m)
        right_all = self._mean(h, right_all_m)
        glob = self._mean(h, other)
        side = left_all - right_all
        dist = (idx - c).abs()
        rts = []
        rvalid = []
        for r in self.radii:
            lm = other & (dist <= r)
            lmask = lm & (idx < c)
            rmask = lm & (idx > c)
            local = self._mean(h, lm)
            left = self._mean(h, lmask)
            right = self._mean(h, rmask)
            rel = torch.cat([
                center, local, left, right,
                center - local, center - left, center - right,
                left - right, (left - right).abs(),
            ], dim=-1)
            rts.append(self.radius_relation(rel))
            rvalid.append(lm.any(dim=1))
        rt = torch.stack(rts, 1)
        rv = torch.stack(rvalid, 1)
        rid = torch.arange(len(self.radii), device=x.device)
        rt = rt + self.radius_emb(rid).unsqueeze(0)
        empty = ~rv.any(1)
        if empty.any():
            rv = rv.clone()
            rv[empty, 0] = True
        gl = self.radius_gate(rt).squeeze(-1).masked_fill(~rv, -1e4)
        gw = F.softmax(gl, dim=1)
        radius_fused = (rt * gw.unsqueeze(-1)).sum(1)
        radius_max = rt.masked_fill(~rv.unsqueeze(-1), -1e4).max(1).values

        ce = center.unsqueeze(1).expand(-1, L, -1)
        delta = h - ce
        signed = ((idx - c) + (MAX_LEN - 1)).clamp(0, 2 * MAX_LEN - 2).long().squeeze(0)
        side_id = torch.zeros(L, device=x.device, dtype=torch.long)
        side_id[pos < c] = 1
        side_id[pos > c] = 2
        sh = self.site_relation(torch.cat([
            delta, delta.abs(), h * ce, h,
        ], dim=-1)) + self.side_emb(side_id).unsqueeze(0)
        slog = self.site_score(sh).squeeze(-1) + self.signed_distance_bias(signed).view(1, L)
        slog = slog.masked_fill(~other, -1e4)
        att = F.softmax(slog, dim=1) * other.float()
        att = att / att.sum(1, keepdim=True).clamp_min(1e-8)
        sp = (self.site_value(sh) * att.unsqueeze(-1)).sum(1) * other.any(1).float().unsqueeze(-1)
        feat = torch.cat([
            self.proj_center(center),
            self.proj_global_ctx(glob),
            radius_fused,
            radius_max,
            sp,
            self.proj_side(side),
            self.proj_left(left_all),
            self.proj_right(right_all),
        ], dim=-1)
        return self.head(feat)


def _deploy_trace_model(branch_name: str, model: nn.Module) -> nn.Module:
    if branch_name == 'esm_sitecontrast_v2':
        safe = _DeploySafeESMSiteContrastV2(model)
        safe = safe.to(CONFIG['DEVICE'])
        safe.eval()
        return safe
    return model


def _deploy_cache_signature(branch_name: str, final_pth_path: str,
                            reference_csv: str) -> Dict:
    return {
        'export_version': int(DEPLOY_EXPORT_VERSION),
        'branch': str(branch_name),
        'final_pth_sha256': _deploy_sha256_file(final_pth_path),
        'reference_sha256': _deploy_sha256_file(reference_csv),
    }


def _deploy_cache_paths(branch_name: str):
    os.makedirs(DEPLOY_CACHE_DIR, exist_ok=True)
    return (
        os.path.join(DEPLOY_CACHE_DIR, f'{branch_name}.ts'),
        os.path.join(DEPLOY_CACHE_DIR, f'{branch_name}.json'),
    )


def _deploy_cache_valid(branch_name: str, final_pth_path: str,
                        reference_csv: str) -> bool:
    if not bool(DEPLOY_RESUME):
        return False
    ts_path, meta_path = _deploy_cache_paths(branch_name)
    if not (os.path.isfile(ts_path) and os.path.isfile(meta_path)):
        return False
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            got = json.load(f)
        return got == _deploy_cache_signature(
            branch_name, final_pth_path, reference_csv
        )
    except Exception:
        return False


def _mark_deploy_cache_done(branch_name: str, final_pth_path: str,
                            reference_csv: str):
    _, meta_path = _deploy_cache_paths(branch_name)
    tmp = meta_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(
            _deploy_cache_signature(branch_name, final_pth_path, reference_csv),
            f, indent=2, ensure_ascii=False,
        )
    os.replace(tmp, meta_path)

def _trace_expert_to_file(
    branch_name: str,
    model: nn.Module,
    samples: List[Sample],
    preprocess: Dict,
    uid2idx: Optional[dict],
    base_kvoc: Optional[Dict[int, KmerVocab]],
    out_path: str,
):
    stats = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in (preprocess.get('stats', {}) or {}).items()
    }
    active_kvoc = None
    if branch_name == 'kmer':
        active_kvoc = _active_kmer_vocabs_for_final(base_kvoc, preprocess)

    n_example = max(1, min(int(DEPLOY_TRACE_BATCH_SIZE), len(samples)))
    ds = build_branch_dataset(
        branch_name,
        samples[:n_example],
        stats=stats,
        uid2idx=uid2idx,
        kmer_vocabs=active_kvoc,
    )
    dl = DataLoader(ds, batch_size=n_example, shuffle=False, num_workers=0)
    batch = next(iter(dl))
    x = _deploy_prepare_input(branch_name, batch, CONFIG['DEVICE'])

    model.eval()
    trace_model = _deploy_trace_model(branch_name, model)
    trace_model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(
            trace_model,
            (x,),
            strict=False,
            check_trace=False,
        )
        traced.eval()
        torch.jit.save(traced, out_path)


def export_deploy_bundle_from_final_checkpoint(
    final_pth_path: str,
    reference_samples: List[Sample],
    reference_csv: str,
    deploy_bundle_path: str,
) -> str:





    if not os.path.isfile(final_pth_path):
        raise FileNotFoundError(final_pth_path)
    if not reference_samples:
        raise RuntimeError('reference_samples 为空')

    ckpt = safe_torch_load(final_pth_path, map_location='cpu')
    required = {
        'expert_order', 'modality_order', 'expert_state_dicts',
        'expert_preprocess', 'kmer_vocab', 'fusion', 'config',
    }
    missing_keys = sorted(required - set(ckpt.keys()))
    if missing_keys:
        raise RuntimeError(f'FINAL.pth 缺少部署字段: {missing_keys}')
    if list(ckpt['expert_order']) != list(FINAL_EXPERTS):
        raise RuntimeError(
            f"expert order mismatch: ckpt={ckpt['expert_order']} runtime={FINAL_EXPERTS}"
        )
    if list(ckpt['modality_order']) != list(FUSED_MODALITIES):
        raise RuntimeError(
            f"modality order mismatch: ckpt={ckpt['modality_order']} runtime={FUSED_MODALITIES}"
        )

    ref = _reference_table_for_deploy(reference_csv, reference_samples)
    base_kvoc = _restore_kmer_vocabs_from_checkpoint(ckpt['kmer_vocab'])
    uid2idx = build_uid2idx(CONFIG['ESM_H5'])


    _STRUCTURE_STORES.clear()
    prepare_structure_store(reference_samples, source='main', pdb_dir=CONFIG['PDB_DIR'])

    tmp_bundle = deploy_bundle_path + '.tmp'
    if os.path.exists(tmp_bundle):
        os.remove(tmp_bundle)
    os.makedirs(os.path.dirname(os.path.abspath(deploy_bundle_path)), exist_ok=True)

    verification = []
    scripted_raw = {}
    scripted_aligned = {}

    try:
        with tempfile.TemporaryDirectory(prefix='deeppalm_deploy_') as td:
            expert_files = {}

            for branch in FINAL_EXPERTS:
                print(f'\n[DEPLOY] tracing expert: {branch}')
                model = build_branch_model(
                    branch,
                    kmer_vocabs=base_kvoc if branch == 'kmer' else None,
                )
                state = ckpt['expert_state_dicts'][branch]
                incompatible = model.load_state_dict(state, strict=True)
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise RuntimeError(
                        f'{branch}: state_dict mismatch: {incompatible}'
                    )
                model.eval()

                preprocess = ckpt['expert_preprocess'][branch]
                cache_ts_path, _ = _deploy_cache_paths(branch)

                if _deploy_cache_valid(
                    branch, final_pth_path, reference_csv
                ):
                    ts_path = cache_ts_path
                    print(f'[DEPLOY RESUME] HIT {branch}; skip tracing -> {ts_path}')
                else:
                    ts_path = cache_ts_path
                    tmp_ts = ts_path + '.tmp'
                    if os.path.exists(tmp_ts):
                        os.remove(tmp_ts)
                    _trace_expert_to_file(
                        branch,
                        model,
                        reference_samples,
                        preprocess,
                        uid2idx if is_esm_branch(branch) else None,
                        base_kvoc if branch == 'kmer' else None,
                        tmp_ts,
                    )

                    probe = torch.jit.load(tmp_ts, map_location=CONFIG['DEVICE'])
                    del probe
                    os.replace(tmp_ts, ts_path)

                expert_files[branch] = ts_path


                scripted = torch.jit.load(ts_path, map_location=CONFIG['DEVICE'])
                raw = _predict_script_module(
                    scripted,
                    branch,
                    reference_samples,
                    preprocess,
                    uid2idx if is_esm_branch(branch) else None,
                    base_kvoc if branch == 'kmer' else None,
                )
                align = preprocess.get('score_alignment', {'center': 0.0, 'scale': 1.0})
                aligned = apply_score_alignment(raw, align)
                scripted_raw[branch] = raw
                scripted_aligned[branch] = aligned

                ref_col = f'expert_{branch}'
                if ref_col not in ref.columns:
                    raise RuntimeError(f'reference CSV 缺少列: {ref_col}')
                verification.append(
                    _assert_deploy_close(
                        f'expert_{branch}',
                        aligned,
                        ref[ref_col].to_numpy(dtype=float),
                    )
                )
                _mark_deploy_cache_done(
                    branch, final_pth_path, reference_csv
                )

                del scripted
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            expert_mat = np.stack(
                [scripted_aligned[b] for b in FINAL_EXPERTS], axis=1
            ).astype(np.float32)
            modality_mat = apply_saved_first_stage(
                expert_mat,
                ckpt['fusion']['params'],
            )
            final_prob = apply_saved_fusion(
                expert_mat,
                ckpt['fusion']['method'],
                ckpt['fusion']['params'],
            ).astype(np.float32)

            for j, modality in enumerate(FUSED_MODALITIES):
                ref_col = f'modality_{modality}'
                if ref_col in ref.columns:
                    verification.append(
                        _assert_deploy_close(
                            f'modality_{modality}',
                            modality_mat[:, j],
                            ref[ref_col].to_numpy(dtype=float),
                        )
                    )

            verification.append(
                _assert_deploy_close(
                    'prob_selected_final',
                    final_prob,
                    ref['prob_selected_final'].to_numpy(dtype=float),
                )
            )


            expected_final = ref['prob_selected_final'].to_numpy(dtype=float)
            threshold_checks = {}
            for thr in DEPLOY_VERIFY_THRESHOLDS:
                old_pred = expected_final >= float(thr)
                new_pred = final_prob >= float(thr)
                mismatch = int(np.sum(old_pred != new_pred))
                threshold_checks[str(thr)] = {
                    'threshold': float(thr),
                    'mismatch_count': mismatch,
                    'match_fraction': float(np.mean(old_pred == new_pred)),
                }
                print(
                    f'[DEPLOY VERIFY] threshold={thr}: class mismatch={mismatch}/{len(final_prob)}'
                )
                if mismatch != 0:
                    raise RuntimeError(
                        f'deploy classification mismatch at threshold={thr}: {mismatch}'
                    )


            path_keys = {
                'CSV_PATH', 'ESM_H5', 'PDB_DIR', 'AAINDEX1_PATH', 'AAINDEX_PCA_PATH',
                'OUT_DIR', 'MODEL_DIR', 'CACHE_DIR', 'STRUCTURE_CACHE_DIR', 'FINAL_MODEL_PATH',
            }
            deploy_config = {
                k: copy.deepcopy(v)
                for k, v in ckpt.get('config', {}).items()
                if k not in path_keys
            }
            metadata = {
                'format': 'DeepPalmV14DeployBundle',
                'format_version': 1,
                'model_name': ckpt.get('model_name', 'DeepPalm V14 FINAL'),
                'description': ckpt.get('description'),
                'expert_order': list(FINAL_EXPERTS),
                'modality_order': list(FUSED_MODALITIES),
                'expert_preprocess': copy.deepcopy(ckpt['expert_preprocess']),
                'kmer_vocab': copy.deepcopy(ckpt['kmer_vocab']),
                'fusion': copy.deepcopy(ckpt['fusion']),
                'final_epochs': copy.deepcopy(ckpt.get('final_epochs', {})),
                'config': deploy_config,
                'deployment_test_metrics': copy.deepcopy(ckpt.get('deployment_test_metrics')),
                'source_final_pth_sha256': _deploy_sha256_file(final_pth_path),
                'reference_prediction_sha256': _deploy_sha256_file(reference_csv),
                'verification': verification,
                'threshold_checks': threshold_checks,
            }

            meta_path = os.path.join(td, 'metadata.pt')
            torch.save(metadata, meta_path)

            ref_npz = os.path.join(td, 'reference_outputs.npz')
            np.savez_compressed(
                ref_npz,
                uids=np.asarray([s.uid for s in reference_samples], dtype=object),
                final_prob=final_prob.astype(np.float32),
                expert_mat=expert_mat.astype(np.float32),
                modality_mat=modality_mat.astype(np.float32),
            )

            manifest = {
                'format': 'DeepPalmV14DeployBundle',
                'format_version': 1,
                'created_by': os.path.basename(__file__),
                'source_final_pth': os.path.abspath(final_pth_path),
                'source_final_pth_sha256': metadata['source_final_pth_sha256'],
                'reference_predictions': os.path.abspath(reference_csv),
                'reference_prediction_sha256': metadata['reference_prediction_sha256'],
                'expert_order': list(FINAL_EXPERTS),
                'modality_order': list(FUSED_MODALITIES),
                'fusion_method': ckpt['fusion']['method'],
                'verification_passed': True,
                'max_abs_tolerance': float(DEPLOY_VERIFY_MAX_ABS),
                'mean_abs_tolerance': float(DEPLOY_VERIFY_MEAN_ABS),
                'threshold_checks': threshold_checks,
                'files': {},
            }

            files_to_pack = {
                'metadata.pt': meta_path,
                'reference/reference_outputs.npz': ref_npz,
            }
            for branch, ts_path in expert_files.items():
                files_to_pack[f'experts/{branch}.ts'] = ts_path

            for arcname, fpath in files_to_pack.items():
                manifest['files'][arcname] = {
                    'size_bytes': int(os.path.getsize(fpath)),
                    'sha256': _deploy_sha256_file(fpath),
                }

            manifest_path = os.path.join(td, 'manifest.json')
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

            with zipfile.ZipFile(tmp_bundle, 'w', compression=zipfile.ZIP_STORED) as zf:
                zf.write(manifest_path, 'manifest.json')
                for arcname, fpath in files_to_pack.items():
                    zf.write(fpath, arcname)

        os.replace(tmp_bundle, deploy_bundle_path)


        with zipfile.ZipFile(deploy_bundle_path, 'r') as zf:
            names = set(zf.namelist())
            must = {'manifest.json', 'metadata.pt', 'reference/reference_outputs.npz'} | {
                f'experts/{b}.ts' for b in FINAL_EXPERTS
            }
            missing = must - names
            if missing:
                raise RuntimeError(f'deploy bundle 打包不完整: {sorted(missing)}')

        print('\n' + '=' * 86)
        print('[DEPLOY EXPORT] PASS')
        print('[DEPLOY EXPORT] 不重新训练；FINAL.pth 未被修改。')
        print('[DEPLOY EXPORT] bundle ->', deploy_bundle_path)
        print('[DEPLOY EXPORT] sha256 ->', _deploy_sha256_file(deploy_bundle_path))
        print('=' * 86)
        return deploy_bundle_path

    except Exception:
        if os.path.exists(tmp_bundle):
            os.remove(tmp_bundle)
        if os.path.exists(deploy_bundle_path):

            try:
                os.remove(deploy_bundle_path)
            except Exception:
                pass
        raise


def _configure_checkpoint_runtime(ckpt: Dict):

    current_paths = {
        'CSV_PATH': CONFIG['CSV_PATH'],
        'ESM_H5': CONFIG['ESM_H5'],
        'PDB_DIR': CONFIG['PDB_DIR'],
        'AAINDEX1_PATH': CONFIG['AAINDEX1_PATH'],
        'AAINDEX_PCA_PATH': CONFIG['AAINDEX_PCA_PATH'],
        'OUT_DIR': CONFIG['OUT_DIR'],
        'MODEL_DIR': CONFIG['MODEL_DIR'],
        'CACHE_DIR': CONFIG['CACHE_DIR'],
        'STRUCTURE_CACHE_DIR': CONFIG['STRUCTURE_CACHE_DIR'],
        'FINAL_MODEL_PATH': CONFIG['FINAL_MODEL_PATH'],
    }
    CONFIG.update(copy.deepcopy(ckpt.get('config', {})))
    CONFIG.update(current_paths)
    CONFIG['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    CONFIG['AMP'] = False
    CONFIG['NUM_WORKERS_ESM'] = 0
    CONFIG['NUM_WORKERS_OTHER'] = 0
    CONFIG['PIN_MEMORY'] = bool(torch.cuda.is_available())

    global AAINDEX1_PATH, PHYSICO_CHEMICAL_FEATURES, SEQ_FEAT_DIM, _PHYSCHEM_PCA_LOOKUP
    AAINDEX1_PATH = CONFIG['AAINDEX1_PATH']
    PHYSICO_CHEMICAL_FEATURES = build_physico_chemical_features(AAINDEX1_PATH)
    SEQ_FEAT_DIM = len(AAINDEX_IDS)
    _PHYSCHEM_PCA_LOOKUP = None


def export_existing_main():




    print('=' * 86)
    print('DeepPalm V14 — EXPORT EXISTING FINAL.pth ONLY')
    print('NO TRAINING / NO PARAMETER UPDATE')
    print('=' * 86)

    if not os.path.isfile(EXISTING_FINAL_MODEL_PATH):
        raise FileNotFoundError(EXISTING_FINAL_MODEL_PATH)
    if not os.path.isfile(EXISTING_REFERENCE_PREDICTIONS):
        raise FileNotFoundError(EXISTING_REFERENCE_PREDICTIONS)

    ckpt = safe_torch_load(EXISTING_FINAL_MODEL_PATH, map_location='cpu')
    _configure_checkpoint_runtime(ckpt)

    print('[EXPORT ONLY] model     =', EXISTING_FINAL_MODEL_PATH)
    print('[EXPORT ONLY] reference =', EXISTING_REFERENCE_PREDICTIONS)
    print('[EXPORT ONLY] output    =', DEPLOY_BUNDLE_PATH)
    print('[EXPORT ONLY] device    =', CONFIG['DEVICE'])

    samples = summarize_and_dedup(load_csv(CONFIG['CSV_PATH']))
    train_all, test_samples = split_train_test(
        samples,
        float(CONFIG['TEST_RATIO']),
        int(CONFIG['SPLIT_SEED']),
    )
    del train_all

    _STRUCTURE_STORES.clear()
    prepare_structure_store(samples, source='main', pdb_dir=CONFIG['PDB_DIR'])
    uidmap = build_uid2idx(CONFIG['ESM_H5'])
    missing = [s.uid for s in test_samples if s.uid not in uidmap]
    if missing:
        raise RuntimeError(
            f'export reference test 中 {len(missing)} 个 UID 缺失 ESM H5: {missing[:20]}'
        )

    export_deploy_bundle_from_final_checkpoint(
        final_pth_path=EXISTING_FINAL_MODEL_PATH,
        reference_samples=test_samples,
        reference_csv=EXISTING_REFERENCE_PREDICTIONS,
        deploy_bundle_path=DEPLOY_BUNDLE_PATH,
    )


def auto_resume_main():







    final_pth = CONFIG['FINAL_MODEL_PATH']
    reference_csv = os.path.join(
        CONFIG['OUT_DIR'], 'FINA', 'single_model_test_predictions_FINAL.csv'
    )

    if os.path.isfile(final_pth):
        if not os.path.isfile(reference_csv):
            raise FileNotFoundError(
                '[AUTO RESUME] 检测到 FINAL.pth 已存在，为防止误重训，程序不会进入训练。\n'
                '但缺少部署一致性验证所需的 reference predictions：\n'
                f'{reference_csv}\n'
                '请先修正 OUT_DIR / reference 路径后重新运行。'
            )
        print('=' * 86)
        print('[AUTO RESUME] FINAL.pth 已存在；跳过全部训练，直接继续 DEPLOY 导出。')
        print('[AUTO RESUME] model     =', final_pth)
        print('[AUTO RESUME] reference =', reference_csv)
        print('[AUTO RESUME] deploy    =', DEPLOY_BUNDLE_PATH)
        print('=' * 86)

        ckpt = safe_torch_load(final_pth, map_location='cpu')
        _configure_checkpoint_runtime(ckpt)

        samples = summarize_and_dedup(load_csv(CONFIG['CSV_PATH']))
        train_all, test_samples = split_train_test(
            samples,
            float(CONFIG['TEST_RATIO']),
            int(CONFIG['SPLIT_SEED']),
        )
        del train_all

        _STRUCTURE_STORES.clear()
        prepare_structure_store(
            samples, source='main', pdb_dir=CONFIG['PDB_DIR']
        )
        uidmap = build_uid2idx(CONFIG['ESM_H5'])
        missing = [s.uid for s in test_samples if s.uid not in uidmap]
        if missing:
            raise RuntimeError(
                f'auto-resume reference test 中 {len(missing)} 个 UID 缺失 ESM H5: '
                f'{missing[:20]}'
            )

        export_deploy_bundle_from_final_checkpoint(
            final_pth_path=final_pth,
            reference_samples=test_samples,
            reference_csv=reference_csv,
            deploy_bundle_path=DEPLOY_BUNDLE_PATH,
        )
        return

    print('[AUTO RESUME] 未发现完整 FINAL checkpoint + reference predictions；进入训练流程。')
    train_main()

def train_main():
    print_path_config()
    print('==> DeepPalm V14 TRAIN + AUTO-FUSION reproducible pipeline')
    print('    base experts =', FINAL_EXPERTS)
    print('    fused modalities =', FUSED_MODALITIES)
    print('    fusion candidates =', CONFIG['FUSION_CANDIDATES'])
    print('    selection = grouped training OOF only; test labels never select')
    print('    CV      =', CONFIG['N_REPEATS'], 'x', CONFIG['N_FOLDS'])
    print('    mining  = none (random fixed-ratio negatives only)')
    print('    deploy  = one final .pth containing 7 full-train experts + fusion')
    write_reproducibility_manifest()

    samples = summarize_and_dedup(load_csv(CONFIG['CSV_PATH']))
    train_all, test_samples = split_train_test(
        samples, CONFIG['TEST_RATIO'], CONFIG['SPLIT_SEED']
    )
    train_groups = np.asarray([_base_uid(s.uid) for s in train_all], dtype=object)
    test_groups = np.asarray([_base_uid(s.uid) for s in test_samples], dtype=object)
    if set(train_groups.tolist()) & set(test_groups.tolist()):
        raise RuntimeError('[FATAL] train/test protein overlap detected')

    y_train = np.asarray([s.label for s in train_all], dtype=int)
    y_test = np.asarray([s.label for s in test_samples], dtype=int)
    uids_tr = [s.uid for s in train_all]
    uids_te = [s.uid for s in test_samples]
    print(f'[Split] train={len(train_all)} test={len(test_samples)} protein-overlap=0')


    prepare_structure_store(samples, source='main', pdb_dir=CONFIG['PDB_DIR'])
    store = get_esm_ram_store(CONFIG['ESM_H5'])
    uidmap = store['uid2idx'] if store is not None else build_uid2idx(CONFIG['ESM_H5'])
    missing = [s.uid for s in samples if s.uid not in uidmap]
    if missing:
        raise RuntimeError(f'{len(missing)} CSV IDs missing in main H5: {missing[:20]}')

    kvoc = {}
    for k in CONFIG['KMERS']:
        voc = KmerVocab(k, max_size=None)
        for s0 in train_all:
            voc.add_seq(s0.seq)
        mc = CONFIG.get('KMER_MIN_COUNT', {})
        min_count = int(mc.get(k, mc.get(str(k), 1))) if isinstance(mc, dict) else int(mc)
        voc.finalize(min_count=min_count)
        kvoc[k] = voc
        print(f'[Kmer] k={k} vocab={voc.size} min_count={min_count}')




    repeat_results = []
    repeat_folds = []
    global CURRENT_REPEAT
    for rep0 in range(int(CONFIG['N_REPEATS'])):
        CURRENT_REPEAT = rep0 + 1
        folds = build_folds(
            train_all,
            int(CONFIG['N_FOLDS']),
            cv_seed=int(CONFIG['CV_SEED']) + rep0,
        )
        repeat_folds.append(folds)
        save_split_manifest(
            os.path.join(CONFIG['OUT_DIR'], f'split_manifest_rep{CURRENT_REPEAT}.csv'),
            samples, train_all, test_samples, folds=folds,
        )
        rr = {}
        print(f'\n========== Repeat {CURRENT_REPEAT}/{CONFIG["N_REPEATS"]} ==========')
        for branch in FINAL_EXPERTS:
            print(f'\n[Train] {branch}')
            rr[branch] = train_one_branch(
                branch,
                train_all,
                test_samples,
                folds,
                kmer_vocabs=kvoc if branch == 'kmer' else None,
            )
        repeat_results.append(rr)


    raw_oof = {
        b: np.mean(np.stack([rr[b].oof_prob for rr in repeat_results]), axis=0).astype(np.float32)
        for b in FINAL_EXPERTS
    }
    raw_test = {
        b: np.mean(np.stack([rr[b].test_prob for rr in repeat_results]), axis=0).astype(np.float32)
        for b in FINAL_EXPERTS
    }
    aligned_oof = {
        b: np.mean(np.stack([
            rr[b].aligned_oof_prob if rr[b].aligned_oof_prob is not None else rr[b].oof_prob
            for rr in repeat_results
        ]), axis=0).astype(np.float32)
        for b in FINAL_EXPERTS
    }
    aligned_test = {
        b: np.mean(np.stack([
            rr[b].aligned_test_prob if rr[b].aligned_test_prob is not None else rr[b].test_prob
            for rr in repeat_results
        ]), axis=0).astype(np.float32)
        for b in FINAL_EXPERTS
    }
    expert_oof = aligned_oof if CONFIG.get('USE_ALIGNED_FOR_FUSION', True) else raw_oof
    expert_test = aligned_test if CONFIG.get('USE_ALIGNED_FOR_FUSION', True) else raw_test





    expert_oof_mat = np.stack([expert_oof[b] for b in FINAL_EXPERTS], axis=1)
    expert_test_mat = np.stack([expert_test[b] for b in FINAL_EXPERTS], axis=1)
    shared_meta_seed = int(CONFIG['META_SHARED_SEED'])
    fusion_selection = fit_and_select_fusion(
        expert_oof_mat,
        y_train,
        train_groups,
        expert_test_mat,
        seed=shared_meta_seed,
    )
    selected_fusion_method = fusion_selection['selected_method']
    final_meta = fusion_selection['selected']
    final_oof = final_meta['oof_prob']
    final_test = final_meta['test_prob']
    modality_oof_mat = fusion_selection['modality_oof']
    modality_test_mat = fusion_selection['modality_test']

    FINA = os.path.join(CONFIG['OUT_DIR'], 'FINA')
    os.makedirs(FINA, exist_ok=True)

    save_meta_fold_manifest(
        os.path.join(FINA, 'meta_fold_manifest_FINAL.csv'),
        uids_tr, train_groups, y_train, shared_meta_seed,
    )


    branch_rows = []
    for b in FINAL_EXPERTS:
        macro, sd = aggregate_macro_fold_auc(repeat_results, b)
        branch_rows.append({
            'branch': b,
            'macro_fold_auc': float(macro),
            'fold_auc_sd': float(sd),
            'raw_oof_auc': float(roc_auc_score(y_train, raw_oof[b])),
            'aligned_oof_auc': float(roc_auc_score(y_train, aligned_oof[b])),
            'raw_test_auc': float(roc_auc_score(y_test, raw_test[b])),
            'aligned_test_auc': float(roc_auc_score(y_test, aligned_test[b])),
        })
    pd.DataFrame(branch_rows).to_csv(
        os.path.join(FINA, 'branch_metrics_FINAL.csv'), index=False
    )


    fused_branch_rows = []
    for j, modality in enumerate(FUSED_MODALITIES):
        fused_branch_rows.append({
            'modality': modality,
            'oof_auc': float(roc_auc_score(y_train, modality_oof_mat[:, j])),
            'oof_auprc': float(average_precision_score(y_train, modality_oof_mat[:, j])),
            'test_auc': float(roc_auc_score(y_test, modality_test_mat[:, j])),
            'test_auprc': float(average_precision_score(y_test, modality_test_mat[:, j])),
        })
    pd.DataFrame(fused_branch_rows).to_csv(
        os.path.join(FINA, 'fused_modality_metrics_FINAL.csv'), index=False
    )


    df_oof = pd.DataFrame({
        'uniprotid': uids_tr,
        'base_uniprot': train_groups,
        'label': y_train,
    })
    df_test = pd.DataFrame({
        'uniprotid': uids_te,
        'base_uniprot': test_groups,
        'label': y_test,
    })
    for b in FINAL_EXPERTS:
        df_oof[f'raw_{b}'] = raw_oof[b]
        df_oof[f'aligned_{b}'] = aligned_oof[b]
        df_oof[f'expert_{b}'] = expert_oof[b]
        df_test[f'raw_{b}'] = raw_test[b]
        df_test[f'aligned_{b}'] = aligned_test[b]
        df_test[f'expert_{b}'] = expert_test[b]
    for j, modality in enumerate(FUSED_MODALITIES):
        df_oof[f'modality_{modality}'] = modality_oof_mat[:, j]
        df_test[f'modality_{modality}'] = modality_test_mat[:, j]
    fusion_candidate_rows = []
    for method, result in fusion_selection['candidates'].items():
        df_oof[f'prob_{method}'] = result['oof_prob']
        df_test[f'prob_{method}'] = result['test_prob']
        test_candidate_metrics = calc_metrics(y_test, result['test_prob'])
        result['test_metrics'] = test_candidate_metrics
        fusion_candidate_rows.append({
            'method': method,
            'selected_by_training_oof': method == selected_fusion_method,
            'oof_auc': float(result['oof_metrics']['auc']),
            'oof_auprc': float(result['oof_metrics']['auprc']),
            'test_auc_final_report_only': float(test_candidate_metrics['auc']),
            'test_auprc_final_report_only': float(test_candidate_metrics['auprc']),
            'C': float(result['params']['second_stage']['C']),
            'n_features': int(result['params']['second_stage']['n_features']),
        })
    df_oof['prob_selected_final'] = final_oof
    df_test['prob_selected_final'] = final_test

    df_oof.to_csv(os.path.join(FINA, 'train_OOF_predictions_FINAL.csv'), index=False)
    df_test.to_csv(os.path.join(FINA, 'test_predictions_FULL_FINAL.csv'), index=False)
    df_test[['uniprotid', 'label', 'prob_selected_final']].to_csv(
        os.path.join(FINA, 'test_predictions_FINAL.csv'), index=False
    )
    pd.DataFrame(fusion_candidate_rows).to_csv(
        os.path.join(FINA, 'fusion_candidates_FINAL.csv'), index=False
    )


    fusion_params = {
        'expert_order': FINAL_EXPERTS,
        'modality_order': FUSED_MODALITIES,
        'first_stage_metrics': fusion_selection['first_stage_metrics'],
        'selected_fusion_method': selected_fusion_method,
        'shared_meta_seed': shared_meta_seed,
        'uses_fold_score_alignment': bool(CONFIG.get('USE_ALIGNED_FOR_FUSION', True)),
        'fusion': {
            method: result['params']
            for method, result in fusion_selection['candidates'].items()
        },
        'selection_rule': fusion_selection['selection_rule'],
        'selection_uses_test': False,
        'candidate_metrics': fusion_candidate_rows,
    }
    with open(os.path.join(FINA, 'fusion_params_FINAL.json'), 'w', encoding='utf-8') as f:
        json.dump(fusion_params, f, indent=2, ensure_ascii=False)

    save_kmer_vocab_v3(kvoc)

    oof_metrics = calc_metrics(y_train, final_oof)
    test_metrics = calc_metrics(y_test, final_test)
    summary = {
        'version': 'DeepPalm V14 reproducible train-and-auto-fusion',
        'experts': FINAL_EXPERTS,
        'base_experts': FINAL_EXPERTS,
        'fused_modalities': FUSED_MODALITIES,
        'final_fusion': selected_fusion_method,
        'fusion_selection_rule': fusion_selection['selection_rule'],
        'selection_uses_test': False,
        'fusion_candidates': fusion_candidate_rows,
        'n_repeats': int(CONFIG['N_REPEATS']),
        'n_folds': int(CONFIG['N_FOLDS']),
        'oof_metrics': oof_metrics,
        'test_metrics': test_metrics,
        'branch_metrics': branch_rows,
        'fused_modality_metrics': fused_branch_rows,
        'first_stage_metrics': fusion_selection['first_stage_metrics'],
    }
    with open(os.path.join(FINA, 'v14_final_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(CONFIG['OUT_DIR'], 'report.json'), 'w', encoding='utf-8') as f:
        json.dump({'config': CONFIG, **summary}, f, indent=2, ensure_ascii=False, default=str)






    deployment_metrics = None
    if bool(CONFIG.get('TRAIN_SINGLE_FINAL_MODEL', True)):
        print('\n' + '=' * 82)
        print('TRAIN ONE DEPLOYABLE DeepPalm_V14_FINAL.pth')
        print('=' * 82)

        final_epochs = choose_final_epochs(repeat_results)
        final_expert_results = {}

        for branch in FINAL_EXPERTS:
            cached_final = _try_load_final_expert_resume(
                branch, final_epochs[branch]
            )
            if cached_final is not None:
                final_expert_results[branch] = cached_final
                continue

            result = train_single_final_branch(
                branch,
                train_all,
                test_samples,
                epochs=final_epochs[branch],
                kmer_vocabs=kvoc if branch == 'kmer' else None,
            )
            final_expert_results[branch] = result
            _save_final_expert_resume(
                branch, final_epochs[branch], result
            )

        single_train_mat = np.stack([
            final_expert_results[b]['train_prob']
            for b in FINAL_EXPERTS
        ], axis=1)
        single_test_mat = np.stack([
            final_expert_results[b]['eval_prob']
            for b in FINAL_EXPERTS
        ], axis=1)

        single_train_modality_mat = apply_saved_first_stage(
            single_train_mat, final_meta['params']
        )
        single_test_modality_mat = apply_saved_first_stage(
            single_test_mat, final_meta['params']
        )
        single_train_prob = apply_saved_fusion(
            single_train_mat,
            selected_fusion_method,
            final_meta['params'],
        )
        single_test_prob = apply_saved_fusion(
            single_test_mat,
            selected_fusion_method,
            final_meta['params'],
        )

        deployment_metrics = calc_metrics(y_test, single_test_prob)

        df_single = pd.DataFrame({
            'uniprotid': uids_te,
            'base_uniprot': test_groups,
            'label': y_test,
        })
        for b in FINAL_EXPERTS:
            df_single[f'expert_{b}'] = final_expert_results[b]['eval_prob']
        for j, modality in enumerate(FUSED_MODALITIES):
            df_single[f'modality_{modality}'] = single_test_modality_mat[:, j]
        df_single['prob_selected_final'] = single_test_prob
        df_single.to_csv(
            os.path.join(FINA, 'single_model_test_predictions_FINAL.csv'),
            index=False,
        )

        pd.DataFrame([
            {
                'branch': b,
                'final_epochs': final_epochs[b],
                'test_auc': float(roc_auc_score(y_test, final_expert_results[b]['eval_prob'])),
                'test_auprc': float(average_precision_score(y_test, final_expert_results[b]['eval_prob'])),
            }
            for b in FINAL_EXPERTS
        ]).to_csv(
            os.path.join(FINA, 'single_model_branch_metrics_FINAL.csv'),
            index=False,
        )

        single_fused_rows = [
            {
                'modality': modality,
                'test_auc': float(roc_auc_score(y_test, single_test_modality_mat[:, j])),
                'test_auprc': float(average_precision_score(y_test, single_test_modality_mat[:, j])),
            }
            for j, modality in enumerate(FUSED_MODALITIES)
        ]
        pd.DataFrame(single_fused_rows).to_csv(
            os.path.join(FINA, 'single_model_fused_modality_metrics_FINAL.csv'),
            index=False,
        )

        single_summary = {
            'model_file': CONFIG['FINAL_MODEL_PATH'],
            'fusion_method': selected_fusion_method,
            'final_epochs': final_epochs,
            'test_metrics': deployment_metrics,
            'cv_ensemble_reference_test_metrics': test_metrics,
            'fused_modality_metrics': single_fused_rows,
            'note': (
                'The single deployable checkpoint contains seven base experts trained once on the full training split. '
                'Their outputs are reduced to four modality-level probabilities (physchem/ESM/kmer/structure) before final fusion. '
                'Its test metric can differ from the 3x5 CV ensemble reference.'
            ),
        }
        with open(
            os.path.join(FINA, 'single_model_summary_FINAL.json'),
            'w', encoding='utf-8'
        ) as f:
            json.dump(single_summary, f, indent=2, ensure_ascii=False, default=str)

        save_single_final_checkpoint(
            final_expert_results=final_expert_results,
            final_epochs=final_epochs,
            kvoc=kvoc,
            fusion_method=selected_fusion_method,
            fusion_params=final_meta['params'],
            cv_summary={
                'oof_metrics': oof_metrics,
                'cv_ensemble_test_metrics': test_metrics,
                'branch_metrics': branch_rows,
                'fused_modality_metrics': fused_branch_rows,
                'first_stage_metrics': fusion_selection['first_stage_metrics'],
                'fusion_candidates': fusion_candidate_rows,
            },
            deployment_metrics=deployment_metrics,
        )






        export_deploy_bundle_from_final_checkpoint(
            final_pth_path=CONFIG['FINAL_MODEL_PATH'],
            reference_samples=test_samples,
            reference_csv=os.path.join(FINA, 'single_model_test_predictions_FINAL.csv'),
            deploy_bundle_path=DEPLOY_BUNDLE_PATH,
        )

        print('\n[SINGLE FINAL] TEST AUC   =', deployment_metrics['auc'])
        print('[SINGLE FINAL] TEST AUPRC =', deployment_metrics['auprc'])
        print('[SINGLE FINAL] MODEL FILE =', CONFIG['FINAL_MODEL_PATH'])

    write_result_checksums(
        FINA,
        selected_method=selected_fusion_method,
        include_model=deployment_metrics is not None,
    )

    print('\n========== DeepPalm V14 FINAL ==========', flush=True)
    for r in branch_rows:
        print(
            f"  - {r['branch']:20s} macro={r['macro_fold_auc']:.4f} "
            f"alignedOOF={r['aligned_oof_auc']:.4f} test={r['aligned_test_auc']:.4f}"
        )
    print('[FINAL] fusion =', selected_fusion_method)
    print('[FINAL] OOF AUC =', oof_metrics['auc'])
    print('[FINAL] OOF AUPRC =', oof_metrics['auprc'])
    print('[FINAL] TEST AUC =', test_metrics['auc'])
    print('[FINAL] TEST AUPRC =', test_metrics['auprc'])
    print('[FINAL] outputs ->', FINA)
    if deployment_metrics is not None:
        print('[FINAL] single deployable model ->', CONFIG['FINAL_MODEL_PATH'])
        print('[FINAL] single-model TEST AUC ->', deployment_metrics['auc'])

if __name__ == '__main__':
    mode = str(RUN_MODE).strip().lower()
    if mode == 'train':
        train_main()
    elif mode == 'export_existing':
        export_existing_main()
    elif mode == 'auto_resume':
        auto_resume_main()
    else:
        raise ValueError(
            f"RUN_MODE={RUN_MODE!r} 不支持；只能是 'train'、'export_existing' 或 'auto_resume'"
        )
