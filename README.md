# Deep-Palm

Deep-Palm is a multi-view deep learning framework for protein **S-palmitoylation site prediction**.

The model integrates four complementary types of information:

- AAindex-derived protein physicochemical properties
- ESM-2 protein language model embeddings
- sequence k-mer features
- ESMFold-predicted spatial-structure features

This repository provides the files required for both **model training** and **prediction using the pretrained Deep-Palm model**.

---

## Repository structure

```text
Deep-Palm/
├── Train_the_Model/
│   ├── train.py
│   ├── input.csv
│   ├── embedding.h5
│   ├── esmfold.pdb/
│   ├── aaindex1.txt
│   └── AAindex_PCA.csv
│
├── Using_the_Model/
│   ├── predict.py
│   ├── DeepPalm_DEPLOY.dpalm
│   ├── DeepPalm.pth
│   ├── aaindex1.txt
│   ├── AAindex_PCA.csv
│   ├── esm2_embedding.py
│   └── esmfold_structure.py
│
├── environment.yml
├── .gitattributes
└── README.md
```

### Main files

- `Train_the_Model/train.py`: main Deep-Palm training script.
- `Train_the_Model/input.csv`: input data used for model training.
- `Train_the_Model/embedding.h5`: precomputed ESM-2 embedding features.
- `Train_the_Model/esmfold.pdb/`: precomputed ESMFold structure files.
- `aaindex1.txt`: AAindex database file used to construct physicochemical features.
- `AAindex_PCA.csv`: PCA transformation information used by the physicochemical branch.
- `DeepPalm.pth`: trained Deep-Palm model checkpoint.
- `DeepPalm_DEPLOY.dpalm`: deployable Deep-Palm model bundle used by `predict.py`.
- `esm2_embedding.py`: script for generating ESM-2 embedding features.
- `esmfold_structure.py`: script for generating ESMFold-predicted structures.
- `predict.py`: script for prediction using the pretrained Deep-Palm model.

---

## Download

Some large files in this repository are managed using **Git LFS**.

Install Git LFS before cloning the repository:

```bash
git lfs install
git clone https://github.com/DML666666/Deep-Palm.git
cd Deep-Palm
```

If necessary, large files can also be retrieved manually after cloning:

```bash
git lfs pull
```

---

## Runtime environment

Deep-Palm was developed using a CUDA-enabled PyTorch environment.

A tested runtime included:

```text
Python        3.11.11
PyTorch       2.2.1+cu121
CUDA          12.1
h5py          3.15.1
pandas        2.3.3
numpy         1.26.4
tqdm          4.67.1
scikit-learn  1.6.1
matplotlib    3.10.0
biopython     1.83
```

The ESM package is additionally required for ESM-2 and ESMFold feature generation.

Because the original runtime was assembled from an existing ESM/ESMFold environment, users may need to install the required packages manually according to their local CUDA and PyTorch configuration.

---

## External pretrained models

Deep-Palm uses external pretrained models to generate embedding and structural features.

### ESM-2

ESM-2 embeddings are generated using:

```text
esm2_t36_3B_UR50D
```

The corresponding pretrained weight file is typically:

```text
esm2_t36_3B_UR50D.pt
```

The supplied `esm2_embedding.py` script uses representations from the last four ESM-2 layers.

### ESMFold

Predicted structural features are generated using:

```python
esm.pretrained.esmfold_v1()
```

The ESM-2 and ESMFold models are external pretrained resources and are not part of the Deep-Palm prediction model itself.

The official ESM and ESMFold implementation is available from:

https://github.com/facebookresearch/esm

---

# Training the model

The `Train_the_Model/` directory contains the files required to reproduce Deep-Palm model training.

```text
Train_the_Model/
├── train.py
├── input.csv
├── embedding.h5
├── esmfold.pdb/
├── aaindex1.txt
└── AAindex_PCA.csv
```

The current training script uses relative paths:

```python
CSV_PATH = "input.csv"
ESM_H5 = "embedding.h5"
PDB_DIR = "esmfold.pdb"
AAINDEX1_PATH = "aaindex1.txt"
AAINDEX_PCA_PATH = "AAindex_PCA.csv"
```

Therefore, training can be started directly from the `Train_the_Model` directory:

```bash
cd Train_the_Model
python train.py
```

The training pipeline includes repeated cross-validation, protein-level data splitting, multiple feature branches, and final multi-view fusion.

The supplied precomputed ESM-2 embeddings and ESMFold structures can be used directly for model training.

---

# Using the pretrained model

The `Using_the_Model/` directory contains the deployable model and scripts required for Deep-Palm prediction.

The general workflow is:

```text
Candidate 31-aa sequence windows
            │
            ├── esm2_embedding.py
            │        ↓
            │   embedding.h5
            │
            ├── esmfold_structure.py
            │        ↓
            │   esmfold.pdb/
            │
            └── predict.py
                     +
               DeepPalm_DEPLOY.dpalm
                     ↓
              prediction_results.csv
```

---

## Step 1. Prepare the input file

Prepare a CSV or TSV file containing at least two columns:

```text
ID
Window
```

Example:

```csv
ID,Window
candidate_site_001,AAAAAAAAAAAAAAACAAAAAAAAAAAAAAA
candidate_site_002,GGGGGGGGGGGGGGGCGGGGGGGGGGGGGGG
```

Requirements:

- `ID` must uniquely identify each candidate site.
- `Window` should contain a 31-residue amino-acid sequence.
- The candidate cysteine must be located at the central position.
- For a 31-residue window, the central cysteine is residue **16**.
- The same ID must be used consistently in the input file, ESM-2 embedding file, and PDB filename.
- Duplicate IDs are not allowed.

For example:

```text
candidate_site_001
```

must correspond to:

```text
candidate_site_001.pdb
```

in the structure directory.

---

## Step 2. Generate ESM-2 embeddings

Open:

```text
Using_the_Model/esm2_embedding.py
```

and modify the `CONFIG` section.

For example:

```python
CONFIG = {
    "CSV": "/path/to/input.csv",
    "OUT": "/path/to/embedding.h5",

    "ID_COL": "ID",
    "SEQ_COL": "Window",
    "COL3_COL": "",
    "COL3_PLACEHOLDER": "",

    "MODEL_PATH": "/path/to/esm2_t36_3B_UR50D.pt",
    "MODEL_ID": "esm2_t36_3B_UR50D",

    "DEVICE": "cuda",
    "BATCH_SIZE": 8,
    "LAYERS": "last4",
    "CENTER_INDEX": 15,

    "STORE_FP16": True,
    "SAVE_CLS": True,
    "SAVE_MASK_STATS": True,
}
```

Then run:

```bash
cd Using_the_Model
python esm2_embedding.py
```

The generated HDF5 file contains the ESM-2 representations required by Deep-Palm.

---

## Step 3. Generate ESMFold structures

Open:

```text
Using_the_Model/esmfold_structure.py
```

and set the input and output paths:

```python
INPUT_CSV = r"/path/to/input.csv"
OUTPUT_DIR = r"/path/to/esmfold.pdb"
NUM_WORKERS = 1
```

Then run:

```bash
python esmfold_structure.py
```

The script generates one PDB file for each candidate site:

```text
esmfold.pdb/
├── candidate_site_001.pdb
├── candidate_site_002.pdb
└── ...
```

`NUM_WORKERS` should be adjusted according to available GPU memory. A single worker is the safest option for a typical single-GPU environment.

---

## Step 4. Run Deep-Palm prediction

Open:

```text
Using_the_Model/predict.py
```

and modify the file paths at the beginning of the script:

```python
MODEL_PATH = r"DeepPalm_DEPLOY.dpalm"

INPUT_CSV = r"input.csv"
INPUT_ESM_H5 = r"embedding.h5"
INPUT_PDB_DIR = r"esmfold.pdb"

AAINDEX1_PATH = r"aaindex1.txt"
AAINDEX_PCA_PATH = r"AAindex_PCA.csv"

OUTPUT_CSV = r"prediction_results.csv"
```

The prediction script automatically uses GPU acceleration when CUDA is available:

```python
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
```

Run:

```bash
cd Using_the_Model
python predict.py
```

---

## Prediction output

The prediction output contains two columns:

```csv
ID,probability
candidate_site_001,0.9231
candidate_site_002,0.1847
```

where:

- `ID` is the identifier of the candidate site.
- `probability` is the Deep-Palm prediction score.

A higher probability indicates stronger model support for the candidate site being an S-palmitoylation site.

---

## Important notes

1. The same candidate sequences and IDs must be used for ESM-2 embedding generation, ESMFold structure generation, and Deep-Palm prediction.

2. The candidate sequence should be a 31-residue cysteine-centered window.

3. The central residue must be cysteine.

4. Input IDs must be unique.

5. Each input ID must have a corresponding entry in the generated `embedding.h5`.

6. Each input ID must have a corresponding PDB file in the structure directory.

7. `DeepPalm_DEPLOY.dpalm` is the deployable model used by `predict.py`.

8. Users do not need to retrain Deep-Palm when using the supplied pretrained model.

9. ESM-2 and ESMFold are used only to generate input features and are separate from the trained Deep-Palm model.

---

## Code availability

The source code, pretrained model, and processed files required for model training and prediction are available in this repository:

https://github.com/DML666666/Deep-Palm
