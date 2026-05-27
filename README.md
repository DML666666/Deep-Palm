# Deep-Palm

Deep-Palm is a multi-view deep learning framework for protein S-palmitoylation site prediction. It integrates sequence-derived features, protein-property features, protein language model embeddings, and predicted spatial-structure features.

## Repository structure

```text
Deep-Palm/
├── Train_the_Model/
│   ├── train.py
│   ├── input.csv
│   ├── embedding.h5
│   ├── esmfold.pdb/
│   ├── aaindex1.txt
│   ├── uniprotid_species.csv
│   ├── esm2_embedding.py
│   └── esm2_structure.py
├── Using_the_Model/
│   ├── predict.py
│   ├── Deep-Palm.pth
│   ├── kmer_vocab.json
│   ├── aaindex1.txt
│   ├── esm2_embedding.py
│   └── esm2_structure.py
├── environment.yml
└── .gitattributes
```

`Deep-Palm.pth` and `kmer_vocab.json` are the pretrained Deep-Palm resources used for prediction.  
`aaindex1.txt` is required by `predict.py` to construct protein-property features.

## Download and environment setup

The repository contains large files managed through Git LFS. Install Git LFS before cloning the repository:

```bash
git lfs install
git clone https://github.com/DML666666/Deep-Palm.git
cd Deep-Palm
```

Create the conda environment using the provided environment file:

```bash
conda env create -f environment.yml
```

Activate the environment name specified in `environment.yml` before running the scripts.

## External pretrained models required for feature generation

Prediction with Deep-Palm requires embedding and structure features generated from the candidate sequence windows.

- **Embedding features:** generated using **ESM-2 t36 3B UR50D** (`esm2_t36_3B_UR50D.pt`). The supplied `esm2_embedding.py` script averages representations from the last four ESM-2 layers.
- **Structure features:** generated using **ESMFold v1** through `esm.pretrained.esmfold_v1()` in `esm2_structure.py`.

The ESM-2 and ESMFold weights are external pretrained model resources and are not the Deep-Palm prediction model. Ensure that they are available in your local environment before feature generation. The official ESM and ESMFold implementation is available from the [facebookresearch/esm](https://github.com/facebookresearch/esm) repository.

---

## Training the model

The `Train_the_Model/` directory contains the files used for Deep-Palm model training.

### Included training resources

```text
Train_the_Model/
├── train.py                 # model training script
├── input.csv                # training input table
├── embedding.h5             # precomputed ESM-2 embedding features
├── esmfold.pdb/             # precomputed ESMFold-predicted PDB structures
├── aaindex1.txt             # AAindex resource for protein-property features
├── uniprotid_species.csv    # auxiliary species information used in data preparation
├── esm2_embedding.py        # embedding feature generation script
└── esm2_structure.py        # structure feature generation script
```

### Reproduce model training

1. Open `Train_the_Model/train.py`.
2. In the configuration section, set the corresponding file paths to the files provided in `Train_the_Model/`:
   - training input table: `input.csv`
   - ESM embedding features: `embedding.h5`
   - predicted structures: `esmfold.pdb/`
   - AAindex resource: `aaindex1.txt`
3. Run the training script:

```bash
cd Train_the_Model
python train.py
```

The supplied precomputed features can be used directly. To regenerate the features, configure and run `esm2_embedding.py` and `esm2_structure.py` using the same input sequences.

---

## Using the pretrained model

The `Using_the_Model/` directory provides the pretrained model and scripts for predicting S-palmitoylation sites in user-provided candidate sequences.

### Workflow overview

```text
Your candidate sequence windows
        │
        ├── esm2_embedding.py  ──>  your_embedding.h5
        │       (ESM-2 t36 3B UR50D)
        │
        ├── esm2_structure.py  ──>  your_esmfold_pdb/*.pdb
        │       (ESMFold v1)
        │
        └── predict.py + Deep-Palm.pth + kmer_vocab.json + aaindex1.txt
                              ──>  Deep-Palm prediction scores
```

### Step 1. Prepare your candidate sequence file

Prepare a CSV file containing two columns named `ID` and `Window`:

```csv
ID,Window
candidate_site_001,AAAAAAAAAAAAAAACAAAAAAAAAAAAAAA
candidate_site_002,GGGGGGGGGGGGGGGCGGGGGGGGGGGGGGG
```

Requirements:

- `ID` may be any unique identifier or placeholder for each candidate site.
- `Window` must contain a **31-residue amino acid sequence window**.
- The candidate cysteine must be located at the central position of the window, that is, position 16.
- Labels, species information, and experimental annotations are not required for prediction.
- The same `ID` values must be used consistently for embedding generation, structure generation, and prediction.

The column name `Window` is used here because it is directly compatible with the supplied feature-generation scripts.

### Step 2. Generate ESM-2 embedding features

Open `Using_the_Model/esm2_embedding.py` and set the paths in the `CONFIG` section. For example:

```python
CONFIG = {
    "CSV": "./candidate_sites.csv",
    "OUT": "./features/your_embedding.h5",

    "ID_COL": "ID",
    "SEQ_COL": "Window",
    "COL3_COL": "",
    "COL3_PLACEHOLDER": "",

    "MODEL_PATH": "path/to/esm2_t36_3B_UR50D.pt",
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
```

Run:

```bash
cd Using_the_Model
python esm2_embedding.py
```

This script generates an HDF5 file containing the ESM-2 embedding features used by Deep-Palm.

### Step 3. Generate ESMFold-predicted structure files

Open `Using_the_Model/esm2_structure.py` and set:

```python
INPUT_CSV = r"./candidate_sites.csv"
OUTPUT_DIR = r"./features/your_esmfold_pdb"
NUM_WORKERS = 1
```

Run:

```bash
python esm2_structure.py
```

This script uses ESMFold v1 to generate one PDB structure file for each candidate sequence window:

```text
features/your_esmfold_pdb/
├── candidate_site_001.pdb
└── candidate_site_002.pdb
```

`NUM_WORKERS = 1` is recommended for typical single-GPU environments because each worker loads an ESMFold model instance.

### Step 4. Predict S-palmitoylation sites using Deep-Palm

Open `Using_the_Model/predict.py` and set the paths in its `CONFIG` section:

```python
CONFIG = {
    # ---- your input and output ----
    "INPUT_CSV": "./candidate_sites.csv",
    "OUTPUT_CSV": "./prediction_results.csv",

    # ---- pretrained Deep-Palm resources ----
    "FUSED_CKPT": "./Deep-Palm.pth",
    "KMER_VOCAB_JSON": "./kmer_vocab.json",
    "AAINDEX1_PATH": "./aaindex1.txt",

    # ---- features generated from your candidate sequences ----
    "ESM_H5": "./features/your_embedding.h5",
    "PDB_DIR": "./features/your_esmfold_pdb",

    # ---- runtime ----
    "BATCH_SIZE": 64,
    "NUM_WORKERS": 0,
    "DEVICE": "cuda",
    "STRICT": True,
    "THRESHOLD": 0.5,
}
```

Run:

```bash
python predict.py
```

The prediction output file contains the Deep-Palm score for each candidate site:

```text
ID,Window,Deep-Palm_score,prediction
```

- `Deep-Palm_score`: model output score for the candidate S-palmitoylation site.
- `prediction`: binary prediction generated using the threshold specified in `predict.py`.

### Important notes for prediction

- Use the same candidate CSV file to generate the embedding features, generate the PDB structures, and perform prediction.
- Do not provide full-length protein sequences to this workflow. Each row must contain a 31-residue cysteine-centered sequence window.
- The IDs in the candidate CSV file must match the IDs stored in the generated HDF5 file and the PDB filenames.
- `Deep-Palm.pth` is the trained Deep-Palm prediction model. `esm2_t36_3B_UR50D.pt` and ESMFold v1 are external pretrained models used only to generate input features.

## Code availability

The source code, pretrained Deep-Palm model, and processed files provided for model training and prediction are available at:

https://github.com/DML666666/Deep-Palm
