# HyQuRP

## Introduction
HyQuRP is a hybrid quantum–classical neural network for 3D point clouds that maintains rotational and permutational equivariance in its representations, enabling rotation- and permutation-invariant classification. We show that HyQuRP outperforms most classical and quantum state-of-the-art models on various datasets in the sparse point regime.

**Key Results**

| Dataset | #Points | Params | HyQuRP (acc) | Best baseline (acc) |
|---|---:|---:|---:|---:|
| ModelNet | 6 | ~1K | XX.XX | XX.XX |
| ShapeNet | 6 | ~1K | XX.XX | XX.XX |
| Sydney Urban Objects | 6 | ~1K | XX.XX | XX.XX |

---

## Installation

### Requirements
- Python >= 3.10
- (Optional) CUDA >= 12.x
- Framework: (PyTorch or JAX) + (PennyLane, etc.)

### Setup
```bash
git clone https://github.com/<USER>/<REPO>.git
cd <REPO>
pip install -r requirements.txt
```

---

## Usage

### Data
There are three data type in data folder(ModelNet, ShapeNet, Sydney Urban Objects).

### Make matrix

### Run baselines

### Run HyQuRP

