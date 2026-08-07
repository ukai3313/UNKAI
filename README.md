# UNKAI

UNKAI is a protein-pair classification model for predicting whether two proteins are associated with the same enzymatic reaction.

The model uses per-residue protein embeddings, attention pooling, and a multilayer perceptron classifier. For a protein pair, each protein embedding is pooled independently, and the absolute difference between the two pooled representations is used for binary classification.

## Model overview

UNKAI currently provides two model variants:

* **Original**

  * Attention pooling
  * Pair representation: `|v1 - v2|`
  * MLP: `2560 -> 1599 -> 781 -> 117 -> 1`

* **Seen-unseen**

  * Attention pooling with attention dropout
  * Pair representation: `|v1 - v2|`
  * MLP: `2560 -> 1408 -> 640 -> 512 -> 1`
  * Designed for evaluation where one side of validation/test pairs belongs to a cluster observed during training and the other side belongs to an unseen cluster.

Pretrained model weights are distributed through Hugging Face.

## Input

UNKAI expects one per-residue embedding file for each protein.

Expected filename:

```text
<UNIPROT_ACCESSION>_embedding.npy
```

Expected array shape:

```text
(L, 2560)
```

or

```text
(1, L, 2560)
```

where `L` is the protein sequence length.

## Installation

```bash
git clone https://github.com/ukai3313/UNKAI.git
cd UNKAI
pip install -r requirements.txt
```

## Inference

Add the source directory to `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src"
```

### Original model

```bash
python -m unkai.predict \
  --model original \
  --protein1 Q6GZV6 \
  --protein2 Q6GZN7 \
  --embeddings-dir /path/to/embeddings \
  --checkpoint /path/to/original/model.pth
```

### Seen-unseen model

```bash
python -m unkai.predict \
  --model seen_unseen \
  --protein1 Q6GZV6 \
  --protein2 Q6GZN7 \
  --embeddings-dir /path/to/embeddings \
  --checkpoint /path/to/seen_unseen/model.pth
```

Example output:

```text
Protein 1   : Q6GZV6
Protein 2   : Q6GZN7
Model       : original
Probability : 0.002518
Prediction  : 0
```

`Prediction = 1` indicates that the two proteins are predicted to be associated with the same enzymatic reaction.

## Datasets

Three dataset variants are provided separately through Hugging Face:

* **original**

  * Random pair-level train/validation/test split.

* **seen_unseen**

  * Training pairs contain two proteins from training-assigned clusters.
  * Validation pairs contain one training-cluster protein and one validation-cluster protein.
  * Test pairs contain one training-cluster protein and one test-cluster protein.

* **strict**

  * Protein clusters are assigned independently to train, validation, and test.
  * Both proteins in a pair must belong to clusters assigned to the same split.
  * This prevents cluster overlap across train/validation/test.

No exact protein-pair duplicates are shared across train, validation, and test in the released datasets.

## Repository structure

```text
UNKAI/
├── src/
│   └── unkai/
│       ├── __init__.py
│       ├── model.py
│       └── predict.py
├── examples/
├── requirements.txt
├── README.md
└── LICENSE
```

## Model weights

Pretrained weights:

* Hugging Face model repository: `ukaikotaro/UNKAI`

## Dataset

Released datasets:

* Hugging Face dataset repository: `ukaikotaro/UNKAI-dataset`

## Embeddings

UNKAI requires precomputed per-residue protein embeddings with dimensionality 2560.

The embedding dataset is planned to be distributed separately.

## Citation

Citation information will be added here.

## License

License information will be added here.

