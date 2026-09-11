# PathoVerse AI Data

This directory contains the datasets, metadata, and processed artifacts used by
PathoVerse AI.

Large datasets and generated processed data are intentionally excluded from Git
because pathology datasets and Whole-Slide Images can require several gigabytes
of storage.

## Expected Directory Structure

```text
data/
├── raw/
│   ├── medmnist/
│   ├── patchcamelyon/
│   └── wsi/
│
├── processed/
│
└── metadata/
````

## PatchCamelyon

PatchCamelyon (PCam) is used for pathology image classification experiments.

The local dataset is expected under:

```text
data/raw/patchcamelyon/
```

The official PCam data contains:

* Training set
* Validation set
* Test set
* 96 × 96 RGB pathology image patches

The raw HDF5 files are intentionally excluded from Git.

## PathMNIST

MedMNIST PathMNIST is used for pathology dataset experimentation and validation.

The local dataset is expected under:

```text
data/raw/medmnist/
```

The raw dataset is intentionally excluded from Git.

## Whole-Slide Images

Whole-Slide Images are expected under:

```text
data/raw/wsi/
```

The current WSI demonstration uses:

```text
CMU-1-Small-Region.svs
```

This is an Aperio SVS pathology slide used for:

* WSI metadata extraction
* Tissue detection
* Tissue-mask generation
* Tile extraction
* Foundation-model embedding
* Multiple Instance Learning
* Attention heatmap generation
* Retrieval experiments

The raw WSI file is intentionally excluded from Git.

## Metadata

Lightweight metadata and validation artifacts are stored under:

```text
data/metadata/
```

These files provide dataset statistics, validation information, tile metadata,
and other lightweight information required for understanding the experiments.

## Important

Do not place large raw datasets or generated processed datasets into Git.

The repository `.gitignore` excludes:

```text
data/raw/
data/processed/
```

See the main project `README.md` for complete setup and benchmark information.

```
