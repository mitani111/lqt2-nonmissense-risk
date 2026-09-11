# lqt2-nonmissense-risk

Code for **"Risk Stratification of Non-Missense KCNH2 Variants Using Deep
Generative Models pretrained on Public Data"**.

A variational autoencoder (VAE) with a DeepSurv-style survival head is
pretrained without supervision on publicly available *KCNH2* variant records
and then fine-tuned on a clinically annotated LQT2 cohort. This repository
contains the scripts for model training, inference, evaluation and
latent-space analysis, together with the hyperparameter search space.

---

## Requirements

```
pip install -r requirements.txt
```

The analyses were run with **Python 3.9** and **PyTorch 2.5.1** on macOS
(Apple M1 Max, 32 GB unified memory) using the Metal Performance Shaders (MPS)
backend. Results are seeded (`seed = 42`) but are not
bit-wise reproducible across backends.

## Notation

The article writes the coefficient of the KL term as **beta** and that of the
survival term as **lambda**. The command-line flags predate that notation:

| article | command-line flag | applies to |
|---|---|---|
| beta | `--alphaZ` | KL divergence term, both stages |
| lambda | `--betaZ` | Cox partial-likelihood term, fine-tuning only |

Run directories are named from these values with the decimal point removed,
so `--alphaZ 0.2 --betaZ 50.0` gives `finetune_z02_500/`.

---

## Data

The clinical cohort cannot be redistributed. As stated in the article, the data
underlying it are available on reasonable request to the corresponding author.
The pretraining set consists of *KCNH2* variant records retrieved from the
NCBI databases.

### Expected data layout

Both roots are given on the command line (`--data-root`, `--out-root`) or
through the environment variables `LQT2_DATA_ROOT` and `LQT2_OUT_ROOT`.

```
<data-root>/
  in_data/
    metadata.csv               one row per record; ID, Age, Event, Male, Female, MSA
    metadata_with_split.csv    the same table plus `assign` (CV1 / CV2 / CV3 / HO)
  tensors/
    all/       array_<ID>.npy  float (1380, 8) input tensor, every record
    labels/    array_<ID>.npy  int   (1380,)   reconstruction target
    latent/                    clinical cohort only (used by --split full)
    test/train/, test/test/    training set / holdout set
    train/val1/{train,test}/   training set, cross-validation fold 1
    train/val2/{train,test}/   training set, cross-validation fold 2
    train/val3/{train,test}/   training set, cross-validation fold 3
```

Record IDs follow a fixed convention: `ID < 999` clinical cohort,
`ID = 1000` reference sequence, `ID > 9999` public records.

---

## Input representation and preprocessing

1. **Variant to protein sequence.** Each variant is translated into its amino
   acid sequence using human genome build **GRCh38.p14** as the reference.
2. **Alignment.** All sequences are aligned with **MAFFT** (default settings).
   The resulting alignment has a fixed length of **1,380** positions.
3. **Residue encoding.** Every aligned residue is replaced by **8 prespecified
   AAindex descriptors**, each min-max scaled to [0, 1] over the 20 standard
   amino acids. The channel order used by the models is:

   | channel | AAindex accession | property |
   |---|---|---|
   | 0 | `KYTJ820101` | hydrophobicity (Kyte-Doolittle) |
   | 1 | `GRAR740103` | side-chain volume (Grantham) |
   | 2 | `CHOP780201` | alpha-helix propensity (Chou-Fasman) |
   | 3 | `CHOP780202` | beta-sheet propensity (Chou-Fasman) |
   | 4 | `CHOP780101` | beta-turn propensity (Chou-Fasman) |
   | 5 | `GRAR740102` | polarity (Grantham) |
   | 6 | `ZIMJ680104` | isoelectric point (charge) |
   | 7 | `VINM940101` | flexibility |

   Positions that correspond to alignment gaps are filled with a zero vector.
4. **Reconstruction target.** The same alignment is stored as a length-1,380
   vector of integer class labels over **21 classes** (20 amino acids + gap).
5. **Covariates.** Sex is one-hot encoded in two dimensions and enters the model only at the survival head.

---


## Hyperparameter search

Two coefficients are tuned. Since beta is shared between the two stages, a
fine-tuning run with a given beta starts from the pretraining run with that
same beta.

```
beta   in {0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0}     (--alphaZ)
lambda in {0.1, 0.5, 1.0, 10.0, 50.0, 100.0}       (--betaZ)
```

42 configurations tuned by grid search with 3-fold cross-validation in the
training set. The validation C-index is computed at every epoch on each
fold and the mean C-index across folds is monitored with early stopping
after **10 epochs without improvement**; the configuration with the highest
mean validation C-index is selected.

---

## KCNH2 structural domains

hERG1a is divided into seven consecutive domains, with boundaries based on the
full-length hERG cryo-EM structure. Each variant is assigned to the domain that
contains the **first alignment position at which its sequence diverges from the
reference**; for frameshift, nonsense and splice-site variants this marks the
first altered position, not the extent of the alteration.

| domain | protein residues | alignment positions |
|---|---|---|
| EAGD (eag/PAS) | 1-135 | 1-135 |
| N-linker | 136-398 | 136-473 |
| VSD | 399-545 | 474-629 |
| PD | 546-670 | 630-754 |
| C-linker | 671-736 | 755-822 |
| CNBHD | 737-863 | 823-956 |
| C-tail | 864-1159 | 957-1380 |

Among non-missense variants, the composition of the clusters within each domain
is reported with **Wilson 95% confidence intervals**, and domain-specific
p-values are calculated with a **chi-square test** per domain (a 2 x k table of
that domain against the remaining domains). Benjamini-Hochberg adjusted values
are written to the output table as well, for reference.
