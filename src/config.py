import os
import shutil

import numpy as np
import torch

# Reproducibility
SEED = 42

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Input tensor shape
MSA_LENGTH = 1380      # fixed length of the multiple sequence alignment
N_DESCRIPTORS = 8      # AAindex descriptors per residue (see README)
N_AA_CLASSES = 21      # 20 amino acids + alignment gap

# Network widths (identical in the pre-training and fine-tuning stages)
BRANCH_DIMS = [MSA_LENGTH, 800]                       # per-descriptor branch
TRUNK_DIMS = [BRANCH_DIMS[-1] * N_DESCRIPTORS,        # 6400
              3000, 1000, 200, 100]                   # last entry = latent dim
LATENT_DIM = TRUNK_DIMS[-1]

# KCNH2 (hERG1a, UniProt Q12809) structural domains.
DOMAIN_LABELS = ["EAGD", "N-linker", "VSD", "PD", "C-linker", "CNBHD", "C-tail"]

# Boundaries in the original (un-aligned) protein sequence, 1-based.
DOMAIN_BOUNDS_PROTEIN = {
    "EAGD": (1, 135),
    "N-linker": (136, 398),
    "VSD": (399, 545),
    "PD": (546, 670),
    "C-linker": (671, 736),
    "CNBHD": (737, 863),
    "C-tail": (864, 1159),
}

# The same boundaries expressed in alignment (MSA) coordinates, 1-based.
DOMAIN_BOUNDS_MSA = {
    "EAGD": (1, 135),
    "N-linker": (136, 473),
    "VSD": (474, 629),
    "PD": (630, 754),
    "C-linker": (755, 822),
    "CNBHD": (823, 956),
    "C-tail": (957, 1380),
}

# Record-ID convention used throughout the pipeline
CLINICAL_ID_MAX = 999      # ID  <  999   : clinical cohort (with outcome)
PUBLIC_ID_MIN = 10000      # ID  > 9999   : public variant records
                           # 1000 <= ID <= 9999 : reference (control) sequence


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def data_root(cli_value=None):
    root = cli_value or os.environ.get("LQT2_DATA_ROOT")
    return os.path.abspath(os.path.expanduser(root))


def out_root(cli_value=None):
    root = cli_value or os.environ.get("LQT2_OUT_ROOT")
    return os.path.abspath(os.path.expanduser(root))


def resolve_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(device, seed=SEED):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    elif device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def tag(value):
    return str(float(value)).replace(".", "")


def early_stop(curve, patience=EARLY_STOPPING_PATIENCE):
    best, best_epoch, wait = -np.inf, 0, 0
    for epoch, value in enumerate(curve):
        if np.isnan(value):
            wait += 1
        elif value > best:
            best, best_epoch, wait = value, epoch, 0
        else:
            wait += 1
        if wait > patience:
            break
    return best, best_epoch


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path
