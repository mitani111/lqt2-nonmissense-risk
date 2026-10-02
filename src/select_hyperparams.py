# Hyper-parameter selection from the three validation splits

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from config import EARLY_STOPPING_PATIENCE, early_stop, out_root, tag

ALPHA_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
BETA_GRID = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
CV_SPLITS = ["cv1", "cv2", "cv3"]


def parse_args():
    p = argparse.ArgumentParser(description="Select alpha, beta and the epoch")
    p.add_argument("--out-root", default=None)
    p.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    p.add_argument("--save-csv", default=None,
                   help="optional path for the full grid summary")
    return p.parse_args()


def mean_val_curve(oroot, alpha, beta):
    curves = []
    for split in CV_SPLITS:
        path = os.path.join(oroot, f"finetune_z{tag(alpha)}_{tag(beta)}",
                            split, "loss", "val_cindex_epoch.npy")
        if not os.path.exists(path):
            return None
        curves.append(np.load(path))
    length = min(len(c) for c in curves)
    return np.nanmean(np.vstack([c[:length] for c in curves]), axis=0)

def main():
    args = parse_args()
    oroot = out_root(args.out_root)

    rows = []
    for alpha, beta in itertools.product(ALPHA_GRID, BETA_GRID):
        curve = mean_val_curve(oroot, alpha, beta)
        if curve is None:
            continue
        best, best_epoch = early_stop(curve, args.patience)
        rows.append({"alphaZ": alpha, "betaZ": beta,
                     "best_val_c_index": best, "best_epoch": best_epoch})

    summary = (pd.DataFrame(rows)
                 .sort_values("best_val_c_index", ascending=False)
                 .reset_index(drop=True))
    if args.save_csv:
        summary.to_csv(args.save_csv, index=False)

    best = summary.iloc[0]
    print(summary.to_string(index=False))
    print("\nSelected configuration:")
    print(f"  alphaZ = {best['alphaZ']}")
    print(f"  betaZ  = {best['betaZ']}")
    print(f"  epoch  = {int(best['best_epoch'])}")
    print(f"  mean validation C-index = {best['best_val_c_index']:.4f}")

if __name__ == "__main__":
    main()
