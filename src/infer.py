# Export risk scores and latent representations for every record.

import argparse
import os

import numpy as np
import pandas as pd
import torch

from config import (BRANCH_DIMS, LATENT_DIM, N_DESCRIPTORS, TRUNK_DIMS,
                    data_root, out_root, resolve_device, tag)
from finetune import SPLITS
from models import Encoder, Encoder_last, Pyz


def parse_args():
    p = argparse.ArgumentParser(description="Export risk scores and latent codes")
    p.add_argument("--stage", choices=["finetune", "pretrain"], default="finetune")
    p.add_argument("--split", choices=sorted(SPLITS), default="holdout")
    p.add_argument("--alphaZ", type=float, default=1.0)
    p.add_argument("--betaZ", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=151,
                   help="number of fine-tuning checkpoints to score (0 .. epochs-1)")
    p.add_argument("--pretrain-epoch", type=int, default=260,
                   help="checkpoint to score when --stage pretrain")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out-root", default=None)
    p.add_argument("--tensor-dir", default=None)
    p.add_argument("--meta-csv", default=None,
                   help="metadata table including the split assignment "
                        "(default <data-root>/in_data/metadata_with_split.csv)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_models(load_path, device, with_survival_head=True):
    branches = []
    for k in range(N_DESCRIPTORS):
        m = Encoder(BRANCH_DIMS[0], BRANCH_DIMS[1]).to(device)
        m.load_state_dict(torch.load(os.path.join(load_path, f"encoder_1_{k}.pth"),
                                     map_location=device, weights_only=True))
        m.eval()
        branches.append(m)

    encoder_2 = Encoder(TRUNK_DIMS[0], TRUNK_DIMS[1]).to(device)
    encoder_2.load_state_dict(torch.load(os.path.join(load_path, "encoder_2.pth"),
                                         map_location=device, weights_only=True))
    encoder_2.eval()

    encoder_3 = Encoder(TRUNK_DIMS[1], TRUNK_DIMS[2]).to(device)
    encoder_3.load_state_dict(torch.load(os.path.join(load_path, "encoder_3.pth"),
                                         map_location=device, weights_only=True))
    encoder_3.eval()

    encoder_4 = Encoder_last(TRUNK_DIMS[2], TRUNK_DIMS[3], TRUNK_DIMS[4]).to(device)
    encoder_4.load_state_dict(torch.load(os.path.join(load_path, "encoder_4.pth"),
                                         map_location=device, weights_only=True))
    encoder_4.eval()

    pyz = None
    if with_survival_head:
        pyz = Pyz(LATENT_DIM).to(device)
        pyz.load_state_dict(torch.load(os.path.join(load_path, "pyz.pth"),
                                       map_location=device, weights_only=True))
        pyz.eval()

    return branches, encoder_2, encoder_3, encoder_4, pyz


def main():
    args = parse_args()

    droot = data_root(args.data_root)
    oroot = out_root(args.out_root)
    tensor_dir = args.tensor_dir or os.path.join(droot, "tensors")
    meta_csv = args.meta_csv or os.path.join(droot, "in_data",
                                             "metadata_with_split.csv")
    if args.stage == "pretrain":
        run_path = os.path.join(oroot, f"pretrain_z{tag(args.alphaZ)}")
        epochs = [args.pretrain_epoch]
    else:
        run_path = os.path.join(oroot,
                                f"finetune_z{tag(args.alphaZ)}_{tag(args.betaZ)}",
                                args.split)
        epochs = list(range(args.epochs))
    with_survival_head = args.stage == "finetune"
    save_dir = os.path.join(run_path, "validation")
    os.makedirs(save_dir, exist_ok=True)

    device = resolve_device(args.device)

    # ---------------- inputs (all records, loaded once) -----------------
    dir_path = os.path.join(tensor_dir, "all")
    filelist = sorted(os.listdir(dir_path))
    ids = np.array([int(f[6:-4]) for f in filelist], dtype=np.int64)
    arrays = torch.from_numpy(
        np.stack([np.load(os.path.join(dir_path, f)).astype(np.float32)
                  for f in filelist])
    ).to(device)

    df = pd.read_csv(meta_csv)
    df["sequence"] = df["MSA"].str.replace("-", "", regex=False)
    base_df = (df.set_index("ID")
                 .loc[ids, ["Age", "Event", "Male", "Female", "MSA",
                            "sequence", "assign"]]
                 .reset_index()
                 .rename(columns={"Age": "age", "Event": "event",
                                  "Male": "male", "Female": "female",
                                  "assign": "HoldOut"})
                 .reset_index(drop=True))

    covariates = torch.from_numpy(
        base_df[["male", "female"]].to_numpy(np.float32)
    ).to(device)
    feature_cols = [f"feature{i}" for i in range(LATENT_DIM)]

    # ---------------- inference ------------------------------------------
    for epoch in epochs:
        load_path = os.path.join(run_path, f"epoch_{epoch}")
        branches, encoder_2, encoder_3, encoder_4, pyz = load_models(
            load_path, device, with_survival_head=with_survival_head)

        with torch.inference_mode():
            x = torch.cat([branches[k](arrays[:, :, k])
                           for k in range(N_DESCRIPTORS)], dim=1)
            x = encoder_2(x)
            x = encoder_3(x)
            x_mu, _ = encoder_4(x)
            risk = None
            if pyz is not None:
                risk = pyz(torch.cat([x_mu, covariates], dim=1)).squeeze()

        parts = [base_df]
        if risk is not None:
            parts.append(pd.DataFrame({"hz_hat": risk.detach().cpu().numpy()
                                       .reshape(-1).astype(np.float32)}))
        parts.append(pd.DataFrame(x_mu.detach().cpu().numpy()
                                  .reshape(len(base_df), -1).astype(np.float32),
                                  columns=feature_cols))
        pd.concat(parts, axis=1).to_csv(
            os.path.join(save_dir, f"df_epoch{epoch}.csv"), index=False)


if __name__ == "__main__":
    main()
