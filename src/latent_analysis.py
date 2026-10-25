import argparse
import os
import re
import string

import Levenshtein
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE

from config import (CLINICAL_ID_MAX, EARLY_STOPPING_PATIENCE, LATENT_DIM,
                    PUBLIC_ID_MIN, SEED, early_stop, out_root, tag)

matplotlib.use("Agg")

TSNE_PERPLEXITY = 5
GMM_N_INIT = 5
COVARIANCE_TYPE = "full"
MISSENSE_LABEL = "Missense (reference)"


def parse_args():
    p = argparse.ArgumentParser(description="Cluster and visualise the latent space")
    p.add_argument("--alphaZ", type=float, default=None)
    p.add_argument("--betaZ", type=float, default=None)
    p.add_argument("--split", default="full",
                   help="fine-tuning run to analyse (default: the whole cohort)")
    p.add_argument("--epoch", type=int, default=None,
                   help="epoch to analyse; by default it is derived from the KLz curve")
    p.add_argument("--table", default=None,
                   help="analyse this exported table instead of resolving a run directory")
    p.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                   help="epochs without improvement")
    p.add_argument("--n-clusters", type=int, default=2)
    p.add_argument("--missense-ids", default="",
                   help="comma-separated record IDs that carry several missense "
                        "substitutions and are therefore relabelled as missense")
    p.add_argument("--out-root", default=None)
    p.add_argument("--fig-dir", default=None)
    return p.parse_args()


def select_epoch_from_loss(loss_dir, patience=EARLY_STOPPING_PATIENCE):
    curve = np.load(os.path.join(loss_dir, "loss_epoch.npy"))
    _, epoch = early_stop(-curve, patience)
    return int(epoch), curve


def annotate_variants(df, missense_ids):
    control = df[(df["ID"] > CLINICAL_ID_MAX) & (df["ID"] < PUBLIC_ID_MIN)]
    if control.empty:
        raise SystemExit("No reference (control) record found in the exported table.")
    ctrl_msa = str(control["MSA"].iloc[0])
    ctrl_seq = ctrl_msa.replace("-", "")

    df = df.copy()
    df["distance_msa"] = [Levenshtein.distance(ctrl_msa, str(s)) for s in df["MSA"]]
    df["distance_orig"] = [Levenshtein.distance(ctrl_seq, str(s)) for s in df["sequence"]]

    df["MutationType"] = df["VariantType"]
    if missense_ids:
        df.loc[df["ID"].isin(missense_ids), "MutationType"] = "missense"
    df["ctrl_msa"] = ctrl_msa
    return df


def name_clusters(labels, risk=None):
    if risk is None:
        order = sorted({int(l) for l in labels})
    else:
        order = (pd.Series(np.asarray(risk)).groupby(labels).mean()
                 .sort_values(ascending=False).index)
    mapping = {int(c): string.ascii_uppercase[i] for i, c in enumerate(order)}
    return np.array([mapping[int(l)] for l in labels])


def display_name(value):
    return value if value == MISSENSE_LABEL else f"Cluster {value}"


def scatter(embedding, values, label, fig_path, discrete=False):
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    if discrete:
        for value in sorted(pd.unique(values)):
            mask = np.asarray(values) == value
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=11, alpha=0.85,
                       linewidths=0, rasterized=True, label=display_name(value))
        ax.legend(frameon=False, markerscale=2.0)
    else:
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=values, cmap="plasma",
                        s=11, alpha=0.85, linewidths=0, rasterized=True)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label(label)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def km_by_cluster(df_clinical, fig_path, title):
    if df_clinical["cluster_label"].nunique() < 2:
        print(f"[skipped] {title}: fewer than two clusters are represented")
        return None

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for cluster, group in df_clinical.groupby("cluster_label"):
        kmf = KaplanMeierFitter()
        kmf.fit(group["age"], group["event"], label=display_name(cluster))
        kmf.plot(ax=ax, show_censors=True, ci_show=False)
    result = multivariate_logrank_test(df_clinical["age"],
                                       df_clinical["cluster_label"],
                                       df_clinical["event"])
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Event-free survival")
    ax.set_title(f"{title} (log-rank p = {result.p_value:.4g})")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return result


def resolve_table(args):
    """Locate the exported table to analyse and the directory to write to."""
    if args.table:
        epoch_match = re.search(r"df_epoch(\d+)\.csv$", os.path.basename(args.table))
        epoch = int(epoch_match.group(1)) if epoch_match else -1
        run_path = os.path.dirname(os.path.dirname(os.path.abspath(args.table)))
        return os.path.abspath(args.table), epoch, run_path

    if args.alphaZ is None or args.betaZ is None:
        raise SystemExit("Pass --alphaZ and --betaZ, or --table.")

    run_path = os.path.join(out_root(args.out_root),
                            f"finetune_z{tag(args.alphaZ)}_{tag(args.betaZ)}",
                            args.split)
    epoch = args.epoch
    if epoch is None:
        epoch, _ = select_epoch_from_loss(os.path.join(run_path, "loss"),
                                         args.patience)
        print(f"Analysis epoch from the KLz curve: {epoch}")
    return os.path.join(run_path, "validation", f"df_epoch{epoch}.csv"), epoch, run_path


def main():
    args = parse_args()
    table_path, epoch, run_path = resolve_table(args)
    fig_dir = args.fig_dir or os.path.join(run_path, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    loss_dir = os.path.join(run_path, "loss")
    if args.table is None and os.path.exists(os.path.join(loss_dir, "loss_epoch.npy")):
        curve = np.load(os.path.join(loss_dir, "loss_epoch.npy"))
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(curve)
        ax.axvline(epoch, linestyle="--", color="red", alpha=0.8,
                   label=f"selected epoch = {epoch}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("$L_{finetune}$")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "loss_early_stopping.png"), dpi=300)
        plt.close(fig)

    df = pd.read_csv(table_path)
    missense_ids = [int(x) for x in args.missense_ids.split(",") if x.strip()]
    df = annotate_variants(df, missense_ids)

    # variants only: the clinical cohort and the public records
    df_variant = df[(df["ID"] <= CLINICAL_ID_MAX) |
                    (df["ID"] >= PUBLIC_ID_MIN)].reset_index(drop=True)
    feature_cols = [f"feature{i}" for i in range(LATENT_DIM)]
    features = df_variant[feature_cols].to_numpy()

    is_nm = (df_variant["MutationType"] == "non-missense").to_numpy()
    is_ms = (df_variant["MutationType"] == "missense").to_numpy()
    features_nm = features[is_nm]

    gmm = GaussianMixture(n_components=args.n_clusters,
                          covariance_type=COVARIANCE_TYPE,
                          n_init=GMM_N_INIT, random_state=SEED)
    labels = gmm.fit_predict(features_nm)

    risk = (df_variant.loc[is_nm, "hz_hat"].to_numpy()
            if "hz_hat" in df_variant.columns else None)

    df_variant["cluster"] = pd.array([pd.NA] * len(df_variant), dtype="Int64")
    df_variant.loc[is_nm, "cluster"] = labels
    df_variant["cluster_label"] = np.where(is_ms, MISSENSE_LABEL, "unclustered")
    df_variant.loc[is_nm, "cluster_label"] = name_clusters(labels, risk)

    embedding = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY,
                     learning_rate="auto", init="pca",
                     early_exaggeration=12.0,
                     random_state=SEED).fit_transform(features)
    df_variant["tsne1"] = embedding[:, 0]
    df_variant["tsne2"] = embedding[:, 1]

    scatter(embedding, df_variant["distance_orig"], "Levenshtein distance",
            os.path.join(fig_dir, "tsne_levenshtein.png"))
    scatter(embedding, df_variant["cluster_label"], "Cluster",
            os.path.join(fig_dir, "tsne_cluster.png"), discrete=True)
    if "hz_hat" in df_variant.columns:
        scatter(embedding, np.exp(df_variant["hz_hat"]), "Estimated risk",
                os.path.join(fig_dir, "tsne_risk.png"))

    clinical = df_variant[df_variant["ID"] <= CLINICAL_ID_MAX].reset_index(drop=True)
    non_missense = clinical[clinical["MutationType"] == "non-missense"]
    restricted = km_by_cluster(non_missense,
                               os.path.join(fig_dir, "km_cluster_nonmissense.png"),
                               "Non-missense clusters")
    overall = km_by_cluster(
        clinical[clinical["MutationType"].isin(["non-missense", "missense"])],
        os.path.join(fig_dir, "km_cluster_with_missense.png"),
        "Non-missense clusters and missense reference")

    keep = ["ID", "age", "event", "male", "female", "MSA", "sequence", "ctrl_msa",
            "distance_orig", "distance_msa", "MutationType",
            "cluster", "cluster_label", "tsne1", "tsne2"]
    if "hz_hat" in df_variant.columns:
        keep.insert(8, "hz_hat")
    out_csv = os.path.join(run_path, f"latent_clusters_epoch{epoch}.csv")
    df_variant[keep].to_csv(out_csv, index=False)

    print(f"epoch            : {epoch}")
    print(f"clusters         : {args.n_clusters}")
    for name, res in (("non-missense clusters", restricted),
                      ("clusters vs missense reference", overall)):
        if res is not None:
            print(f"log-rank p ({name}) : {res.p_value:.4g}")
    print(f"per-record table : {out_csv}")
    print(pd.crosstab(df_variant["MutationType"], df_variant["cluster_label"]))


if __name__ == "__main__":
    main()
