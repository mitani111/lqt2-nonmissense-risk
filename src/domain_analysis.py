import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from config import DOMAIN_BOUNDS_MSA, DOMAIN_LABELS

matplotlib.use("Agg")

def parse_args():
    p = argparse.ArgumentParser(description="Cluster composition per structural domain")
    p.add_argument("--clusters-csv", required=True,
                   help="table written by latent_analysis.py")
    p.add_argument("--fig-dir", default=None)
    p.add_argument("--alpha-ci", type=float, default=0.05)
    return p.parse_args()


def first_divergence(reference, target):
    for i, (a, b) in enumerate(zip(reference, target)):
        if a != b:
            return i
    return min(len(reference), len(target))


def assign_domain(position):
    for label in DOMAIN_LABELS:
        lo, hi = DOMAIN_BOUNDS_MSA[label]
        if lo <= position <= hi:
            return label
    return "out_of_range"


def stars(p_value):
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def composition_figure(prop, low, high, table, clusters, fig_path):
    x = np.arange(len(DOMAIN_LABELS))
    width = 0.8 / max(len(clusters), 1)
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=300)

    for i, cluster in enumerate(clusters):
        y = prop[cluster].values
        yerr = np.vstack([y - low[cluster].values, high[cluster].values - y])
        ax.bar(x + (i - (len(clusters) - 1) / 2) * width, y, width,
               yerr=yerr, capsize=3, label=f"Cluster {cluster}")

    for i, label in enumerate(DOMAIN_LABELS):
        mark = table.loc[table["domain"] == label, "signif"].iloc[0]
        if mark:
            top = max(high[c].values[i] for c in clusters)
            ax.text(x[i], top + 1.5, mark, ha="center", fontsize=12)

    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_LABELS, rotation=20)
    ax.set_xlabel("Protein domain")
    ax.set_ylabel("Within-cluster composition (%)")
    ax.legend(frameon=False)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    df = pd.read_csv(args.clusters_csv)
    fig_dir = args.fig_dir or os.path.dirname(os.path.abspath(args.clusters_csv))
    os.makedirs(fig_dir, exist_ok=True)

    cluster_col = "cluster_label" if "cluster_label" in df.columns else "cluster"

    # ---------------- domain assignment ----------------------------------
    df["mutation_point"] = [first_divergence(str(c), str(m))
                            for c, m in zip(df["ctrl_msa"], df["MSA"])]
    df = df[df["mutation_point"] >= 0].reset_index(drop=True)
    # alignment positions are reported 1-based
    df["domain"] = [assign_domain(p + 1) for p in df["mutation_point"]]

    non_missense = df[df["MutationType"] == "non-missense"].reset_index(drop=True)
    clusters = sorted(non_missense[cluster_col].unique())

    # ---------------- cluster composition per domain ---------------------
    counts = (non_missense.groupby(["domain", cluster_col]).size()
              .unstack(cluster_col, fill_value=0)
              .reindex(DOMAIN_LABELS, fill_value=0)
              .reindex(columns=clusters, fill_value=0))

    totals = counts.sum(axis=0)
    prop = counts.divide(totals, axis=1) * 100
    low, high = prop.copy(), prop.copy()
    for cluster in clusters:
        if totals[cluster] > 0:
            lo, hi = proportion_confint(count=counts[cluster].values,
                                        nobs=totals[cluster],
                                        alpha=args.alpha_ci, method="wilson")
            low[cluster], high[cluster] = lo * 100.0, hi * 100.0
        else:
            low[cluster], high[cluster] = 0.0, 0.0

    # one chi-square test per domain: that domain against the remaining ones
    pvals = []
    for label in DOMAIN_LABELS:
        inside = counts.loc[label, clusters].values
        outside = (counts.sum(axis=0) - counts.loc[label, clusters]).values
        pvals.append(chi2_contingency(np.vstack([inside, outside]))[1])
    pvals = np.asarray(pvals)

    table = pd.DataFrame({"domain": DOMAIN_LABELS,
                          "p_value": pvals,
                          "signif": [stars(p) for p in pvals]})
    for cluster in clusters:
        table[f"n_cluster{cluster}"] = counts[cluster].values
        table[f"pct_cluster{cluster}"] = prop[cluster].values

    composition_figure(prop, low, high, table, clusters,
                       os.path.join(fig_dir, "domain_composition.png"))


if __name__ == "__main__":
    main()
