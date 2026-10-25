import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
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
    return "n.s."


def composition_figure(prop, low, high, table, clusters, fig_path):
    x = np.arange(len(DOMAIN_LABELS))
    width = 0.8 / max(len(clusters), 1)
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=300)

    for i, cluster in enumerate(clusters):
        y = prop[cluster].values
        yerr = np.clip(np.vstack([y - low[cluster].values,
                                  high[cluster].values - y]), 0.0, None)
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
    ax.set_ylabel("Cluster proportion within region (%)")
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
    df["mutation_point"] = [first_divergence(str(c), str(m)) + 1
                            for c, m in zip(df["ctrl_msa"], df["MSA"])]
    df["domain"] = [assign_domain(p) for p in df["mutation_point"]]
    non_missense = df[df["MutationType"] == "non-missense"].reset_index(drop=True)
    clusters = sorted(non_missense[cluster_col].unique())

    # ---------------- cluster composition per domain ---------------------
    counts = (non_missense.groupby(["domain", cluster_col]).size()
              .unstack(cluster_col, fill_value=0)
              .reindex(DOMAIN_LABELS, fill_value=0)
              .reindex(columns=clusters, fill_value=0))

    totals = counts.sum(axis=1)
    prop = (counts.divide(totals, axis=0) * 100).fillna(0.0)
    low, high = prop.copy(), prop.copy()
    for label in DOMAIN_LABELS:
        n = int(totals[label])
        for cluster in clusters:
            if n == 0:
                low.loc[label, cluster] = high.loc[label, cluster] = 0.0
            else:
                lo, hi = proportion_confint(count=int(counts.loc[label, cluster]),
                                            nobs=n, alpha=args.alpha_ci,
                                            method="wilson")
                low.loc[label, cluster] = lo * 100.0
                high.loc[label, cluster] = hi * 100.0
    cluster_totals = counts.sum(axis=0)
    pvals, methods = [], []
    for label in DOMAIN_LABELS:
        inside = counts.loc[label, clusters].astype(int).values
        outside = (cluster_totals - counts.loc[label, clusters]).astype(int).values
        table_2x2 = np.vstack([inside, outside])
        _, p, _, expected = chi2_contingency(table_2x2)
        if (expected < 5).any():
            _, p = fisher_exact(table_2x2)
            methods.append("fisher")
        else:
            methods.append("chi2")
        pvals.append(p)
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
