import argparse
import json
import os

import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

from config import out_root, tag

DEV_LABELS = ["CV1", "CV2", "CV3"]   # values of the `assign` column
HOLDOUT_LABEL = "HO"
MAX_FOLLOW_UP = 40                   # administrative censoring at 40 years

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a fine-tuned model on the hold-out set")
    p.add_argument("--alphaZ", type=float, required=True)
    p.add_argument("--betaZ", type=float, required=True)
    p.add_argument("--epoch", type=int, required=True,
                   help="checkpoint index of the df_epoch<E>.csv table to score")
    p.add_argument("--split", default="holdout")
    p.add_argument("--out-root", default=None)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--alpha-ci", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def breslow_baseline_survival(df):
    d = df.sort_values("age").reset_index()
    times = np.unique(d["age"].values)
    survival, cum_hazard = [], 0.0
    for t in times:
        events_at_t = ((d["age"] == t) & (d["event"] == 1)).sum()
        risk_sum = d.loc[d["age"] >= t, "w"].sum()
        cum_hazard += events_at_t / risk_sum
        survival.append(np.exp(-cum_hazard))
    return np.array(survival), times


def survival_matrix(df, baseline_survival, times, horizon=MAX_FOLLOW_UP):
    preds = np.ones((len(df), horizon))
    for i, row in df.iterrows():
        surv = baseline_survival ** row["w"]
        for j, t in enumerate(times):
            t0 = int(t)
            t1 = int(times[j + 1]) if j + 1 < len(times) else -1
            preds[i, t0:t1] = surv[j]
    return preds


def main():
    args = parse_args()
    oroot = out_root(args.out_root)
    run_path = os.path.join(oroot,
                            f"finetune_z{tag(args.alphaZ)}_{tag(args.betaZ)}",
                            args.split)
    table = os.path.join(run_path, "validation", f"df_epoch{args.epoch}.csv")
    df_epoch = pd.read_csv(table)

    df_train = df_epoch[df_epoch["HoldOut"].isin(DEV_LABELS)].reset_index(drop=True)
    df_val = df_epoch[df_epoch["HoldOut"] == HOLDOUT_LABEL].reset_index(drop=True)
    df_train["w"] = np.exp(df_train["hz_hat"])
    df_val["w"] = np.exp(df_val["hz_hat"])

    baseline_survival, times = breslow_baseline_survival(df_train)
    prob_train = survival_matrix(df_train, baseline_survival, times)
    prob_val = survival_matrix(df_val, baseline_survival, times)

    y_train = Surv.from_arrays(event=df_train["event"].values,
                               time=df_train["age"].values)
    y_val = Surv.from_arrays(event=df_val["event"].values,
                             time=df_val["age"].values)
    train_max_time = float(y_train["time"].max())

    grid = np.arange(MAX_FOLLOW_UP)
    summary = {
        "alphaZ": args.alphaZ,
        "betaZ": args.betaZ,
        "epoch": args.epoch,
        "n_holdout": int(len(df_val)),
        "n_holdout_events": int(df_val["event"].sum()),
        "c_index": float(concordance_index_censored(
            df_val["event"].astype(bool), df_val["age"], df_val["hz_hat"])[0]),
        "IBS_holdout": float(integrated_brier_score(y_train, y_val, prob_val, grid)),
        "IBS_development": float(integrated_brier_score(y_train, y_train, prob_train, grid)),
    }

    rng = np.random.default_rng(args.seed)
    n = len(df_val)
    cindex_scores, ibs_scores = [], []

    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_df = df_val.iloc[idx, :].reset_index(drop=True)
        events = b_df["event"].astype(bool)
        if events.sum() == 0:    # C-index is undefined without events
            continue
        durations = b_df["age"]

        lower = int(np.ceil(durations.min()))
        upper = int(np.ceil(min(durations.max(), train_max_time)))
        eval_times = np.arange(lower, upper, dtype=float)
        if len(eval_times) == 0:
            continue

        cidx = concordance_index_censored(events, durations, b_df["hz_hat"])[0]
        ibs = integrated_brier_score(
            y_train,
            Surv.from_arrays(event=events, time=durations),
            prob_val[idx][:, eval_times.astype(int)],
            eval_times,
        )
        cindex_scores.append(float(cidx))
        ibs_scores.append(float(ibs))

    cindex_scores = np.asarray(cindex_scores)
    ibs_scores = np.asarray(ibs_scores)
    q = [args.alpha_ci / 2, 1 - args.alpha_ci / 2]
    c_lo, c_hi = np.quantile(cindex_scores, q)
    i_lo, i_hi = np.quantile(ibs_scores, q)

    summary.update({
        "n_bootstrap_requested": int(args.n_bootstrap),
        "n_bootstrap_valid": int(len(cindex_scores)),
        "bootstrap_mean_c_index": float(cindex_scores.mean()),
        "bootstrap_std_c_index": float(cindex_scores.std(ddof=1)),
        "c_index_ci_lower": float(c_lo),
        "c_index_ci_upper": float(c_hi),
        "bootstrap_mean_IBS": float(ibs_scores.mean()),
        "bootstrap_std_IBS": float(ibs_scores.std(ddof=1)),
        "IBS_ci_lower": float(i_lo),
        "IBS_ci_upper": float(i_hi),
    })

    out_dir = os.path.join(run_path, "validation")
    with open(os.path.join(out_dir, "holdout_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame({"c_index": cindex_scores, "ibs": ibs_scores}).to_csv(
        os.path.join(out_dir, "bootstrap_distribution.csv"), index=False)


if __name__ == "__main__":
    main()
