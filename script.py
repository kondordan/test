"""End-to-end uplift pipeline: predict how clients interact with discounts.

Task recap
----------
We have a client-communication history and a purchases table. We must build an
uplift ML model that predicts how clients (by e-mail) react to discounts, with
**click_rate** as the target metric, using only data strictly before
**01.08.2026**, while following good ML practice (no overfitting), and report an
estimate of the forecast quality.

Pipeline
--------
1. Load data (synthetic generator if no CSVs are present -- same schema).
2. Enforce the 01.08.2026 cutoff.
3. Leakage-free feature engineering.
4. Time-based train/test split (most recent period held out).
5. Cross-validated model selection across S-/T-/Class-transformation learners.
6. Train the winner, evaluate on the time-based holdout (Qini / AUUC / uplift@30%)
   with a bootstrap 95% CI -> the forecast-quality estimate.
7. Persist plots, a metrics report and per-client uplift scores.

Usage
-----
    python script.py                       # uses/creates data in ./data
    python script.py --comm a.csv --purch b.csv   # bring your own data
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_generation import DISCOUNT_COL, generate  # noqa: E402
from features import build_dataset  # noqa: E402
from uplift_model import (  # noqa: E402
    bootstrap_qini_ci,
    build_preprocessor,
    cross_val_uplift,
    evaluate_uplift,
    make_uplift_model,
    response_auc,
)
from sklift.metrics import (  # noqa: E402
    perfect_qini_curve,
    qini_curve,
    uplift_by_percentile,
)

CUTOFF = pd.Timestamp("2026-08-01")
TEST_SPLIT_DATE = pd.Timestamp("2026-04-01")  # last ~4 months -> time-based holdout
DATA_DIR = "data"
OUT_DIR = "outputs"


def load_or_generate(comm_path: str | None, purch_path: str | None):
    if comm_path and purch_path and os.path.exists(comm_path) and os.path.exists(purch_path):
        print(f"Loading real data: {comm_path}, {purch_path}")
        comm = pd.read_csv(comm_path)
        purch = pd.read_csv(purch_path)
    else:
        default_comm = os.path.join(DATA_DIR, "communications.csv")
        default_purch = os.path.join(DATA_DIR, "purchases.csv")
        if os.path.exists(default_comm) and os.path.exists(default_purch):
            print("Loading cached synthetic data from ./data")
            comm = pd.read_csv(default_comm)
            purch = pd.read_csv(default_purch)
        else:
            print("No data found -> generating synthetic dataset (same schema).")
            os.makedirs(DATA_DIR, exist_ok=True)
            comm, purch = generate()
            comm.to_csv(default_comm, index=False)
            purch.to_csv(default_purch, index=False)
    return comm, purch


def enforce_cutoff(comm: pd.DataFrame, purch: pd.DataFrame):
    comm = comm.copy()
    purch = purch.copy()
    comm["mailing_date"] = pd.to_datetime(comm["mailing_date"], errors="coerce")
    purch["date_booking"] = pd.to_datetime(purch["date_booking"], errors="coerce")

    n0 = len(comm)
    comm = comm[comm["mailing_date"] < CUTOFF].reset_index(drop=True)
    purch = purch[purch["date_booking"] < CUTOFF].reset_index(drop=True)
    print(f"Cutoff {CUTOFF.date()}: kept {len(comm)}/{n0} mailings, {len(purch)} purchases.")
    return comm, purch


def _plot_qini(y_true, uplift, treatment, path):
    """Qini curve vs. random and perfect baselines (sklift.viz is incompatible
    with sklearn>=1.4, so we draw it ourselves from the metric curves)."""
    x_act, y_act = qini_curve(y_true, uplift, treatment)
    x_prf, y_prf = perfect_qini_curve(y_true, treatment)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_act, y_act, label="model", color="#1f77b4", lw=2)
    ax.plot(x_prf, y_prf, label="perfect", color="#2ca02c", ls="--", lw=1.5)
    ax.plot([0, x_act[-1]], [0, y_act[-1]], label="random", color="grey", ls=":")
    ax.set_xlabel("Number of targeted clients")
    ax.set_ylabel("Cumulative incremental clicks")
    ax.set_title("Qini curve (time-based holdout)")
    ax.legend()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_uplift_by_percentile(y_true, uplift, treatment, path):
    df = uplift_by_percentile(y_true, uplift, treatment, strategy="overall", bins=10)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(df)), df["uplift"].to_numpy(), color="#1f77b4")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index.astype(str), rotation=45, ha="right")
    ax.set_xlabel("Predicted-uplift percentile (high -> low)")
    ax.set_ylabel("Observed click uplift")
    ax.set_title("Observed uplift by predicted-uplift percentile (holdout)")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm", default=None, help="communications CSV path")
    parser.add_argument("--purch", default=None, help="purchases CSV path")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- 1) load + cutoff ----------------
    comm, purch = load_or_generate(args.comm, args.purch)
    comm, purch = enforce_cutoff(comm, purch)

    base_rate = comm["click_date"].notna().mean()
    treated_rate = comm[comm[DISCOUNT_COL] > 0]["click_date"].notna().mean()
    control_rate = comm[comm[DISCOUNT_COL] == 0]["click_date"].notna().mean()
    print(f"\nclick_rate overall={base_rate:.4f} | with discount={treated_rate:.4f} "
          f"| no discount={control_rate:.4f} | naive ATE={treated_rate - control_rate:.4f}")

    # ---------------- 2) features ----------------
    X, treatment, target, num_cols, cat_cols, meta = build_dataset(comm, purch)
    preprocessor = (num_cols, cat_cols)
    print(f"\nFeatures: {len(num_cols)} numeric + {len(cat_cols)} categorical "
          f"-> {X.shape}")

    # ---------------- 3) time-based split ----------------
    is_test = meta["mailing_date"] >= TEST_SPLIT_DATE
    X_train, X_test = X[~is_test], X[is_test]
    t_train, t_test = treatment[~is_test], treatment[is_test]
    y_train, y_test = target[~is_test], target[is_test]
    meta_test = meta[is_test]
    print(f"Time split @ {TEST_SPLIT_DATE.date()}: train={len(X_train)} test={len(X_test)}")

    # ---------------- 4) CV model selection ----------------
    candidates = [
        ("SoloModel", "hgb"),
        ("TwoModels", "hgb"),
        ("ClassTransformation", "hgb"),
        ("SoloModel", "logreg"),
    ]
    print("\n=== Cross-validated model selection (5-fold, train period only) ===")
    cv_results = {}
    for name, base in candidates:
        cv = cross_val_uplift(name, base, preprocessor, X_train, t_train, y_train)
        cv_results[f"{name}+{base}"] = cv
        lo, hi = cv["qini_ci95"]
        print(f"  {name:>20s} [{base:>6s}]  Qini={cv['qini_mean']:.4f} "
              f"+/- {cv['qini_std']:.4f}  (95% CI [{lo:.4f}, {hi:.4f}])  "
              f"uplift@30%={cv['uplift_at_30_mean']:.4f}")

    best_key = max(cv_results, key=lambda k: cv_results[k]["qini_mean"])
    best_name, best_base = best_key.split("+")
    print(f"\nBest model by CV Qini: {best_key}")

    # ---------------- 5) final fit on train, eval on time holdout ----------------
    pre = build_preprocessor(*preprocessor)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)

    model = make_uplift_model(best_name, best_base)
    model.fit(Xtr, y_train.to_numpy(), t_train.to_numpy())
    uplift_test = model.predict(Xte)

    holdout = evaluate_uplift(y_test.to_numpy(), uplift_test, t_test.to_numpy())
    boot_lo, boot_hi, boot_mean = bootstrap_qini_ci(
        y_test.to_numpy(), uplift_test, t_test.to_numpy()
    )
    resp_auc = response_auc(preprocessor, X_train, y_train, X_test, y_test)

    print("\n=== Forecast quality on TIME-BASED HOLDOUT ===")
    print(f"  Qini AUC        : {holdout.qini:.4f}  "
          f"(bootstrap 95% CI [{boot_lo:.4f}, {boot_hi:.4f}])")
    print(f"  AUUC            : {holdout.auuc:.4f}")
    print(f"  Uplift @ top 30%: {holdout.uplift_at_30:.4f}  "
          f"(extra click_rate vs random targeting)")
    print(f"  Response ROC-AUC: {resp_auc:.4f}  (secondary click-prediction diagnostic)")

    # ---------------- 6) plots ----------------
    try:
        _plot_qini(y_test.to_numpy(), uplift_test, t_test.to_numpy(),
                   os.path.join(OUT_DIR, "qini_curve.png"))
        _plot_uplift_by_percentile(y_test.to_numpy(), uplift_test, t_test.to_numpy(),
                                   os.path.join(OUT_DIR, "uplift_by_percentile.png"))
        print(f"\nSaved plots to {OUT_DIR}/qini_curve.png, {OUT_DIR}/uplift_by_percentile.png")
    except Exception as e:  # plotting must never break the pipeline
        print(f"[warn] plotting failed: {e}")

    # ---------------- 7) per-client scores + segments ----------------
    scores = meta_test.copy()
    scores["predicted_uplift"] = uplift_test
    scores["clicked"] = y_test.to_numpy()
    scores["had_discount"] = t_test.to_numpy()
    # aggregate to client level: mean predicted discount-responsiveness
    client_scores = (
        scores.groupby("email")["predicted_uplift"].mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"predicted_uplift": "discount_responsiveness"})
    )
    client_scores.to_csv(os.path.join(OUT_DIR, "client_uplift_scores.csv"), index=False)

    # Validate that high-score clients really react more (decile lift table).
    scores["uplift_decile"] = pd.qcut(scores["predicted_uplift"].rank(method="first"),
                                      10, labels=False)
    rows = []
    for d, g in scores.groupby("uplift_decile"):
        tr = g[g["had_discount"] == 1]["clicked"].mean()
        ct = g[g["had_discount"] == 0]["clicked"].mean()
        rows.append({
            "decile": int(d),
            "n": len(g),
            "click_rate_discount": round(float(tr), 4) if len(g[g["had_discount"] == 1]) else None,
            "click_rate_no_discount": round(float(ct), 4) if len(g[g["had_discount"] == 0]) else None,
            "observed_uplift": (round(float(tr - ct), 4)
                                if len(g[g["had_discount"] == 1]) and len(g[g["had_discount"] == 0])
                                else None),
            "mean_pred_uplift": round(float(g["predicted_uplift"].mean()), 4),
        })
    lift_table = pd.DataFrame(rows)
    lift_table.to_csv(os.path.join(OUT_DIR, "decile_lift_table.csv"), index=False)
    print("\nDecile lift table (observed click uplift by predicted-uplift decile):")
    print(lift_table.to_string(index=False))

    # ---------------- 8) report ----------------
    report = {
        "cutoff": str(CUTOFF.date()),
        "n_mailings": int(len(comm)),
        "click_rate_overall": round(float(base_rate), 4),
        "click_rate_with_discount": round(float(treated_rate), 4),
        "click_rate_no_discount": round(float(control_rate), 4),
        "naive_ate": round(float(treated_rate - control_rate), 4),
        "best_model": best_key,
        "cv_qini_mean": round(cv_results[best_key]["qini_mean"], 4),
        "cv_qini_ci95": [round(v, 4) for v in cv_results[best_key]["qini_ci95"]],
        "holdout": {k: round(v, 4) for k, v in holdout.as_dict().items()},
        "holdout_qini_bootstrap_ci95": [round(boot_lo, 4), round(boot_hi, 4)],
        "response_roc_auc": round(resp_auc, 4),
    }
    with open(os.path.join(OUT_DIR, "metrics_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved metrics report -> {OUT_DIR}/metrics_report.json")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
