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
import logging
import os
import sys
import time
from contextlib import contextmanager

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

logger = logging.getLogger("uplift")


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure root logging so every module (script + src/*) streams to console.

    Format includes a timestamp, level and logger name so each stage and any
    failure can be traced. Optionally also mirrors logs to ``log_file``.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,  # override any handler the libraries may have installed
    )
    # Keep noisy third-party libraries from flooding the console at DEBUG.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    # Route Python warnings through logging (so they are timestamped/formatted)
    # and silence a harmless, repetitive deprecation emitted from inside sklift.
    import warnings
    logging.captureWarnings(True)
    warnings.filterwarnings("ignore", message=".*stable_cumsum is deprecated.*")


@contextmanager
def log_stage(name: str):
    """Log START/DONE (with elapsed seconds) around a pipeline stage and surface
    any exception with a full traceback before re-raising it."""
    logger.info("[START] %s", name)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        logger.exception("[FAILED] %s (after %.2fs)", name, time.perf_counter() - t0)
        raise
    else:
        logger.info("[DONE]  %s (%.2fs)", name, time.perf_counter() - t0)


# Only the columns the model actually needs are read from disk. Enriched
# exports often carry dozens of extra columns and millions of rows; reading the
# whole file is the usual cause of the loader stalling / running out of memory.
COMM_DATE_COLS = ["mailing_date", "open_date", "click_date"]
COMM_NUM_COLS = [DISCOUNT_COL, "claim_sum_usd", "adult", "pax"]
COMM_CAT_COLS = ["mailing_name", "state_name"]
COMM_NEEDED = ["email"] + COMM_DATE_COLS + COMM_NUM_COLS + COMM_CAT_COLS

PURCH_DATE_COLS = ["date_booking"]
PURCH_NUM_COLS = ["claim_sum_usd", "adult", "pax"]
PURCH_CAT_COLS = ["state_name"]
PURCH_NEEDED = ["email"] + PURCH_DATE_COLS + PURCH_NUM_COLS + PURCH_CAT_COLS


def _dtype_map(num_cols, cat_cols):
    # email stays object (high cardinality -> category union across chunks is
    # costly and complicates merges); low-cardinality strings -> category.
    dmap = {c: "float32" for c in num_cols}
    dmap.update({c: "category" for c in cat_cols})
    dmap["email"] = "object"
    return dmap


def read_csv_optimized(path, needed, date_cols, num_cols, cat_cols,
                       chunksize=1_000_000, sep=","):
    """Memory-friendly CSV reader.

    - reads ONLY the needed columns (``usecols``) -> skips enriched extras;
    - applies compact dtypes (float32 / category) to cut memory;
    - parses just the required date columns;
    - streams in chunks so peak memory stays bounded, logging progress.
    """
    logger.info("reading %s (only %d needed columns, chunksize=%d) ...",
                os.path.basename(path), len(needed), chunksize)
    dmap = _dtype_map(num_cols, cat_cols)  # dates parsed separately, not here

    parts, total = [], 0
    reader = pd.read_csv(
        path, sep=sep,
        usecols=lambda c: c in set(needed),
        dtype=dmap,
        chunksize=chunksize,
        low_memory=False,
    )
    for i, ch in enumerate(reader, 1):
        for dc in date_cols:
            if dc in ch.columns:
                ch[dc] = pd.to_datetime(ch[dc], errors="coerce")
        parts.append(ch)
        total += len(ch)
        logger.info("  chunk %d: +%d rows (total=%d)", i, len(ch), total)

    if not parts:
        raise ValueError(f"{path} produced no rows.")
    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info("read done: %s, ~%.0f MB in memory", df.shape, mem_mb)
    return df


def load_or_generate(comm_path: str | None, purch_path: str | None,
                     sep: str = ",", chunksize: int = 1_000_000):
    if comm_path and purch_path and os.path.exists(comm_path) and os.path.exists(purch_path):
        logger.info("Loading real data: comm=%s, purch=%s", comm_path, purch_path)
        _check_columns_exist(comm_path, COMM_NEEDED, "communications", sep)
        _check_columns_exist(purch_path, PURCH_NEEDED, "purchases", sep)
        comm = read_csv_optimized(comm_path, COMM_NEEDED, COMM_DATE_COLS,
                                  COMM_NUM_COLS, COMM_CAT_COLS, chunksize, sep)
        purch = read_csv_optimized(purch_path, PURCH_NEEDED, PURCH_DATE_COLS,
                                   PURCH_NUM_COLS, PURCH_CAT_COLS, chunksize, sep)
    else:
        default_comm = os.path.join(DATA_DIR, "communications.csv")
        default_purch = os.path.join(DATA_DIR, "purchases.csv")
        if os.path.exists(default_comm) and os.path.exists(default_purch):
            logger.info("Loading cached synthetic data from ./%s", DATA_DIR)
            comm = read_csv_optimized(default_comm, COMM_NEEDED, COMM_DATE_COLS,
                                      COMM_NUM_COLS, COMM_CAT_COLS, chunksize, sep)
            purch = read_csv_optimized(default_purch, PURCH_NEEDED, PURCH_DATE_COLS,
                                       PURCH_NUM_COLS, PURCH_CAT_COLS, chunksize, sep)
        else:
            logger.warning("No data files found -> generating synthetic dataset "
                           "(same schema). Pass --comm/--purch to use real data.")
            os.makedirs(DATA_DIR, exist_ok=True)
            comm, purch = generate()
            comm.to_csv(default_comm, index=False)
            purch.to_csv(default_purch, index=False)
    logger.info("Loaded communications=%s, purchases=%s", comm.shape, purch.shape)
    return comm, purch


def _check_columns_exist(path, needed, label, sep=","):
    """Read only the header to fail fast (and clearly) on schema mismatches,
    without loading the whole file."""
    header = pd.read_csv(path, nrows=0, sep=sep)
    miss = [c for c in needed if c not in header.columns]
    if miss:
        raise ValueError(
            f"{label} file '{os.path.basename(path)}' is missing required "
            f"columns: {miss}. Found columns: {list(header.columns)[:30]}"
            + (" ..." if len(header.columns) > 30 else "")
        )
    logger.info("%s schema OK (%d required columns present, %d total in file).",
                label, len(needed), len(header.columns))


def enforce_cutoff(comm: pd.DataFrame, purch: pd.DataFrame):
    comm = comm.copy()
    purch = purch.copy()
    comm["mailing_date"] = pd.to_datetime(comm["mailing_date"], errors="coerce")
    purch["date_booking"] = pd.to_datetime(purch["date_booking"], errors="coerce")

    n_bad = int(comm["mailing_date"].isna().sum())
    if n_bad:
        logger.warning("%d rows have unpar. mailing_date and will be dropped by the "
                       "cutoff filter.", n_bad)

    n0 = len(comm)
    comm = comm[comm["mailing_date"] < CUTOFF].reset_index(drop=True)
    purch = purch[purch["date_booking"] < CUTOFF].reset_index(drop=True)
    logger.info("Cutoff %s: kept %d/%d mailings, %d purchases.",
                CUTOFF.date(), len(comm), n0, len(purch))
    if len(comm) == 0:
        raise ValueError(f"No mailings left before cutoff {CUTOFF.date()}. "
                         "Check mailing_date values / adjust CUTOFF.")
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


def run_pipeline(args) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    pipeline_t0 = time.perf_counter()

    # ---------------- 1) load + cutoff ----------------
    with log_stage("1/8 load data + enforce cutoff"):
        comm, purch = load_or_generate(args.comm, args.purch,
                                       sep=args.sep, chunksize=args.chunksize)
        comm, purch = enforce_cutoff(comm, purch)

        base_rate = comm["click_date"].notna().mean()
        treated_rate = comm[comm[DISCOUNT_COL] > 0]["click_date"].notna().mean()
        control_rate = comm[comm[DISCOUNT_COL] == 0]["click_date"].notna().mean()
        n_treated = int((comm[DISCOUNT_COL] > 0).sum())
        n_control = int((comm[DISCOUNT_COL] == 0).sum())
        logger.info("treatment balance: discount=%d (%.1f%%), no-discount=%d (%.1f%%)",
                    n_treated, 100 * n_treated / len(comm),
                    n_control, 100 * n_control / len(comm))
        logger.info("click_rate overall=%.4f | discount=%.4f | no-discount=%.4f | "
                    "naive ATE=%.4f", base_rate, treated_rate, control_rate,
                    treated_rate - control_rate)
        if n_treated == 0 or n_control == 0:
            raise ValueError("Uplift needs BOTH groups: mailings with AND without "
                             "a discount. One of the groups is empty.")

    # ---------------- 2) features ----------------
    with log_stage("2/8 build leakage-free features"):
        X, treatment, target, num_cols, cat_cols, meta = build_dataset(comm, purch)
        preprocessor = (num_cols, cat_cols)
        logger.info("feature matrix: %s (%d numeric + %d categorical)",
                    X.shape, len(num_cols), len(cat_cols))

    # ---------------- 3) time-based split ----------------
    with log_stage("3/8 time-based train/test split"):
        is_test = meta["mailing_date"] >= TEST_SPLIT_DATE
        X_train, X_test = X[~is_test], X[is_test]
        t_train, t_test = treatment[~is_test], treatment[is_test]
        y_train, y_test = target[~is_test], target[is_test]
        meta_test = meta[is_test]
        logger.info("split @ %s -> train=%d (clicks=%.3f), test=%d (clicks=%.3f)",
                    TEST_SPLIT_DATE.date(), len(X_train), y_train.mean(),
                    len(X_test), y_test.mean() if len(X_test) else float("nan"))
        if len(X_test) == 0:
            raise ValueError(f"Holdout is empty: no mailings on/after "
                             f"{TEST_SPLIT_DATE.date()}. Adjust TEST_SPLIT_DATE.")

    # ---------------- 4) CV model selection ----------------
    candidates = [
        ("SoloModel", "hgb"),
        ("TwoModels", "hgb"),
        ("ClassTransformation", "hgb"),
        ("SoloModel", "logreg"),
    ]
    cv_results = {}
    with log_stage("4/8 cross-validated model selection (5-fold)"):
        # On very large training sets, model SELECTION is done on a random
        # sample for speed/memory; the WINNER is later refit on the FULL train.
        Xcv, tcv, ycv = X_train, t_train, y_train
        if len(X_train) > args.max_cv_rows:
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(X_train), size=args.max_cv_rows, replace=False)
            Xcv = X_train.iloc[sample_idx]
            tcv = t_train.iloc[sample_idx]
            ycv = y_train.iloc[sample_idx]
            logger.info("train has %d rows -> CV model selection on a %d-row sample "
                        "(winner refit on full train).", len(X_train), args.max_cv_rows)
        for i, (name, base) in enumerate(candidates, 1):
            logger.info("CV candidate %d/%d: %s [%s] ...", i, len(candidates), name, base)
            cv = cross_val_uplift(name, base, preprocessor, Xcv, tcv, ycv)
            cv_results[f"{name}+{base}"] = cv
            lo, hi = cv["qini_ci95"]
            logger.info("  -> Qini=%.4f +/- %.4f (95%% CI [%.4f, %.4f]) uplift@30%%=%.4f",
                        cv["qini_mean"], cv["qini_std"], lo, hi, cv["uplift_at_30_mean"])
        best_key = max(cv_results, key=lambda k: cv_results[k]["qini_mean"])
        best_name, best_base = best_key.split("+")
        logger.info("best model by CV Qini: %s (%.4f)",
                    best_key, cv_results[best_key]["qini_mean"])

    # ---------------- 5) final fit + holdout evaluation ----------------
    with log_stage(f"5/8 fit {best_key} on train, evaluate on holdout"):
        pre = build_preprocessor(*preprocessor)
        Xtr = pre.fit_transform(X_train)
        Xte = pre.transform(X_test)
        logger.info("transformed matrices: train=%s test=%s", Xtr.shape, Xte.shape)

        model = make_uplift_model(best_name, best_base)
        model.fit(Xtr, y_train.to_numpy(), t_train.to_numpy())
        uplift_test = model.predict(Xte)
        logger.info("predicted uplift on holdout: mean=%.4f std=%.4f min=%.4f max=%.4f",
                    float(np.mean(uplift_test)), float(np.std(uplift_test)),
                    float(np.min(uplift_test)), float(np.max(uplift_test)))

        holdout = evaluate_uplift(y_test.to_numpy(), uplift_test, t_test.to_numpy())
        boot_lo, boot_hi, boot_mean = bootstrap_qini_ci(
            y_test.to_numpy(), uplift_test, t_test.to_numpy()
        )
        resp_auc = response_auc(preprocessor, X_train, y_train, X_test, y_test)
        logger.info("HOLDOUT Qini=%.4f (bootstrap 95%% CI [%.4f, %.4f])",
                    holdout.qini, boot_lo, boot_hi)
        logger.info("HOLDOUT AUUC=%.4f | uplift@30%%=%.4f | response ROC-AUC=%.4f",
                    holdout.auuc, holdout.uplift_at_30, resp_auc)

    # ---------------- 6) plots ----------------
    with log_stage("6/8 save plots"):
        try:
            _plot_qini(y_test.to_numpy(), uplift_test, t_test.to_numpy(),
                       os.path.join(OUT_DIR, "qini_curve.png"))
            _plot_uplift_by_percentile(y_test.to_numpy(), uplift_test, t_test.to_numpy(),
                                       os.path.join(OUT_DIR, "uplift_by_percentile.png"))
            logger.info("saved plots -> %s/qini_curve.png, %s/uplift_by_percentile.png",
                        OUT_DIR, OUT_DIR)
        except Exception:  # plotting must never break the pipeline
            logger.exception("plotting failed (continuing without plots)")

    # ---------------- 7) per-client scores + decile lift table ----------------
    with log_stage("7/8 per-client scores + decile lift table"):
        scores = meta_test.copy()
        scores["predicted_uplift"] = uplift_test
        scores["clicked"] = y_test.to_numpy()
        scores["had_discount"] = t_test.to_numpy()
        client_scores = (
            scores.groupby("email")["predicted_uplift"].mean()
            .sort_values(ascending=False)
            .reset_index()
            .rename(columns={"predicted_uplift": "discount_responsiveness"})
        )
        client_scores.to_csv(os.path.join(OUT_DIR, "client_uplift_scores.csv"), index=False)
        logger.info("wrote %d per-client uplift scores -> %s/client_uplift_scores.csv",
                    len(client_scores), OUT_DIR)

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
        logger.info("decile lift table (observed click uplift by predicted-uplift decile):\n%s",
                    lift_table.to_string(index=False))

    # ---------------- 8) report ----------------
    with log_stage("8/8 write metrics report"):
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
        logger.info("saved metrics report -> %s/metrics_report.json", OUT_DIR)
        logger.info("FINAL REPORT:\n%s", json.dumps(report, indent=2, ensure_ascii=False))

    logger.info("PIPELINE FINISHED OK in %.2fs", time.perf_counter() - pipeline_t0)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm", default=None, help="communications CSV path")
    parser.add_argument("--purch", default=None, help="purchases CSV path")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="console log verbosity (default INFO)")
    parser.add_argument("--log-file", default=None,
                        help="optional path to also write logs to a file")
    parser.add_argument("--max-cv-rows", type=int, default=300_000,
                        help="cap rows used for CV model selection on big data "
                             "(winner is always refit on the full train set)")
    parser.add_argument("--sep", default=",",
                        help="CSV separator of the input files (default ',')")
    parser.add_argument("--chunksize", type=int, default=1_000_000,
                        help="rows per chunk while streaming large CSVs")
    args = parser.parse_args()

    setup_logging(args.log_level, args.log_file)
    logger.info("uplift pipeline starting | cutoff=%s | holdout>=%s | log-level=%s",
                CUTOFF.date(), TEST_SPLIT_DATE.date(), args.log_level)
    try:
        run_pipeline(args)
    except Exception:
        logger.exception("PIPELINE ABORTED due to an unhandled error (see traceback above)")
        sys.exit(1)


if __name__ == "__main__":
    main()
