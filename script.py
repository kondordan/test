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
    choose_uplift_threshold,
    cross_val_uplift,
    evaluate_uplift,
    predict_ensemble,
    response_auc,
    save_ensemble,
    train_ensemble,
)
from sklift.metrics import (  # noqa: E402
    perfect_qini_curve,
    qini_curve,
    uplift_by_percentile,
)

CUTOFF = pd.Timestamp("2026-06-01")  # train on mailings/purchases STRICTLY before this
TEST_SPLIT_DATE = pd.Timestamp("2026-04-01")  # last ~2 months -> time-based holdout
DATA_DIR = "data"
OUT_DIR = "outputs"
MODEL_DIR = "models"

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

# date_begin is the trip START date. It is NOT used as a model feature, but is
# needed to exclude clients who booked before the cutoff yet have not travelled
# yet (trip starts on/after the cutoff) from the send-recommendation list.
PURCH_DATE_COLS = ["date_booking", "date_begin"]
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


def _estimate_total_rows(path, sep=",") -> int:
    """Cheaply estimate the row count from file size and the average length of
    the first ~2000 data lines (reads only a few KB)."""
    size = os.path.getsize(path)
    lengths = []
    with open(path, "rb") as f:
        f.readline()  # header
        for _ in range(2000):
            line = f.readline()
            if not line:
                break
            lengths.append(len(line))
    if not lengths:
        return 0
    avg = sum(lengths) / len(lengths)
    return max(1, int(size / avg))


def read_csv_optimized(path, needed, date_cols, num_cols, cat_cols,
                       chunksize=1_000_000, sep=",",
                       sample_rows=0, sample_frac=None,
                       email_filter=None, seed=42):
    """Memory-friendly CSV reader for very large files.

    - reads ONLY the needed columns (``usecols``) -> skips enriched extras;
    - compact dtypes (float32 / category) to cut memory;
    - streams in chunks; **samples rows on the fly** so only the kept subset is
      ever held in memory (essential for 100M+ row files);
    - optional ``email_filter`` keeps only rows for a given set of clients;
    - parses dates only on the (smaller) kept rows.

    Sampling fraction is taken from ``sample_frac`` if given, else derived from
    ``sample_rows`` (target row count) via an estimated total. ``sample_rows<=0``
    and ``sample_frac=None`` => read everything.
    """
    rng = np.random.default_rng(seed)
    frac = None
    if sample_frac is not None:
        frac = float(sample_frac)
    elif sample_rows and sample_rows > 0:
        total_est = _estimate_total_rows(path, sep)
        frac = min(1.0, sample_rows / total_est) if total_est else 1.0
        logger.info("estimated ~%d rows in %s -> sampling fraction %.5f "
                    "(target ~%d rows)", total_est, os.path.basename(path),
                    frac, sample_rows)
    if frac is not None and frac >= 1.0:
        frac = None  # no point sampling

    logger.info("reading %s (only %d needed columns, chunksize=%d, sampling=%s) ...",
                os.path.basename(path), len(needed), chunksize,
                f"{frac:.5f}" if frac else "off")
    dmap = _dtype_map(num_cols, cat_cols)
    if email_filter is not None:
        email_filter = np.asarray(list(email_filter))

    parts, total_read, kept = [], 0, 0
    reader = pd.read_csv(
        path, sep=sep,
        usecols=lambda c: c in set(needed),
        dtype=dmap,
        chunksize=chunksize,
        low_memory=False,
    )
    for i, ch in enumerate(reader, 1):
        total_read += len(ch)
        if email_filter is not None:
            ch = ch[ch["email"].isin(email_filter)]
        if frac is not None and len(ch):
            ch = ch[rng.random(len(ch)) < frac]
        if len(ch):
            ch = ch.copy()
            for dc in date_cols:
                if dc in ch.columns:
                    ch[dc] = pd.to_datetime(ch[dc], errors="coerce")
            parts.append(ch)
            kept += len(ch)
        if i % 20 == 0:
            logger.info("  ... read %d rows, kept %d", total_read, kept)

    if not parts:
        raise ValueError(f"{path}: no rows left after filtering/sampling "
                         f"(read {total_read} rows).")
    df = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
    del parts
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info("read done: %s (from %d source rows), ~%.0f MB in memory",
                df.shape, total_read, mem_mb)
    return df


def load_or_generate(comm_path: str | None, purch_path: str | None,
                     sep: str = ",", chunksize: int = 1_000_000,
                     sample_rows: int = 0, sample_frac: float | None = None):
    def _read_pair(comm_p, purch_p, label):
        logger.info("%s: comm=%s, purch=%s", label, comm_p, purch_p)
        _check_columns_exist(comm_p, COMM_NEEDED, "communications", sep)
        _check_columns_exist(purch_p, PURCH_NEEDED, "purchases", sep)
        # 1) read (and sample) the communications table
        comm = read_csv_optimized(comm_p, COMM_NEEDED, COMM_DATE_COLS,
                                  COMM_NUM_COLS, COMM_CAT_COLS, chunksize, sep,
                                  sample_rows=sample_rows, sample_frac=sample_frac)
        # 2) read purchases ONLY for the sampled clients (full history, bounded
        #    memory) -- purchase aggregates must stay complete per client.
        emails = set(comm["email"].unique())
        logger.info("filtering purchases to %d sampled clients ...", len(emails))
        purch = read_csv_optimized(purch_p, PURCH_NEEDED, PURCH_DATE_COLS,
                                   PURCH_NUM_COLS, PURCH_CAT_COLS, chunksize, sep,
                                   email_filter=emails)
        return comm, purch

    if comm_path and purch_path and os.path.exists(comm_path) and os.path.exists(purch_path):
        comm, purch = _read_pair(comm_path, purch_path, "Loading real data")
    else:
        default_comm = os.path.join(DATA_DIR, "communications.csv")
        default_purch = os.path.join(DATA_DIR, "purchases.csv")
        if os.path.exists(default_comm) and os.path.exists(default_purch):
            comm, purch = _read_pair(default_comm, default_purch,
                                     "Loading cached synthetic data")
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
    # Avoid full-frame .copy() here to keep peak memory low on big data; the
    # boolean filtering below already returns new frames. Dates were parsed at
    # read time, but re-coerce defensively (cheap on already-datetime columns).
    if not pd.api.types.is_datetime64_any_dtype(comm["mailing_date"]):
        comm["mailing_date"] = pd.to_datetime(comm["mailing_date"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(purch["date_booking"]):
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
                                       sep=args.sep, chunksize=args.chunksize,
                                       sample_rows=args.sample_rows,
                                       sample_frac=args.sample_frac)
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

    # ---------------- 3) train/test split (time-based, with fallback) ----------------
    with log_stage("3/10 train/test split"):
        (X_train, X_test, t_train, t_test, y_train, y_test, meta_test,
         split_kind) = make_holdout_split(X, treatment, target, meta)
        logger.info("split=%s -> train=%d (treated=%d, clicks=%.4f), "
                    "test=%d (treated=%d, clicks=%.4f)",
                    split_kind, len(X_train), int(t_train.sum()), y_train.mean(),
                    len(X_test), int(t_test.sum()),
                    y_test.mean() if len(X_test) else float("nan"))
        # Keep the raw (unbalanced) train; balancing (and any ratio sweep) is
        # applied inside model selection below.
        X_train_raw, t_train_raw, y_train_raw = X_train, t_train, y_train

    # ---------------- 4) CV model selection ----------------
    candidates = [
        ("SoloModel", "hgb"),
        ("TwoModels", "hgb"),
        ("ClassTransformation", "hgb"),
        ("SoloModel", "logreg"),
    ]
    # Build the grid of balance ratios to consider.
    if not args.balance:
        ratio_grid = [None]                       # no balancing
    elif args.balance_ratios:
        ratio_grid = [float(x) for x in args.balance_ratios.split(",") if x.strip()]
    else:
        ratio_grid = [args.balance_ratio]          # single ratio, no sweep

    cv_results = {}
    sweep_rows = []
    with log_stage("4/10 cross-validated selection (model x balance-ratio)"):
        rng = np.random.default_rng(42)
        for ratio in ratio_grid:
            # balance the raw train at this ratio (None = keep natural)
            if ratio is None:
                Xb, tb, yb = X_train_raw, t_train_raw, y_train_raw
            else:
                Xb, tb, yb = balance_training_set(X_train_raw, t_train_raw,
                                                  y_train_raw, ratio)
            # cap rows used for CV (selection only; winner refit on full train)
            if len(Xb) > args.max_cv_rows:
                idx = rng.choice(len(Xb), size=args.max_cv_rows, replace=False)
                Xb, tb, yb = Xb.iloc[idx], tb.iloc[idx], yb.iloc[idx]
            for name, base in candidates:
                cv = cross_val_uplift(name, base, preprocessor, Xb, tb, yb)
                key = (ratio, name, base)
                cv_results[key] = cv
                lo, hi = cv["qini_ci95"]
                logger.info("  ratio=%s %s[%s]: Qini=%.4f +/- %.4f "
                            "(95%% CI [%.4f, %.4f]) uplift@30%%=%.4f",
                            ratio, name, base, cv["qini_mean"], cv["qini_std"],
                            lo, hi, cv["uplift_at_30_mean"])
                sweep_rows.append({"balance_ratio": ratio, "model": f"{name}+{base}",
                                   "cv_qini_mean": round(cv["qini_mean"], 4),
                                   "cv_qini_std": round(cv["qini_std"], 4)})

        best_key_tuple = max(cv_results, key=lambda k: cv_results[k]["qini_mean"])
        best_ratio, best_name, best_base = best_key_tuple
        best_key = f"{best_name}+{best_base}"
        if len(ratio_grid) > 1:
            logger.info("ratio sweep results:\n%s",
                        pd.DataFrame(sweep_rows).to_string(index=False))
        logger.info("BEST by CV Qini: model=%s, balance_ratio=%s (Qini=%.4f)",
                    best_key, best_ratio, cv_results[best_key_tuple]["qini_mean"])

        # Materialise the FULL training set at the chosen ratio for the ensemble.
        if best_ratio is None:
            X_train, t_train, y_train = X_train_raw, t_train_raw, y_train_raw
        else:
            X_train, t_train, y_train = balance_training_set(
                X_train_raw, t_train_raw, y_train_raw, best_ratio)
        best_cv = cv_results[best_key_tuple]

    # ---------------- 5) train 3-model ensemble + holdout evaluation ----------------
    with log_stage(f"5/10 train {args.n_models}-model ensemble ({best_key}) + holdout eval"):
        members = train_ensemble(best_name, best_base, preprocessor,
                                 X_train, t_train, y_train,
                                 n_models=args.n_models)
        uplift_test = predict_ensemble(members, X_test)
        logger.info("ensemble uplift on holdout: mean=%.4f std=%.4f min=%.4f max=%.4f",
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

        # Optimal send-threshold: cutoff on predicted uplift that maximises
        # cumulative incremental clicks on the holdout (peak of the uplift curve).
        if args.send_threshold is not None:
            send_threshold = float(args.send_threshold)
            thr_info = {"targeted_fraction": float("nan")}
            logger.info("send threshold set manually = %.5f", send_threshold)
        else:
            send_threshold, thr_info = choose_uplift_threshold(
                y_test.to_numpy(), uplift_test, t_test.to_numpy())
            logger.info("optimal send threshold = %.5f (would target ~%.1f%% of "
                        "the holdout; previously threshold was 0)",
                        send_threshold, 100 * thr_info.get("targeted_fraction", float("nan")))

    # ---------------- 6) persist the ensemble to disk ----------------
    with log_stage("6/10 save trained ensemble to disk"):
        model_path = os.path.join(MODEL_DIR, "uplift_ensemble.joblib")
        save_ensemble(model_path, members, best_name, best_base,
                      num_cols, cat_cols,
                      extra={"cutoff": str(CUTOFF.date()),
                             "test_split_date": str(TEST_SPLIT_DATE.date()),
                             "holdout_qini": round(holdout.qini, 4),
                             "send_threshold": send_threshold})

    # ---------------- 7) plots ----------------
    with log_stage("7/10 save plots"):
        try:
            _plot_qini(y_test.to_numpy(), uplift_test, t_test.to_numpy(),
                       os.path.join(OUT_DIR, "qini_curve.png"))
            _plot_uplift_by_percentile(y_test.to_numpy(), uplift_test, t_test.to_numpy(),
                                       os.path.join(OUT_DIR, "uplift_by_percentile.png"))
            logger.info("saved plots -> %s/qini_curve.png, %s/uplift_by_percentile.png",
                        OUT_DIR, OUT_DIR)
        except Exception:  # plotting must never break the pipeline
            logger.exception("plotting failed (continuing without plots)")

    # ---------------- 8) per-client scores + decile lift table ----------------
    with log_stage("8/10 per-client scores + decile lift table"):
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

    # ---------------- 9) send recommendations for every email ----------------
    with log_stage("9/10 build send-recommendations per email"):
        reco = build_send_recommendations(members, X, meta, comm, purch,
                                          threshold=send_threshold)
        reco_path = os.path.join(OUT_DIR, "send_recommendations.csv")
        reco.to_csv(reco_path, index=False)
        n_send = int(reco["recommend_send"].sum())
        n_excl = int(reco["has_upcoming_trip"].sum())
        logger.info("recommendations for %d emails -> %s", len(reco), reco_path)
        logger.info("  recommend SEND uplift mailing: %d (%.1f%%)",
                    n_send, 100 * n_send / max(1, len(reco)))
        logger.info("  excluded (bought but not yet travelled by cutoff): %d", n_excl)

    # ---------------- 10) report ----------------
    with log_stage("10/10 write metrics report"):
        report = {
            "cutoff": str(CUTOFF.date()),
            "n_mailings": int(len(comm)),
            "click_rate_overall": round(float(base_rate), 4),
            "click_rate_with_discount": round(float(treated_rate), 4),
            "click_rate_no_discount": round(float(control_rate), 4),
            "naive_ate": round(float(treated_rate - control_rate), 4),
            "best_model": best_key,
            "holdout_split": split_kind,
            "balanced_training": bool(args.balance),
            "balance_ratio": best_ratio,
            "balance_ratio_tuned": bool(args.balance and args.balance_ratios),
            "n_train_rows": int(len(X_train)),
            "n_ensemble_models": args.n_models,
            "cv_qini_mean": round(best_cv["qini_mean"], 4),
            "cv_qini_ci95": [round(v, 4) for v in best_cv["qini_ci95"]],
            "holdout": {k: round(v, 4) for k, v in holdout.as_dict().items()},
            "holdout_qini_bootstrap_ci95": [round(boot_lo, 4), round(boot_hi, 4)],
            "response_roc_auc": round(resp_auc, 4),
            "send_threshold": round(float(send_threshold), 5),
            "send_threshold_auto": bool(args.send_threshold is None),
            "n_emails_scored": int(len(reco)),
            "n_recommended_send": int(reco["recommend_send"].sum()),
            "n_excluded_upcoming_trip": int(reco["has_upcoming_trip"].sum()),
        }
        with open(os.path.join(OUT_DIR, "metrics_report.json"), "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("saved metrics report -> %s/metrics_report.json", OUT_DIR)
        logger.info("FINAL REPORT:\n%s", json.dumps(report, indent=2, ensure_ascii=False))

    logger.info("PIPELINE FINISHED OK in %.2fs%s",
                time.perf_counter() - pipeline_t0, _peak_mem_str())
    return report


MIN_TEST_TREATED = 50  # need at least this many treated rows in the holdout to
                       # evaluate uplift (both arms required by Qini/uplift metrics)


def make_holdout_split(X, treatment, target, meta):
    """Prefer a TIME-BASED (out-of-time) holdout. But discounts are rare and can
    be clustered in time -- if the recent window contains no/too-few treated
    mailings, uplift is not evaluable there. In that case fall back to a
    TREATMENT-stratified random holdout so evaluation is possible, with a clear
    warning. Feature engineering stays leakage-free either way.

    Returns (X_train, X_test, t_train, t_test, y_train, y_test, meta_test, kind).
    """
    is_test = (meta["mailing_date"] >= TEST_SPLIT_DATE).to_numpy()
    t = treatment.to_numpy()

    time_ok = (
        is_test.any() and (~is_test).any()
        and int(t[is_test].sum()) >= MIN_TEST_TREATED
        and int((t[is_test] == 0).sum()) > 0
        and int(t[~is_test].sum()) > 0          # train must also have treated
    )
    if time_ok:
        sel = is_test
        kind = "time-based"
    else:
        n_recent_treated = int(t[is_test].sum()) if is_test.any() else 0
        logger.warning(
            "time-based holdout (>= %s) is not usable for uplift eval "
            "(treated in window=%d < %d, or one arm missing). Discounts are rare/"
            "time-clustered -> falling back to a treatment-stratified random "
            "holdout. (Features remain leakage-free.)",
            TEST_SPLIT_DATE.date(), n_recent_treated, MIN_TEST_TREATED)
        from sklearn.model_selection import train_test_split
        strat = (treatment.astype(str) + "_" + target.astype(str)).to_numpy()
        idx = np.arange(len(X))
        try:
            _, test_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                           stratify=strat)
        except ValueError:
            # extremely rare class for stratification -> stratify on treatment only
            _, test_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                           stratify=t)
        sel = np.zeros(len(X), dtype=bool)
        sel[test_idx] = True
        kind = "stratified-random (time-based unusable)"

    X_train, X_test = X[~sel], X[sel]
    t_train, t_test = treatment[~sel], treatment[sel]
    y_train, y_test = target[~sel], target[sel]
    meta_test = meta[sel]

    if int(t_test.to_numpy().sum()) < MIN_TEST_TREATED or int(t_train.to_numpy().sum()) == 0:
        logger.warning("holdout still has few treated rows (test treated=%d). "
                       "Uplift metrics will be noisy; rely also on CV metrics.",
                       int(t_test.to_numpy().sum()))
    return X_train, X_test, t_train, t_test, y_train, y_test, meta_test, kind


def balance_training_set(X_train, t_train, y_train, ratio: float, seed: int = 42):
    """Undersample the control (no-promo) class of the TRAINING set.

    Keeps ALL promo (discount) mailings and ``ratio`` x as many randomly chosen
    no-promo mailings. Useful when promo is very rare (here ~0.8%): a balanced
    training set lets the learners actually see the treatment signal. Applied to
    TRAINING data only -- the holdout is left in its natural distribution so the
    reported metrics stay honest.
    """
    t = t_train.to_numpy()
    treated_pos = np.where(t == 1)[0]
    control_pos = np.where(t == 0)[0]
    n_treated = len(treated_pos)
    if n_treated == 0:
        logger.warning("balance: no promo rows in train -> skipping balancing.")
        return X_train, t_train, y_train

    n_keep = int(round(ratio * n_treated))
    rng = np.random.default_rng(seed)
    if n_keep >= len(control_pos):
        logger.warning("balance: requested %d no-promo rows but only %d available "
                       "-> keeping all no-promo (ratio effectively %.2f).",
                       n_keep, len(control_pos), len(control_pos) / max(1, n_treated))
        keep_control = control_pos
    else:
        keep_control = rng.choice(control_pos, size=n_keep, replace=False)

    keep = np.sort(np.concatenate([treated_pos, keep_control]))
    Xb, tb, yb = X_train.iloc[keep], t_train.iloc[keep], y_train.iloc[keep]
    logger.info("balanced TRAIN: promo=%d, no-promo=%d (ratio 1:%.2f), total=%d "
                "(was %d)", n_treated, len(keep_control),
                len(keep_control) / max(1, n_treated), len(keep), len(X_train))
    return Xb, tb, yb


def emails_with_upcoming_trip(purch: pd.DataFrame, cutoff: pd.Timestamp) -> set:
    """Clients who booked a tour before the cutoff but whose trip starts on/after
    the cutoff (i.e. they have NOT travelled yet). They already have an upcoming
    trip, so an uplift discount mailing is pointless -> exclude them.

    All kept purchases already satisfy ``date_booking < cutoff`` (enforced
    earlier); here we additionally require ``date_begin >= cutoff``.
    """
    if "date_begin" not in purch.columns:
        logger.warning("purchases has no date_begin -> cannot detect upcoming "
                       "trips; no clients excluded on that basis.")
        return set()
    db = pd.to_datetime(purch["date_begin"], errors="coerce")
    upcoming = purch.loc[db >= cutoff, "email"].astype(str).unique()
    return set(upcoming)


def build_send_recommendations(members, X, meta, comm, purch,
                               threshold: float = 0.0) -> pd.DataFrame:
    """For every e-mail in the dataset decide whether to send an uplift mailing.

    - score every mailing row with the ensemble, average per e-mail to get the
      client's discount-responsiveness (uplift);
    - exclude clients who already bought a tour but have not travelled yet as of
      the cutoff;
    - recommend sending when uplift >= ``threshold`` (optimal cutoff chosen on
      the holdout, see choose_uplift_threshold) and the client is not excluded.
    """
    uplift_all = predict_ensemble(members, X)
    df = pd.DataFrame({
        "email": meta["email"].astype(str).to_numpy(),
        "predicted_uplift": uplift_all,
    })
    per_email = (df.groupby("email")["predicted_uplift"].mean()
                 .reset_index()
                 .rename(columns={"predicted_uplift": "mean_predicted_uplift"}))

    upcoming = emails_with_upcoming_trip(purch, CUTOFF)
    per_email["has_upcoming_trip"] = per_email["email"].isin(upcoming)
    per_email["recommend_send"] = (
        (~per_email["has_upcoming_trip"])
        & (per_email["mean_predicted_uplift"] >= threshold)
    )
    per_email = per_email.sort_values(
        ["recommend_send", "mean_predicted_uplift"], ascending=[False, False]
    ).reset_index(drop=True)
    per_email["mean_predicted_uplift"] = per_email["mean_predicted_uplift"].round(5)
    return per_email


def _peak_mem_str() -> str:
    """Best-effort peak-RSS reporter (Unix) for the final log line."""
    try:
        import resource
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return f" | peak memory ~{peak_kb / 1024:.0f} MB"
    except Exception:
        return ""


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
    parser.add_argument("--sample-rows", type=int, default=3_000_000,
                        help="target number of mailing rows to keep (uniform "
                             "sample-on-read for huge files). 0 = use all rows.")
    parser.add_argument("--sample-frac", type=float, default=None,
                        help="explicit sampling fraction for the communications "
                             "file (overrides --sample-rows)")
    parser.add_argument("--n-models", type=int, default=3,
                        help="number of models in the ensemble, each trained on "
                             "a different sampled subset (default 3)")
    parser.add_argument("--balance", action="store_true",
                        help="balance the TRAINING set: keep all promo (discount) "
                             "mailings + (balance-ratio x) as many non-promo ones. "
                             "Off by default; holdout stays untouched.")
    parser.add_argument("--balance-ratio", type=float, default=2.0,
                        help="non-promo : promo ratio when --balance is on "
                             "(default 2.0 = twice as many non-promo as promo)")
    parser.add_argument("--balance-ratios", type=str, default=None,
                        help="comma-separated grid to TUNE the ratio by CV Qini, "
                             "e.g. '1,2,3,5'. Requires --balance. Overrides "
                             "--balance-ratio; the best ratio is chosen automatically.")
    parser.add_argument("--send-threshold", type=float, default=None,
                        help="uplift cutoff above which to recommend sending. "
                             "Default = auto (optimal cutoff chosen on the holdout).")
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
