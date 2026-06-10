"""Leakage-free feature engineering for the uplift model.

Golden rule for uplift / response modelling: a feature for a mailing sent on
``mailing_date`` may only use information available *strictly before* that date.

In particular we NEVER use ``open_date`` / ``click_date`` / ``purchase_date`` /
``trip_date`` of the current row as features (those are the outcome / future).
Past mailings of the same client *are* allowed (their outcomes are known by the
time the current mailing is sent), but only via an expanding window shifted by 1.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from data_generation import DISCOUNT_COL

logger = logging.getLogger(__name__)

DATE_COLS_COMM = ["mailing_date", "open_date", "click_date", "purchase_date", "trip_date"]
DATE_COLS_PURCH = ["date_booking", "date_begin"]

# Columns that describe the FUTURE / OUTCOME of the current mailing. They must
# never be used as model inputs.
LEAKAGE_COLS = ["open_date", "click_date", "purchase_date", "trip_date"]


def _ensure_datetime(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


_HIST_COLS = [
    "hist_n_purchases", "hist_total_spend", "hist_avg_spend", "hist_avg_pax",
    "hist_avg_adult", "hist_days_since_last", "hist_days_since_first",
    "hist_visited_offer_state",
]


def _empty_hist(index) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=index, columns=_HIST_COLS)
    out["hist_days_since_last"] = np.nan
    out["hist_days_since_first"] = np.nan
    return out


def _purchase_history_features(comm: pd.DataFrame, purch: pd.DataFrame) -> pd.DataFrame:
    """For every communication row, aggregate the client's purchases booked
    strictly before ``mailing_date`` (as-of aggregation, no leakage).

    Fully vectorised via ``merge_asof`` so it scales to millions of rows instead
    of looping in Python.
    """
    if len(purch) == 0:
        return _empty_hist(comm.index)

    p = purch[["email", "date_booking", "claim_sum_usd", "adult", "pax", "state_name"]].copy()
    p["date_booking"] = pd.to_datetime(p["date_booking"], errors="coerce")
    p = p.dropna(subset=["date_booking"])
    # Categorical merge keys cause dtype-mismatch errors -> use plain strings.
    p["email"] = p["email"].astype(str)
    p["state_name"] = p["state_name"].astype(str)
    p = p.sort_values("date_booking", kind="mergesort")

    g = p.groupby("email", sort=False)
    p["_cum_count"] = g.cumcount() + 1
    p["_cum_spend"] = g["claim_sum_usd"].cumsum()
    p["_cum_pax"] = g["pax"].cumsum()
    p["_cum_adult"] = g["adult"].cumsum()

    left = pd.DataFrame({
        "_row": np.arange(len(comm)),
        "email": comm["email"].astype(str).to_numpy(),
        "mailing_date": pd.to_datetime(comm["mailing_date"], errors="coerce").to_numpy(),
        "offer_state": comm["state_name"].astype(str).to_numpy(),
    }).sort_values("mailing_date", kind="mergesort")

    right = p[["email", "date_booking", "_cum_count", "_cum_spend", "_cum_pax", "_cum_adult"]]
    # backward + no exact match => only purchases STRICTLY before mailing_date.
    merged = pd.merge_asof(
        left, right,
        left_on="mailing_date", right_on="date_booking",
        by="email", direction="backward", allow_exact_matches=False,
    )

    # Earliest booking per client (valid prior history whenever count > 0).
    first_book = g["date_booking"].min().rename("_first_booking")
    merged = merged.merge(first_book, on="email", how="left")

    # Earliest booking per (client, destination) -> "already visited offer state".
    state_first = (p.groupby(["email", "state_name"], observed=True)["date_booking"]
                   .min().rename("_state_first").reset_index())
    merged = merged.merge(state_first, left_on=["email", "offer_state"],
                          right_on=["email", "state_name"], how="left")

    cnt = merged["_cum_count"]
    has = cnt.notna()
    cnt_safe = cnt.where(has)
    md = merged["mailing_date"]

    out = pd.DataFrame(index=merged.index)
    out["hist_n_purchases"] = cnt.fillna(0.0)
    out["hist_total_spend"] = merged["_cum_spend"].fillna(0.0)
    out["hist_avg_spend"] = (merged["_cum_spend"] / cnt_safe).fillna(0.0)
    out["hist_avg_pax"] = (merged["_cum_pax"] / cnt_safe).fillna(0.0)
    out["hist_avg_adult"] = (merged["_cum_adult"] / cnt_safe).fillna(0.0)
    out["hist_days_since_last"] = (md - merged["date_booking"]).dt.total_seconds() / 86400.0
    dsf = (md - merged["_first_booking"]).dt.total_seconds() / 86400.0
    out["hist_days_since_first"] = dsf.where(has)
    out["hist_visited_offer_state"] = (
        merged["_state_first"].notna() & (merged["_state_first"] < md)
    ).astype(float)

    out["_row"] = merged["_row"].to_numpy()
    out = out.sort_values("_row").drop(columns="_row").reset_index(drop=True)
    out.index = comm.index
    return out[_HIST_COLS]


def _past_mailing_features(comm: pd.DataFrame) -> pd.DataFrame:
    """Expanding (shifted) per-client engagement from PRIOR mailings only."""
    df = comm.sort_values(["email", "mailing_date"]).copy()
    df["_opened"] = df["open_date"].notna().astype(float)
    df["_clicked"] = df["click_date"].notna().astype(float)

    grp = df.groupby("email", sort=False)
    # Number of PRIOR mailings for this client (current row excluded).
    past_n = grp.cumcount()
    # Sum over prior rows = cumulative sum including current minus current value.
    prior_opened = grp["_opened"].cumsum() - df["_opened"]
    prior_clicked = grp["_clicked"].cumsum() - df["_clicked"]
    # Expanding mean of prior rows (NaN when there is no history yet).
    # This vectorised form is robust across pandas versions (no groupby.apply).
    denom = past_n.replace(0, np.nan)
    df["past_n_mailings"] = past_n
    df["past_open_rate"] = prior_opened / denom
    df["past_click_rate"] = prior_clicked / denom

    df = df.drop(columns=["_opened", "_clicked"])
    return df[["past_n_mailings", "past_open_rate", "past_click_rate"]].reindex(comm.index)


def build_dataset(comm: pd.DataFrame, purch: pd.DataFrame):
    """Return (X, treatment, target, meta) ready for the uplift pipeline.

    - treatment : 1 if the mailing carried a discount, else 0.
    - target    : 1 if the mailing was clicked (click_rate is the target metric).
    """
    comm = _ensure_datetime(comm, DATE_COLS_COMM)
    purch = _ensure_datetime(purch, DATE_COLS_PURCH)
    logger.info("building features for %d mailings using %d purchases ...",
                len(comm), len(purch))

    treatment = (comm[DISCOUNT_COL].fillna(0) > 0).astype(int)
    target = comm["click_date"].notna().astype(int)

    # ---- offer / calendar features (known at send time) ----
    feat = pd.DataFrame(index=comm.index)
    feat["offer_claim_sum_usd"] = comm["claim_sum_usd"].astype(float)
    feat["offer_adult"] = comm["adult"].astype(float)
    feat["offer_pax"] = comm["pax"].astype(float)
    feat["mailing_month"] = comm["mailing_date"].dt.month.astype(float)
    feat["mailing_quarter"] = comm["mailing_date"].dt.quarter.astype(float)
    feat["mailing_dow"] = comm["mailing_date"].dt.dayofweek.astype(float)
    feat["discount_rub"] = comm[DISCOUNT_COL].fillna(0).astype(float)

    # ---- client purchase-history features (leakage-free) ----
    logger.debug("computing as-of purchase-history features ...")
    feat = pd.concat([feat, _purchase_history_features(comm, purch)], axis=1)

    # ---- client prior-mailing engagement (leakage-free, expanding) ----
    logger.debug("computing prior-mailing engagement features ...")
    feat = pd.concat([feat, _past_mailing_features(comm)], axis=1)

    # ---- categoricals ----
    cat = pd.DataFrame(index=comm.index)
    cat["mailing_name"] = comm["mailing_name"].astype(str)
    cat["offer_state_name"] = comm["state_name"].astype(str)

    # Downcast numeric features to float32 to roughly halve memory on big data.
    feat = feat.astype("float32")

    X = pd.concat([feat, cat], axis=1)

    numeric_cols = feat.columns.tolist()
    categorical_cols = cat.columns.tolist()
    logger.info("features ready: %s | target click_rate=%.4f | treated share=%.4f | "
                "~%.0f MB", X.shape, target.mean(), treatment.mean(),
                X.memory_usage(deep=True).sum() / 1e6)

    meta = pd.DataFrame({
        "email": comm["email"].to_numpy(),
        "mailing_date": comm["mailing_date"].to_numpy(),
    }, index=comm.index)

    return X, treatment, target, numeric_cols, categorical_cols, meta
