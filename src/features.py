"""Leakage-free feature engineering for the uplift model.

Golden rule for uplift / response modelling: a feature for a mailing sent on
``mailing_date`` may only use information available *strictly before* that date.

In particular we NEVER use ``open_date`` / ``click_date`` / ``purchase_date`` /
``trip_date`` of the current row as features (those are the outcome / future).
Past mailings of the same client *are* allowed (their outcomes are known by the
time the current mailing is sent), but only via an expanding window shifted by 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_generation import DISCOUNT_COL

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


def _purchase_history_features(comm: pd.DataFrame, purch: pd.DataFrame) -> pd.DataFrame:
    """For every communication row, aggregate the client's purchases that were
    booked strictly before ``mailing_date`` (as-of aggregation, no leakage)."""
    purch = purch.sort_values("date_booking")

    # Pre-index purchases per email as numpy arrays for fast as-of lookups.
    per_email: dict[str, dict] = {}
    for email, g in purch.groupby("email", sort=False):
        booking = g["date_booking"].to_numpy()
        per_email[email] = {
            "booking": booking,
            "claim_cum": np.concatenate([[0.0], np.cumsum(g["claim_sum_usd"].to_numpy())]),
            "adult_cum": np.concatenate([[0.0], np.cumsum(g["adult"].to_numpy())]),
            "pax_cum": np.concatenate([[0.0], np.cumsum(g["pax"].to_numpy())]),
            # earliest booking date per destination -> for "visited offer state".
            "state_first": g.groupby("state_name")["date_booking"].min().to_dict(),
        }

    n = len(comm)
    out = {
        "hist_n_purchases": np.zeros(n),
        "hist_total_spend": np.zeros(n),
        "hist_avg_spend": np.zeros(n),
        "hist_avg_pax": np.zeros(n),
        "hist_avg_adult": np.zeros(n),
        "hist_days_since_last": np.full(n, np.nan),
        "hist_days_since_first": np.full(n, np.nan),
        "hist_visited_offer_state": np.zeros(n),
    }

    emails = comm["email"].to_numpy()
    mdates = comm["mailing_date"].to_numpy()
    ostates = comm["state_name"].to_numpy()

    for i in range(n):
        rec = per_email.get(emails[i])
        if rec is None:
            continue
        md = mdates[i]
        # number of purchases strictly before the mailing date
        k = int(np.searchsorted(rec["booking"], md, side="left"))
        if k == 0:
            continue
        out["hist_n_purchases"][i] = k
        total = rec["claim_cum"][k]
        out["hist_total_spend"][i] = total
        out["hist_avg_spend"][i] = total / k
        out["hist_avg_pax"][i] = rec["pax_cum"][k] / k
        out["hist_avg_adult"][i] = rec["adult_cum"][k] / k
        out["hist_days_since_last"][i] = (md - rec["booking"][k - 1]) / np.timedelta64(1, "D")
        out["hist_days_since_first"][i] = (md - rec["booking"][0]) / np.timedelta64(1, "D")
        first = rec["state_first"].get(ostates[i])
        out["hist_visited_offer_state"][i] = 1.0 if (first is not None and first < md) else 0.0

    return pd.DataFrame(out, index=comm.index)


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
    feat = pd.concat([feat, _purchase_history_features(comm, purch)], axis=1)

    # ---- client prior-mailing engagement (leakage-free, expanding) ----
    feat = pd.concat([feat, _past_mailing_features(comm)], axis=1)

    # ---- categoricals ----
    cat = pd.DataFrame(index=comm.index)
    cat["mailing_name"] = comm["mailing_name"].astype(str)
    cat["offer_state_name"] = comm["state_name"].astype(str)

    X = pd.concat([feat, cat], axis=1)

    numeric_cols = feat.columns.tolist()
    categorical_cols = cat.columns.tolist()

    meta = pd.DataFrame({
        "email": comm["email"].to_numpy(),
        "mailing_date": comm["mailing_date"].to_numpy(),
    }, index=comm.index)

    return X, treatment, target, numeric_cols, categorical_cols, meta
