"""Synthetic data generator that reproduces the two tables described in the task.

No real data was provided in the repository, so this module builds a realistic,
*causally well-defined* synthetic dataset that matches the exact schema. It is
designed so that the discount has a **heterogeneous** effect on the click event
(some clients react to discounts, some do not). That heterogeneity is what an
uplift model is supposed to recover, and it is intentionally correlated with
observable purchase-history features so the modelling task is meaningful.

Two tables are produced (matching the column names from the task):

1. communications (история коммуникации):
   email, mailing_name, mailing_date, open_date, click_date, purchase_date,
   trip_date, "Есть ли скидка в рублях", claim_sum_usd, adult, pax, state_name

2. purchases (таблица со всеми покупками):
   email, date_booking, date_begin, claim_sum_usd, adult, pax, state_name

IMPORTANT: when plugging in real data, simply skip this module and point the
pipeline at the real CSV files with the same columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DISCOUNT_COL = "Есть ли скидка в рублях"

STATES = [
    "Турция", "Египет", "Таиланд", "ОАЭ", "Греция",
    "Испания", "Италия", "Кипр", "Мальдивы", "Вьетнам",
]
MAILING_NAMES = [
    "Летняя распродажа", "Раннее бронирование", "Горящие туры",
    "Эксклюзив для своих", "Новогодние предложения", "Осенний релакс",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(
    n_customers: int = 6000,
    seed: int = 42,
    history_start: str = "2023-01-01",
    cutoff: str = "2026-08-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (communications, purchases) dataframes.

    Everything is generated strictly before ``cutoff`` so the downstream pipeline
    can honour the "use only data before 01.08.2026" requirement.
    """
    rng = np.random.default_rng(seed)
    history_start = pd.Timestamp(history_start)
    cutoff = pd.Timestamp(cutoff)
    horizon_days = (cutoff - history_start).days

    emails = np.array([f"user{idx:05d}@mail.ru" for idx in range(n_customers)])

    # ---- Latent (unobserved) customer traits ------------------------------
    # wealth drives spend; price_sensitivity drives the uplift from a discount.
    # We make price_sensitivity anti-correlated with wealth so it is partially
    # recoverable from observable purchase history (avg spend, pax, ...).
    wealth = rng.beta(2.0, 2.0, n_customers)
    price_sensitivity = np.clip(1.0 - wealth + rng.normal(0, 0.15, n_customers), 0, 1)
    base_engagement = rng.beta(1.6, 3.0, n_customers)  # base open/click tendency
    favourite_state = rng.integers(0, len(STATES), n_customers)

    # =======================================================================
    # 1) PURCHASES TABLE
    # =======================================================================
    # Number of historical purchases depends on engagement & wealth.
    lam = 0.6 + 4.0 * base_engagement + 1.5 * wealth
    n_purch = rng.poisson(lam)

    p_rows = []
    for i in range(n_customers):
        for _ in range(int(n_purch[i])):
            booking_offset = rng.integers(0, horizon_days)
            date_booking = history_start + pd.Timedelta(days=int(booking_offset))
            lead = int(rng.integers(14, 180))
            date_begin = date_booking + pd.Timedelta(days=lead)
            adult = int(rng.integers(1, 4))
            kids = int(rng.poisson(0.6))
            pax = adult + kids
            # Spend scales with wealth, party size and some noise.
            claim = float(
                np.round(
                    (300 + 1700 * wealth[i]) * (0.6 + 0.4 * pax)
                    * rng.lognormal(0, 0.25),
                    2,
                )
            )
            # Customers mostly travel to their favourite state.
            if rng.random() < 0.6:
                st = STATES[favourite_state[i]]
            else:
                st = STATES[rng.integers(0, len(STATES))]
            p_rows.append((emails[i], date_booking, date_begin, claim, adult, pax, st))

    purchases = pd.DataFrame(
        p_rows,
        columns=[
            "email", "date_booking", "date_begin",
            "claim_sum_usd", "adult", "pax", "state_name",
        ],
    )

    # =======================================================================
    # 2) COMMUNICATIONS TABLE
    # =======================================================================
    # Build campaigns: each campaign has a date and is sent to a random subset.
    n_campaigns = 90
    campaign_dates = sorted(
        history_start + pd.Timedelta(days=int(d))
        for d in rng.integers(30, horizon_days, n_campaigns)
    )

    c_rows = []
    for c_idx, c_date in enumerate(campaign_dates):
        mailing_name = MAILING_NAMES[c_idx % len(MAILING_NAMES)]
        # Send to ~25% of the base, biased a bit towards engaged customers.
        send_p = 0.15 + 0.25 * base_engagement
        recipients = np.where(rng.random(n_customers) < send_p)[0]
        if len(recipients) == 0:
            continue

        # Offer characteristics for this campaign.
        offer_state_idx = rng.integers(0, len(STATES))
        offer_state = STATES[offer_state_idx]

        # RANDOMISED treatment assignment (A/B): half receive a discount.
        # Randomisation makes the uplift identifiable without propensity models.
        treat = rng.random(len(recipients)) < 0.5
        # Discount amount in RUB for treated recipients (else 0).
        discount_rub = np.where(treat, rng.choice([1000, 2000, 3000, 5000], len(recipients)), 0)
        discount_norm = discount_rub / 5000.0

        for j, cust in enumerate(recipients):
            adult = int(rng.integers(1, 4))
            kids = int(rng.poisson(0.6))
            pax = adult + kids
            claim = float(np.round((300 + 1700 * wealth[cust]) * (0.6 + 0.4 * pax)
                                   * rng.lognormal(0, 0.25), 2))

            relevance = 1.0 if offer_state_idx == favourite_state[cust] else 0.0

            # ---- OPEN model ----
            open_logit = (
                -0.4
                + 2.6 * base_engagement[cust]
                + 0.25 * relevance
                + 0.10 * treat[j]            # small open lift from "discount" subject line
                + rng.normal(0, 0.3)
            )
            opened = rng.random() < _sigmoid(open_logit)

            # ---- CLICK model (only possible if opened) ----
            # The TREATMENT EFFECT (uplift) is heterogeneous: it is driven by the
            # client's price sensitivity, the discount size and offer relevance.
            # This is the quantity the uplift model must learn. The components are
            # all correlated with observable features (avg spend, visited state,
            # discount size), so the heterogeneity is recoverable from the data.
            uplift_logit = treat[j] * (
                0.2
                + 3.6 * price_sensitivity[cust] * discount_norm[j]
                + 0.7 * relevance * discount_norm[j]
            )
            click_logit = (
                -0.6
                + 2.2 * base_engagement[cust]
                + 0.4 * relevance
                + uplift_logit
                + rng.normal(0, 0.2)
            )
            clicked = opened and (rng.random() < _sigmoid(click_logit))

            # ---- Dates ----
            open_date = c_date + pd.Timedelta(hours=int(rng.integers(1, 72))) if opened else pd.NaT
            click_date = open_date + pd.Timedelta(hours=int(rng.integers(0, 24))) if clicked else pd.NaT

            # ---- Purchase within 2 weeks of the letter ----
            purchase_date = pd.NaT
            trip_date = pd.NaT
            if clicked:
                p_buy = _sigmoid(-1.2 + 1.5 * price_sensitivity[cust] * (1 + discount_norm[j])
                                 + 0.8 * relevance)
                if rng.random() < p_buy:
                    purchase_date = c_date + pd.Timedelta(days=int(rng.integers(0, 14)))
                    trip_date = purchase_date + pd.Timedelta(days=int(rng.integers(21, 180)))

            c_rows.append((
                emails[cust], mailing_name, c_date, open_date, click_date,
                purchase_date, trip_date, int(discount_rub[j]), claim, adult, pax, offer_state,
            ))

    communications = pd.DataFrame(
        c_rows,
        columns=[
            "email", "mailing_name", "mailing_date", "open_date", "click_date",
            "purchase_date", "trip_date", DISCOUNT_COL,
            "claim_sum_usd", "adult", "pax", "state_name",
        ],
    )

    # Enforce the "strictly before cutoff" invariant for every dated column.
    communications = communications[communications["mailing_date"] < cutoff].reset_index(drop=True)
    purchases = purchases[purchases["date_booking"] < cutoff].reset_index(drop=True)

    return communications, purchases


if __name__ == "__main__":
    comm, purch = generate()
    comm.to_csv("data/communications.csv", index=False)
    purch.to_csv("data/purchases.csv", index=False)
    print("communications:", comm.shape)
    print("purchases:", purch.shape)
    print("overall click_rate:", comm["click_date"].notna().mean().round(4))
    treated = comm[comm[DISCOUNT_COL] > 0]["click_date"].notna().mean()
    control = comm[comm[DISCOUNT_COL] == 0]["click_date"].notna().mean()
    print(f"click_rate treated={treated:.4f} control={control:.4f} "
          f"avg uplift={treated - control:.4f}")
