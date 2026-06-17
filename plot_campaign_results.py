"""Графики метрик в зависимости от размера группы A (таргетинг по uplift).

Группа A = email с самыми большими значениями `mean_predicted_uplift`
(остальные — группа B). Для каждого размера группы A (k = сколько верхних по
uplift email мы берём) считаются метрики ПО ГРУППЕ A и рисуются как линии в
зависимости от k. Вертикальными линиями отмечаются размеры группы A:
10тыс, 30тыс, 80тыс, 95тыс (настраивается через --milestones).

Вход — CSV. Ожидаемые колонки:
    email, mean_predicted_uplift, open_date, click_date  [, claim_id]

Правила:
    - открытие есть, если open_date  > 0;
    - клик есть,     если click_date > 0;
    - покупка есть,  если claim_id   > 0  (если колонка claim_id присутствует).

Метрики по группе A (нормированы на размер группы k):
    - количество кликов (накопл.)
    - OpenRate  = открыли / k
    - ClickRate = кликнули / k
    - CTOR      = кликнули / открыли
    - [если есть claim_id] количество покупок и Conversion = покупки / k

Строятся:
    1) вовлечённость                       -> outputs/campaign_engagement.png
    2) + покупки (если есть claim_id)       -> outputs/campaign_engagement_with_purchases.png

Запуск:
    python plot_campaign_results.py --results outputs/campaign_results.csv
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

DEFAULT_MILESTONES = [10_000, 30_000, 80_000, 95_000]
MILESTONE_COLORS = ["#d62728", "#9467bd", "#ff7f0e", "#1f77b4"]


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _thousands(v, _pos=None) -> str:
    return f"{int(v):,}".replace(",", " ")


def prepare(df: pd.DataFrame, claim_col: str | None):
    """Дедуп по email и сортировка по убыванию предсказанного uplift."""
    d = pd.DataFrame({
        "email": df["email"].astype(str),
        "uplift": _num(df["mean_predicted_uplift"]),
        "opened": _num(df["open_date"]) > 0,
        "clicked": _num(df["click_date"]) > 0,
    })
    if claim_col:
        d["purchased"] = _num(df[claim_col]) > 0

    agg = {"uplift": "mean", "opened": "max", "clicked": "max"}
    if claim_col:
        agg["purchased"] = "max"
    per = d.groupby("email", as_index=False).agg(agg)
    per = per.sort_values("uplift", ascending=False, kind="mergesort").reset_index(drop=True)
    return per


def cumulative_metrics(per: pd.DataFrame, has_purch: bool) -> dict:
    """Накопленные метрики группы A для k = 1..N (A = топ-k по uplift)."""
    n = len(per)
    k = np.arange(1, n + 1)
    cum_open = np.cumsum(per["opened"].to_numpy())
    cum_click = np.cumsum(per["clicked"].to_numpy())
    out = {
        "k": k,
        "clicks": cum_click,
        "open_rate": cum_open / k,
        "click_rate": cum_click / k,
        "ctor": np.divide(cum_click, np.maximum(cum_open, 1)),
    }
    if has_purch:
        cum_purch = np.cumsum(per["purchased"].to_numpy())
        out["purchases"] = cum_purch
        out["conversion"] = cum_purch / k
    return out


def _grid(n: int, milestones: list[int], n_points: int = 1500) -> np.ndarray:
    """Сетка значений k для рисования (равномерная + точные milestone-точки)."""
    base = np.linspace(1, n, num=min(n, n_points)).astype(int)
    ms = np.array([m for m in milestones if 1 <= m <= n], dtype=int)
    return np.unique(np.concatenate([base, ms])) if len(ms) else np.unique(base)


def _add_milestones(ax, milestones: list[int], n: int):
    for m, c in zip(milestones, MILESTONE_COLORS):
        if m > n:
            continue
        ax.axvline(m, color=c, ls="--", lw=1.6, alpha=0.9,
                   label=f"A = {_thousands(m)} email")


def _style_x(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("Размер группы A — число email с наибольшим uplift")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


def plot_chart(metrics: dict, milestones: list[int], with_purch: bool, out_path: str):
    n = int(metrics["k"][-1])
    g = _grid(n, milestones)
    idx = g - 1

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True)

    # --- верх: относительные метрики (%) ---
    ax_top.plot(g, metrics["open_rate"][idx] * 100, color="#1f77b4", lw=2.2, label="OpenRate")
    ax_top.plot(g, metrics["click_rate"][idx] * 100, color="#2ca02c", lw=2.2, label="ClickRate")
    ax_top.plot(g, metrics["ctor"][idx] * 100, color="#8c564b", lw=2.2, label="CTOR (клик/открытие)")
    if with_purch:
        ax_top.plot(g, metrics["conversion"][idx] * 100, color="#e377c2", lw=2.2,
                    label="Conversion (покупки/группа)")
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}%"))
    ax_top.set_ylabel("Доля по группе A, %")
    ax_top.set_title("Относительные метрики группы A в зависимости от её размера",
                     fontsize=12, fontweight="bold")
    _add_milestones(ax_top, milestones, n)
    _style_x(ax_top)
    ax_top.legend(loc="upper right", fontsize=9, ncol=2)

    # --- низ: абсолютные счётчики ---
    ax_bot.plot(g, metrics["clicks"][idx], color="#2ca02c", lw=2.2, label="Количество кликов")
    if with_purch:
        ax_bot.plot(g, metrics["purchases"][idx], color="#e377c2", lw=2.2,
                    label="Количество покупок")
    ax_bot.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax_bot.set_ylabel("Количество (накопленно)")
    ax_bot.set_title("Абсолютные счётчики группы A в зависимости от её размера",
                     fontsize=12, fontweight="bold")
    _add_milestones(ax_bot, milestones, n)
    _style_x(ax_bot)
    ax_bot.legend(loc="upper left", fontsize=9)

    title = ("Метрики группы A (таргетинг по uplift) в зависимости от размера группы"
             + (" — с покупками" if with_purch else ""))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved chart -> {out_path}")


def milestone_table(metrics: dict, milestones: list[int], has_purch: bool) -> pd.DataFrame:
    n = int(metrics["k"][-1])
    rows = []
    for m in milestones:
        if m > n:
            rows.append({"group_A_size": m, "note": "превышает число email"})
            continue
        i = m - 1
        row = {
            "group_A_size": m,
            "clicks": int(metrics["clicks"][i]),
            "open_rate": round(float(metrics["open_rate"][i]), 4),
            "click_rate": round(float(metrics["click_rate"][i]), 4),
            "ctor": round(float(metrics["ctor"][i]), 4),
        }
        if has_purch:
            row["purchases"] = int(metrics["purchases"][i])
            row["conversion"] = round(float(metrics["conversion"][i]), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/campaign_results.csv",
                        help="CSV с результатами (email, mean_predicted_uplift, "
                             "open_date, click_date [, claim_id])")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--sep", default=",")
    parser.add_argument("--milestones", default=",".join(map(str, DEFAULT_MILESTONES)),
                        help="размеры группы A для вертикальных линий, через запятую")
    args = parser.parse_args()

    df = pd.read_csv(args.results, sep=args.sep)
    required = ["email", "mean_predicted_uplift", "open_date", "click_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"в файле нет обязательных колонок: {missing}. "
                         f"Найдены: {list(df.columns)}")
    claim_col = "claim_id" if "claim_id" in df.columns else None
    milestones = [int(x) for x in str(args.milestones).split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    per = prepare(df, claim_col)
    print(f"уникальных email: {len(per):,}".replace(",", " "))

    # График 1 — вовлечённость
    m_eng = cumulative_metrics(per, has_purch=False)
    print("\n=== Метрики на ключевых размерах группы A (вовлечённость) ===")
    print(milestone_table(m_eng, milestones, has_purch=False).to_string(index=False))
    plot_chart(m_eng, milestones, with_purch=False,
               out_path=os.path.join(args.out_dir, "campaign_engagement.png"))

    # График 2 — с покупками (если есть claim_id)
    if claim_col:
        m_all = cumulative_metrics(per, has_purch=True)
        print("\n=== Метрики на ключевых размерах группы A (с покупками) ===")
        print(milestone_table(m_all, milestones, has_purch=True).to_string(index=False))
        plot_chart(m_all, milestones, with_purch=True,
                   out_path=os.path.join(args.out_dir, "campaign_engagement_with_purchases.png"))
    else:
        print("\n[i] колонки 'claim_id' нет -> график с покупками пропущен.")


if __name__ == "__main__":
    main()
