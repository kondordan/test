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
    """Накопленные счётчики (для k=1..N, где A = топ-k по uplift)."""
    n = len(per)
    cum_open = np.cumsum(per["opened"].to_numpy())
    cum_click = np.cumsum(per["clicked"].to_numpy())
    out = {"k": np.arange(1, n + 1), "n": n,
           "cum_open": cum_open, "cum_click": cum_click}
    if has_purch:
        out["cum_purch"] = np.cumsum(per["purchased"].to_numpy())
    return out


def _ab_at(metrics: dict, g: np.ndarray, has_purch: bool) -> dict:
    """Метрики групп A (топ-g) и B (остальные n-g) на сетке размеров g."""
    n = metrics["n"]
    idx = g - 1
    nb = (n - g).astype(float)
    nb_safe = np.where(nb > 0, nb, np.nan)        # B пустая при g==n -> NaN

    opens_a = metrics["cum_open"][idx].astype(float)
    clicks_a = metrics["cum_click"][idx].astype(float)
    tot_open = float(metrics["cum_open"][-1])
    tot_click = float(metrics["cum_click"][-1])
    opens_b = tot_open - opens_a
    clicks_b = tot_click - clicks_a

    res = {
        "open_rate": (opens_a / g, opens_b / nb_safe),
        "click_rate": (clicks_a / g, clicks_b / nb_safe),
        "ctor": (clicks_a / np.maximum(opens_a, 1),
                 np.where(opens_b > 0, clicks_b / np.maximum(opens_b, 1), np.nan)),
        "clicks": (clicks_a, clicks_b),
    }
    if has_purch:
        purch_a = metrics["cum_purch"][idx].astype(float)
        tot_purch = float(metrics["cum_purch"][-1])
        purch_b = tot_purch - purch_a
        res["conversion"] = (purch_a / g, purch_b / nb_safe)
        res["purchases"] = (purch_a, purch_b)
    return res


def _grid(n: int, milestones: list[int], n_points: int = 1500) -> np.ndarray:
    """Сетка значений k для рисования (равномерная + точные milestone-точки)."""
    base = np.linspace(1, n, num=min(n, n_points)).astype(int)
    ms = np.array([m for m in milestones if 1 <= m <= n], dtype=int)
    return np.unique(np.concatenate([base, ms])) if len(ms) else np.unique(base)


def _add_milestones(ax, milestones: list[int], upper: float):
    """Вертикальные линии-ориентиры размера группы A (без засорения легенды).

    Линии за пределами видимой области (``upper``) не рисуются, чтобы не было
    «висящих» подписей вне осей."""
    for m, c in zip(milestones, MILESTONE_COLORS):
        if m > upper:
            continue
        ax.axvline(m, color=c, ls="--", lw=1.4, alpha=0.7)
        ax.text(m, 0.99, f" {m // 1000}к", color=c, fontsize=8, fontweight="bold",
                rotation=90, va="top", ha="left",
                transform=ax.get_xaxis_transform())


def _style_x(ax):
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("Размер группы A — число email с наибольшим uplift")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


def _ab_legend(ax, metric_specs, loc, ncol):
    """Легенда: цвет = метрика, стиль линии = группа (A сплошная / B пунктир)."""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=c, lw=2.2, label=name) for name, c in metric_specs]
    handles += [
        Line2D([], [], color="#444", lw=2.2, ls="-", label="Группа A (топ по uplift)"),
        Line2D([], [], color="#444", lw=1.8, ls="--", label="Группа B (остальные)"),
    ]
    ax.legend(handles=handles, loc=loc, fontsize=9, ncol=ncol, framealpha=0.9)


def _plot_ab(ax, g, a_vals, b_vals, color):
    ax.plot(g, a_vals, color=color, lw=2.2, ls="-")          # A — сплошная
    ax.plot(g, b_vals, color=color, lw=1.8, ls="--", alpha=0.9)  # B — пунктир


def plot_chart(metrics: dict, milestones: list[int], with_purch: bool, out_path: str,
               xmax: float | None = None, ymax_pct: float | None = None,
               ymax_count: float | None = None):
    n = metrics["n"]
    g = _grid(n, milestones)
    ab = _ab_at(metrics, g, with_purch)
    x_upper = xmax if xmax is not None else n   # видимая граница по X (для milestone)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True)

    # --- верх: относительные метрики (%), A vs B ---
    rate_specs = [("OpenRate", "#1f77b4"), ("ClickRate", "#2ca02c"),
                  ("CTOR (клик/открытие)", "#8c564b")]
    _plot_ab(ax_top, g, ab["open_rate"][0] * 100, ab["open_rate"][1] * 100, "#1f77b4")
    _plot_ab(ax_top, g, ab["click_rate"][0] * 100, ab["click_rate"][1] * 100, "#2ca02c")
    _plot_ab(ax_top, g, ab["ctor"][0] * 100, ab["ctor"][1] * 100, "#8c564b")
    if with_purch:
        _plot_ab(ax_top, g, ab["conversion"][0] * 100, ab["conversion"][1] * 100, "#e377c2")
        rate_specs.append(("Conversion (покупки/группа)", "#e377c2"))
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.1f}%"))
    ax_top.set_ylabel("Доля, %")
    ax_top.set_title("Относительные метрики: группа A vs группа B",
                     fontsize=12, fontweight="bold")
    _add_milestones(ax_top, milestones, x_upper)
    _style_x(ax_top)
    _ab_legend(ax_top, rate_specs, loc="upper right", ncol=2)

    # --- низ: абсолютные счётчики, A vs B ---
    cnt_specs = [("Клики", "#2ca02c")]
    _plot_ab(ax_bot, g, ab["clicks"][0], ab["clicks"][1], "#2ca02c")
    if with_purch:
        _plot_ab(ax_bot, g, ab["purchases"][0], ab["purchases"][1], "#e377c2")
        cnt_specs.append(("Покупки", "#e377c2"))
    ax_bot.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax_bot.set_ylabel("Количество")
    ax_bot.set_title("Абсолютные счётчики: группа A vs группа B",
                     fontsize=12, fontweight="bold")
    _add_milestones(ax_bot, milestones, x_upper)
    _style_x(ax_bot)
    _ab_legend(ax_bot, cnt_specs, loc="center right", ncol=1)

    # --- настраиваемые пределы осей (по умолчанию авто) ---
    if xmax is not None:
        ax_top.set_xlim(0, xmax)
        ax_bot.set_xlim(0, xmax)
    if ymax_pct is not None:
        ax_top.set_ylim(0, ymax_pct)
    if ymax_count is not None:
        ax_bot.set_ylim(0, ymax_count)

    title = ("Сравнение групп A и B в зависимости от размера группы A"
             + (" — с покупками" if with_purch else ""))
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved chart -> {out_path}")


def milestone_table(metrics: dict, milestones: list[int], has_purch: bool) -> pd.DataFrame:
    """Метрики обеих групп (A=топ-m, B=остальные) на ключевых размерах m."""
    n = metrics["n"]
    valid = np.array([m for m in milestones if 1 <= m <= n], dtype=int)
    if valid.size == 0:
        return pd.DataFrame([{"note": "все milestone превышают число email"}])
    ab = _ab_at(metrics, valid, has_purch)
    rows = []
    for j, m in enumerate(valid):
        for gi, grp in enumerate(("A", "B")):
            row = {
                "group_A_size": m, "group": grp,
                "clicks": int(ab["clicks"][gi][j]),
                "open_rate": round(float(ab["open_rate"][gi][j]), 4),
                "click_rate": round(float(ab["click_rate"][gi][j]), 4),
                "ctor": round(float(ab["ctor"][gi][j]), 4),
            }
            if has_purch:
                row["purchases"] = int(ab["purchases"][gi][j])
                row["conversion"] = round(float(ab["conversion"][gi][j]), 4)
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
    # --- настраиваемые пределы осей (по умолчанию None = авто/весь диапазон) ---
    # График вовлечённости:
    parser.add_argument("--eng-xmax", type=float, default=None,
                        help="макс. по оси X (число email) для графика вовлечённости")
    parser.add_argument("--eng-ymax", type=float, default=None,
                        help="макс. по оси Y в %% (верхняя панель) для графика вовлечённости, напр. 80")
    parser.add_argument("--eng-ymax-count", type=float, default=None,
                        help="макс. по оси Y (счётчики, нижняя панель) для графика вовлечённости")
    # График с покупками:
    parser.add_argument("--pur-xmax", type=float, default=None,
                        help="макс. по оси X (число email) для графика с покупками")
    parser.add_argument("--pur-ymax", type=float, default=None,
                        help="макс. по оси Y в %% (верхняя панель) для графика с покупками")
    parser.add_argument("--pur-ymax-count", type=float, default=None,
                        help="макс. по оси Y (счётчики, нижняя панель) для графика с покупками")
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
               out_path=os.path.join(args.out_dir, "campaign_engagement.png"),
               xmax=args.eng_xmax, ymax_pct=args.eng_ymax, ymax_count=args.eng_ymax_count)

    # График 2 — с покупками (если есть claim_id)
    if claim_col:
        m_all = cumulative_metrics(per, has_purch=True)
        print("\n=== Метрики на ключевых размерах группы A (с покупками) ===")
        print(milestone_table(m_all, milestones, has_purch=True).to_string(index=False))
        plot_chart(m_all, milestones, with_purch=True,
                   out_path=os.path.join(args.out_dir, "campaign_engagement_with_purchases.png"),
                   xmax=args.pur_xmax, ymax_pct=args.pur_ymax, ymax_count=args.pur_ymax_count)
    else:
        print("\n[i] колонки 'claim_id' нет -> график с покупками пропущен.")


if __name__ == "__main__":
    main()
