"""Графики результатов uplift-кампании: группа A vs группа B.

Вход — CSV с результатами рассылки. Ожидаемые колонки:
    email, flag, sent_date, open_date, click_date  [, claim]
(колонка mean_predicted_uplift не обязательна и не используется в метриках).

Правила (как в задаче):
    - flag = ИСТИНА  -> группа A (мы отправили рекомендацию);
      flag = ЛОЖЬ    -> группа B (не отправляли);
    - открытие есть, если open_date  > 0;
    - клик есть,     если click_date > 0;
    - покупка есть,  если claim      > 0  (если колонка claim присутствует).

Метрики на группу (нормируются на число уникальных email в группе):
    - количество уникальных email
    - количество кликов
    - OpenRate  = открыли / уник. email
    - ClickRate = кликнули / уник. email
    - CTOR      = кликнули / открыли  (click-to-open rate)
    - [если есть claim] количество покупок и Conversion = покупки / размер группы

Строятся два графика:
    1) вовлечённость (без покупок)               -> outputs/campaign_engagement.png
    2) вовлечённость + покупки (если есть claim)  -> outputs/campaign_engagement_with_purchases.png

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

# Бизнес-понятные подписи и цвета групп.
LABEL_A = "Группа A — отправили рекомендацию (ИСТИНА)"
LABEL_B = "Группа B — не отправляли (ЛОЖЬ)"
COLOR_A = "#2e9e5b"   # зелёный
COLOR_B = "#9aa0a6"   # серый

_TRUE_TOKENS = {"истина", "true", "1", "1.0", "да", "yes", "y", "a", "а", "t"}


def _is_group_a(value) -> bool:
    return str(value).strip().lower() in _TRUE_TOKENS


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def compute_metrics(df: pd.DataFrame, has_claim: bool) -> pd.DataFrame:
    """Сводка метрик по группам A и B (одна строка на группу)."""
    d = pd.DataFrame({"email": df["email"].astype(str)})
    d["group"] = np.where(df["flag"].map(_is_group_a), "A", "B")
    d["opened"] = _num(df["open_date"]) > 0
    d["clicked"] = _num(df["click_date"]) > 0
    if has_claim:
        d["purchased"] = _num(df["claim"]) > 0

    # дедуп: один email = один клиент (открыл/кликнул/купил, если было хоть раз)
    agg = {"opened": "max", "clicked": "max"}
    if has_claim:
        agg["purchased"] = "max"
    per = d.groupby(["group", "email"], as_index=False).agg(agg)

    rows = []
    for grp in ["A", "B"]:
        g = per[per["group"] == grp]
        n = len(g)
        opens = int(g["opened"].sum())
        clicks = int(g["clicked"].sum())
        row = {
            "group": grp,
            "unique_emails": n,
            "opens": opens,
            "clicks": clicks,
            "open_rate": opens / n if n else 0.0,
            "click_rate": clicks / n if n else 0.0,
            "ctor": clicks / opens if opens else 0.0,
        }
        if has_claim:
            purch = int(g["purchased"].sum())
            row["purchases"] = purch
            row["conversion"] = purch / n if n else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def _grouped_bars(ax, categories, vals_a, vals_b, *, percent: bool, title: str):
    x = np.arange(len(categories))
    w = 0.38
    b1 = ax.bar(x - w / 2, vals_a, w, label=LABEL_A, color=COLOR_A)
    b2 = ax.bar(x + w / 2, vals_b, w, label=LABEL_B, color=COLOR_B)

    if percent:
        fmt = lambda v: f"{v * 100:.1f}%"
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v * 100:.0f}%")
        ax.set_ylabel("Доля, %")
        top = max([*vals_a, *vals_b, 0.0001])
        ax.set_ylim(0, top * 1.25)
    else:
        fmt = lambda v: f"{int(round(v)):,}".replace(",", " ")
        ax.set_ylabel("Количество")
        top = max([*vals_a, *vals_b, 1])
        ax.set_ylim(0, top * 1.25)

    for bars, vals in ((b1, vals_a), (b2, vals_b)):
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height(),
                    fmt(v), ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def _plot(metrics: pd.DataFrame, with_purchases: bool, out_path: str):
    A, B = metrics.loc["A"], metrics.loc["B"]

    # левая панель — абсолютные счётчики, правая — относительные ставки (в %)
    count_cats = ["Уник. email", "Клики"]
    a_counts = [A["unique_emails"], A["clicks"]]
    b_counts = [B["unique_emails"], B["clicks"]]
    rate_cats = ["OpenRate", "ClickRate", "CTOR"]
    a_rates = [A["open_rate"], A["click_rate"], A["ctor"]]
    b_rates = [B["open_rate"], B["click_rate"], B["ctor"]]

    if with_purchases:
        count_cats = ["Уник. email", "Клики", "Покупки"]
        a_counts = [A["unique_emails"], A["clicks"], A["purchases"]]
        b_counts = [B["unique_emails"], B["clicks"], B["purchases"]]
        rate_cats = ["OpenRate", "ClickRate", "CTOR", "Conversion\n(покупки/группа)"]
        a_rates = [A["open_rate"], A["click_rate"], A["ctor"], A["conversion"]]
        b_rates = [B["open_rate"], B["click_rate"], B["ctor"], B["conversion"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    _grouped_bars(ax1, count_cats, a_counts, b_counts, percent=False,
                  title="Абсолютные показатели")
    _grouped_bars(ax2, rate_cats, a_rates, b_rates, percent=True,
                  title="Относительные показатели (нормировано на уник. email)")

    title = ("Результаты uplift-кампании: A (рекомендация) vs B (контроль)"
             + (" — с покупками" if with_purchases else ""))
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # общая легенда снизу + подпись с размерами групп
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    subtitle = (f"Размер групп: A = {int(A['unique_emails']):,} email, "
                f"B = {int(B['unique_emails']):,} email"
                .replace(",", " "))
    fig.text(0.5, 0.91, subtitle, ha="center", fontsize=10, color="#555")

    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved chart -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/campaign_results.csv",
                        help="CSV с результатами кампании")
    parser.add_argument("--out-dir", default="outputs",
                        help="куда сохранять графики")
    parser.add_argument("--sep", default=",", help="разделитель CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.results, sep=args.sep)
    required = ["email", "flag", "open_date", "click_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"в файле нет обязательных колонок: {missing}. "
                         f"Найдены: {list(df.columns)}")
    has_claim = "claim" in df.columns
    os.makedirs(args.out_dir, exist_ok=True)

    # График 1 — вовлечённость (без покупок)
    m1 = compute_metrics(df, has_claim=False)
    print("\n=== Метрики (вовлечённость) ===")
    print(m1.to_string())
    _plot(m1, with_purchases=False,
          out_path=os.path.join(args.out_dir, "campaign_engagement.png"))

    # График 2 — добавляем покупки, если есть колонка claim
    if has_claim:
        m2 = compute_metrics(df, has_claim=True)
        print("\n=== Метрики (вовлечённость + покупки) ===")
        print(m2.to_string())
        _plot(m2, with_purchases=True,
              out_path=os.path.join(args.out_dir, "campaign_engagement_with_purchases.png"))
    else:
        print("\n[i] колонки 'claim' нет -> график с покупками пропущен.")


if __name__ == "__main__":
    main()
