"""Таблица и рисунки по классификатору эпохи — для отчёта ментору.  [без torch]

Читает готовые файлы run_era_cases / run_era_cv из папки схемы и ничего не обучает (секунды),
поэтому запускается заново после каждого прогона (например, когда досчитается вариант с 3 seed).

    python -m aa.era.report                                   # data/processed/era_coarse/
    python -m aa.era.report --authorship mualaqat_p_values.csv  # + общая таблица со всеми методами

Выход (data/processed/era_coarse/report/):
  era_p_values.csv          p (p_Холм) по муаллакам — тот же формат, что таблица методов авторства
  era_summary.csv           подробно: P(доисламская) муаллаки, ожидаемая для поэмы дивана, p, вывод
  mualaqat_all_methods.csv  [с --authorship] методы авторства + методы эпохи в одной таблице
  fig1_mualaqat_vs_diwan.png  муаллака против типичной поэмы дивана той же длины (BERT | n-граммы)
  fig2_segments.png           P(доисламская) по сегментам каждой муаллаки
  fig3_diwans.png             доля сегментов дивана, отнесённых к доисламской эпохе, по поэтам
  fig4_classifier.png         качество классификатора: балансная точность с ДИ и матрица ошибок
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aa import config

NAMES = {"antar-bin-shaddad": "Антара", "al-asha": "аль-Аша", "tarafa-ibn-al-abd": "Тарафа",
         "abid-ibn-al-abras": "Абид", "labid-ibn-rabiah": "Лабид", "imru-alqays": "Имру аль-Кайс",
         "nabgh-alzpiani": "ан-Набига", "zuhair-bin-abi-salma": "Зухайр", "harith-bin-halzh": "Харис",
         "amr-ibn-kulthum": "Амр"}
ORDER = list(NAMES)                       # порядок строк — как в таблице методов авторства
METHOD_NAMES = {"bert": "BERT (эпоха)", "ngram": "n-граммы (эпоха)"}
CLASS_NAMES = {"pre_islamic": "доисламская", "early_islamic": "ранний ислам\nи Омейяды",
               "abbasid": "Аббасиды", "late_classical": "поздняя\nклассика", "modern": "новое время",
               "islamic_era": "исламская"}

# палитра (dataviz: reference palette, проверена validate_palette.js — 2 слота, все проверки PASS)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


# ---------- таблицы ----------

def _cell(r) -> str:
    if r is None or str(r.get("эталон")) != "ok" or not np.isfinite(r.get("p", np.nan)):
        return "нельзя оценить"
    return f"{r['p']:.3f} ({r['p_Холм']:.2f})"


def p_table(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slug in ORDER:
        g = cases[cases["приписан"] == slug]
        if g.empty:
            continue
        row = {"Муаллака": NAMES[slug], "id": str(g["муаллака_id"].iloc[0]), "сегм.": int(g["сегментов"].max())}
        for m, col in METHOD_NAMES.items():
            gm = g[g["метод"] == m]
            if len(gm) or m == "bert":
                row[col] = _cell(gm.iloc[0]) if len(gm) else "нет данных"
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(cases: pd.DataFrame) -> pd.DataFrame:
    b = cases[cases["метод"] == "bert"].set_index("приписан")
    rows = []
    for slug in ORDER:
        if slug not in b.index:
            continue
        r = b.loc[slug]
        rows.append({"Муаллака": NAMES[slug], "сегм.": int(r["сегментов"]), "seed": int(r["seed"]),
                     "P(доисл.) муаллаки": r["P_pre"],
                     "P(доисл.) типичной поэмы дивана": r["ожидаемая_P_pre"],
                     "граница 5%": r["граница_5%"], "доля сегм. «доисламская»": r["доля_сегм_pre"],
                     "p": r["p"], "p Холма": r["p_Холм"], "вывод": r["вывод"]})
    return pd.DataFrame(rows)


def combined_table(authorship: pd.DataFrame, era: pd.DataFrame) -> pd.DataFrame:
    a = authorship.copy()
    a["id"] = a["id"].astype(str)
    cols = ["id"] + [c for c in METHOD_NAMES.values() if c in era.columns]
    return a.merge(era[cols], on="id", how="left")


# ---------- рисунки ----------

def _style(plt) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False,
    })


def fig_mualaqat(cases: pd.DataFrame, path: Path, plt) -> None:
    methods = [m for m in METHOD_NAMES if (cases["метод"] == m).any()]
    slugs = [s for s in ORDER if s in set(cases["приписан"])]
    n_seed = int(cases["seed"].max())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.6 * len(methods), 0.48 * len(slugs) + 1.9),
                             sharey=True, squeeze=False)
    y = np.arange(len(slugs))[::-1]
    for ax, m in zip(axes[0], methods):
        color = BLUE if m == "bert" else ORANGE
        g = cases[cases["метод"] == m].set_index("приписан")
        for yi, s in zip(y, slugs):
            r = g.loc[s]
            ok = str(r["эталон"]) == "ok" and np.isfinite(r["p"])
            if ok:
                ax.plot([r["ожидаемая_P_pre"], r["P_pre"]], [yi, yi], color=MUTED, lw=2, zorder=1,
                        solid_capstyle="round")
                ax.plot([r["граница_5%"]] * 2, [yi - 0.28, yi + 0.28], color=INK2, lw=1.6, zorder=2)
                ax.scatter(r["ожидаемая_P_pre"], yi, s=64, facecolor=SURFACE, edgecolor=INK2, lw=1.6, zorder=3)
                label = f"p = {r['p']:.3f}"
            else:
                label = "нет эталона"
            ax.scatter(r["P_pre"], yi, s=72, color=color, edgecolor=SURFACE, lw=1.5, zorder=4)
            ax.text(1.03, yi, label, va="center", ha="left", fontsize=9, color=INK2,
                    transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.7, len(slugs) - 0.3)
        ax.set_yticks(y, [NAMES[s] for s in slugs])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("P(доисламская эпоха), среднее по сегментам")
        ax.set_title(METHOD_NAMES[m].replace(" (эпоха)", ""), loc="left")
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="муаллака: BERT"),
         plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=8, label="муаллака: n-граммы"),
         plt.Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, mew=1.6, ms=8,
                    label="типичная поэма дивана той же длины (медиана эталона)"),
         plt.Line2D([], [], marker="|", ls="", color=INK2, ms=12, mew=1.6,
                    label="граница 5%: левее — значимо менее доисламская, чем диван")]
    fig.legend(handles=h[:len(methods)] + h[2:], loc="lower left", ncol=2, fontsize=9,
               bbox_to_anchor=(0.01, 0.0))
    fig.suptitle("Муаллака против поэм дивана того же поэта", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.01, 0.925, f"p — односторонний («муаллака менее доисламская, чем поэма дивана»), "
             f"запусков: {n_seed}", fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.10, 0.95, 0.92))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_segments(segs: pd.DataFrame, diwan: pd.DataFrame, path: Path, plt) -> None:
    b = segs[segs["метод"] == "bert"]
    dw = diwan[diwan["метод"] == "bert"].set_index("author")
    slugs = [s for s in ORDER if s in set(b["author"])]
    ncol = 5
    nrow = int(np.ceil(len(slugs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.7 * nrow + 1.1), sharey=True, squeeze=False)
    for ax, s in zip(axes.flat, slugs):
        g = b[b["author"] == s].sort_values("сегмент_№")
        if s in dw.index:
            ax.axhline(dw.loc[s, "P_pre_сегм"], color=INK2, ls=(0, (4, 3)), lw=1.3)
        ax.plot(g["сегмент_№"], g["p_pre_islamic"], color=BLUE, lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1.2)
        ax.set_title(NAMES[s], loc="left")
        ax.set_ylim(0, 1)
        ax.set_xticks(g["сегмент_№"])
        ax.tick_params(axis="x", labelsize=8)
    for ax in axes.flat[len(slugs):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("P(доисламская)")
    h = [plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=5, label="сегмент муаллаки (8 бейтов)"),
         plt.Line2D([], [], color=INK2, ls=(0, (4, 3)), lw=1.3, label="средняя по сегментам дивана поэта")]
    fig.legend(handles=h, loc="lower left", ncol=2, fontsize=9, bbox_to_anchor=(0.01, 0.0))
    fig.suptitle("P(доисламская) по сегментам муаллак (BERT)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_diwans(diwan: pd.DataFrame, per_class: pd.DataFrame | None, path: Path, plt) -> None:
    piv = diwan.pivot_table(index="author", columns="метод", values="доля_сегм_pre")
    piv = piv.sort_values("bert")
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(piv) + 1.9))
    y = np.arange(len(piv))
    if "ngram" in piv:
        for yi, (a, b) in enumerate(zip(piv["bert"], piv["ngram"])):
            ax.plot([a, b], [yi, yi], color=GRID, lw=2, zorder=1)
        ax.scatter(piv["ngram"], y, s=64, color=ORANGE, edgecolor=SURFACE, lw=1.5, zorder=3, label="n-граммы")
    ax.scatter(piv["bert"], y, s=72, color=BLUE, edgecolor=SURFACE, lw=1.5, zorder=4, label="BERT")
    for yi, v in enumerate(piv["bert"]):
        ax.text(v, yi + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    if per_class is not None and (per_class["класс"] == "pre_islamic").any():
        rec = float(per_class.loc[per_class["класс"] == "pre_islamic", "полнота"].iloc[0])
        ax.axvline(rec, color=INK2, ls=(0, (4, 3)), lw=1.2)
        ax.text(rec + 0.01, -0.45, f"доисламские поэты APCD\nв CV (BERT): {rec:.2f}", fontsize=8.5,
                color=INK2, va="bottom", ha="left")
    ax.set_yticks(y, [NAMES.get(a, a) for a in piv.index])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(piv) - 0.3)
    ax.set_xlabel("доля сегментов дивана, отнесённых к доисламской эпохе")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Насколько «доисламскими» модель считает диваны наших поэтов", loc="left", fontsize=12)
    fig.text(0.01, 0.01, "Никто из 10 поэтов в обучении модели не участвовал; диваны — aldiwan, без копий муаллак.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_classifier(metrics: pd.DataFrame, confusion: pd.DataFrame, path: Path, plt) -> None:
    seg = metrics[metrics["уровень"] == "сегменты"].set_index("метод")
    names = {"bert": "BERT", "ngram": "n-граммы", "meter": "только метр", "majority": "самый частый класс"}
    order = [m for m in ["majority", "meter", "ngram", "bert"] if m in seg.index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(order))
    for yi, m in zip(y, order):
        v = seg.loc[m, "balanced_acc"]
        lo, hi = map(float, str(seg.loc[m, "balanced_acc_ci95"]).split("–"))
        ax1.barh(yi, v, height=0.55, color=BLUE if m == "bert" else "#b7b6af", zorder=2)
        ax1.plot([lo, hi], [yi, yi], color=INK, lw=1.4, zorder=3)
        ax1.text(hi + 0.012, yi, f"{v:.2f}", va="center", fontsize=9, color=INK)
    chance = float(seg["случайный_уровень_balanced_acc"].iloc[0])
    ax1.axvline(chance, color=INK2, ls=(0, (4, 3)), lw=1.2)
    ax1.set_ylim(-0.5, len(order) - 0.05)
    ax1.text(chance + 0.008, len(order) - 0.4, "случайный уровень", fontsize=8.5, color=INK2, va="center")
    ax1.set_yticks(y, [names[m] for m in order])
    ax1.grid(axis="y", visible=False)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("балансная точность по сегментам (линия — 95% ДИ, бутстреп по поэтам)")
    ax1.set_title(f"Качество: CV по поэтам, {int(seg['поэтов'].iloc[0])} тестовых поэтов", loc="left")

    cm = confusion.to_numpy(dtype=float)
    share = cm / cm.sum(axis=1, keepdims=True)
    classes = [c.replace("истина:", "") for c in confusion.index]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    ax2.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            ax2.text(j, i, f"{share[i, j]:.0%}", ha="center", va="center", fontsize=9,
                     color="white" if share[i, j] > 0.45 else INK)
    labels = [CLASS_NAMES.get(c, c) for c in classes]
    ax2.set_xticks(range(len(classes)), [lb.replace("\n", " ") for lb in labels], fontsize=8.5,
                   rotation=30, ha="right", rotation_mode="anchor")
    ax2.set_yticks(range(len(classes)), labels, fontsize=8.5)
    ax2.set_xlabel("предсказанная эпоха")
    ax2.set_ylabel("истинная эпоха")
    ax2.grid(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_title("Матрица ошибок BERT (доля строки)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Таблица и рисунки по классификатору эпохи")
    ap.add_argument("--dir", default=str(config.DATA_PROCESSED / "era_coarse"),
                    help="папка с результатами run_era_cases / run_era_cv")
    ap.add_argument("--authorship", default=None, help="таблица методов авторства (mualaqat_p_values.csv)")
    ap.add_argument("--out-dir", default=None, help="по умолчанию <dir>/report")
    args = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    d = Path(args.dir)
    out = Path(args.out_dir or d / "report")
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    cases = pd.read_csv(d / "era_cases.csv", dtype={"муаллака_id": str})
    era = p_table(cases)
    era.to_csv(out / "era_p_values.csv", index=False)
    summary_table(cases).round(3).to_csv(out / "era_summary.csv", index=False)
    made["era_p_values"] = era
    if args.authorship:
        comb = combined_table(pd.read_csv(args.authorship, dtype={"id": str}), era)
        comb.to_csv(out / "mualaqat_all_methods.csv", index=False)
        made["combined"] = comb

    fig_mualaqat(cases, out / "fig1_mualaqat_vs_diwan.png", plt)
    diwan = pd.read_csv(d / "era_diwan_summary.csv")
    if (d / "era_cases_segments.csv").exists():
        fig_segments(pd.read_csv(d / "era_cases_segments.csv"), diwan, out / "fig2_segments.png", plt)
    per_class = pd.read_csv(d / "era_cv_per_class.csv") if (d / "era_cv_per_class.csv").exists() else None
    fig_diwans(diwan, per_class, out / "fig3_diwans.png", plt)
    if (d / "era_cv_metrics.csv").exists() and (d / "era_cv_confusion.csv").exists():
        fig_classifier(pd.read_csv(d / "era_cv_metrics.csv"), pd.read_csv(d / "era_cv_confusion.csv", index_col=0),
                       out / "fig4_classifier.png", plt)
    info = d / "era_cases_train_info.json"
    if info.exists():
        made["train_info"] = json.loads(info.read_text(encoding="utf-8"))
    print(f"[ok] таблицы и рисунки: {out}")
    print(era.to_string(index=False))
    return made


if __name__ == "__main__":
    main()
