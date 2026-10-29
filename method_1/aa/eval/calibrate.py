"""Калибровка порога уверенности.  [ТВОЯ РАБОТА — ядро процедуры из ТЗ]

Цель проекта — не просто «кто автор», а «насколько уверенно», чтобы ловить
неверную атрибуцию (в статье спорная поэма Имру аль-Кайса дала 0.71 против
обычных 0.99). Для этого смотрим распределения confidence у правильных и
ошибочных предсказаний и выбираем порог, выше которого вывод считаем надёжным.

На входе — preds_df из harness.run_cv (колонки: correct, confidence).
Порог ищем по Youden's J (макс. TPR−FPR) на задаче correct vs incorrect.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class Threshold:
    value: float
    auc: float
    kept_correct: float    # доля верных предсказаний выше порога
    kept_incorrect: float  # доля ошибок, пробившихся выше порога (хотим низкой)

    def label(self, confidence: float) -> str:
        return "надёжно" if confidence >= self.value else "под сомнением"

    def __str__(self) -> str:
        return (f"порог={self.value:.3f}  AUC={self.auc:.3f}  "
                f"верных удержано={self.kept_correct:.1%}  ошибок просочилось={self.kept_incorrect:.1%}")


def calibrate_threshold(preds: pd.DataFrame) -> Threshold:
    y = preds["correct"].astype(int).to_numpy()
    s = preds["confidence"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        # все предсказания верны или все ошибочны — порог не определить осмысленно
        return Threshold(value=float(np.median(s)), auc=float("nan"),
                         kept_correct=float((s >= np.median(s)).mean()), kept_incorrect=0.0)

    fpr, tpr, thr = roc_curve(y, s)
    # roc_curve добавляет первым порогом +inf («никого не принимать»). Если уверенность
    # совсем не разделяет верные/ошибочные (J=0 везде), argmax выбрал бы его -> порог=inf.
    finite = np.isfinite(thr)
    fpr, tpr, thr = fpr[finite], tpr[finite], thr[finite]
    j = tpr - fpr
    best = thr[int(np.argmax(j))]
    above = s >= best
    return Threshold(
        value=float(best),
        auc=float(roc_auc_score(y, s)),
        kept_correct=float((above & (y == 1)).sum() / max((y == 1).sum(), 1)),
        kept_incorrect=float((above & (y == 0)).sum() / max((y == 0).sum(), 1)),
    )


def plot_distributions(preds: pd.DataFrame, out_path: str = "confidence_dist.png") -> str | None:
    """Гистограммы confidence для верных и ошибочных. Требует matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib не установлен — график пропущен.")
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for flag, name in [(True, "верные"), (False, "ошибочные")]:
        sub = preds.loc[preds["correct"] == flag, "confidence"]
        if len(sub):
            ax.hist(sub, bins=20, alpha=0.6, label=name)
    thr = calibrate_threshold(preds)
    ax.axvline(thr.value, color="k", ls="--", label=f"порог {thr.value:.2f}")
    ax.set_xlabel("confidence"); ax.set_ylabel("частота"); ax.legend()
    ax.set_title("Распределение уверенности модели")
    fig.tight_layout(); fig.savefig(out_path, dpi=120)
    print(f"[ok] график: {out_path}")
    return out_path


if __name__ == "__main__":
    from aa.data.make_toy import make_toy
    from aa.eval.harness import run_cv
    from aa.model import stub

    _, preds = run_cv(make_toy(), stub.fit, stub.predict, n_splits=3)
    print(calibrate_threshold(preds))
