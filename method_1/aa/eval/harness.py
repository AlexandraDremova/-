"""Оценка: кросс-валидация + агрегация F1 + таблица предсказаний.  [ТВОЯ РАБОТА]

Модель-агностично: харнесс зовёт только fit_fn(train_df) и predict_fn(text, ens).
Любая модель с контрактом interface.py сюда подставляется без изменений (стаб — тоже).

КЛЮЧЕВАЯ ДЕТАЛЬ: делим по poem_id (GroupKFold), а не по строкам. Сегменты одной
поэмы не должны попасть и в train, и в test — иначе F1 завышен за счёт утечки.

    from aa.eval.harness import run_cv
    from aa.model import stub
    report, preds = run_cv(df, stub.fit, stub.predict)
"""
from __future__ import annotations
from dataclasses import dataclass, field

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupKFold

import logging
import time

from aa import config
from aa.log import fmt_duration
from aa.model.interface import check_prediction

log = logging.getLogger(__name__)


@dataclass
class CVReport:
    macro_f1: float
    accuracy: float
    per_fold_f1: list[float] = field(default_factory=list)
    n_authors: int = 0
    n_segments: int = 0

    def __str__(self) -> str:
        folds = ", ".join(f"{x:.3f}" for x in self.per_fold_f1)
        return (f"macro-F1={self.macro_f1:.3f}  acc={self.accuracy:.3f}  "
                f"(folds: {folds})  authors={self.n_authors}  segments={self.n_segments}")


def run_cv(df, fit_fn=None, predict_fn=None, n_splits: int = config.N_SPLITS,
           predict_many=None, fit_predict_many=None, groups=None):
    """Возвращает (CVReport, preds_df). preds_df идёт в calibrate.py.

    Три режима (по убыванию предпочтения для реального ансамбля):
      fit_predict_many(train_df, test_texts) -> list[Prediction]  — потоково, без сохранения моделей
      fit_fn + predict_many(texts, ensemble)                       — батчем по сохранённым моделям
      fit_fn + predict_fn(text, ensemble)                          — по одному тексту (стаб)

    groups: группы для GroupKFold (по умолчанию poem_id). Поэмы с общими/похожими бейтами
    стоит объединять в одну группу (aa.data.dupes.poem_groups), чтобы они не разъезжались
    по train/test и не завышали оценку.
    """
    if fit_predict_many is None and (fit_fn is None or (predict_fn is None and predict_many is None)):
        raise ValueError("Передай fit_predict_many, либо fit_fn вместе с predict_fn/predict_many.")
    df = df.reset_index(drop=True)
    groups = df["poem_id"] if groups is None else pd.Series(list(groups), index=df.index)
    n_groups = groups.nunique()
    n_splits = min(n_splits, n_groups)
    if n_splits < 2:
        raise ValueError(f"Для CV нужно ≥2 поэмы, есть {n_groups}.")

    gkf = GroupKFold(n_splits=n_splits)
    rows, per_fold = [], []

    t0 = time.time()
    for fold, (tr, te) in enumerate(gkf.split(df, df["author"], groups)):
        train_df, test_df = df.iloc[tr], df.iloc[te]
        log.info("=== фолд %d/%d: train %d сегм., test %d сегм. ===",
                 fold + 1, n_splits, len(train_df), len(test_df))
        texts = test_df["text"].tolist()
        if fit_predict_many is not None:
            raw = fit_predict_many(train_df, texts)
        else:
            ensemble = fit_fn(train_df)
            raw = (predict_many(texts, ensemble) if predict_many is not None
                   else [predict_fn(t, ensemble) for t in texts])
        preds_fold = [check_prediction(p) for p in raw]

        y_true, y_pred = [], []
        for seg, p in zip(test_df.itertuples(index=False), preds_fold):
            y_true.append(str(seg.author))
            y_pred.append(p["pred_author"])
            row = {
                "fold": fold,
                "poem_id": seg.poem_id,
                "true_author": str(seg.author),
                "pred_author": p["pred_author"],
                "confidence": float(p["confidence"]),
                "correct": str(seg.author) == p["pred_author"],
            }
            # Если модель отдала полную раскладку — сохраняем уверенность в НАСТОЯЩЕМ авторе
            # (эталон для муалак) и попарные вероятности (попарная точность, как в статье).
            if "scores" in p:
                row["true_score"] = p["scores"].get(str(seg.author))
            if "pair_probs" in p:
                row["pair_probs"] = p["pair_probs"]
            rows.append(row)
        per_fold.append(f1_score(y_true, y_pred, average="macro", zero_division=0))
        elapsed = time.time() - t0
        log.info("=== фолд %d/%d: macro-F1=%.3f | прошло %s, осталось ~%s ===",
                 fold + 1, n_splits, per_fold[-1], fmt_duration(elapsed),
                 fmt_duration(elapsed / (fold + 1) * (n_splits - fold - 1)))

    preds = pd.DataFrame(rows)
    report = CVReport(
        macro_f1=f1_score(preds["true_author"], preds["pred_author"], average="macro", zero_division=0),
        accuracy=accuracy_score(preds["true_author"], preds["pred_author"]),
        per_fold_f1=per_fold,
        n_authors=int(df["author"].nunique()),
        n_segments=int(len(df)),
    )
    return report, preds


def poem_level_vote(preds: pd.DataFrame) -> pd.DataFrame:
    """Свести предсказания сегментов к одному на поэму (голосование).
    Атрибуция в проекте — на уровне поэмы, так что финальные цифры считаем здесь."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        winner = g["pred_author"].mode().iloc[0]
        return pd.Series({
            "true_author": g["true_author"].iloc[0],
            "pred_author": winner,
            "confidence": g.loc[g["pred_author"] == winner, "confidence"].mean(),
            "correct": g["true_author"].iloc[0] == winner,
        })
    return preds.groupby("poem_id", group_keys=False).apply(_agg).reset_index()


if __name__ == "__main__":
    # Самопроверка на игрушечных данных + стаб-модели
    from aa.data.make_toy import make_toy
    from aa.model import stub

    df = make_toy()
    report, preds = run_cv(df, stub.fit, stub.predict, n_splits=3)
    print("segment-level:", report)
    poem = poem_level_vote(preds)
    print(f"poem-level acc: {poem['correct'].mean():.3f}")
