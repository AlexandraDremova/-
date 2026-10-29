"""A6+A7 — кросс-валидация реального ансамбля на диванах (муалаки, их копии и бейты исключены).

    python -m aa.model.run_cv --data data/processed/dataset.parquet --n-splits 3 \
        --epochs 30 --early-stop-loss 0.01

Что считает и сохраняет (data/processed/):
  cv_predictions.csv  — предсказание на каждый тестовый сегмент (+ уверенность в настоящем авторе)
  cv_pairwise.csv     — точность каждой попарной модели на её двух авторах (сопоставимо со статьёй)
  cv_poem_scores.csv  — ЭТАЛОН для A8: уверенность в настоящем авторе на каждой невиденной поэме
  cv_confusion.csv    — матрица ошибок 10-классовой атрибуции
  cv_pairs.csv        — сводка обучения пар (эпохи, точность, откаты в fp32)
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from sklearn.metrics import classification_report

from aa import config
from aa.data.dupes import poem_groups
from aa.data.prepare import prepare_training_data
from aa.eval.calibrate import calibrate_threshold
from aa.eval.harness import poem_level_vote, run_cv
from aa.eval.reference import pairwise_accuracy, poem_true_scores, summarize_reference
from aa.log import setup_logging
from aa.model import ensemble
from aa.model.voting import breakdown

log = logging.getLogger("aa.run_cv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_PROCESSED / "dataset.parquet"))
    ap.add_argument("--n-splits", type=int, default=config.N_SPLITS)
    ap.add_argument("--max-opponents", type=int, default=None,
                    help="ограничить оппонентов на поэта (ускоряет; по умолчанию — все пары)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--early-stop-loss", type=float, default=config.EARLY_STOP_LOSS,
                    help="остановить пару, когда train-loss ниже порога 3 эпохи подряд (напр. 0.01)")
    ap.add_argument("--no-amp", action="store_true", help="отключить bf16 (обучать в fp32)")
    args = ap.parse_args()

    setup_logging("run_cv")
    log.info("параметры: %s", vars(args))
    out_dir = config.DATA_PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(args.data)
    # Кросс-валидация идёт РОВНО на тех данных, на которых в A8 обучается ансамбль:
    # без муалак, без их копий в диванах и без сегментов с их бейтами (точными и вариантами).
    # Иначе эталон для муаллаки включал бы её саму (копия муаллаки — обычная поэма дивана).
    df, _, _ = prepare_training_data(raw)
    # Поэмы с общими/похожими бейтами — в одну группу, чтобы не попадали в разные фолды.
    groups = poem_groups(df, config.FUZZY_DUP_THRESHOLD)
    merged = df.groupby(groups)["poem_id"].nunique()
    log.info("групп для CV: %d (поэм %d); объединено поэм с общими/похожими бейтами: %d в %d группах",
             groups.nunique(), df["poem_id"].nunique(), int(merged[merged > 1].sum()), int((merged > 1).sum()))
    biggest = groups.value_counts().max() / len(df)
    if biggest > 1 / (2 * args.n_splits):
        log.warning("самая крупная группа поэм — %.0f%% сегментов: фолды будут несбалансированы. "
                    "Проверь порог FUZZY_DUP_THRESHOLD.", 100 * biggest)
    log.info("датасет: %d сегментов, %d авторов, %d поэм",
             len(df), df["author"].nunique(), df["poem_id"].nunique())
    log.info("сегментов на поэта: %s", df["author"].value_counts().to_dict())

    pair_report: list[dict] = []
    fold_no = {"i": 0}

    def fit_predict(train_df, texts):
        fold_no["i"] += 1
        rep: list[dict] = []
        per_text = ensemble.fit_collect(train_df, texts, max_opponents=args.max_opponents,
                                        epochs=args.epochs, early_stop_loss=args.early_stop_loss,
                                        amp=not args.no_amp, report=rep)
        pair_report.extend({"fold": fold_no["i"], **r} for r in rep)
        pd.DataFrame(pair_report).to_csv(out_dir / "cv_pairs.csv", index=False)
        return [{**breakdown(pp), "pair_probs": pp} for pp in per_text]

    report, preds = run_cv(df, n_splits=args.n_splits, fit_predict_many=fit_predict, groups=groups)

    # 1) посегментные предсказания (без громоздких попарных вероятностей)
    preds.drop(columns=["pair_probs"]).to_csv(out_dir / "cv_predictions.csv", index=False)

    # 2) попарная точность — метрика, сопоставимая со статьёй
    pw = pairwise_accuracy(preds)
    pw.to_csv(out_dir / "cv_pairwise.csv", index=False)

    # 3) эталон уверенности в настоящем авторе на невиденных поэмах — для A8
    ref = poem_true_scores(preds)
    ref.to_csv(out_dir / "cv_poem_scores.csv", index=False)

    poem = poem_level_vote(preds)
    cm = pd.crosstab(preds["true_author"], preds["pred_author"],
                     rownames=["истинный"], colnames=["предсказан"])
    cm.to_csv(out_dir / "cv_confusion.csv")

    log.info("=== ИТОГИ ===")
    log.info("попарные модели (сопоставимо со статьёй): средняя точность %.3f, "
             "средняя сбалансированная %.3f, медиана %.3f, худшая пара %.3f",
             pw["acc"].mean(), pw["balanced_acc"].mean(), pw["balanced_acc"].median(),
             pw["balanced_acc"].min())
    worst = (pw.groupby(["a", "b"])["balanced_acc"].mean().sort_values().head(5).round(3))
    log.info("5 самых трудных пар (среднее по фолдам):\n%s", worst.to_string())
    log.info("10-классовая атрибуция, сегменты: %s", report)
    log.info("10-классовая атрибуция, поэмы (голосование сегментов): accuracy %.3f",
             poem["correct"].mean())
    log.info("калибровка (уверенность победителя, верно/неверно): %s", calibrate_threshold(preds))
    log.info("F1 по поэтам (сегменты):\n%s",
             classification_report(preds["true_author"], preds["pred_author"], zero_division=0))
    log.info("матрица ошибок:\n%s", cm.to_string())
    log.info("эталон для муалак — уверенность в настоящем авторе на невиденных поэмах:\n%s",
             summarize_reference(ref).to_string(index=False))
    log.info("файлы: %s", ", ".join(str(out_dir / f) for f in
             ["cv_predictions.csv", "cv_pairwise.csv", "cv_poem_scores.csv",
              "cv_confusion.csv", "cv_pairs.csv"]))


if __name__ == "__main__":
    main()
