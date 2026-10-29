"""Диагностика бинарной пары: учится ли один CAMeLBERT различать двух поэтов,
и если нет — коллапс это или инверсия меток.

Запуск из корня репозитория (venv, torch+transformers):
    python diagnose_pair.py                          # 20 эпох
    python diagnose_pair.py --epochs 100 --early-stop-loss 0.01
    python diagnose_pair.py --a imru-alqays --b al-asha

Лог пишется в консоль и в logs/diagnose_<время>.log (прогресс по эпохам: loss, train_acc, ETA).

Как читать (ключевое — средние вероятности по истинному классу):
  mean P(a)|true a  >>  mean P(a)|true b   -> модель РАЗЛИЧАЕТ (хорошо)
  mean P(a)|true a  ~=  mean P(a)|true b   -> КОЛЛАПС (не учится)
  mean P(a)|true a  <   mean P(a)|true b   -> ИНВЕРСИЯ меток (баг)
И по логу эпох:
  train_acc растёт к 1.0  -> модель учится на train (если test плохой — не обобщает)
  train_acc стоит ~0.5    -> не учится вообще (lr / данные)
"""
from __future__ import annotations
import argparse
import logging

import numpy as np
import pandas as pd

from aa import config
from aa.log import setup_logging
from aa.model.finetune import PairClassifier

log = logging.getLogger("aa.diagnose")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_PROCESSED / "dataset.parquet"))
    ap.add_argument("--a", default="imru-alqays")
    ap.add_argument("--b", default="labid-ibn-rabiah")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--test-poems", type=int, default=2)
    ap.add_argument("--early-stop-loss", type=float, default=None)
    ap.add_argument("--no-amp", action="store_true", help="отключить bf16 (обучать в fp32)")
    args = ap.parse_args()

    setup_logging("diagnose")
    log.info("параметры: %s", vars(args))

    df = pd.read_parquet(args.data)
    df = df[~df["poem_id"].str.startswith("mualaka::")]
    avail = sorted(df["author"].unique())
    for name in (args.a, args.b):
        if name not in avail:
            raise SystemExit(f"Нет поэта '{name}'. Доступные:\n  " + "\n  ".join(avail))

    sub = df[df["author"].isin([args.a, args.b])]
    poems = sub[["poem_id", "author"]].drop_duplicates()
    test_poems = set(poems.groupby("author").tail(args.test_poems)["poem_id"])
    train = sub[~sub["poem_id"].isin(test_poems)]
    test = sub[sub["poem_id"].isin(test_poems)].reset_index(drop=True)
    if test.empty or train.empty:
        raise SystemExit("Мало данных — уменьши --test-poems.")

    log.info("TRAIN сегментов по авторам: %s", train["author"].value_counts().to_dict())
    log.info("TEST  сегментов по авторам: %s", test["author"].value_counts().to_dict())

    clf = PairClassifier.train(args.a, args.b, train, epochs=args.epochs,
                               early_stop_loss=args.early_stop_loss, amp=not args.no_amp)
    probs = np.array(clf.predict_proba_batch(test["text"].tolist()))  # P(a)

    true_a = (test["author"].values == args.a)
    pred_a = probs >= 0.5
    acc = float((pred_a == true_a).mean())
    mean_a, mean_b = probs[true_a].mean(), probs[~true_a].mean()

    log.info("--- вероятности P(%s) по истинному классу ---", args.a)
    log.info("  mean P(%s) | true %s: %.3f", args.a, args.a, mean_a)
    log.info("  mean P(%s) | true %s: %.3f", args.a, args.b, mean_b)
    log.info("  предсказано '%s': %d / %d (истинных '%s' в тесте: %d)",
             args.a, int(pred_a.sum()), len(probs), args.a, int(true_a.sum()))
    log.info(">>> бинарная точность: %.3f", acc)

    sep = mean_a - mean_b
    if sep > 0.2:
        log.info(">>> модель РАЗЛИЧАЕТ поэтов. Если acc низкая — дело в дисбалансе теста/пороге.")
    elif sep < -0.2:
        log.info(">>> ИНВЕРСИЯ: вероятности перевёрнуты. Это баг — пришли лог.")
    else:
        log.info(">>> КОЛЛАПС: классы не разделяются. Смотри train_acc в логе эпох и пришли лог.")


if __name__ == "__main__":
    main()
