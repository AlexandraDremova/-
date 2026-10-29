"""Кросс-валидация классификатора эпохи ПО ПОЭТАМ + сравнение с базовыми линиями.

Для каждого фолда (поэты фолда — тест, остальные — обучение, из них ~10% поэтов — валидация для
ранней остановки):
  1. выборка ограничивается (сегментов на поэта и на класс), набор «по кругу» по поэтам;
  2. CAMeLBERT (aa.era.model) учится на обучении, выбирает лучшую эпоху по валидации;
  3. на тех же сегментах теста — базовые линии: символьные n-граммы, «только метр», самый частый класс.

Итог (data/processed/era_<схема>/):
  era_cv_predictions.csv   по сегменту: фолд, поэт, поэма, метка, вероятности всех методов
  era_cv_metrics.csv       метод × уровень (сегменты / поэты): балансная точность, macro-F1, точность,
                           «точно или соседняя эпоха», 95% интервал (бутстреп по поэтам),
                           p против случайного (перестановка эпох между поэтами);
                           для класса pre_islamic — AUC P(доисламская)
  era_cv_compare.csv       BERT против каждой базовой линии: разность, 95% интервал, p, p Холма
  era_cv_per_class.csv     BERT: полнота / точность / F1 по классам
  era_cv_confusion.csv     BERT: матрица ошибок (строки — истина)
  era_cv_by_original_era.csv  BERT: куда уходят сегменты каждой ИСХОДНОЙ эпохи APCD (видно мухадрамов и т.п.)
  era_cv_train_info.json   ход обучения по фолдам

    python -m aa.era.run_era_cv --data data/processed/era_dataset.parquet --scheme coarse
    python -m aa.era.run_era_cv ... --folds 1          # быстро: только первый фолд (~20% поэтов)
    python -m aa.era.run_era_cv ... --resume           # продолжить прерванный прогон
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aa import config
from aa.era import baselines, watchdog
from aa.era.apcd import read_table
from aa.era.labels import ERA_NAMES_RU, PRE_ISLAMIC
from aa.era.metrics import (calibrate_sets, confusion, paired_compare, per_class_table, poet_votes,
                            summarize)
from aa.era.split import apply_scheme, cap_sample, class_weights, poet_folds, split_val_poets
from aa.eval.reference import holm
from aa.log import fmt_duration, setup_logging

log = logging.getLogger("aa.era.run_cv")

METHODS = ("bert", "ngram", "meter", "majority")


def _prob_cols(method: str, classes: list[str]) -> list[str]:
    return [f"{method}_p_{c}" for c in classes]


def run_fold(df: pd.DataFrame, fold_of: dict, k: int, classes: list[str], args, bert_fn,
             with_baselines: bool = True) -> tuple[pd.DataFrame, dict]:
    test = df[df["author"].map(fold_of) == k]
    rest = df[df["author"].map(fold_of) != k]
    train, val = split_val_poets(rest, args.val_frac, seed=args.seed + 100 + k)
    train_s = cap_sample(train, args.max_seg_per_poet, args.max_seg_per_class, seed=args.seed + k)
    val_s = cap_sample(val, args.eval_seg_per_poet, args.eval_seg_per_class // 3, seed=args.seed + k)
    test_s = cap_sample(test, args.eval_seg_per_poet, args.eval_seg_per_class, seed=args.seed + k)
    log.info("фолд %d: обучение %d сегм. / %d поэтов, валидация %d / %d, тест %d / %d",
             k, len(train_s), train_s["author"].nunique(), len(val_s), val_s["author"].nunique(),
             len(test_s), test_s["author"].nunique())

    w = class_weights(train_s["label"], classes)
    cfg = dict(checkpoint=args.checkpoint, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
               patience=args.patience, attn=args.attn, class_weights=w, seed=args.seed + k)
    texts = {"test": list(test_s["text"]), "val": list(val_s["text"])}
    y_val = val_s["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    probs, info = bert_fn(train_s, val_s, classes, texts, **cfg)
    probs, t_bert = calibrate_sets(probs, y_val)
    out = test_s[["poem_id", "author", "label", "period", "meter", "n_bayts"]].copy()
    out.insert(0, "fold", k)
    out["seg_no"] = test_s.groupby("poem_id").cumcount().to_numpy()
    out[_prob_cols("bert", classes)] = probs["test"]
    temps = {"bert": t_bert}
    finite = np.isfinite(probs["test"]).all(axis=1)
    if with_baselines:
        t0 = time.time()
        watchdog.beat(f"фолд {k}: базовые линии")
        ng = baselines.ngram_fit_predict(train_s, classes, texts, seed=args.seed + k)
        ng, temps["ngram"] = calibrate_sets(ng, y_val)
        out[_prob_cols("ngram", classes)] = ng["test"]
        me = baselines.meter_fit_predict(train_s, classes, {"test": list(test_s["meter"])}, seed=args.seed + k)
        out[_prob_cols("meter", classes)] = me["test"]
        out[_prob_cols("majority", classes)] = baselines.majority_predict(train_s, classes, len(test_s))
        log.info("фолд %d: базовые линии посчитаны за %s", k, fmt_duration(time.time() - t0))
    if not finite.all():
        log.warning("фолд %d: %d тестовых сегментов с NaN у BERT исключены (у всех методов — для парности)",
                    k, int((~finite).sum()))
        out = out[finite]
    log.info("фолд %d: температура калибровки (по валидационным поэтам): %s", k,
             {m: round(t, 2) for m, t in temps.items()})
    info = dict(info or {}, fold=k, temperature=temps, class_weights=dict(zip(classes, np.round(w, 3).tolist())),
                n_train=len(train_s), n_val=len(val_s), n_test=len(test_s))
    return out.reset_index(drop=True), info


def evaluate(preds: pd.DataFrame, classes: list[str], n_boot: int, n_perm: int, seed: int = 0) -> dict:
    """Все таблицы итога по объединённым предсказаниям фолдов."""
    from sklearn.metrics import roc_auc_score
    C = len(classes)
    cidx = {c: i for i, c in enumerate(classes)}
    y = preds["label"].map(cidx).to_numpy()
    poets = preds["author"].astype(str).to_numpy()
    methods = [m for m in METHODS if f"{m}_p_{classes[0]}" in preds.columns]
    pred = {m: preds[_prob_cols(m, classes)].to_numpy().argmax(axis=1) for m in methods}

    rows = []
    poet_label = preds.groupby("author")["label"].first().map(cidx)
    for m in methods:
        P = preds[_prob_cols(m, classes)].to_numpy()
        seg = summarize(y, pred[m], poets, C, n_boot, n_perm, seed)
        names, pv = poet_votes(P, poets)
        pl = poet_label.reindex(names).to_numpy()
        poet = summarize(pl, pv, names, C, n_boot, n_perm, seed)
        for level, s in (("сегменты", seg), ("поэты", poet)):
            row = {"метод": m, "уровень": level, **s}
            if PRE_ISLAMIC in cidx and m != "majority":
                j = cidx[PRE_ISLAMIC]
                if level == "сегменты":
                    row["AUC_P_pre_islamic"] = round(roc_auc_score(y == j, P[:, j]), 4)
                else:
                    mean_p = pd.Series(P[:, j]).groupby(poets).mean().reindex(names).to_numpy()
                    row["AUC_P_pre_islamic"] = round(roc_auc_score(pl == j, mean_p), 4)
            rows.append(row)
    metrics = pd.DataFrame(rows)

    cmp_rows = []
    if "bert" in methods:
        for other in [m for m in methods if m not in ("bert", "majority")]:
            for metric in ("balanced_acc", "macro_f1"):
                cmp_rows.append({"метод_A": "bert", "метод_B": other,
                                 **paired_compare(y, pred["bert"], pred[other], poets, C, metric,
                                                  n_boot, n_perm, seed)})
    compare = pd.DataFrame(cmp_rows)
    if len(compare):
        compare["p_Холм"] = holm(compare["p_двусторонний"]).round(4)

    main = "bert" if "bert" in methods else methods[0]
    cm = confusion(y, pred[main], C)
    per_class = per_class_table(cm, classes)
    per_class.insert(1, "поэтов", [int(preds.loc[preds["label"] == c, "author"].nunique()) for c in classes])
    conf = pd.DataFrame(cm.astype(int), index=[f"истина:{c}" for c in classes],
                        columns=[f"предсказано:{c}" for c in classes])
    by_orig = (pd.crosstab(preds["period"].rename("исходная_эпоха"),
                           pd.Series(np.array(classes)[pred[main]], index=preds.index, name="предсказано"),
                           normalize="index").round(3))
    by_orig.insert(0, "сегментов", preds.groupby("period").size())
    by_orig.insert(0, "эпоха", [ERA_NAMES_RU.get(k, k) for k in by_orig.index])
    return {"metrics": metrics, "compare": compare, "per_class": per_class, "confusion": conf,
            "by_original_era": by_orig}


def main(argv=None, bert_fn=None) -> dict:
    ap = argparse.ArgumentParser(description="CV классификатора эпохи по поэтам (APCD)")
    ap.add_argument("--data", default=str(config.DATA_PROCESSED / "era_dataset.parquet"))
    ap.add_argument("--scheme", default=config.ERA_SCHEME, choices=["binary", "coarse", "full"])
    ap.add_argument("--out-dir", default=None, help="по умолчанию data/processed/era_<схема>/")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--folds", type=int, default=0, help="сколько фолдов прогнать (0 = все)")
    ap.add_argument("--resume", action="store_true", help="пропустить фолды, уже записанные на диск")
    ap.add_argument("--checkpoint", default=config.CHECKPOINT)
    ap.add_argument("--epochs", type=int, default=config.ERA_EPOCHS)
    ap.add_argument("--lr", type=float, default=config.ERA_LEARNING_RATE)
    ap.add_argument("--batch-size", type=int, default=config.ERA_BATCH_SIZE)
    ap.add_argument("--patience", type=int, default=config.ERA_PATIENCE)
    ap.add_argument("--attn", default=config.ERA_ATTN, choices=["sdpa", "eager"],
                    help="реализация внимания BERT (по умолчанию eager; sdpa давала NaN с паддингом)")
    ap.add_argument("--max-seg-per-poet", type=int, default=config.ERA_MAX_SEG_PER_POET)
    ap.add_argument("--max-seg-per-class", type=int, default=config.ERA_MAX_SEG_PER_CLASS)
    ap.add_argument("--eval-seg-per-poet", type=int, default=config.ERA_EVAL_SEG_PER_POET)
    ap.add_argument("--eval-seg-per-class", type=int, default=config.ERA_EVAL_SEG_PER_CLASS)
    ap.add_argument("--min-poets-per-class", type=int, default=config.ERA_MIN_POETS_PER_CLASS)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--watchdog-min", type=float, default=0,
                    help="завершить процесс (код 3), если столько минут нет прогресса; 0 = выкл")
    args = ap.parse_args(argv)
    setup_logging(f"era_cv_{args.scheme}")
    if bert_fn is None:
        from aa.era.model import bert_fit_predict as bert_fn
    watchdog.start(args.watchdog_min)

    out_dir = Path(args.out_dir or config.DATA_PROCESSED / f"era_{args.scheme}")
    out_dir.mkdir(parents=True, exist_ok=True)
    df, classes = apply_scheme(read_table(args.data), args.scheme, args.min_poets_per_class)
    log.info("схема %s: классы %s", args.scheme, classes)
    log.info("поэтов / сегментов по классам:\n%s",
             df.groupby("label").agg(поэтов=("author", "nunique"), сегментов=("text", "size"))
               .reindex(classes).to_string())
    fold_of = poet_folds(df, args.n_splits, args.seed)
    n_run = args.folds or args.n_splits

    parts, infos, failed, t0 = [], [], [], time.time()
    for k in range(n_run):
        part_file = out_dir / f"era_cv_predictions_fold{k}.csv"
        if args.resume and part_file.exists():
            log.info("фолд %d: беру готовый %s", k, part_file)
            parts.append(pd.read_csv(part_file))
            continue
        tk = time.time()
        try:
            part, info = run_fold(df, fold_of, k, classes, args, bert_fn, with_baselines=not args.no_baselines)
        except Exception:  # noqa: BLE001 — один сломанный фолд не должен убивать весь прогон
            log.exception("фолд %d НЕ посчитан — пропускаю его и иду дальше (перезапустить: та же команда "
                          "с --resume)", k)
            failed.append(k)
            continue
        part.to_csv(part_file, index=False)
        parts.append(part)
        infos.append(info)
        done = k + 1
        el = time.time() - t0
        log.info("фолд %d готов за %s; всего прошло %s, осталось ~%s", k, fmt_duration(time.time() - tk),
                 fmt_duration(el), fmt_duration(el / done * (n_run - done)))

    if failed:
        log.warning("не посчитаны фолды %s; итог — по фолдам %s", failed, sorted(set(range(n_run)) - set(failed)))
    if not parts:
        raise SystemExit("ни один фолд не посчитан — см. лог выше")
    preds = pd.concat(parts, ignore_index=True)
    preds.to_csv(out_dir / "era_cv_predictions.csv", index=False)
    if infos:
        (out_dir / "era_cv_train_info.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1,
                                                                   default=str), encoding="utf-8")
    res = evaluate(preds, classes, n_boot=args.n_boot, n_perm=args.n_boot, seed=args.seed)
    res["metrics"].to_csv(out_dir / "era_cv_metrics.csv", index=False)
    res["compare"].to_csv(out_dir / "era_cv_compare.csv", index=False)
    res["per_class"].to_csv(out_dir / "era_cv_per_class.csv", index=False)
    res["confusion"].to_csv(out_dir / "era_cv_confusion.csv")
    res["by_original_era"].to_csv(out_dir / "era_cv_by_original_era.csv")

    cols = ["метод", "уровень", "поэтов", "balanced_acc", "balanced_acc_ci95", "balanced_acc_p_vs_chance",
            "macro_f1", "adjacent_acc", "AUC_P_pre_islamic", "случайный_уровень_balanced_acc"]
    log.info("ИТОГ (%s, фолды %s из %d):\n%s", args.scheme, sorted(preds["fold"].unique().tolist()), args.n_splits,
             res["metrics"][[c for c in cols if c in res["metrics"].columns]].to_string(index=False))
    if len(res["compare"]):
        log.info("BERT против базовых линий (парно, по поэтам):\n%s", res["compare"].to_string(index=False))
    log.info("BERT по классам:\n%s", res["per_class"].to_string(index=False))
    log.info("BERT: куда уходят исходные эпохи APCD:\n%s", res["by_original_era"].to_string())
    log.info("файлы: %s", out_dir)
    return res


if __name__ == "__main__":
    main()
