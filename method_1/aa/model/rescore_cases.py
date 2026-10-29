"""Пересчитать p-value A8 по уже сохранённым результатам — без обучения моделей (около минуты).

Что берёт (всё уже есть после run_cv и run_cases):
    data/processed/cv_predictions.csv (+ cv_pairwise.csv)   — эталон: невиденные в CV поэмы авторов;
    data/processed/mualaqat_cases_per_seed.csv              — уверенность муалак по seed.
Что пишет (старые файлы не трогает):
    mualaqat_cases_v2.csv                       — итог: p по seed объединены, поправка Холма, вывод;
    mualaqat_cases_v2_per_seed.csv              — p каждого seed;
    mualaqat_cases_v2_reference_fit.csv         — оценки модели эталона по авторам (ICC, τ, σ, связь с длиной);
    mualaqat_cases_v2_calibration.csv           — p каждой настоящей поэмы против эталона из остальных поэм;
    mualaqat_cases_v2_calibration_summary.csv   — доля p ≤ 0.05 и ≤ 0.20 на настоящих поэмах (≈ 5% и 20%).

    python -m aa.model.rescore_cases
    # проверки устойчивости — в отдельные файлы, чтобы не перезаписать основной результат:
    python -m aa.model.rescore_cases --scale prob --out data/processed/mualaqat_cases_v2_prob.csv
    python -m aa.model.rescore_cases --method bootstrap --out data/processed/mualaqat_cases_v2_boot.csv

Что изменилось по сравнению с прежней таблицей (см. aa/eval/reference.py):
  * эталон — «одна поэма из k сегментов», а не смесь нескольких коротких поэм: сегменты одной поэмы
    похожи друг на друга, и прежний эталон был слишком узким (p-value занижены);
  * p-value трёх seed объединяются правилом Рюгера (1.5 × медиана), корректным при любой зависимости;
  * поправка Холма на число оцениваемых муалак (у Хариса и Амра мало данных — их нет в поправке).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from aa import config
from aa.eval.reference import calibration_summary, loo_calibration
from aa.log import fmt_duration, setup_logging
from aa.model.run_cases import (ALPHA, ReferenceCache, combine_seeds, load_reference, log_final,
                                log_reference_fit, rescore)

log = logging.getLogger("aa.rescore_cases")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference-dir", default=str(config.DATA_PROCESSED), help="папка с результатами run_cv")
    ap.add_argument("--per-seed", default=str(config.DATA_PROCESSED / "mualaqat_cases_per_seed.csv"))
    ap.add_argument("--out", default=str(config.DATA_PROCESSED / "mualaqat_cases_v2.csv"))
    ap.add_argument("--n-boot", type=int, default=10000, help="значений в эталоне на муаллаку")
    ap.add_argument("--scale", choices=["logit", "prob"], default="logit")
    ap.add_argument("--eps", type=float, default=0.01, help="обрезка уверенности перед логитом")
    ap.add_argument("--method", choices=["bayes", "bootstrap"], default="bayes")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--calibration-boot", type=int, default=2000,
                    help="значений в эталоне при проверке на настоящих поэмах")
    ap.add_argument("--no-calibration", action="store_true", help="не проверять эталон на настоящих поэмах")
    args = ap.parse_args()

    setup_logging("rescore_cases")
    log.info("параметры: %s", vars(args))
    t_start = time.time()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    reference, reference_segments, author_stats, _ = load_reference(Path(args.reference_dir))
    if reference_segments is None:
        raise SystemExit("Нет cv_predictions.csv с колонкой true_score — сначала запусти run_cv.")
    fit = log_reference_fit(reference_segments, args.scale, args.eps)
    fit.to_csv(out.with_name(out.stem + "_reference_fit.csv"), index=False)

    per_seed = pd.read_csv(args.per_seed)
    need = {"муаллака_id", "приписан", "увер_в_приписанном", "сегментов", "seed", "предсказан",
            "доля_сегм_за_приписанного"}
    missing = need - set(per_seed.columns)
    if missing:
        raise SystemExit(f"В {args.per_seed} нет колонок {sorted(missing)} — это не вывод run_cases.")
    if "сегм_автора_в_обучении" not in per_seed.columns:
        log.warning("в таблице нет «сегм_автора_в_обучении» — правило «мало данных» не применится")
    log.info("муалак: %d, seed: %s", per_seed["муаллака_id"].nunique(),
             sorted(int(x) for x in per_seed["seed"].unique()))

    cache = ReferenceCache(reference_segments, n_boot=args.n_boot, seed=config.SEED, scale=args.scale,
                           eps=args.eps, method=args.method)
    per_seed_v2 = rescore(per_seed, reference, cache)
    per_seed_path = out.with_name(out.stem + "_per_seed.csv")
    per_seed_v2.to_csv(per_seed_path, index=False)
    final = combine_seeds(per_seed_v2, author_stats, alpha=args.alpha)
    final.to_csv(out, index=False)
    log_final(final, per_seed["seed"].nunique())

    if not args.no_calibration:
        log.info("проверка эталона на настоящих поэмах: каждая поэма автора по очереди — «спорная», "
                 "эталон — по остальным его поэмам (%d значений на поэму)", args.calibration_boot)
        t0 = time.time()

        def progress(i, total, author):
            el = time.time() - t0
            log.info("  проверка: %s (%d/%d) | прошло %s, осталось ~%s", author, i, total,
                     fmt_duration(el), fmt_duration(el / i * (total - i)))

        cal = loo_calibration(reference_segments, n_boot=args.calibration_boot, seed=config.SEED,
                              scale=args.scale, eps=args.eps, method=args.method, progress=progress)
        cal.to_csv(out.with_name(out.stem + "_calibration.csv"), index=False)
        summ = calibration_summary(cal)
        summ.to_csv(out.with_name(out.stem + "_calibration_summary.csv"), index=False)
        log.info("доля ложных тревог на НАСТОЯЩИХ поэмах (у честного эталона ≈ 0.05 и ≈ 0.20):\n%s",
                 summ.to_string(index=False))
        row = summ[(summ["длина_поэмы"] == "все") & (summ["эталон"] == "новый")]
        if len(row):
            share = float(row["доля_p≤0.05"].iloc[0])
            lo = float(str(row["ДИ95_0.05"].iloc[0]).split("–")[0])
            if lo > args.alpha:
                log.warning("новый эталон даёт p ≤ 0.05 у %.1f%% настоящих поэм — больше 5%% и за пределами "
                            "случайности: p-value муалак могут быть занижены, выводы осторожнее", share * 100)
            else:
                log.info("новый эталон: p ≤ 0.05 у %.1f%% настоящих поэм — согласуется с 5%%", share * 100)
        log.info("в диванах могут быть и поэмы с ошибочной атрибуцией: небольшое превышение 5% ожидаемо "
                 "даже у идеального эталона")

    log.info("готово за %s. Файлы: %s и %s_*.csv рядом с ним", fmt_duration(time.time() - t_start), out,
             out.with_name(out.stem))


if __name__ == "__main__":
    main()
