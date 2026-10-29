"""Эпоха муаллак: настолько ли муаллака «доисламская», насколько доисламские поэмы дивана её поэта?

Вопрос тот же, что у A8 (типичность муаллаки для своего поэта), но признак другой: не «похожа на поэта»,
а P(доисламская) по классификатору эпохи. Если муаллака (или её часть — напр. вступление) написана или
сильно переработана позже, она должна выглядеть МЕНЕЕ доисламской, чем поэмы того же поэта.

Почему это честно:
  * модель эпох учится на APCD БЕЗ наших 10 поэтов и без бейтов муаллак и диванов
    (aa.era.apcd: исключение по имени + точные и нечёткие совпадения бейтов). Значит, и муаллака,
    и поэмы диванов для неё одинаково невиденные, и эталон из поэм дивана сопоставим с муаллакой;
  * эталон — тот же «одна невиденная поэма из k сегментов» (aa/eval/reference.py, байесовский,
    с учётом сходства сегментов одной поэмы), p = доля эталона ≤ P(доисламская) муаллаки;
  * несколько seed (своя валидация, выборка и инициализация) -> правило Рюгера; поправка Холма —
    по муаллакам, отдельно для каждого метода;
  * второй метод для сравнения — символьные n-граммы на тех же данных (--no-ngram отключает);
  * вероятности калибруются температурой по валидационным поэтам APCD: дообученный BERT переуверен,
    и без этого P(доисламская) у большинства сегментов дивана > 0.99 и эталон на логит-шкале
    (обрезка [0.01, 0.99]) вырождается. Решение argmax калибровка не меняет.
Диваны берутся очищенными так же, как в A8 (aa.data.prepare): без копий муаллак и вариантов их бейтов.

Итог (data/processed/era_<схема>/):
  era_cases.csv              по (муаллака, метод): P(доисламская), ожидаемая для поэмы той же длины,
                             p по seed, p (Рюгер), p Холма, вывод
  era_cases_per_seed.csv     то же по seed
  era_cases_segments.csv     по сегменту муаллаки: номер, P всех классов (среднее по seed), разброс
  era_diwan_summary.csv      по поэту: средняя P(доисламская) поэм дивана, доля сегментов, отнесённых
                             к доисламской эпохе (проверка переноса модели на тексты aldiwan)
  era_cases_reference_fit.csv  оценки модели эталона (ICC, связь с длиной поэмы)

    python -m aa.era.run_era_cases --era-data data/processed/era_dataset.parquet \
        --data data/processed/dataset.parquet --scheme coarse --seeds 3
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
from aa.data.prepare import prepare_training_data
from aa.era import baselines, watchdog
from aa.era.apcd import read_table
from aa.era.labels import PRE_ISLAMIC, is_project_poet
from aa.era.metrics import calibrate_sets
from aa.era.split import apply_scheme, cap_sample, class_weights, split_val_poets
from aa.eval.reference import combine_seed_pvalues, holm, reference_fit_table
from aa.log import fmt_duration, setup_logging
from aa.model.run_cases import ReferenceCache

log = logging.getLogger("aa.era.run_cases")

ALPHA, WEAK = 0.05, 0.20
REF_EPS = 0.01          # как в эталоне A8: вероятности обрезаются до [0.01, 0.99] перед логитом
VERDICTS = ("значимо менее доисламская, чем диван (p_Холм ≤ 0.05)",
            "менее доисламская, чем диван (p ≤ 0.05, но не после поправки)",
            "слабый сигнал (p ≤ 0.20)",
            "в пределах нормы дивана",
            "нет эталона")


def verdict(p: float, p_holm: float) -> str:
    if p is None or not np.isfinite(p):
        return VERDICTS[4]
    if np.isfinite(p_holm) and p_holm <= ALPHA:
        return VERDICTS[0]
    if p <= ALPHA:
        return VERDICTS[1]
    if p <= WEAK:
        return VERDICTS[2]
    return VERDICTS[3]


def poems_of(df: pd.DataFrame, scores: np.ndarray) -> dict[str, list[list[float]]]:
    """{поэт: [[P сегментов поэмы 1], ...]} в порядке сегментов."""
    d = df[["author", "poem_id"]].copy()
    d["s"] = scores
    d = d[np.isfinite(d["s"])]          # сегменты с NaN (редко) не входят в эталон
    out: dict[str, list[list[float]]] = {}
    for (a, _), g in d.groupby(["author", "poem_id"], sort=False):
        out.setdefault(str(a), []).append(g["s"].tolist())
    return out


def score_seed(seed: int, method: str, mual: pd.DataFrame, diwan: pd.DataFrame, p_mual: np.ndarray,
               p_diwan: np.ndarray, classes: list[str], n_boot: int) -> tuple[pd.DataFrame, dict]:
    j = classes.index(PRE_ISLAMIC)
    ref = poems_of(diwan, p_diwan[:, j])
    cache = ReferenceCache(ref, n_boot=n_boot, seed=seed)
    rows = []
    for pid, g in mual.groupby("poem_id", sort=False):
        idx = g.index.to_numpy()
        idx = idx[np.isfinite(p_mual[idx]).all(axis=1)]
        a = str(g["author"].iloc[0])
        if idx.size == 0:
            log.warning("%s (%s, seed %d): все сегменты муаллаки дали NaN — пропуск", a, method, seed)
            continue
        val = float(p_mual[idx, j].mean())
        seg_a = p_diwan[(diwan["author"] == a).to_numpy(), j]
        sat = float(np.mean((seg_a < REF_EPS) | (seg_a > 1 - REF_EPS))) if seg_a.size else 0.0
        if sat > 0.5:
            log.warning("%s (%s, seed %d): у %.0f%% сегментов дивана P(доисламская) за границей [%.2f, %.2f] — "
                        "на логит-шкале они обрезаются, эталон огрублён", a, method, seed, 100 * sat,
                        REF_EPS, 1 - REF_EPS)
        try:
            cols = cache.columns(a, val, idx.size)
        except (ValueError, FloatingPointError) as e:   # все значения у границы -> нулевой разброс
            log.warning("%s (%s, seed %d): эталон не построен (%s)", a, method, seed, e)
            cols = {"p": np.nan, "ожидаемая_увер": np.nan, "граница_5%": np.nan, "ICC_автора": np.nan,
                    "эталон": "вырожден: почти все P у границы"}
        rows.append({"муаллака_id": str(pid).replace("mualaka::", ""), "приписан": a, "метод": method,
                     "seed": seed, "сегментов": int(idx.size), "P_pre": round(val, 4),
                     "доля_сегм_pre": round(float((p_mual[idx].argmax(1) == j).mean()), 3),
                     "p": cols["p"], "ожидаемая_P_pre": cols["ожидаемая_увер"],
                     "граница_5%": cols["граница_5%"], "ICC_поэта": cols["ICC_автора"],
                     "поэм_в_эталоне": len(ref.get(a, [])), "эталон": cols["эталон"]})
    return pd.DataFrame(rows), ref


def save_seed(path: Path, results: dict, info: dict, temps: dict) -> None:
    arrays = {f"{m}__{k}": np.asarray(v) for m, pr in results.items() for k, v in pr.items() if k != "val"}
    np.savez_compressed(path, **arrays,
                        _meta=np.array(json.dumps({"info": info, "temps": temps}, ensure_ascii=False, default=str)))


def load_seed(path: Path, n_mual: int, n_diwan: int) -> tuple[dict, dict, dict]:
    z = np.load(path, allow_pickle=False)
    results: dict = {}
    for key in z.files:
        if key == "_meta":
            continue
        m, k = key.split("__", 1)
        results.setdefault(m, {})[k] = z[key]
    if any(r["mual"].shape[0] != n_mual or r["diwan"].shape[0] != n_diwan for r in results.values()):
        raise SystemExit(f"{path}: сохранено для другого набора муаллак/диванов — удали файл или запусти без --resume")
    meta = json.loads(str(z["_meta"]))
    return results, meta["info"], meta["temps"]


def combine(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mid, method), g in per_seed.groupby(["муаллака_id", "метод"], sort=False):
        ps = g["p"].astype(float)
        rows.append({"муаллака_id": mid, "приписан": g["приписан"].iloc[0], "метод": method,
                     "сегментов": int(g["сегментов"].iloc[0]), "seed": len(g),
                     "P_pre": round(g["P_pre"].mean(), 3), "P_pre_мин": round(g["P_pre"].min(), 3),
                     "P_pre_макс": round(g["P_pre"].max(), 3),
                     "ожидаемая_P_pre": round(float(g["ожидаемая_P_pre"].median()), 3),
                     "граница_5%": round(float(g["граница_5%"].median()), 3),
                     "доля_сегм_pre": round(g["доля_сегм_pre"].mean(), 2),
                     "p_по_seed": " / ".join(f"{x:.3f}" for x in ps),
                     "p": combine_seed_pvalues(ps), "ICC_поэта": g["ICC_поэта"].median(),
                     "поэм_в_эталоне": int(g["поэм_в_эталоне"].iloc[0]), "эталон": g["эталон"].iloc[0]})
    out = pd.DataFrame(rows)
    out["p_Холм"] = np.nan
    for method, idx in out.groupby("метод").groups.items():
        ok = [i for i in idx if np.isfinite(out.at[i, "p"])]
        if ok:
            out.loc[ok, "p_Холм"] = holm(out.loc[ok, "p"].to_numpy())
    out["вывод"] = [verdict(p, ph) for p, ph in zip(out["p"], out["p_Холм"])]
    out["p"] = out["p"].round(4)
    out["p_Холм"] = out["p_Холм"].round(4)
    return out.sort_values(["метод", "p"]).reset_index(drop=True)


def main(argv=None, bert_fn=None) -> dict:
    ap = argparse.ArgumentParser(description="P(доисламская) муаллак против эталона из дивана поэта")
    ap.add_argument("--era-data", default=str(config.DATA_PROCESSED / "era_dataset.parquet"))
    ap.add_argument("--data", default=str(config.DATA_PROCESSED / "dataset.parquet"),
                    help="датасет проекта: муаллаки + диваны aldiwan")
    ap.add_argument("--scheme", default=config.ERA_SCHEME, choices=["binary", "coarse", "full"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seeds", type=int, default=3)
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
    ap.add_argument("--no-ngram", action="store_true", help="не считать второй метод (n-граммы)")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                    help="не калибровать вероятности температурой по валидационным поэтам")
    ap.add_argument("--save-model", default=None, help="куда сохранить модель первого seed (~0.5 ГБ)")
    ap.add_argument("--resume", action="store_true",
                    help="не обучать заново seed, чьи предсказания уже сохранены (era_cases_seed<N>.npz)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--watchdog-min", type=float, default=0,
                    help="завершить процесс (код 3), если столько минут нет прогресса; 0 = выкл")
    args = ap.parse_args(argv)
    setup_logging(f"era_cases_{args.scheme}")
    if bert_fn is None:
        from aa.era.model import bert_fit_predict as bert_fn
    watchdog.start(args.watchdog_min)

    out_dir = Path(args.out_dir or config.DATA_PROCESSED / f"era_{args.scheme}")
    out_dir.mkdir(parents=True, exist_ok=True)

    era, classes = apply_scheme(read_table(args.era_data), args.scheme, args.min_poets_per_class)
    if PRE_ISLAMIC not in classes:
        raise SystemExit(f"в схеме {args.scheme} после фильтров нет класса {PRE_ISLAMIC}")
    leaked = sorted({a for a in era["author"].unique() if is_project_poet(a)})
    if leaked:
        raise SystemExit(f"в данных эпох есть поэты проекта {leaked}: пересобери era_dataset (aa.era.apcd)")
    log.info("схема %s: классы %s; поэтов %d, сегментов %d", args.scheme, classes,
             era["author"].nunique(), len(era))

    diwan, mual, _ = prepare_training_data(read_table(args.data), log_details=False)
    diwan = diwan[~diwan["poem_id"].astype(str).str.startswith("apcd::")].reset_index(drop=True)
    mual = mual.reset_index(drop=True)
    if mual.empty:
        raise SystemExit("в датасете проекта нет муаллак (poem_id 'mualaka::…')")
    log.info("муаллак: %d (%d сегментов); диваны: %d поэм, %d сегментов", mual["poem_id"].nunique(),
             len(mual), diwan["poem_id"].nunique(), len(diwan))

    texts = {"mual": list(mual["text"]), "diwan": list(diwan["text"])}
    per_seed, seg_probs, infos, diwan_rows, fits, failed = [], [], [], [], [], []
    fitted: set = set()
    t0 = time.time()
    for s in range(args.seeds):
        seed = args.seed + s
        train, val = split_val_poets(era, args.val_frac, seed=seed + 100)
        train_s = cap_sample(train, args.max_seg_per_poet, args.max_seg_per_class, seed=seed)
        val_s = cap_sample(val, args.eval_seg_per_poet, args.eval_seg_per_class // 3, seed=seed)
        w = class_weights(train_s["label"], classes)
        log.info("seed %d: обучение %d сегм. / %d поэтов, валидация %d / %d", seed, len(train_s),
                 train_s["author"].nunique(), len(val_s), val_s["author"].nunique())
        cfg = dict(checkpoint=args.checkpoint, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                   patience=args.patience, attn=args.attn, class_weights=w, seed=seed)
        seed_texts = dict(texts, val=list(val_s["text"]))
        y_val = val_s["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
        cache = out_dir / f"era_cases_seed{seed}.npz"
        if args.resume and cache.exists():
            results, info, temps = load_seed(cache, len(mual), len(diwan))
            log.info("seed %d: беру сохранённые предсказания %s (без обучения)", seed, cache)
        else:
            try:
                probs, info = bert_fn(train_s, val_s, classes, seed_texts,
                                      save_to=args.save_model if (s == 0 and args.save_model) else None, **cfg)
            except Exception:  # noqa: BLE001 — один сломанный seed не должен убивать весь прогон
                log.exception("seed %d НЕ посчитан — пропускаю и иду дальше", seed)
                failed.append(seed)
                continue
            results, temps = {}, {}
            results["bert"], temps["bert"] = calibrate_sets(probs, y_val) if args.calibrate else (probs, 1.0)
            if not args.no_ngram:
                watchdog.beat(f"seed {seed}: n-граммы")
                ng = baselines.ngram_fit_predict(train_s, classes, seed_texts, seed=seed)
                results["ngram"], temps["ngram"] = calibrate_sets(ng, y_val) if args.calibrate else (ng, 1.0)
            save_seed(cache, results, info, temps)
            log.info("seed %d: предсказания сохранены в %s", seed, cache)
        log.info("seed %d: температура калибровки (по валидационным поэтам): %s", seed,
                 {m: round(t, 2) for m, t in temps.items()})
        infos.append(dict(info or {}, seed=seed, temperature=temps))
        for method, pr in results.items():
            watchdog.beat(f"seed {seed}: эталон {method}")
            tab, ref = score_seed(seed, method, mual, diwan, pr["mual"], pr["diwan"], classes, args.n_boot)
            per_seed.append(tab)
            sp = pd.DataFrame(pr["mual"], columns=[f"p_{c}" for c in classes])
            sp["метод"], sp["seed"], sp["seg"] = method, seed, np.arange(len(mual))
            seg_probs.append(sp)
            j = classes.index(PRE_ISLAMIC)
            d = diwan[["author", "poem_id"]].assign(P_pre=pr["diwan"][:, j],
                                                    is_pre=pr["diwan"].argmax(1) == j)
            d = d[np.isfinite(pr["diwan"]).all(axis=1)]
            pm = d.groupby(["author", "poem_id"]).agg(P_pre=("P_pre", "mean"))
            dsum = d.groupby("author").agg(сегментов=("P_pre", "size"), доля_сегм_pre=("is_pre", "mean"),
                                           P_pre_сегм=("P_pre", "mean"))
            dsum["поэм"] = pm.groupby("author").size()
            dsum["P_pre_поэм_медиана"] = pm.groupby("author")["P_pre"].median()
            diwan_rows.append(dsum.reset_index().assign(метод=method, seed=seed))
            if method not in fitted:          # оценки эталона — по первому удачному seed
                fits.append(reference_fit_table(ref).assign(метод=method))
                fitted.add(method)
        el = time.time() - t0
        log.info("seed %d готов; прошло %s, осталось ~%s", seed, fmt_duration(el),
                 fmt_duration(el / (s + 1) * (args.seeds - s - 1)))

    if failed:
        log.warning("не посчитаны seed %s; итог — по остальным", failed)
    if not per_seed:
        raise SystemExit("ни один seed не посчитан — см. лог выше")
    per_seed = pd.concat(per_seed, ignore_index=True)
    final = combine(per_seed)

    sp = pd.concat(seg_probs, ignore_index=True)
    pcols = [f"p_{c}" for c in classes]
    seg = sp.groupby(["метод", "seg"])[pcols].mean().reset_index()
    seg["P_pre_мин"] = sp.groupby(["метод", "seg"])[f"p_{PRE_ISLAMIC}"].min().to_numpy()
    seg["P_pre_макс"] = sp.groupby(["метод", "seg"])[f"p_{PRE_ISLAMIC}"].max().to_numpy()
    meta = mual[["poem_id", "author", "text"]].assign(seg=np.arange(len(mual)))
    meta["сегмент_№"] = mual.groupby("poem_id").cumcount().to_numpy() + 1
    meta["начало_текста"] = meta["text"].str[:60]
    seg = meta.drop(columns=["text"]).merge(seg, on="seg").drop(columns=["seg"])
    seg["предсказано"] = np.array(classes)[seg[pcols].to_numpy().argmax(1)]
    seg = seg.sort_values(["метод", "poem_id", "сегмент_№"], kind="stable").reset_index(drop=True)

    diwan_sum = (pd.concat(diwan_rows).groupby(["метод", "author"])
                 [["поэм", "сегментов", "P_pre_сегм", "P_pre_поэм_медиана", "доля_сегм_pre"]]
                 .mean().round(3).reset_index())

    final.to_csv(out_dir / "era_cases.csv", index=False)
    per_seed.to_csv(out_dir / "era_cases_per_seed.csv", index=False)
    seg.round(4).to_csv(out_dir / "era_cases_segments.csv", index=False)
    diwan_sum.to_csv(out_dir / "era_diwan_summary.csv", index=False)
    pd.concat(fits).to_csv(out_dir / "era_cases_reference_fit.csv", index=False)
    (out_dir / "era_cases_train_info.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1,
                                                                  default=str), encoding="utf-8")

    show = ["метод", "приписан", "сегментов", "P_pre", "ожидаемая_P_pre", "граница_5%", "p_по_seed", "p",
            "p_Холм", "вывод"]
    log.info("ИТОГ: P(доисламская) муаллак против поэм дивана (%s, %d seed):\n%s", args.scheme, args.seeds,
             final[show].to_string(index=False))
    log.info("диваны (проверка переноса модели на aldiwan; поэты в обучении не участвовали):\n%s",
             diwan_sum.to_string(index=False))
    log.info("файлы: %s; всего %s", out_dir, fmt_duration(time.time() - t0))
    return {"final": final, "per_seed": per_seed, "segments": seg, "diwan": diwan_sum}


if __name__ == "__main__":
    main()
