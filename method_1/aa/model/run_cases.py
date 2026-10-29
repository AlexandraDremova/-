"""A8 — атрибуция 10 муалак: главный результат проекта.

Вопрос: насколько муаллака похожа на поэмы приписываемого ей поэта — по сравнению с тем,
насколько похожи на него его собственные (невиденные моделью) поэмы из дивана.

Процедура:
  1. Данные: диваны без муалак, без их точных копий и без вариантов их бейтов
     (та же подготовка, что в run_cv — aa.data.prepare).
  2. Для каждого seed: ансамбль учится на случайных 2/3 диванов (по поэмам, по каждому автору) —
     столько же, сколько модели эталона в CV, поэтому шкалы уверенности сопоставимы.
  3. Уверенность в приписанном авторе = среднее по сегментам муаллаки.
  4. Эталон «одна поэма из k сегментов» (k = число сегментов муаллаки) по невиденным в CV поэмам
     того же автора: модель «эффект поэмы + разброс сегментов внутри поэмы», байесовский
     предсказательный эталон (aa/eval/reference.py, single_poem_reference). Сегменты одной поэмы похожи друг на друга,
     поэтому разброс средней по одной поэме не сводится к разбросу смеси нескольких коротких поэм.
  5. p-value = доля эталона ≤ уверенности муаллаки (с поправкой +1). По seed p объединяются
     правилом Рюгера (1.5 × медиана для 3 seed), по муаллакам — поправка Холма на число
     оцениваемых муалак.

Вывод (не по голосованию — голосование систематически перекошено к Зухайру/ан-Набиге/Абиду, см. CV):
  p_Холм ≤ 0.05 — аномально низкая похожесть на автора; p ≤ 0.05 без поправки — ниже типичной;
  p ≤ 0.20 — слабый сигнал; иначе — в пределах нормы; у автора < 30 обучающих сегментов — нельзя оценить.

Пересчитать готовые результаты без обучения: python -m aa.model.rescore_cases (около минуты).

    python -m aa.model.run_cases --data data/processed/dataset.parquet --seeds 3 \
        --epochs 30 --early-stop-loss 0.01
Требует результатов run_cv (cv_predictions.csv, cv_poem_scores.csv, cv_pairwise.csv) в data/processed/.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from aa import config
from aa.data.prepare import prepare_training_data, split_and_decontaminate, subsample_poems  # noqa: F401
from aa.eval.reference import (author_cv_stats, combine_seed_pvalues, holm, length_matched_reference,
                               p_lower, percentile_in, reference_fit_table, segment_scores_by_poem,
                               single_poem_reference)
from aa.log import fmt_duration, setup_logging
from aa.model import ensemble

log = logging.getLogger("aa.run_cases")

MIN_TRAIN = config.EXAMPLES_PER_POET // 2     # меньше — «нельзя оценить»
ALPHA, BELOW = 0.05, 0.20                     # уровень значимости и граница «слабого сигнала»
ANOMALY = ALPHA                               # старое имя, оставлено для совместимости

VERDICTS = ("аномально низкая (p_Холм ≤ 0.05)",
            "ниже типичной (p ≤ 0.05, но не после поправки)",
            "слабый сигнал (p ≤ 0.20)",
            "в пределах нормы",
            "нет эталона",
            "нельзя оценить: мало данных")


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def conclusion(p: float, p_holm: float | None, n_train: int | None, min_train: int = MIN_TRAIN,
               alpha: float = ALPHA) -> str:
    """Вывод по муаллаке: p — объединённый по seed p-value, p_holm — он же с поправкой Холма."""
    if n_train is not None and _finite(n_train) and n_train < min_train:
        return VERDICTS[5]
    if not _finite(p):
        return VERDICTS[4]
    if _finite(p_holm) and p_holm <= alpha:
        return VERDICTS[0]
    if p <= alpha:
        return VERDICTS[1]
    if p <= BELOW:
        return VERDICTS[2]
    return VERDICTS[3]


def seed_category(p: float, alpha: float = ALPHA) -> str:
    """Категория одного seed — для колонки «устойчивость»."""
    if not _finite(p):
        return "нет эталона"
    return f"p ≤ {alpha:g}" if p <= alpha else (f"p ≤ {BELOW:g}" if p <= BELOW else f"p > {BELOW:g}")


class ReferenceCache:
    """Эталоны по (автор, число сегментов). Одинаковы для всех seed и муалак одной длины,
    поэтому считаются один раз. Хранит и новый эталон, и старый (только для сравнения)."""

    def __init__(self, reference_segments: dict, n_boot: int = 10000, seed: int = 0,
                 scale: str = "logit", eps: float = 0.01, old_boot: int = 2000, method: str = "bayes"):
        self.segments = reference_segments or {}
        self.n_boot, self.seed, self.scale, self.eps, self.old_boot = n_boot, seed, scale, eps, old_boot
        self.method = method
        self._new: dict = {}
        self._old: dict = {}

    def new(self, author: str, k: int):
        key = (author, k)
        if key not in self._new:
            self._new[key] = single_poem_reference(self.segments.get(author, []), k, n_boot=self.n_boot,
                                                   seed=self.seed, scale=self.scale, eps=self.eps,
                                                   method=self.method)
        return self._new[key]

    def old(self, author: str, k: int) -> np.ndarray:
        key = (author, k)
        if key not in self._old:
            self._old[key] = length_matched_reference(self.segments.get(author, []), k,
                                                      n_boot=self.old_boot, seed=self.seed)
        return self._old[key]

    def columns(self, author: str, value: float, k: int) -> dict:
        draws, fit = self.new(author, k)
        old = self.old(author, k)
        has = draws.size > 0
        return {
            "p": p_lower(value, draws),
            "граница_5%": round(float(np.quantile(draws, ALPHA)), 3) if has else np.nan,
            "ожидаемая_увер": round(float(np.median(draws)), 3) if has else np.nan,
            "ICC_автора": round(fit.icc, 3) if fit.ok else np.nan,
            "эталон": "ok" if fit.ok else fit.reason,
            "перцентиль_старый": round(percentile_in(value, old), 3) if old.size else np.nan,
        }


def aggregate_cases(mualaka_df: pd.DataFrame, analyze, threshold: float | None = None,
                    train_counts: dict | None = None,
                    min_train: int = MIN_TRAIN,
                    reference: pd.DataFrame | None = None,
                    ref_alpha: float = 0.05, min_ref_poems: int = 5,
                    reference_segments: dict | None = None,
                    n_boot: int = 10000, boot_seed: int = 0,
                    ref_cache: ReferenceCache | None = None) -> pd.DataFrame:
    """Сводит посегментные раскладки к одной строке на муаллаку (один seed). [без torch]

    analyze: функция texts -> list[{pred_author, confidence, scores}] (вызывается ОДИН раз на все
    сегменты всех муалак) либо готовый список раскладок, выровненный по строкам mualaka_df.
    reference: эталон по поэмам (cv_poem_scores.csv); reference_segments: {автор: [[сегменты поэмы]]}
    из cv_predictions.csv — для эталона «одна поэма из k сегментов» (или готовый ref_cache).
    """
    if ref_cache is None and reference_segments is not None:
        ref_cache = ReferenceCache(reference_segments, n_boot=n_boot, seed=boot_seed)
    mualaka_df = mualaka_df.reset_index(drop=True)
    all_res = analyze(mualaka_df["text"].tolist()) if callable(analyze) else list(analyze)
    if len(all_res) != len(mualaka_df):
        raise ValueError(f"Раскладок {len(all_res)}, а сегментов {len(mualaka_df)}.")
    rows = []
    for pid, idx in mualaka_df.groupby("poem_id", sort=False).groups.items():
        g = mualaka_df.loc[idx]
        attributed = str(g["author"].iloc[0])
        res = [all_res[i] for i in idx]
        n = len(res) or 1
        predicted = Counter(r["pred_author"] for r in res).most_common(1)[0][0]
        conf_pred = sum(r["confidence"] for r in res) / n
        conf_attr = sum(r["scores"].get(attributed, 0.0) for r in res) / n
        share_attr = sum(r["pred_author"] == attributed for r in res) / n
        verdict = "совпадает" if predicted == attributed else "РАСХОЖДЕНИЕ"
        flags = []
        if threshold is not None and conf_attr < threshold:
            flags.append("низкая уверенность")
        n_train = (train_counts or {}).get(attributed)
        if n_train is not None and n_train < min_train:
            flags.append(f"мало данных об авторе ({n_train} сегм.)")
        row = {
            "муаллака_id": str(pid).replace("mualaka::", ""),
            "приписан": attributed,
            "предсказан": predicted,
            "увер_в_приписанном": round(conf_attr, 3),
            "доля_сегм_за_приписанного": round(share_attr, 2),
            "увер_предсказанного": round(conf_pred, 3),
            "сегментов": len(g),
        }
        if train_counts is not None:
            row["сегм_автора_в_обучении"] = n_train
        if reference is not None:
            ref_auth = reference.loc[reference["author"] == attributed, "true_score"]
            pct_auth = percentile_in(conf_attr, ref_auth)
            row["поэм_автора_в_эталоне"] = int(len(ref_auth))
            row["перцентиль_у_автора"] = round(pct_auth, 3)
            row["перцентиль_у_всех"] = round(percentile_in(conf_attr, reference["true_score"]), 3)
            if len(ref_auth) < min_ref_poems:
                flags.append(f"эталон автора мал ({len(ref_auth)} поэм)")
            elif pct_auth <= ref_alpha:
                flags.append(f"ниже {1 - ref_alpha:.0%} поэм автора")
        if ref_cache is not None:
            row.update(ref_cache.columns(attributed, conf_attr, len(g)))
        row.update({"вердикт": verdict, "флаг": "; ".join(flags)})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("увер_в_приписанном").reset_index(drop=True)


# колонки старой версии (до 28.10): в пересчитанной таблице их заменяют p и перцентиль_старый
_OLD_REF_COLUMNS = ["перцентиль_по_длине", "граница_5%_по_длине", "медиана_эталона_по_длине"]


def rescore(table: pd.DataFrame, reference: pd.DataFrame | None, reference_segments,
            n_boot: int = 10000) -> pd.DataFrame:
    """Пересчитать p-value сохранённой таблицы seed по ТЕКУЩЕМУ эталону (для --resume и для
    rescore_cases). Уверенность в приписанном авторе берётся из таблицы и не меняется.
    reference_segments: {автор: [[сегменты поэмы]]} или готовый ReferenceCache."""
    t = table.copy()
    t["муаллака_id"] = t["муаллака_id"].astype(str)
    cache = reference_segments
    if cache is not None and not isinstance(cache, ReferenceCache):
        cache = ReferenceCache(cache, n_boot=n_boot)
    new_cols: dict[str, list] = {}
    for i, r in t.iterrows():
        a, v, k = str(r["приписан"]), float(r["увер_в_приписанном"]), int(r["сегментов"])
        if reference is not None:
            ref_auth = reference.loc[reference["author"] == a, "true_score"]
            t.at[i, "перцентиль_у_автора"] = round(percentile_in(v, ref_auth), 3)
            t.at[i, "перцентиль_у_всех"] = round(percentile_in(v, reference["true_score"]), 3)
        if cache is not None:
            for col, val in cache.columns(a, v, k).items():
                new_cols.setdefault(col, []).append(val)
    for col, vals in new_cols.items():
        t[col] = vals
    if cache is not None:
        t = t.drop(columns=[c for c in _OLD_REF_COLUMNS if c in t.columns])
    return t


def segment_details(mualaka_df: pd.DataFrame, all_res: list[dict]) -> pd.DataFrame:
    """Посегментная раскладка: кто победил в каждом сегменте и с какими вероятностями всех авторов."""
    mualaka_df = mualaka_df.reset_index(drop=True)
    rows = []
    for row, r in zip(mualaka_df.itertuples(index=False), all_res):
        top = sorted(r["scores"].items(), key=lambda kv: -kv[1])[:3]
        rows.append({
            "муаллака_id": str(row.poem_id).replace("mualaka::", ""),
            "приписан": row.author,
            "предсказан": r["pred_author"],
            "увер_предсказанного": r["confidence"],
            "увер_в_приписанном": r["scores"].get(str(row.author)),
            "голоса": json.dumps(r["votes"], ensure_ascii=False),
            "топ3": "; ".join(f"{a}={s:.2f}" for a, s in top),
            "начало_текста": str(row.text)[:60],
        })
    return pd.DataFrame(rows)


def combine_seeds(per_seed: pd.DataFrame, author_stats: pd.DataFrame | None = None,
                  min_train: int = MIN_TRAIN, alpha: float = ALPHA) -> pd.DataFrame:
    """Итог по муаллаке из таблиц нескольких seed (колонка 'seed'). [без torch]

    p по seed объединяются правилом Рюгера (combine_seed_pvalues), затем поправка Холма — по всем
    муаллакам, которые можно оценить (у автора ≥ min_train обучающих сегментов и есть эталон).
    """
    has_p = "p" in per_seed.columns
    rows = []
    for mid, g in per_seed.groupby("муаллака_id", sort=False):
        a = g["приписан"].iloc[0]
        n_train = g["сегм_автора_в_обучении"].iloc[0] if "сегм_автора_в_обучении" in g else None
        ps = g["p"].astype(float) if has_p else pd.Series([np.nan] * len(g))
        p_comb = combine_seed_pvalues(ps)
        cats = [seed_category(x, alpha) for x in ps]
        row = {
            "муаллака_id": mid, "приписан": a, "сегментов": int(g["сегментов"].iloc[0]),
            "сегм_автора_в_обучении": n_train,
        }
        if author_stats is not None and a in author_stats.index:
            row.update(author_stats.loc[a].to_dict())

        def med(col):
            return round(float(g[col].median()), 3) if col in g and g[col].notna().any() else np.nan

        row.update({
            "увер_в_приписанном": round(g["увер_в_приписанном"].mean(), 3),
            "увер_мин": round(g["увер_в_приписанном"].min(), 3),
            "увер_макс": round(g["увер_в_приписанном"].max(), 3),
            "доля_сегм_за_приписанного": round(g["доля_сегм_за_приписанного"].mean(), 2),
            "ожидаемая_увер": med("ожидаемая_увер"),
            "граница_5%": med("граница_5%"),
            "ICC_автора": med("ICC_автора"),
            "p_seed_мин": float(ps.min()) if ps.notna().any() else np.nan,
            "p_seed_макс": float(ps.max()) if ps.notna().any() else np.nan,
            "_p": p_comb,
            "перцентиль_старый": med("перцентиль_старый"),
            "ближайший_по_голосам": Counter(g["предсказан"]).most_common(1)[0][0],
            "устойчивость": ("устойчиво" if len(set(cats)) == 1 else
                             "меняется между seed: p = " + " / ".join(f"{x:.3f}" for x in ps)),
            "эталон": g["эталон"].iloc[0] if "эталон" in g else "",
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    n_tr = pd.to_numeric(out["сегм_автора_в_обучении"], errors="coerce")
    evaluable = (n_tr.isna() | (n_tr >= min_train)) & np.isfinite(out["_p"])
    out["_p_holm"] = np.nan
    if evaluable.any():
        out.loc[evaluable, "_p_holm"] = holm(out.loc[evaluable, "_p"].to_numpy())
    out["вывод"] = [conclusion(p, ph, nt, min_train, alpha)
                    for p, ph, nt in zip(out["_p"], out["_p_holm"], n_tr)]
    out["проверок_в_поправке"] = int(evaluable.sum())
    out["p"] = out["_p"].round(5)
    out["p_Холм"] = out["_p_holm"].round(5)
    out["p_seed_мин"] = out["p_seed_мин"].round(5)
    out["p_seed_макс"] = out["p_seed_макс"].round(5)
    order = {v: i for i, v in enumerate(VERDICTS)}
    out = (out.assign(_o=out["вывод"].map(order)).sort_values(["_o", "_p"])
              .drop(columns=["_o", "_p", "_p_holm"]).reset_index(drop=True))
    first = ["муаллака_id", "приписан", "сегментов", "сегм_автора_в_обучении"]
    tail = ["вывод", "p", "p_Холм", "p_seed_мин", "p_seed_макс", "устойчивость", "проверок_в_поправке",
            "увер_в_приписанном", "увер_мин", "увер_макс", "ожидаемая_увер", "граница_5%", "ICC_автора",
            "доля_сегм_за_приписанного", "ближайший_по_голосам", "перцентиль_старый", "эталон"]
    middle = [c for c in out.columns if c not in first + tail]
    return out[[c for c in first + middle + tail if c in out.columns]]


def load_reference(ref_dir: Path):
    """Результаты run_cv: (cv_poem_scores, {автор: [[сегменты поэмы]]}, статистика авторов, число фолдов)."""
    reference = reference_segments = author_stats = None
    n_folds = None
    if (ref_dir / "cv_predictions.csv").exists():
        cv_preds = pd.read_csv(ref_dir / "cv_predictions.csv")
        if "true_score" in cv_preds.columns:
            reference_segments = segment_scores_by_poem(cv_preds)
        n_folds = cv_preds["fold"].nunique()
        pw = pd.read_csv(ref_dir / "cv_pairwise.csv") if (ref_dir / "cv_pairwise.csv").exists() else None
        author_stats = author_cv_stats(cv_preds, pw)
        log.info("насколько модель узнаёт авторов (из CV):\n%s", author_stats.to_string())
    if (ref_dir / "cv_poem_scores.csv").exists():
        reference = pd.read_csv(ref_dir / "cv_poem_scores.csv")
    if reference_segments is None:
        log.warning("нет %s с true_score — p-value не посчитаются. Сначала запусти run_cv.",
                    ref_dir / "cv_predictions.csv")
    return reference, reference_segments, author_stats, n_folds


def log_reference_fit(reference_segments: dict, scale: str, eps: float) -> pd.DataFrame:
    fit = reference_fit_table(reference_segments, scale, eps)
    log.info("эталон «одна поэма»: оценки по авторам (шкала %s). ICC — насколько сегменты одной поэмы "
             "похожи друг на друга (0 — не похожи, 1 — одинаковы):\n%s", scale, fit.to_string(index=False))
    weak = fit[(fit["p_связи"] < 0.05) & (fit["связь_длины_и_уверенности"].abs() >= 0.3)]
    for r in weak.itertuples(index=False):
        log.warning("%s: уверенность заметно зависит от длины поэмы (Спирмен %.2f, p=%.3f) — "
                    "допущение эталона нарушено, выводы по длинной муаллаке осторожнее",
                    r.автор, r.связь_длины_и_уверенности, r.p_связи)
    return fit


FINAL_SHOW = ["муаллака_id", "приписан", "сегм_автора_в_обучении", "recall_CV", "увер_в_приписанном",
              "ожидаемая_увер", "граница_5%", "ICC_автора", "p_seed_мин", "p_seed_макс", "p", "p_Холм",
              "перцентиль_старый", "вывод", "устойчивость"]


def log_final(final: pd.DataFrame, n_seeds: int) -> None:
    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 320)
    show = [c for c in FINAL_SHOW if c in final.columns]
    n_tests = int(final["проверок_в_поправке"].iloc[0]) if "проверок_в_поправке" in final and len(final) else 0
    log.info("=== ИТОГ: атрибуция муалак (%d seed; поправка Холма на %d муалак) ===\n%s",
             n_seeds, n_tests, final[show].to_string(index=False))
    log.info("p — объединённый по seed p-value (правило Рюгера), p_Холм — с поправкой на число муалак; "
             "перцентиль_старый — прежний эталон из смеси коротких поэм, только для сравнения.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_PROCESSED / "dataset.parquet"))
    ap.add_argument("--seeds", type=int, default=3, help="сколько независимых ансамблей (seed) обучить")
    ap.add_argument("--train-fraction", type=float, default=None,
                    help="доля диванов для обучения; по умолчанию (n-1)/n, как у фолдов CV")
    ap.add_argument("--max-opponents", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--early-stop-loss", type=float, default=config.EARLY_STOP_LOSS)
    ap.add_argument("--no-amp", action="store_true", help="отключить bf16 (обучать в fp32)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="абсолютный порог уверенности (устаревшее; смотри перцентиль)")
    ap.add_argument("--reference-dir", default=str(config.DATA_PROCESSED),
                    help="папка с результатами run_cv")
    ap.add_argument("--out", default=str(config.DATA_PROCESSED / "mualaqat_cases.csv"))
    ap.add_argument("--resume", action="store_true",
                    help="продолжить прерванный прогон: готовые seed взять из *_per_seed.csv")
    ap.add_argument("--n-boot", type=int, default=10000, help="значений в эталоне на муаллаку")
    ap.add_argument("--scale", choices=["logit", "prob"], default="logit",
                    help="шкала модели эталона (logit — основная, prob — проверка устойчивости)")
    ap.add_argument("--eps", type=float, default=0.01, help="обрезка уверенности перед логитом")
    ap.add_argument("--method", choices=["bayes", "bootstrap"], default="bayes",
                    help="как строить эталон (bayes — основной, bootstrap — для сравнения)")
    args = ap.parse_args()

    setup_logging("run_cases")
    log.info("параметры: %s", vars(args))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(args.data)
    train_all, mualaka_df, shared = prepare_training_data(raw)
    if mualaka_df.empty:
        raise SystemExit("В датасете нет муалак (poem_id 'mualaka::…').")
    if len(shared):
        shared.to_csv(out.with_name(out.stem + "_shared_bayts.csv"), index=False)
    train_counts = train_all["author"].value_counts().to_dict()
    log.info("обучающих сегментов на поэта (до подвыборки): %s", train_counts)

    # --- эталон из CV ---
    reference, reference_segments, author_stats, n_folds = load_reference(Path(args.reference_dir))
    ref_cache = None
    if reference_segments is not None:
        log_reference_fit(reference_segments, args.scale, args.eps)
        ref_cache = ReferenceCache(reference_segments, n_boot=args.n_boot, seed=config.SEED,
                                   scale=args.scale, eps=args.eps, method=args.method)

    fraction = args.train_fraction
    if fraction is None:
        fraction = (n_folds - 1) / n_folds if n_folds and n_folds > 1 else 1.0
    log.info("каждый ансамбль учится на доле диванов %.3f (как модели эталона в CV)", fraction)

    # --- несколько независимых ансамблей ---
    tables, seg_tables, pair_reports = [], [], []
    per_seed_path = out.with_name(out.stem + "_per_seed.csv")
    seg_path = out.with_name(out.stem + "_segments.csv")
    pairs_path = out.with_name(out.stem + "_pairs.csv")
    wanted = [config.SEED + s for s in range(args.seeds)]
    done: set[int] = set()
    if args.resume and per_seed_path.exists():
        prev = pd.read_csv(per_seed_path)
        if {"seed", "увер_в_приписанном", "сегментов"} <= set(prev.columns):
            prev = prev[prev["seed"].isin(wanted)]
            done = set(int(x) for x in prev["seed"].unique())
            if done:
                tables.append(rescore(prev, reference, ref_cache))
                if seg_path.exists():
                    sp = pd.read_csv(seg_path)
                    seg_tables.append(sp[sp["seed"].isin(done)])
                if pairs_path.exists():
                    pp = pd.read_csv(pairs_path)
                    pair_reports += pp[pp["seed"].isin(done)].to_dict("records")
                log.info("--resume: взяты готовые seed %s из %s (p-value пересчитаны по текущему эталону)",
                         sorted(done), per_seed_path)
        else:
            log.warning("--resume: в %s нет нужных колонок — считаю все seed заново", per_seed_path)
    texts = mualaka_df["text"].tolist()
    t0 = time.time()
    todo_seeds = [x for x in wanted if x not in done]
    for s, seed in enumerate(todo_seeds):
        train_df = subsample_poems(train_all, fraction, seed=seed)
        log.info("===== seed %d (%d/%d): обучение на %d сегментах из %d =====",
                 seed, s + 1, len(todo_seeds), len(train_df), len(train_all))
        rep: list[dict] = []
        res = ensemble.fit_analyze_many(train_df, texts, max_opponents=args.max_opponents,
                                        epochs=args.epochs, early_stop_loss=args.early_stop_loss,
                                        amp=not args.no_amp, seed=seed, report=rep)
        pair_reports += [{"seed": seed, **r} for r in rep]
        seg_tables.append(segment_details(mualaka_df, res).assign(seed=seed))
        t = aggregate_cases(mualaka_df, res, threshold=args.threshold, train_counts=train_counts,
                            reference=reference, ref_cache=ref_cache).assign(seed=seed)
        tables.append(t)
        cols = [c for c in ["муаллака_id", "приписан", "увер_в_приписанном", "ожидаемая_увер", "p",
                            "перцентиль_старый", "предсказан"] if c in t.columns]
        log.info("seed %d:\n%s", seed, t[cols].to_string(index=False))
        elapsed = time.time() - t0
        log.info("seed %d готов | прошло %s, осталось ~%s", seed, fmt_duration(elapsed),
                 fmt_duration(elapsed / (s + 1) * (len(todo_seeds) - s - 1)))
        # сохраняем по ходу — если прогон прервётся, готовые seed не пропадут (--resume)
        pd.concat(tables).to_csv(per_seed_path, index=False)
        pd.concat(seg_tables).to_csv(seg_path, index=False)
        pd.DataFrame(pair_reports).to_csv(pairs_path, index=False)

    all_seeds = pd.concat(tables, ignore_index=True)
    all_seeds["муаллака_id"] = all_seeds["муаллака_id"].astype(str)
    final = combine_seeds(all_seeds, author_stats)
    final.to_csv(out, index=False)
    log_final(final, args.seeds)
    log.info("файлы: %s; %s; %s; %s", out, per_seed_path, seg_path, pairs_path)


if __name__ == "__main__":
    main()
