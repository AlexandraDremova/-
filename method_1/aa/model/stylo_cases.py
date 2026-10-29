"""Arud и частые слова на муаллаках: p-value по той же процедуре, что у CAMeLBERT в A8.

Зачем. Эксперимент участника Б (arud_vs_mfw.py) кластеризовал только диваны — p-value для каждой
муаллаки у Arud не было. Здесь признаки Arud (и для сравнения — частые слова) проходят ровно ту же
процедуру, что CAMeLBERT, поэтому p-value можно поставить в одну таблицу:
  1. Те же данные и та же чистка: диваны без муаллак, без их точных копий и без сегментов с вариантами
     их бейтов (aa.data.prepare); те же сегменты по 8 бейтов. Признаки считаются по ОГЛАСОВАННОМУ
     тексту: Arud без огласовок не работает.
  2. Признаки сегмента — доли 100 самых частых Arud-кодов слов (кодировщик из arud_vs_mfw.py участника Б)
     или 100 самых частых слов; словарь строится только по обучающей части. Модель — логистическая
     регрессия на 10 поэтов с весами классов. «Уверенность в поэте» — её вероятность для этого поэта.
  3. Кросс-валидация на 3 фолдах по группам поэм (как run_cv). Уверенность в настоящем поэте на невиденных
     поэмах — эталон. Плюс p-value «модель различает поэтов лучше случайного» (поэты переставляются
     между поэмами, всё обучение повторяется).
  4. Каждая из 3 моделей фолдов (обучена на 2/3 диванов — как ансамбли A8) оценивает муаллаки.
     p — против эталона «одна поэма поэта той же длины», p трёх моделей объединяются по Рюгеру,
     поправка Холма — по муаллакам, которые можно оценить. Всё как в run_cases / rescore_cases.
  5. Проверка на метр: Arud-коды сильно зависят от метра, а муаллака написана одним метром. Поэтому
     ещё один p — против эталона только из поэм поэта ТЕМ ЖЕ метром (если таких поэм хватает).
     Если «аномалия» Arud там пропадает, дело в метре, а не в авторе.
  6. Сводная таблица «муаллака × метод» вместе с CAMeLBERT (mualaqat_cases_v2.csv, если есть) и проверка
     эталона на настоящих поэмах диванов.

Нужно: data/raw/mualakat.json, data/raw/diwans/*.json и arud_vs_mfw.py участника Б в корне репозитория.

    python -m aa.model.stylo_cases                  # arud и mfw, 200 перестановок (несколько минут)
    python -m aa.model.stylo_cases --perms 0        # без теста «лучше случайного» (минута-две)
    python -m aa.model.stylo_cases --modes arud     # только Arud
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from aa import config
from aa.data.build_dataset import _poem_id, as_poem_list, index_authors_by_id, load_diwans, load_mualakat
from aa.data.dupes import poem_groups
from aa.data.prepare import prepare_training_data
from aa.data.preprocess import clean
from aa.eval.reference import (calibration_summary, combine_seed_pvalues, holm, loo_calibration, p_lower,
                               reference_fit_table, single_poem_reference)
from aa.log import fmt_duration, setup_logging
from aa.model.run_cases import MIN_TRAIN, conclusion

log = logging.getLogger("aa.stylo_cases")

MODE_NAMES = {"arud": "Arud", "mfw": "частые слова"}
METHOD_ORDER = ["CAMeLBERT", "Arud", "частые слова"]
POET_RU = {"antar-bin-shaddad": "Антара", "al-asha": "аль-Аша", "tarafa-ibn-al-abd": "Тарафа",
           "abid-ibn-al-abras": "Абид", "labid-ibn-rabiah": "Лабид", "zuhair-bin-abi-salma": "Зухайр",
           "nabgh-alzpiani": "ан-Набига", "imru-alqays": "Имру аль-Кайс", "harith-bin-halzh": "Харис",
           "amr-ibn-kulthum": "Амр"}
HARAKAT = set("ًٌٍَُِّْٰ")
VOWEL_GAP_WARN = 0.10


# ---------------------------------------------------------------- данные

def import_arud():
    """Кодировщик Arud и нормализация слов берутся из arud_vs_mfw.py участника Б без изменений."""
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import arud_vs_mfw as base
    except ImportError as e:
        raise SystemExit("Не найден arud_vs_mfw.py участника Б — положи его в корень репозитория "
                         "(рядом с папкой aa/).") from e
    return lru_cache(maxsize=None)(base.encode_word_arud_strict), lru_cache(maxsize=None)(base.normalize_for_mfw)


def segment_pairs(raw_text: str, window: int = config.SEGMENT_WINDOW, stride: int = config.SEGMENT_STRIDE,
                  min_bayts: int = config.MIN_SEGMENT_BAYTS) -> list[tuple[str, str]]:
    """Те же окна, что aa.data.segment.segment_poem(clean(текст)), но для каждого сегмента — пара
    (очищенный текст — для чистки от дублей, огласованный текст тех же бейтов — для признаков)."""
    lines = []
    for line in str(raw_text).splitlines():
        c = clean(line).strip()
        if c:
            lines.append((c, line.strip()))
    n = len(lines)
    chunks = []
    if 0 < n <= window:
        if n >= min_bayts:
            chunks.append(lines)
    elif n:
        for start in range(0, n - min_bayts + 1, stride):
            chunk = lines[start:start + window]
            if len(chunk) >= min_bayts:
                chunks.append(chunk)
            if start + window >= n:
                break
    return [(" \n".join(c for c, _ in ch), "\n".join(r for _, r in ch)) for ch in chunks]


def _norm_meter(m) -> str | None:
    m = str(m or "").strip()
    return m.replace("بحر ", "").strip() or None


def meter_index(raw_dir: Path) -> dict[str, str]:
    """poem_id (как в load_diwans / load_mualakat) -> метр из поля "meter". У муаллаки без поля
    метр берётся у её копии в диване (тот же aldiwan id)."""
    out, by_id = {}, {}
    for f in sorted((raw_dir / "diwans").glob("*.json")):
        author = f.stem.strip()
        for i, poem in enumerate(as_poem_list(json.loads(f.read_text(encoding="utf-8")))):
            m = _norm_meter(poem.get("meter"))
            if m:
                out[f"diwan::{author}::{_poem_id(poem, i)}"] = m
                if poem.get("id") is not None:
                    by_id[str(poem["id"])] = m
    path = raw_dir / "mualakat.json"
    if path.exists():
        for i, poem in enumerate(as_poem_list(json.loads(path.read_text(encoding="utf-8")))):
            m = _norm_meter(poem.get("meter")) or by_id.get(str(poem.get("id")))
            if m:
                out[f"mualaka::{_poem_id(poem, i)}"] = m
    return out


def load_segments(raw_dir: Path) -> pd.DataFrame:
    diwans_dir = raw_dir / "diwans"
    diw = load_diwans(diwans_dir)
    if diw.empty:
        raise SystemExit(f"В {diwans_dir} нет диванов (*.json).")
    if not (raw_dir / "mualakat.json").exists():
        raise SystemExit(f"Нет {raw_dir / 'mualakat.json'}.")
    amap_file = raw_dir / "mualakat_authors.json"
    author_map = json.loads(amap_file.read_text(encoding="utf-8")) if amap_file.exists() else None
    mual = load_mualakat(raw_dir / "mualakat.json", index_authors_by_id(diwans_dir), author_map,
                         known_authors=set(diw["author"]))
    meters = meter_index(raw_dir)
    rows = []
    for r in pd.concat([diw, mual], ignore_index=True).itertuples(index=False):
        for text, raw in segment_pairs(r.text):
            rows.append({"poem_id": r.poem_id, "author": r.author, "text": text, "raw": raw,
                         "meter": meters.get(r.poem_id)})
    return pd.DataFrame(rows)


def vowel_share(texts) -> float:
    """Доля арабских букв, после которых стоит огласовка (как «огласовано_букв» в arud_vs_mfw_stats)."""
    letters = voweled = 0
    for t in texts:
        for i, ch in enumerate(t):
            if "ء" <= ch <= "ي" and ch != "ـ":
                letters += 1
                voweled += i + 1 < len(t) and t[i + 1] in HARAKAT
    return voweled / letters if letters else float("nan")


# ---------------------------------------------------------------- признаки и модель

def make_vocab(token_lists, top_k: int) -> list[str]:
    c = Counter()
    for toks in token_lists:
        c.update(toks)
    return [t for t, _ in c.most_common(top_k)]


def featurize(token_lists, vocab) -> np.ndarray:
    """Относительные частоты (как build_feature_matrix участника Б: доля от всех токенов сегмента)."""
    idx = {t: j for j, t in enumerate(vocab)}
    x = np.zeros((len(token_lists), len(vocab)))
    for r, toks in enumerate(token_lists):
        for t in toks:
            j = idx.get(t)
            if j is not None:
                x[r, j] += 1
        if toks:
            x[r] /= len(toks)
    return x


def fit(x, y, c_reg: float, seed: int):
    sc = StandardScaler().fit(x)
    clf = LogisticRegression(C=c_reg, max_iter=5000, class_weight="balanced", random_state=seed)
    clf.fit(sc.transform(x), y)
    return sc, clf


def proba(model, x, authors):
    """(вероятность указанного поэта, предсказанный поэт) для каждого сегмента."""
    sc, clf = model
    p = clf.predict_proba(sc.transform(x))
    col = {a: j for j, a in enumerate(clf.classes_)}
    own = np.array([p[i, col[a]] if a in col else np.nan for i, a in enumerate(authors)])
    return own, clf.classes_[p.argmax(axis=1)]


def cross_validate(train: pd.DataFrame, toks, labels, groups, n_splits, top_k, c_reg, seed,
                   keep_models=True):
    """CV по группам поэм. Возвращает (предсказания по сегментам, [(словарь, модель)] фолдов)."""
    rows, models = [], []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(train, labels, groups)):
        vocab = make_vocab([toks[i] for i in tr], top_k)
        model = fit(featurize([toks[i] for i in tr], vocab), labels[tr], c_reg, seed)
        own, pred = proba(model, featurize([toks[i] for i in te], vocab), labels[te])
        rows.append(pd.DataFrame({"fold": fold, "poem_id": train["poem_id"].to_numpy()[te],
                                  "true_author": labels[te], "pred_author": pred, "true_score": own}))
        if keep_models:
            models.append((vocab, model))
    return pd.concat(rows, ignore_index=True), models


def permuted_labels(train: pd.DataFrame, rng) -> np.ndarray:
    """Поэты переставляются между ПОЭМАМИ: все сегменты поэмы получают одного и того же поэта,
    размеры классов сохраняются по числу поэм."""
    poems = train.groupby("poem_id", sort=False)["author"].first()
    shuffled = dict(zip(poems.index, rng.permutation(poems.to_numpy())))
    return train["poem_id"].map(shuffled).to_numpy()


def reference_from_cv(cv: pd.DataFrame, poem_filter=None) -> dict:
    """{поэт: [[уверенности сегментов поэмы], ...]} по невиденным в CV поэмам."""
    out: dict[str, list[list[float]]] = {}
    for (author, pid), g in cv.groupby(["true_author", "poem_id"], sort=False):
        if poem_filter is not None and not poem_filter(pid):
            continue
        vals = [float(v) for v in g["true_score"] if np.isfinite(v)]
        if vals:
            out.setdefault(str(author), []).append(vals)
    return out


# ---------------------------------------------------------------- один метод целиком

def tokenize(texts, encode) -> list[list[str]]:
    return [[t for t in map(encode, s.split()) if t] for s in texts]


def run_method(mode, encode, train, mual, groups, args, out_dir):
    name = MODE_NAMES[mode]
    log.info("===== %s =====", name)
    train_toks, mual_toks = tokenize(train["raw"], encode), tokenize(mual["raw"], encode)
    labels = train["author"].to_numpy()
    train_counts = train["author"].value_counts().to_dict()
    log.info("%s: токенов на сегмент — медиана %d; различных токенов в обучении %d", name,
             int(np.median([len(t) for t in train_toks])), len({t for ts in train_toks for t in ts}))

    t0 = time.time()
    cv, models = cross_validate(train, train_toks, labels, groups, args.n_splits, args.top_k, args.C,
                                config.SEED)
    cv.to_csv(out_dir / f"stylo_{mode}_cv_predictions.csv", index=False)
    bacc = balanced_accuracy_score(cv["true_author"], cv["pred_author"])
    hit = cv["true_author"] == cv["pred_author"]
    recall = hit.groupby(cv["true_author"]).mean()
    log.info("%s, CV за %s: точность по сегментам %.3f, средняя полнота по поэтам %.3f (случайно 0.10)",
             name, fmt_duration(time.time() - t0), hit.mean(), bacc)
    log.info("%s, полнота по поэтам: %s", name, recall.round(2).to_dict())

    p_perm, null_mean = np.nan, np.nan
    if args.perms:
        rng = np.random.default_rng(config.SEED)
        null, t1 = [], time.time()
        step = max(1, args.perms // 10)
        for b in range(args.perms):
            cvp, _ = cross_validate(train, train_toks, permuted_labels(train, rng), groups, args.n_splits,
                                    args.top_k, args.C, config.SEED, keep_models=False)
            null.append(balanced_accuracy_score(cvp["true_author"], cvp["pred_author"]))
            if (b + 1) % step == 0 or b + 1 == args.perms:
                el = time.time() - t1
                log.info("  %s, перестановки: %d/%d | прошло %s, осталось ~%s", name, b + 1, args.perms,
                         fmt_duration(el), fmt_duration(el / (b + 1) * (args.perms - b - 1)))
        null = np.array(null)
        null_mean = float(null.mean())
        p_perm = float((1 + np.sum(null >= bacc)) / (1 + args.perms))
        log.info("%s: средняя полнота %.3f против %.3f ± %.3f при перестановке поэтов -> p = %.4f "
                 "(минимально возможное при %d перестановках: %.4f)", name, bacc, null_mean, null.std(),
                 p_perm, args.perms, 1 / (1 + args.perms))

    ref = reference_from_cv(cv)
    log.info("%s, эталон по поэтам:\n%s", name, reference_fit_table(ref).to_string(index=False))
    meter_of = dict(zip(train["poem_id"], train["meter"]))

    per_model = [proba(model, featurize(mual_toks, vocab), mual["author"].to_numpy()) for vocab, model in models]
    cache: dict = {}
    vowel_diwan: dict = {}
    rows = []
    for pid, g in mual.groupby("poem_id", sort=False):
        idx = g.index.to_numpy()
        author, k, meter = str(g["author"].iat[0]), len(idx), g["meter"].iat[0]
        if (author, k) not in cache:
            cache[(author, k)] = single_poem_reference(ref.get(author, []), k, n_boot=args.n_boot,
                                                       seed=config.SEED)
        draws, rfit = cache[(author, k)]
        confs = [float(np.nanmean(own[idx])) for own, _ in per_model]
        ps = [p_lower(c, draws) for c in confs]
        shares = [float(np.mean(pred[idx] == author)) for _, pred in per_model]

        # тот же метр: эталон только из поэм поэта этим метром
        p_meter, n_meter_poems = np.nan, 0
        if isinstance(meter, str):
            same = reference_from_cv(cv[cv["true_author"] == author],
                                     poem_filter=lambda q, m=meter: meter_of.get(q) == m)
            n_meter_poems = len(same.get(author, []))
            d_m, f_m = single_poem_reference(same.get(author, []), k, n_boot=args.n_boot, seed=config.SEED)
            if f_m.ok:
                p_meter = combine_seed_pvalues([p_lower(c, d_m) for c in confs])
        author_poems = train.loc[train["author"] == author, "poem_id"].unique()
        share_meter = (float(np.mean([meter_of.get(q) == meter for q in author_poems]))
                       if isinstance(meter, str) and len(author_poems) else np.nan)
        if author not in vowel_diwan:
            vowel_diwan[author] = vowel_share(train.loc[train["author"] == author, "raw"])

        rows.append({
            "муаллака_id": str(pid).replace("mualaka::", ""), "приписан": author, "метод": name,
            "сегментов": k, "сегм_автора_в_обучении": int(train_counts.get(author, 0)),
            "увер_в_приписанном": round(float(np.mean(confs)), 3),
            "ожидаемая_увер": round(float(np.median(draws)), 3) if draws.size else np.nan,
            "граница_5%": round(float(np.quantile(draws, 0.05)), 3) if draws.size else np.nan,
            "доля_сегм_за_приписанного": round(float(np.mean(shares)), 2),
            "_p": combine_seed_pvalues(ps),
            "p_модели_мин": min(ps) if draws.size else np.nan,
            "p_модели_макс": max(ps) if draws.size else np.nan,
            "эталон": "ok" if rfit.ok else rfit.reason,
            "метр": meter,
            "доля_поэм_поэта_в_этом_метре": round(share_meter, 2) if np.isfinite(share_meter) else np.nan,
            "поэм_этим_метром_в_эталоне": n_meter_poems,
            "p_тот_же_метр": p_meter,
            "огласовано_муаллака": round(vowel_share(g["raw"]), 2),
            "огласовано_диван": round(vowel_diwan[author], 2),
        })
    res = pd.DataFrame(rows)
    evaluable = (res["сегм_автора_в_обучении"] >= MIN_TRAIN) & np.isfinite(res["_p"])
    res["_p_holm"] = np.nan
    if evaluable.any():
        res.loc[evaluable, "_p_holm"] = holm(res.loc[evaluable, "_p"].to_numpy())
    res["вывод"] = [conclusion(p, ph, nt) for p, ph, nt in
                    zip(res["_p"], res["_p_holm"], res["сегм_автора_в_обучении"])]
    res["p"] = res["_p"].round(5)
    res["p_Холм"] = res["_p_holm"].round(5)
    res["p_тот_же_метр"] = res["p_тот_же_метр"].round(5)
    res["проверок_в_поправке"] = int(evaluable.sum())
    res["CV_средняя_полнота"] = round(bacc, 3)
    res["CV_полнота_при_перестановке"] = round(null_mean, 3) if np.isfinite(null_mean) else np.nan
    res["CV_p_лучше_случайного"] = p_perm
    res = res.drop(columns=["_p", "_p_holm"])

    if mode == "arud":
        gap = (res["огласовано_муаллака"] - res["огласовано_диван"]).abs()
        for r in res[gap > VOWEL_GAP_WARN].itertuples(index=False):
            log.warning("%s: муаллака %s огласована иначе, чем диван поэта (%.2f против %.2f) — Arud-коды "
                        "отчасти меряют огласовку, p для неё ненадёжен", name, r.муаллака_id,
                        r.огласовано_муаллака, r.огласовано_диван)
        for r in res[res["доля_поэм_поэта_в_этом_метре"] < 0.15].itertuples(index=False):
            log.warning("%s: муаллака %s написана метром «%s», а у поэта им написано лишь %.0f%% поэм — "
                        "низкий p Arud может означать «редкий для поэта метр», смотри p_тот_же_метр",
                        name, r.муаллака_id, r.метр, 100 * r.доля_поэм_поэта_в_этом_метре)

    t2 = time.time()
    cal = calibration_summary(loo_calibration(ref, n_boot=2000, seed=config.SEED, old_boot=0))
    cal = cal[cal["эталон"] == "новый"].assign(метод=name)
    log.info("%s, эталон на настоящих поэмах за %s (доля p ≤ 0.05 должна быть ≈ 0.05):\n%s",
             name, fmt_duration(time.time() - t2), cal.to_string(index=False))
    return res, cal


# ---------------------------------------------------------------- сводная таблица

def wide_table(results: pd.DataFrame, camel_path: Path) -> pd.DataFrame:
    """Строки — муаллаки, колонки — «метод: p» и «метод: p_Холм» (Холм — внутри каждого метода)."""
    parts = []
    if camel_path.exists():
        cam = pd.read_csv(camel_path, dtype={"муаллака_id": str})
        parts.append(cam[["муаллака_id", "приписан", "сегментов", "p", "p_Холм"]].assign(метод="CAMeLBERT"))
    else:
        log.warning("нет %s — колонки CAMeLBERT в сводной таблице не будет (сначала rescore_cases)", camel_path)
    parts.append(results[["муаллака_id", "приписан", "сегментов", "p", "p_Холм", "метод"]])
    long = pd.concat(parts, ignore_index=True)
    long["муаллака_id"] = long["муаллака_id"].astype(str)
    wide = long.drop_duplicates("муаллака_id")[["муаллака_id", "приписан", "сегментов"]].reset_index(drop=True)
    for m in [m for m in METHOD_ORDER if m in set(long["метод"])]:
        part = long.loc[long["метод"] == m, ["муаллака_id", "p", "p_Холм"]]
        wide = wide.merge(part.rename(columns={"p": f"{m}: p", "p_Холм": f"{m}: p_Холм"}),
                          on="муаллака_id", how="left")
    wide.insert(0, "поэт", wide["приписан"].map(POET_RU).fillna(wide["приписан"]))
    first = [c for c in wide.columns if c.endswith(": p")][0]
    return wide.sort_values(first, na_position="last", kind="stable").reset_index(drop=True)


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    s = f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"
    return f"**{s}**" if p <= 0.05 else s


def markdown_table(wide: pd.DataFrame) -> str:
    methods = [c[:-3] for c in wide.columns if c.endswith(": p")]
    lines = ["| Муаллака | id | сегм. | " + " | ".join(f"{m}: p (Холм)" for m in methods) + " |",
             "|" + "---|" * (3 + len(methods))]
    for row in wide.to_dict("records"):
        cells = []
        for m in methods:
            p, ph = row[f"{m}: p"], row[f"{m}: p_Холм"]
            # нет p с поправкой — муаллака не вошла в поправку (мало данных или нет эталона)
            cells.append("нельзя оценить" if not (np.isfinite(p) and np.isfinite(ph)) else f"{fmt_p(p)} ({fmt_p(ph)})")
        lines.append(f"| {row['поэт']} | {row['муаллака_id']} | {row['сегментов']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(config.DATA_RAW))
    ap.add_argument("--out-dir", default=str(config.DATA_PROCESSED))
    ap.add_argument("--camelbert", default=None,
                    help="таблица CAMeLBERT для сводки (по умолчанию <out-dir>/mualaqat_cases_v2.csv)")
    ap.add_argument("--modes", default="arud,mfw", help="через запятую: arud, mfw")
    ap.add_argument("--top-k", type=int, default=100, help="размер словаря признаков (как у участника Б)")
    ap.add_argument("--C", type=float, default=1.0, help="обратная сила регуляризации логистической регрессии")
    ap.add_argument("--n-splits", type=int, default=3, help="фолдов CV (у CAMeLBERT было 3)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--perms", type=int, default=200, help="перестановок для p «лучше случайного»; 0 — не считать")
    args = ap.parse_args()

    setup_logging("stylo_cases")
    log.info("параметры: %s", vars(args))
    t_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encode_arud, normalize_word = import_arud()
    encoders = {"arud": encode_arud, "mfw": normalize_word}
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(encoders)
    if unknown:
        raise SystemExit(f"неизвестные режимы: {sorted(unknown)}; есть: arud, mfw")

    segs = load_segments(Path(args.raw_dir))
    is_m = segs["poem_id"].str.startswith("mualaka::")
    log.info("сегментов всего: %d (муаллак: %d); метр известен у %.0f%% сегментов", len(segs), int(is_m.sum()),
             100 * segs["meter"].notna().mean())
    train, mual, _ = prepare_training_data(segs, log_details=False)
    train, mual = train.reset_index(drop=True), mual.reset_index(drop=True)
    groups = poem_groups(train, config.FUZZY_DUP_THRESHOLD)
    log.info("обучающих сегментов на поэта: %s", train["author"].value_counts().to_dict())
    if len(train) != 1204:
        log.info("у CAMeLBERT было 1204 обучающих сегмента, здесь %d — отличие возможно, если менялись "
                 "данные или чистка (camel-tools)", len(train))

    results, cals = [], []
    for mode in modes:
        res, cal = run_method(mode, encoders[mode], train, mual, groups, args, out_dir)
        results.append(res)
        cals.append(cal)
    results = pd.concat(results, ignore_index=True)
    results.to_csv(out_dir / "stylo_cases.csv", index=False)
    pd.concat(cals, ignore_index=True).to_csv(out_dir / "stylo_calibration_summary.csv", index=False)

    pd.set_option("display.unicode.east_asian_width", True)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    show = ["муаллака_id", "приписан", "сегментов", "увер_в_приписанном", "ожидаемая_увер", "граница_5%",
            "p_модели_мин", "p_модели_макс", "p", "p_Холм", "вывод", "метр", "доля_поэм_поэта_в_этом_метре",
            "p_тот_же_метр", "огласовано_муаллака", "огласовано_диван"]
    for name, g in results.groupby("метод", sort=False):
        log.info("=== %s: муаллаки (CV: средняя полнота %.3f, p «лучше случайного» = %s) ===\n%s", name,
                 g["CV_средняя_полнота"].iloc[0], g["CV_p_лучше_случайного"].iloc[0],
                 g[show].to_string(index=False))

    camel = Path(args.camelbert) if args.camelbert else out_dir / "mualaqat_cases_v2.csv"
    wide = wide_table(results, camel)
    wide.to_csv(out_dir / "mualaqat_methods_pvalues.csv", index=False)
    md = markdown_table(wide)
    (out_dir / "mualaqat_methods_pvalues.md").write_text(
        "p — односторонний p-value «муаллака похожа на поэта меньше, чем его типичная поэма той же длины»; "
        "в скобках — с поправкой Холма внутри метода. Жирным — p ≤ 0.05.\n\n" + md + "\n", encoding="utf-8")
    log.info("=== СВОДНАЯ ТАБЛИЦА: p (Холм) по муаллакам и методам ===\n%s", md)
    log.info("готово за %s. Файлы в %s: mualaqat_methods_pvalues.csv/.md, stylo_cases.csv, "
             "stylo_calibration_summary.csv, stylo_*_cv_predictions.csv", fmt_duration(time.time() - t_start),
             out_dir)


if __name__ == "__main__":
    main()
