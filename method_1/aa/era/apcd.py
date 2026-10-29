"""Сборка датасета эпох из APCD.  [чистая логика + CLI; без torch]

APCD (arbml/APCD, ~1.83 млн строк) — по строке на БЕЙТ: era, poet, diwan, qafiyah, meter,
first_shatr, second_shatr, bayt. Границ поэм нет.

Поток:
  1. загрузка (локальный parquet/csv или HuggingFace `datasets`);
  2. псевдо-поэмы: подряд идущие бейты с одинаковыми (poet, diwan, meter, qafiyah) — одна поэма
     (смена размера или рифмы = новая поэма; две соседние поэмы с одним размером и рифмой сольются —
     для разбиения по поэтам это не важно). Считается ДО любых удалений, чтобы дыры не склеивали поэмы;
  3. чистка как в проекте (aa.data.preprocess.clean), эпоха -> канонический ключ (aa.era.labels);
  4. защита от утечек:
     а) поэты проекта (10 авторов муаллак) выбрасываются целиком по имени; похожие имена печатаются;
     б) бейты, которые есть в данных проекта (муаллаки + диваны aldiwan), выбрасываются: точно — во всех
        эпохах, как варианты (Жаккар символьных триграмм ≥ FUZZY_DUP_THRESHOLD) — в ранних эпохах;
     в) бейт, который встречается у РАЗНЫХ поэтов APCD, выбрасывается везде (спорная атрибуция,
        а при разбиении по поэтам — утечка между обучением и тестом); повтор у того же поэта — оставляем
        первый;
  5. нарезка на сегменты по 8 бейтов (aa.data.segment.segment_poem — как у всего проекта).

Итог — таблица сегментов: poem_id, author (поэт APCD), period (ключ эпохи), era_raw, meter, qafiyah,
diwan, text. Схема классов (binary/coarse/full) применяется уже при обучении.

    python -m aa.era.apcd --apcd arbml/APCD --project-data data/processed/dataset.parquet \
        --out data/processed/era_dataset.parquet
    # офлайн: --apcd data/raw/apcd/  (папка с *.parquet, скачанная заранее) или --apcd apcd.csv
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aa import config
from aa.data.dupes import MIN_CHARS, bayts_of, similar_bayt_pairs
from aa.data.preprocess import clean
from aa.data.segment import segment_poem
from aa.era.labels import (ERA_NAMES_RU, canonical_era, is_project_poet, looks_like_project_poet,
                           normalize_ar)
from aa.log import fmt_duration

log = logging.getLogger("aa.era.apcd")

APCD_COLUMNS = ["era", "poet", "diwan", "qafiyah", "meter", "first_shatr", "second_shatr", "bayt"]
ERA_DATA_COLUMNS = ["poem_id", "author", "period", "era_raw", "meter", "qafiyah", "diwan",
                    "n_bayts", "text"]


# ---------- ввод-вывод ----------

def read_table(path, columns=None) -> pd.DataFrame:
    """parquet (файл или папка с *.parquet) или csv — по расширению."""
    p = Path(path)
    if p.is_dir():
        files = sorted(p.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"в {p} нет *.parquet")
        return pd.concat([_read_one(f, columns) for f in files], ignore_index=True)
    return _read_one(p, columns)


def _read_one(p: Path, columns=None) -> pd.DataFrame:
    if p.suffix == ".csv":
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        return df[[c for c in columns if c in df.columns]] if columns else df
    if columns:
        try:
            return pd.read_parquet(p, columns=columns)
        except Exception:  # noqa: BLE001 — каких-то колонок нет
            df = pd.read_parquet(p)
            return df[[c for c in columns if c in df.columns]]
    return pd.read_parquet(p)


def write_table(df: pd.DataFrame, path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_parquet(p, index=False)
    return p


def load_apcd_frame(source: str) -> pd.DataFrame:
    """Локальный путь -> read_table; иначе имя датасета HuggingFace (нужен `pip install datasets`)."""
    if Path(source).exists():
        log.info("APCD: читаю локально %s", source)
        df = read_table(source, columns=APCD_COLUMNS)
    else:
        log.info("APCD: загружаю %s через datasets (первый раз скачивается ~240 МБ)", source)
        from datasets import ClassLabel, load_dataset
        ds = load_dataset(source, split="train")
        df = ds.to_pandas()
        for col, feat in ds.features.items():      # ClassLabel (числа) -> названия классов
            if isinstance(feat, ClassLabel) and col in df.columns:
                df[col] = df[col].map(lambda i, f=feat: f.int2str(int(i)) if i is not None and i >= 0 else "")
    for c in APCD_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[APCD_COLUMNS].copy()
    if pd.api.types.is_integer_dtype(df["era"]):   # локальный файл с номерами классов вместо названий
        raise ValueError("Колонка era — числа, а нужны арабские названия эпох. Скачай arbml/APCD "
                         "(или загрузи его через datasets — номера классов тогда переводятся в названия).")
    return df.astype(str).replace({"None": "", "nan": "", "<NA>": ""})


# ---------- шаги ----------

def assign_pseudo_poems(df: pd.DataFrame) -> pd.Series:
    """Номер псевдо-поэмы: новая поэма при смене poet/diwan/meter/qafiyah в порядке строк."""
    key = (df["poet"].astype(str) + "\x1f" + df["diwan"].astype(str) + "\x1f"
           + df["meter"].astype(str) + "\x1f" + df["qafiyah"].astype(str))
    return (key != key.shift()).cumsum().astype(np.int64)


def bayt_text(df: pd.DataFrame) -> pd.Series:
    """Полный бейт: поле bayt, а если пусто — склейка полустиший."""
    b = df["bayt"].astype(str).str.strip()
    alt = (df["first_shatr"].astype(str).str.strip() + " " + df["second_shatr"].astype(str).str.strip()).str.strip()
    return b.where(b.str.len() > 0, alt)


def clean_bayts(raw: pd.Series) -> pd.Series:
    """Та же чистка, что у проекта; бейт = одна строка."""
    uniq = pd.unique(raw)
    cleaned = {u: " ".join(clean(u).split()) for u in uniq}
    return raw.map(cleaned)


def drop_cross_poet_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Бейт у нескольких поэтов -> выбросить все вхождения; повтор у одного поэта -> оставить первый.
    Ключ — нормализованный бейт (колонка `key`)."""
    long = df["key"].str.len() >= MIN_CHARS        # короткие формулы не трогаем как «повторы»
    n_poets = df[long].groupby("key")["poet"].nunique()
    multi = set(n_poets[n_poets > 1].index)
    cross = df["key"].isin(multi) & long
    rep_same = df.duplicated(subset=["poet", "key"], keep="first") & long & ~cross
    stats = {"бейтов_у_нескольких_поэтов": int(cross.sum()), "повторов_у_того_же_поэта": int(rep_same.sum())}
    return df[~(cross | rep_same)], stats


def project_bayts(project_df: pd.DataFrame) -> list[str]:
    """Нормализованные бейты данных проекта (муаллаки + диваны)."""
    out = []
    for t in project_df["text"]:
        out.extend(normalize_ar(b) for b in bayts_of(t))
    return [b for b in dict.fromkeys(out) if len(b) >= MIN_CHARS]


def exclude_project_bayts(df: pd.DataFrame, proj: list[str], fuzzy_eras, threshold: float,
                          chunk: int = 20000) -> tuple[pd.DataFrame, dict]:
    """Убирает бейты APCD, совпадающие с бейтами проекта: точно (везде) и как варианты (в fuzzy_eras)."""
    if not proj:
        return df, {"точных_совпадений_с_проектом": 0, "вариантов_бейтов_проекта": 0}
    exact = df["key"].isin(set(proj))
    cand = df[~exact & df["period"].isin(set(fuzzy_eras)) & (df["key"].str.len() >= MIN_CHARS)]
    hit_idx: set = set()
    t0 = time.time()
    keys = cand["key"].tolist()
    idx = cand.index.to_numpy()
    for s in range(0, len(keys), chunk):
        pairs = similar_bayt_pairs(keys[s:s + chunk], proj, threshold)
        hit_idx.update(int(idx[s + i]) for i, _, _ in pairs)
        done = min(s + chunk, len(keys))
        el = time.time() - t0
        log.info("  поиск вариантов бейтов проекта: %d/%d бейтов APCD, найдено %d, прошло %s, осталось ~%s",
                 done, len(keys), len(hit_idx), fmt_duration(el), fmt_duration(el / done * (len(keys) - done)))
    fuzzy = df.index.isin(sorted(hit_idx))
    stats = {"точных_совпадений_с_проектом": int(exact.sum()), "вариантов_бейтов_проекта": int(fuzzy.sum())}
    removed = df[exact | fuzzy]
    if len(removed):
        top = removed.groupby(["poet", "period"]).size().sort_values(ascending=False).head(15)
        log.info("бейты проекта в APCD — у каких поэтов больше всего (удалены):\n%s", top.to_string())
    return df[~(exact | fuzzy)], stats


def to_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Псевдо-поэмы -> сегменты по config.SEGMENT_WINDOW бейтов."""
    rows = []
    for (poet, pid), g in df.groupby(["poet", "poem_run"], sort=False):
        first = g.iloc[0]
        for seg in segment_poem("\n".join(g["clean"])):
            rows.append({"poem_id": f"apcd::{poet}::{pid}", "author": poet, "period": first["period"],
                         "era_raw": first["era"], "meter": first["meter"], "qafiyah": first["qafiyah"],
                         "diwan": first["diwan"], "n_bayts": seg.count("\n") + 1, "text": seg})
    return pd.DataFrame(rows, columns=ERA_DATA_COLUMNS)


def build_era_dataset(apcd: pd.DataFrame, project_df: pd.DataFrame | None = None,
                      fuzzy_eras=config.ERA_FUZZY_ERAS, threshold: float = config.FUZZY_DUP_THRESHOLD,
                      extra_exclude: set[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Весь поток на готовом DataFrame APCD. Возвращает (сегменты, сводка по шагам)."""
    summary: dict = {"строк_APCD": len(apcd)}
    df = apcd.copy()
    df["poem_run"] = assign_pseudo_poems(df)
    run_len = df.groupby("poem_run").size()
    summary["псевдо_поэм"] = int(run_len.size)
    summary["бейтов_в_псевдо_поэме_медиана"] = float(run_len.median())
    if run_len.median() < 3:
        log.warning("медиана длины псевдо-поэмы %.1f бейта: похоже, порядок строк APCD перемешан, "
                    "границы поэм восстановить нельзя (сегменты будут из разных поэм)", run_len.median())

    df["clean"] = clean_bayts(bayt_text(df))
    df = df[df["clean"].str.len() > 0]
    summary["пустых_после_чистки"] = summary["строк_APCD"] - len(df)

    df["period"] = df["era"].map(canonical_era)
    unknown = df[df["period"].isna()]["era"].value_counts()
    if len(unknown):
        log.warning("нераспознанные эпохи (выброшены): %s", unknown.to_dict())
    summary["нераспознанная_эпоха"] = int(unknown.sum())
    df = df[df["period"].notna()]

    poets = pd.Series(df["poet"].unique())
    proj_mask = poets.map(lambda n: is_project_poet(n, extra_exclude))
    excluded = sorted(poets[proj_mask])
    near = sorted(p for p in poets[~proj_mask] if looks_like_project_poet(p))
    log.info("поэты проекта, исключённые по имени (%d): %s", len(excluded), excluded)
    if near:
        log.info("похожие имена, НЕ исключены (проверь вручную; при необходимости — --exclude-poets): %s", near)
    summary["исключено_поэтов_проекта"] = len(excluded)
    summary["бейтов_поэтов_проекта"] = int(df["poet"].isin(set(excluded)).sum())
    df = df[~df["poet"].isin(set(excluded))]

    df["key"] = df["clean"].map(normalize_ar)
    df, st = drop_cross_poet_duplicates(df)
    summary.update(st)

    if project_df is not None and len(project_df):
        df, st = exclude_project_bayts(df, project_bayts(project_df), fuzzy_eras, threshold)
        summary.update(st)
    else:
        log.warning("данные проекта не переданы (--project-data): бейты муаллак и диванов, записанные в APCD "
                    "под другими именами, НЕ будут найдены. Для run_era_cases это утечка.")

    segs = to_segments(df)
    summary["сегментов"] = len(segs)
    return segs, summary


def era_table(segs: pd.DataFrame) -> pd.DataFrame:
    t = segs.groupby("period").agg(поэтов=("author", "nunique"), поэм=("poem_id", "nunique"),
                                   сегментов=("text", "size"))
    t["эпоха"] = [ERA_NAMES_RU.get(k, k) for k in t.index]
    return t.reset_index()


def main() -> None:
    from aa.log import setup_logging
    ap = argparse.ArgumentParser(description="APCD -> датасет сегментов с эпохами")
    ap.add_argument("--apcd", default="arbml/APCD", help="имя на HuggingFace или локальный parquet/папка/csv")
    ap.add_argument("--project-data", default=str(config.DATA_PROCESSED / "dataset.parquet"),
                    help="датасет проекта (муаллаки + диваны) — его бейты исключаются из APCD")
    ap.add_argument("--out", default=str(config.DATA_PROCESSED / "era_dataset.parquet"))
    ap.add_argument("--exclude-poets", default="", help="доп. имена поэтов для исключения, через запятую")
    ap.add_argument("--fuzzy-eras", default=",".join(config.ERA_FUZZY_ERAS),
                    help="эпохи, где искать ВАРИАНТЫ бейтов проекта (точные ищутся везде)")
    ap.add_argument("--threshold", type=float, default=config.FUZZY_DUP_THRESHOLD)
    args = ap.parse_args()
    setup_logging("era_apcd")

    t0 = time.time()
    apcd = load_apcd_frame(args.apcd)
    log.info("APCD: %d строк, эпохи: %s", len(apcd), apcd["era"].value_counts().to_dict())
    project = None
    if args.project_data and Path(args.project_data).exists():
        project = read_table(args.project_data)
        log.info("данные проекта: %d сегментов, %d поэм", len(project), project["poem_id"].nunique())
    extra = {normalize_ar(x) for x in args.exclude_poets.split(",") if x.strip()}
    segs, summary = build_era_dataset(apcd, project, fuzzy_eras=[e for e in args.fuzzy_eras.split(",") if e],
                                      threshold=args.threshold, extra_exclude=extra)
    for k, v in summary.items():
        log.info("  %-34s %s", k, v)
    log.info("по эпохам:\n%s", era_table(segs).to_string(index=False))
    write_table(segs, args.out)
    log.info("записано: %s (%s)", args.out, fmt_duration(time.time() - t0))


if __name__ == "__main__":
    main()
