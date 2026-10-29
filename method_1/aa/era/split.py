"""Метки схемы, разбиение ПО ПОЭТАМ и ограничение выборки.  [чистая логика]

Почему по поэтам. Эпоха — свойство поэта: все сегменты поэта имеют одну метку. Если сегменты одного
поэта попадут и в обучение, и в тест, модель может узнавать ПОЭТА (его словарь, темы, рифмы),
а не эпоху, и точность будет завышена. Поэтому фолды строятся на уровне поэтов
(стратифицированно по эпохе), а валидация для ранней остановки — тоже отдельные поэты.

Почему ограничение выборки. В APCD у Аббасидов и нового времени сотни тысяч бейтов, у доисламской
эпохи — на порядок меньше; внутри эпохи несколько плодовитых поэтов дают большую часть текста.
Ограничиваем число сегментов на поэта и на класс, набирая «по кругу» по поэтам, чтобы
в выборке было как можно больше разных поэтов.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aa import config
from aa.era.labels import classes_of, to_class

log = logging.getLogger("aa.era.split")


def apply_scheme(df: pd.DataFrame, scheme: str,
                 min_poets: int = config.ERA_MIN_POETS_PER_CLASS) -> tuple[pd.DataFrame, list[str]]:
    """Добавляет колонку label по схеме; убирает эпохи вне схемы и классы с < min_poets поэтов.
    Поэту, у которого в APCD несколько эпох (редко), даётся самая частая; остальные его сегменты
    выбрасываются. Возвращает (df, классы в хронологическом порядке)."""
    out = df.copy()
    out["label"] = out["period"].map(lambda e: to_class(e, scheme))
    n0 = len(out)
    out = out[out["label"].notna()]
    if n0 - len(out):
        log.info("схема %s: вне схемы %d сегментов (эпохи: %s)", scheme, n0 - len(out),
                 sorted(set(df["period"].dropna()) - set(out["period"])))

    mode = out.groupby("author")["label"].agg(lambda s: s.value_counts().index[0])
    clash = out["label"] != out["author"].map(mode)
    if clash.any():
        log.info("у %d поэтов несколько эпох; выброшено %d сегментов с не основной меткой",
                 out[clash]["author"].nunique(), int(clash.sum()))
        out = out[~clash]

    poets_per_class = out.groupby("label")["author"].nunique()
    small = sorted(poets_per_class[poets_per_class < min_poets].index)
    if small:
        log.warning("классы с < %d поэтов выброшены (их нельзя честно проверить по поэтам): %s",
                    min_poets, {c: int(poets_per_class[c]) for c in small})
        out = out[~out["label"].isin(small)]
    classes = [c for c in classes_of(scheme) if c in set(out["label"])]
    return out.reset_index(drop=True), classes


def poet_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("author").agg(label=("label", "first"), n=("text", "size")).reset_index()


def poet_folds(df: pd.DataFrame, n_splits: int, seed: int) -> dict[str, int]:
    """{поэт -> номер фолда}: стратифицированно по эпохе, на уровне поэтов."""
    from sklearn.model_selection import StratifiedKFold
    pt = poet_table(df)
    min_class = pt["label"].value_counts().min()
    if min_class < n_splits:
        raise ValueError(f"в каком-то классе всего {min_class} поэтов < n_splits={n_splits}; "
                         f"уменьши --n-splits или подними ERA_MIN_POETS_PER_CLASS")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = {}
    for k, (_, te) in enumerate(skf.split(pt["author"], pt["label"])):
        for a in pt["author"].iloc[te]:
            fold[a] = k
    return fold


def split_val_poets(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Часть поэтов (стратифицированно, ≥1 на класс) — в валидацию для ранней остановки."""
    rng = np.random.default_rng(seed)
    pt = poet_table(df)
    val = []
    for _, g in pt.groupby("label"):
        names = g["author"].to_numpy()
        rng.shuffle(names)
        k = max(1, int(round(frac * len(names)))) if len(names) >= 3 else 0
        val.extend(names[:k])
    is_val = df["author"].isin(set(val))
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def cap_sample(df: pd.DataFrame, per_poet: int | None, per_class: int | None, seed: int) -> pd.DataFrame:
    """Не больше per_poet сегментов от поэта и per_class на класс; набор «по кругу» по поэтам:
    сначала по одному случайному сегменту каждого поэта, потом по второму и т.д."""
    if df.empty:
        return df
    rng = np.random.default_rng(seed)
    d = df.copy()
    d["_r"] = rng.random(len(d))
    d["_rank"] = d.sort_values("_r").groupby("author").cumcount()
    if per_poet:
        d = d[d["_rank"] < per_poet]
    d["_r2"] = rng.random(len(d))
    d = d.sort_values(["_rank", "_r2"])
    if per_class:
        d = d.groupby("label", group_keys=False).head(per_class)
    return d.drop(columns=["_r", "_rank", "_r2"]).sort_index().reset_index(drop=True)


def class_weights(labels, classes: list[str]) -> list[float]:
    """Веса классов для cross-entropy: n / (C · n_c). Пустой класс — вес 0."""
    counts = pd.Series(list(labels)).value_counts()
    n, c = len(labels), len(classes)
    return [float(n / (c * counts[k])) if counts.get(k, 0) else 0.0 for k in classes]
