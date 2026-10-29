"""КОНТРАКТ ДАННЫХ — единственный источник правды о форме датасета.

Обе дорожки импортируют отсюда. Любое изменение формата — через этот файл
и через совместное согласование, а не правкой по месту.

Одна строка = один сегмент стихотворения (не целая поэма).
"""
from __future__ import annotations
import pandas as pd

COLUMNS = ["poem_id", "author", "period", "century", "topic", "text"]

DTYPES = {
    "poem_id": "string",   # стабильный id исходной поэмы; сегменты одной поэмы делят его
    "author":  "string",   # нормализованное имя поэта (ключ класса)
    "period":  "string",   # метка периода, напр. "pre-islamic"; может быть <NA>
    "century": "Int64",    # век (по дате смерти автора); может быть <NA>
    "topic":   "Int64",    # id темы ETM; на фазе 1 везде <NA>
    "text":    "string",   # ОЧИЩЕННЫЙ сегмент (диакритика/кашида уже сняты)
}


def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=DTYPES[c]) for c in COLUMNS})


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет контракт и приводит типы. Падает громко при нарушении —
    лучше сломаться на границе, чем передать кривой датасет в модель."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"В датасете нет обязательных колонок: {sorted(missing)}")
    df = df[COLUMNS].copy()
    for c, t in DTYPES.items():
        df[c] = df[c].astype(t)
    if df["text"].str.len().fillna(0).eq(0).any():
        raise ValueError("Есть пустые тексты сегментов — чистка/сегментация выдала мусор.")
    if df[["poem_id", "author"]].isna().any().any():
        raise ValueError("poem_id и author не могут быть пустыми.")
    return df
