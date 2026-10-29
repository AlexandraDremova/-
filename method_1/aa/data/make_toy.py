"""Игрушечный датасет в формате контракта — чтобы eval и тесты бегали без реальных данных.

НЕ арабский и не поэзия: просто синтетические «стили» из разных токенов, чтобы
у стаб-модели (и позже у реальной) было что делить. Нужен только для проверки трубы.

    python -m aa.data.make_toy
"""
from __future__ import annotations
import random

import pandas as pd

from aa import config, schema

_AUTHORS = {
    "poet_A": ["alif", "baa", "taa", "thaa"],
    "poet_B": ["jiim", "haa", "khaa", "daal"],
    "poet_C": ["raa", "zaa", "siin", "shiin"],
}


def _make_segment(vocab: list[str], rng: random.Random, n: int = 20) -> str:
    return " ".join(rng.choice(vocab) for _ in range(n))


def make_toy(poems_per_author: int = 5, segs_per_poem: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for author, vocab in _AUTHORS.items():
        for p in range(poems_per_author):
            poem_id = f"toy::{author}::{p}"
            for _ in range(segs_per_poem):
                rows.append({
                    "poem_id": poem_id,
                    "author": author,
                    "period": "pre-islamic",
                    "century": 6,
                    "topic": pd.NA,
                    "text": _make_segment(vocab, rng),
                })
    return schema.validate(pd.DataFrame(rows))


if __name__ == "__main__":
    df = make_toy()
    out = config.DATA_TOY / "dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"[ok] {out} | строк: {len(df)} | авторов: {df['author'].nunique()} | поэм: {df['poem_id'].nunique()}")
