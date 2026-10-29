"""Сбалансированный отбор примеров для одной бинарной пары.  [чистая логика, без torch]

Метод статьи: на каждую пару (target vs opponent) берём ~60 примеров на поэта.
Доисламских данных мало — если примеров меньше, берём сколько есть (не падаем).
"""
from __future__ import annotations

import zlib

import pandas as pd

from aa import config


def pair_seed(seed: int, target: str, opponent: str) -> int:
    """Детерминированный seed, свой для каждой пары (не зависит от PYTHONHASHSEED).

    Зачем: у крупных поэтов >60 сегментов, а пара берёт 60. С общим seed все пары с
    Имру аль-Кайсом видели бы ОДНИ И ТЕ ЖЕ 60 сегментов; с парным seed ансамбль в целом
    покрывает больше его корпуса."""
    return (seed + zlib.crc32(f"{target}|{opponent}".encode("utf-8"))) % (2**31 - 1)


def pair_frame(
    df: pd.DataFrame,
    target: str,
    opponent: str,
    examples_per_poet: int = config.EXAMPLES_PER_POET,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Строки двух авторов, по <=examples_per_poet на каждого, с меткой label.

    label = 1 для target, 0 для opponent. Возвращает перемешанный df с
    колонками [text, label, author].
    """
    out = []
    for author, label in ((target, 1), (opponent, 0)):
        sub = df[df["author"].astype(str) == str(author)]
        if len(sub) > examples_per_poet:
            sub = sub.sample(n=examples_per_poet, random_state=seed)
        out.append(pd.DataFrame({
            "text": sub["text"].astype(str).to_numpy(),
            "label": label,
            "author": str(author),
        }))
    res = pd.concat(out, ignore_index=True)
    return res.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def author_pairs(
    authors: list[str],
    max_opponents: int | None = config.MAX_OPPONENTS,
    seed: int = config.SEED,
) -> list[tuple[str, str]]:
    """Список неупорядоченных пар авторов.

    max_opponents=None -> полный round-robin (все пары; каждая модель = 1 оппонент).
    int -> на каждого автора не более N оппонентов (дешевле; тот же knob, что крутят в статье).
    """
    import random

    authors = sorted(set(map(str, authors)))
    if max_opponents is None:
        return [(a, b) for i, a in enumerate(authors) for b in authors[i + 1:]]

    rng = random.Random(seed)
    seen: set[frozenset[str]] = set()
    pairs: list[tuple[str, str]] = []
    for a in authors:
        opponents = [x for x in authors if x != a]
        rng.shuffle(opponents)
        for b in opponents[:max_opponents]:
            key = frozenset((a, b))
            if key not in seen:
                seen.add(key)
                pairs.append((a, b))
    return pairs
