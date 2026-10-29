"""Заглушка модели — НЕ учит ничего, но соблюдает контракт interface.py.

Нужна, чтобы дорожка данных/оценки работала до появления реальной модели:
eval-харнесс и калибровка порога полностью отлаживаются на ней.

Поведение подобрано так, чтобы eval выдавал осмысленные (не вырожденные) числа:
предсказание угадывается правильно с некоторой вероятностью, а confidence
коррелирует с правильностью — тогда калибровке порога есть что отделять.

Сокомандник заменит этот модуль на finetune.py + ensemble.py с той же сигнатурой.
"""
from __future__ import annotations
import hashlib
import random
from typing import Any

from aa.model.interface import Prediction


class StubEnsemble:
    def __init__(self, authors: list[str]):
        self.authors = authors


def train_pair(target: str, opponent: str, train_df, **cfg) -> str:
    """Реальная версия обучает CAMeLBERT на двух авторах. Здесь — фиктивный путь."""
    return f"stub://{target}__vs__{opponent}"


def fit(train_df, **cfg) -> StubEnsemble:
    authors = sorted(train_df["author"].astype(str).unique().tolist())
    # реальная версия: для каждого автора train_pair против оппонентов -> ансамбль
    return StubEnsemble(authors)


def _true_author_hint(text: str) -> str | None:
    """В игрушечных данных автор зашит в стиль; для честности стаба подглядываем
    не в метку, а в текст (как будто модель что-то выучила). В реальных данных
    этого хука не будет — тут он только чтобы eval не был чистым рандомом."""
    # toy-словарь -> автор (см. make_toy). Для не-toy вернёт None.
    buckets = {
        "poet_A": {"alif", "baa", "taa", "thaa"},
        "poet_B": {"jiim", "haa", "khaa", "daal"},
        "poet_C": {"raa", "zaa", "siin", "shiin"},
    }
    toks = set(text.split())
    best, score = None, 0
    for author, vocab in buckets.items():
        s = len(toks & vocab)
        if s > score:
            best, score = author, s
    return best


def predict(text: str, ensemble: StubEnsemble) -> Prediction:
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    hint = _true_author_hint(text)
    if hint in ensemble.authors and rng.random() < 0.8:
        return {"pred_author": hint, "confidence": round(rng.uniform(0.8, 0.99), 3)}
    return {
        "pred_author": rng.choice(ensemble.authors),
        "confidence": round(rng.uniform(0.5, 0.8), 3),
    }
