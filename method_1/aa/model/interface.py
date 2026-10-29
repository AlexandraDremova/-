"""КОНТРАКТ МОДЕЛИ — граница между дорожками.

Eval-харнесс зовёт модель ТОЛЬКО через эти две функции и ничего не знает про
её устройство. Сокомандник реализует их в finetune.py/ensemble.py; пока —
заглушки в stub.py с той же сигнатурой.

Единицы метода (из статьи):
  train_pair(target, opponent, train_df) -> путь к одной бинарной модели
  fit(train_df)  -> Ensemble: для каждого автора строит one-vs-one модели
                    (внутри вызывает train_pair), собирает majority-vote ансамбль
  predict(text, ensemble) -> {"pred_author": str, "confidence": float}

confidence ∈ [0,1] — на нём строится калибровка порога (детекция подделок),
поэтому определение confidence фиксируется совместно и не меняется молча.
"""
from __future__ import annotations
from typing import Any, Protocol, TypedDict


class Prediction(TypedDict):
    pred_author: str
    confidence: float


class Model(Protocol):
    def fit(self, train_df) -> Any: ...
    def predict(self, text: str, ensemble: Any) -> Prediction: ...


REQUIRED_KEYS = ("pred_author", "confidence")


def check_prediction(p: dict) -> Prediction:
    """Валидатор ответа модели — eval прогоняет через него каждый predict."""
    if set(REQUIRED_KEYS) - set(p):
        raise ValueError(f"predict() должен вернуть ключи {REQUIRED_KEYS}, вернул {list(p)}")
    if not 0.0 <= float(p["confidence"]) <= 1.0:
        raise ValueError(f"confidence вне [0,1]: {p['confidence']}")
    return p  # type: ignore[return-value]
