"""Majority voting по попарным моделям + определение confidence.  [чистая логика, без torch]

Схема round-robin (one-vs-one): каждая попарная модель (a,b) голосует за одного из
своих двух авторов — за того, кому дала P>0.5. Победитель — автор с макс. числом
голосов (ничьи разрешаются по суммарной вероятности).

ОПРЕДЕЛЕНИЕ CONFIDENCE (зафиксировано с дорожкой данных — на нём строится калибровка):
  confidence = средняя вероятность, которую модели, где участвует победитель,
               присвоили этому победителю. ∈ [0,1].
Интуиция: автора 0.99 уверенно обыгрывает всех оппонентов; спорный текст даёт низкую
среднюю вероятность даже у номинального победителя (ср. 0.71 vs 0.99 в статье).
"""
from __future__ import annotations

import math
from collections import defaultdict


def breakdown(pair_probs: list[tuple[str, str, float]]) -> dict:
    """Полная раскладка голосования по одному тексту.

    pair_probs: список (author_a, author_b, p_a), где p_a = P(author=a) от модели (a,b).
    Возвращает {"pred_author","confidence","votes","scores"}, где scores[author] —
    средняя вероятность, присвоенная автору по всем его парам (для A8 нужна уверенность
    не только победителя, но и приписываемого автора).
    """
    if not pair_probs:
        raise ValueError("Пустой список попарных вероятностей — нечего агрегировать.")

    votes: dict[str, int] = defaultdict(int)
    prob_sum: dict[str, float] = defaultdict(float)
    prob_cnt: dict[str, int] = defaultdict(int)
    for a, b, p_a in pair_probs:
        p_a = float(p_a)
        if not math.isfinite(p_a) or not 0.0 <= p_a <= 1.0:
            # Раньше NaN молча уходил в голос за b (NaN >= 0.5 -> False) и портил scores.
            raise ValueError(f"Пара {a} vs {b}: вероятность {p_a} — не число в [0,1]; голос не учитывается.")
        votes[a if p_a >= 0.5 else b] += 1
        prob_sum[a] += p_a; prob_cnt[a] += 1
        prob_sum[b] += (1.0 - p_a); prob_cnt[b] += 1

    scores = {au: prob_sum[au] / prob_cnt[au] for au in prob_cnt}
    best = max(votes, key=lambda au: (votes[au], scores[au]))  # ничья -> макс средняя вероятность
    return {
        "pred_author": best,
        "confidence": round(scores[best], 4),
        "votes": dict(votes),
        "scores": {k: round(v, 4) for k, v in scores.items()},
    }


def tally(pair_probs: list[tuple[str, str, float]]) -> dict:
    """Короткая форма breakdown: {"pred_author","confidence","votes"}."""
    b = breakdown(pair_probs)
    return {"pred_author": b["pred_author"], "confidence": b["confidence"], "votes": b["votes"]}
