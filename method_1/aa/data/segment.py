"""Нарезка очищенной поэмы на сегменты — окна по N бейтов (строк).

Зачем: у CAMeLBERT лимит 512 токенов, а касыды длинные. Плюс это даёт
несколько обучающих примеров из одной поэмы (метод из статьи работает на
~60 примерах на поэта — без нарезки столько не набрать на доисламских поэтах).

Единица сегмента — бейт (одна строка очищенного текста). Параметры окна/шага
в aa/config.py. Сегменты короче MIN_SEGMENT_BAYTS отбрасываем.
"""
from __future__ import annotations
from aa import config


def to_bayts(poem_text: str) -> list[str]:
    return [ln.strip() for ln in poem_text.splitlines() if ln.strip()]


def segment_poem(
    poem_text: str,
    window: int = config.SEGMENT_WINDOW,
    stride: int = config.SEGMENT_STRIDE,
    min_bayts: int = config.MIN_SEGMENT_BAYTS,
) -> list[str]:
    """Возвращает список сегментов (каждый — строка из `window` бейтов)."""
    bayts = to_bayts(poem_text)
    if not bayts:
        return []
    if len(bayts) <= window:
        return [" \n".join(bayts)] if len(bayts) >= min_bayts else []

    segments: list[str] = []
    for start in range(0, len(bayts) - min_bayts + 1, stride):
        chunk = bayts[start:start + window]
        if len(chunk) >= min_bayts:
            segments.append(" \n".join(chunk))
        if start + window >= len(bayts):
            break
    return segments
