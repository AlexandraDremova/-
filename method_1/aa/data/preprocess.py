"""Чистка арабского текста перед сегментацией.

Снимаем диакритику (tashkeel) и кашиду (tatweel), убираем HTML-мусор и
нормализуем пробелы. Если camel-tools установлен — используем его dediac,
иначе падаем на регулярку: результат почти тот же, пайплайн не блокируется.

ВАЖНО: модель получает текст УЖЕ после этой чистки. На стороне модели чистку
не дублируем (см. teammate_model_track.md).
"""
from __future__ import annotations
import re
import html

try:
    from camel_tools.utils.dediac import dediac_ar
    _HAS_CAMEL = True
except Exception:  # noqa: BLE001 — camel-tools опционален
    _HAS_CAMEL = False

# Диакритика + надстрочные знаки; ـ — кашида (tatweel)
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t ]+")
_NONARABIC_LINE = re.compile(r"[؀-ۿ]")  # содержит ли строка арабские буквы


def strip_diacritics(text: str) -> str:
    if _HAS_CAMEL:
        text = dediac_ar(text)
    else:
        text = _DIACRITICS.sub("", text)
    return text.replace(_TATWEEL, "")


def clean(text: str) -> str:
    """Полная чистка одной поэмы (многострочный текст)."""
    if not text:
        return ""
    text = html.unescape(text)
    text = _TAG.sub(" ", text)
    text = strip_diacritics(text)
    # построчно: нормализуем пробелы, выкидываем строки без арабских букв
    lines = []
    for line in text.splitlines():
        line = _WS.sub(" ", line).strip()
        if line and _NONARABIC_LINE.search(line):
            lines.append(line)
    return "\n".join(lines)


def using_camel_tools() -> bool:
    """Полезно залогировать на старте: честная ли чистка или fallback."""
    return _HAS_CAMEL
