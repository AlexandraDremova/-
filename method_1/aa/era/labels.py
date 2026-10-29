"""Эпохи APCD и схемы классов.  [чистая логика]

В APCD поле `era` — арабская строка, 12 значений (قبل الإسلام, المخضرمين, الإسلامي, الأموي,
بين الدولتين, العباسي, ... , الحديث). Точное написание может гулять (артикль, «العصر», хамза),
поэтому сопоставление идёт по нормализованным подстрокам, а не по точному совпадению.
Нераспознанные значения не теряются молча: build_apcd печатает их со счётом.

Канонические ключи эпох (в хронологическом порядке):
    pre_islamic      قبل الإسلام          джахилия, до ~610
    mukhadram        المخضرمين            застали и джахилию, и ислам (Лабид, Хассан ибн Сабит)
    islamic          الإسلامي             ранний ислам, ~610–661
    umayyad          الأموي               661–750
    between_states   بين الدولتين         «между двумя державами»: застали Омейядов и Аббасидов (~750)
    abbasid          العباسي              750–1258
    andalus_maghreb  المغرب والأندلس      по географии, а не по времени (~711–1492)
    fatimid          الفاطمي              909–1171
    ayyubid          الأيوبي              1171–1260
    mamluk           المملوكي             1250–1517
    ottoman          العثماني             1517–~1900
    modern           الحديث               XIX–XXI вв.

Схемы классов (значение None = эпоха в этой схеме не используется):
    binary  — доисламская против классической исламской (без переходных мухадрамов и без новой поэзии).
              Самая прямая постановка для вопроса о муаллаках.
    coarse  — 5 крупных периодов (по умолчанию): доисламский / ранний ислам и Омейяды /
              Аббасиды / поздняя классика (Андалусия, Фатимиды, Айюбиды, Мамлюки, Османы) / новое время.
    full    — все 12 эпох APCD как есть.
"""
from __future__ import annotations

import re

ERA_KEYS = ["pre_islamic", "mukhadram", "islamic", "umayyad", "between_states", "abbasid",
            "andalus_maghreb", "fatimid", "ayyubid", "mamluk", "ottoman", "modern"]

ERA_NAMES_RU = {
    "pre_islamic": "доисламская", "mukhadram": "мухадрамы", "islamic": "ранний ислам",
    "umayyad": "Омейяды", "between_states": "между Омейядами и Аббасидами", "abbasid": "Аббасиды",
    "andalus_maghreb": "Магриб и Андалусия", "fatimid": "Фатимиды", "ayyubid": "Айюбиды",
    "mamluk": "Мамлюки", "ottoman": "Османы", "modern": "новое время",
}

# (нормализованная подстрока -> ключ). Порядок важен: «قبل الاسلام» проверяется раньше «اسلامي».
_ERA_PATTERNS: list[tuple[str, str]] = [
    ("قبل الاسلام", "pre_islamic"),
    ("جاهلي", "pre_islamic"),
    ("مخضرم", "mukhadram"),
    ("بين الدولتين", "between_states"),
    ("اموي", "umayyad"),
    ("عباسي", "abbasid"),
    ("اندلس", "andalus_maghreb"),
    ("مغرب", "andalus_maghreb"),
    ("فاطمي", "fatimid"),
    ("ايوبي", "ayyubid"),
    ("مملوك", "mamluk"),
    ("عثماني", "ottoman"),
    ("حديث", "modern"),
    ("اسلامي", "islamic"),
]

SCHEMES: dict[str, dict[str, str | None]] = {
    "binary": {
        "pre_islamic": "pre_islamic",
        "mukhadram": None,          # переходные — метка неоднозначна
        "islamic": "islamic_era", "umayyad": "islamic_era", "between_states": "islamic_era",
        "abbasid": "islamic_era", "andalus_maghreb": "islamic_era", "fatimid": "islamic_era",
        "ayyubid": "islamic_era", "mamluk": "islamic_era", "ottoman": "islamic_era",
        "modern": None,             # свободный стих и т.п. — к вопросу о муаллаках не относится
    },
    "coarse": {
        "pre_islamic": "pre_islamic",
        "mukhadram": "early_islamic", "islamic": "early_islamic", "umayyad": "early_islamic",
        "between_states": "abbasid", "abbasid": "abbasid",
        "andalus_maghreb": "late_classical", "fatimid": "late_classical", "ayyubid": "late_classical",
        "mamluk": "late_classical", "ottoman": "late_classical",
        "modern": "modern",
    },
    "full": {k: k for k in ERA_KEYS},
}

# Порядок классов каждой схемы — хронологический (нужен для «ошибки на соседнюю эпоху»).
SCHEME_CLASSES: dict[str, list[str]] = {
    "binary": ["pre_islamic", "islamic_era"],
    "coarse": ["pre_islamic", "early_islamic", "abbasid", "late_classical", "modern"],
    "full": list(ERA_KEYS),   # andalus_maghreb — по географии; его место в порядке условно
}

PRE_ISLAMIC = "pre_islamic"   # класс, вероятность которого — оценка муаллаки


_AR_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_AR_PUNCT = re.compile(r"[^\w\s]")


def normalize_ar(s: str) -> str:
    """Нормализация для СОПОСТАВЛЕНИЯ (не для модели): огласовки и кашида долой,
    أإآٱ->ا, ؤ->و, ئ->ي, ى->ي, ة->ه, пунктуация -> пробел, пробелы схлопнуты."""
    s = _AR_DIAC.sub("", str(s or ""))
    s = re.sub("[أإآٱ]", "ا", s)
    s = s.replace("ؤ", "و").replace("ئ", "ي").replace("ى", "ي").replace("ة", "ه")
    s = _AR_PUNCT.sub(" ", s)
    return " ".join(s.split())


def canonical_era(raw) -> str | None:
    """Строка `era` из APCD -> канонический ключ (или None, если не распознана)."""
    s = normalize_ar(raw)
    if not s or s.lower() == "nan":
        return None
    for pat, key in _ERA_PATTERNS:
        if pat in s:
            return key
    return None


def scheme_map(scheme: str) -> dict[str, str | None]:
    if scheme not in SCHEMES:
        raise ValueError(f"Неизвестная схема {scheme!r}; доступны: {sorted(SCHEMES)}")
    return SCHEMES[scheme]


def to_class(era_key, scheme: str) -> str | None:
    if era_key is None or (isinstance(era_key, float)) or str(era_key) in ("", "<NA>", "nan"):
        return None
    return scheme_map(scheme).get(str(era_key))


def classes_of(scheme: str) -> list[str]:
    return list(SCHEME_CLASSES[scheme])


# ---------- поэты проекта: исключаются из обучения модели эпох ----------
# Точные нормализованные имена (как они обычно пишутся в APCD / aldiwan) ...
PROJECT_POETS_EXACT = {
    "عنتره بن شداد", "عنتره", "الاعشي", "الاعشي الكبير", "اعشي قيس", "ميمون بن قيس",
    "امرو القيس", "امري القيس", "زهير بن ابي سلمي", "زهير بن ابي سلمي المزني", "النابغه الذبياني",
    "لبيد بن ربيعه", "لبيد بن ربيعه العامري", "عبيد بن الابرص", "طرفه بن العبد",
    "عمرو بن كلثوم", "الحارث بن حلزه", "الحارث بن حلزه اليشكري",
}
# ... и подстроки, которые однозначно указывают на них.
PROJECT_POETS_SUBSTR = ["عنتره بن شداد", "امرو القيس", "امري القيس", "زهير بن ابي سلمي",
                        "النابغه الذبياني", "لبيد بن ربيعه", "عبيد بن الابرص", "طرفه بن العبد",
                        "عمرو بن كلثوم", "حلزه"]
# Похожие имена, которые НЕ исключаются автоматически, но печатаются для ручной проверки
# (напр. «أعشى همدان» — другой поэт, омейядский).
PROJECT_POETS_WARN = ["عنتر", "اعشي", "زهير", "نابغه", "لبيد", "طرفه", "كلثوم", "الابرص"]


def is_project_poet(name: str, extra: set[str] | None = None) -> bool:
    n = normalize_ar(name)
    if n in PROJECT_POETS_EXACT or (extra and n in extra):
        return True
    return any(s in n for s in PROJECT_POETS_SUBSTR)


def looks_like_project_poet(name: str) -> bool:
    n = normalize_ar(name)
    return any(s in n for s in PROJECT_POETS_WARN)
