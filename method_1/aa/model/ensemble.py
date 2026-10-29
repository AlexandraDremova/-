"""Ансамбль one-vs-one + majority vote — публичный контракт модели.

Реализует те же fit / predict, что и stub.py, поэтому eval-харнесс не меняется:
    from aa.model import ensemble
    report, preds = run_cv(df, ensemble.fit, ensemble.predict)

Устройство (метод статьи):
  fit    -> для каждой пары авторов обучает бинарный CAMeLBERT (train_pair),
            каждая модель = 1 оппонент (оптимум статьи); набор пар задаёт sampling.author_pairs
  predict-> прогоняет текст через все попарные модели, majority vote (voting.tally),
            confidence = средняя вероятность победителя (см. voting.py)

torch импортируется лениво внутри train_pair/predict_proba, поэтому САМ модуль и
логика голосования тестируются без torch (см. tests/test_model_logic.py).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from aa import config
from aa.log import fmt_duration, mem_usage
from aa.model.finetune import NonFinitePrediction, PairClassifier, train_pair
from aa.model.interface import Prediction
from aa.model.sampling import author_pairs
from aa.model.voting import breakdown, tally

log = logging.getLogger(__name__)


@dataclass
class Ensemble:
    """Манифест попарных моделей. pairs: список {target, opponent, path}."""
    pairs: list[dict] = field(default_factory=list)
    _cache: dict = field(default_factory=dict, repr=False)

    def save(self, path: str) -> str:
        Path(path).write_text(json.dumps({"pairs": self.pairs}, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str) -> "Ensemble":
        return cls(pairs=json.loads(Path(path).read_text(encoding="utf-8"))["pairs"])


def fit(train_df, max_opponents: int | None = config.MAX_OPPONENTS, **cfg) -> Ensemble:
    authors = sorted(train_df["author"].astype(str).unique().tolist())
    if len(authors) < 2:
        raise ValueError("Нужно ≥2 авторов для построения ансамбля.")
    todo = author_pairs(authors, max_opponents=max_opponents)
    log.info("ансамбль: %d авторов -> %d попарных моделей", len(authors), len(todo))
    pairs, t0 = [], time.time()
    for i, (target, opponent) in enumerate(todo, 1):
        log.info("[пара %d/%d] %s vs %s", i, len(todo), target, opponent)
        path = train_pair(target, opponent, train_df, **cfg)
        pairs.append({"target": target, "opponent": opponent, "path": path})
        elapsed = time.time() - t0
        log.info("[пара %d/%d] готово | ансамбль: прошло %s, осталось ~%s",
                 i, len(todo), fmt_duration(elapsed), fmt_duration(elapsed / i * (len(todo) - i)))
    return Ensemble(pairs=pairs)


def _collect_pair_probs(texts: list[str], ensemble: Ensemble, _proba=None):
    """Для каждого текста — список (target, opponent, P(target)) по всем парам.

    ВАЖНО для памяти: внешний цикл по ПАРАМ, внутренний по текстам — в RAM одна модель
    за раз (иначе 45 моделей CAMeLBERT на 10 авторах ≈ 20 ГБ -> OOM).
    _proba(target, opponent, text, path) — подмена вероятностей для тестов без torch.
    """
    per_text: list[list[tuple[str, str, float]]] = [[] for _ in texts]
    if _proba is None:
        log.info("инференс: %d текстов через %d попарных моделей", len(texts), len(ensemble.pairs))
    for pr in ensemble.pairs:
        if _proba is not None:
            probs = [_proba(pr["target"], pr["opponent"], t, pr["path"]) for t in texts]
        else:
            clf = PairClassifier.load(pr["path"])
            probs = clf.predict_proba_batch(texts)   # один проход модели по всем текстам
            clf.free()
        for i, p_a in enumerate(probs):
            per_text[i].append((pr["target"], pr["opponent"], p_a))
    return per_text


def predict(text: str, ensemble: Ensemble, _proba=None) -> Prediction:
    """Один текст. Для батча/оценки используй predict_many — он памяти-эффективен."""
    return predict_many([text], ensemble, _proba=_proba)[0]


def predict_many(texts: list[str], ensemble: Ensemble, _proba=None) -> list[Prediction]:
    """Предсказания {pred_author, confidence} для списка текстов (памяти-эффективно)."""
    out = []
    for pp in _collect_pair_probs(texts, ensemble, _proba):
        b = tally(pp)
        out.append({"pred_author": b["pred_author"], "confidence": float(b["confidence"])})
    return out


def analyze_many(texts: list[str], ensemble: Ensemble, _proba=None) -> list[dict]:
    """Полная раскладка на текст: {pred_author, confidence, votes, scores}. Для A8 (кейсы муалак)."""
    return [breakdown(pp) for pp in _collect_pair_probs(texts, ensemble, _proba)]


# ---------- Потоковый режим: обучили пару -> сразу посчитали тексты -> выгрузили ----------
# Не сохраняет модели на диск (каждая пара ~440 МБ, 45 пар ≈ 20 ГБ) и держит в памяти одну
# модель. Используется в run_cv (A6/A7) и run_cases (A8), где модели после прогона не нужны.

def fit_collect(train_df, texts: list[str], max_opponents: int | None = config.MAX_OPPONENTS,
                _trainer=None, report: list | None = None, **cfg):
    """Для каждого текста — список (target, opponent, P(target)) по всем парам ансамбля.

    _trainer(target, opponent, train_df, **cfg) -> объект с predict_proba_batch/free
    (подмена для тестов без torch; по умолчанию PairClassifier.train).
    report: если передан список — в него добавляется сводка обучения каждой пары (train_info).
    """
    trainer = _trainer or PairClassifier.train
    authors = sorted(train_df["author"].astype(str).unique().tolist())
    if len(authors) < 2:
        raise ValueError("Нужно ≥2 авторов для построения ансамбля.")
    todo = author_pairs(authors, max_opponents=max_opponents)
    log.info("ансамбль (потоково): %d авторов -> %d пар, текстов для оценки: %d",
             len(authors), len(todo), len(texts))
    per_text: list[list[tuple[str, str, float]]] = [[] for _ in texts]
    t0 = time.time()
    for i, (target, opponent) in enumerate(todo, 1):
        log.info("[пара %d/%d] %s vs %s", i, len(todo), target, opponent)
        clf = trainer(target, opponent, train_df, **cfg)
        try:
            probs = clf.predict_proba_batch(texts)
        except NonFinitePrediction as e:
            # модель обучилась без явных NaN, но предсказывает NaN — переобучаем пару в fp32
            log.warning("  %s -> переобучаю пару в fp32", e)
            clf.free()
            clf = trainer(target, opponent, train_df, **{**cfg, "amp": False})
            probs = clf.predict_proba_batch(texts)
            clf.train_info = {**(getattr(clf, "train_info", {}) or {}), "fallback_fp32": True}
        info = dict(getattr(clf, "train_info", {}) or {})
        clf.free()
        for j, p_a in enumerate(probs):
            per_text[j].append((target, opponent, p_a))
        if report is not None:
            report.append({"target": target, "opponent": opponent, **info})
        elapsed = time.time() - t0
        log.info("[пара %d/%d] готово | ансамбль: прошло %s, осталось ~%s | %s",
                 i, len(todo), fmt_duration(elapsed), fmt_duration(elapsed / i * (len(todo) - i)),
                 mem_usage())
    if report is not None:
        fb = [f"{r['target']} vs {r['opponent']}" for r in report[-len(todo):] if r.get("fallback_fp32")]
        if fb:
            log.warning("пары, переобученные в fp32 из-за NaN в bf16: %s", ", ".join(fb))
    return per_text


def fit_predict_many(train_df, texts: list[str], **kw) -> list[Prediction]:
    """Обучить ансамбль на train_df и вернуть {pred_author, confidence} для texts."""
    out = []
    for pp in fit_collect(train_df, texts, **kw):
        b = tally(pp)
        out.append({"pred_author": b["pred_author"], "confidence": float(b["confidence"])})
    return out


def fit_analyze_many(train_df, texts: list[str], **kw) -> list[dict]:
    """Обучить ансамбль и вернуть полную раскладку (votes, scores) для texts. Для A8."""
    return [breakdown(pp) for pp in fit_collect(train_df, texts, **kw)]
