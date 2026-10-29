"""Метрики классификатора эпохи и их статистика.  [чистая логика, numpy]

Всё считается через матрицы ошибок: у каждого тестового поэта своя матрица (C×C),
метрика по выборке — функция суммы матриц. Поэтому бутстреп и перестановки по ПОЭТАМ
(а сегменты одного поэта зависимы, и единица независимости — поэт) стоят миллисекунды.

  * балансная точность (средняя полнота по классам) — главная метрика: классы несбалансированы;
  * macro-F1, точность, «точно или соседняя эпоха» (классы схемы упорядочены по времени);
  * 95% интервал — стратифицированный кластерный бутстреп по поэтам (поэты пересэмплируются
    внутри своего класса, чтобы каждый класс оставался в выборке);
  * p-value против случайного — перестановка меток ЭПОХ между поэтами при фиксированных
    предсказаниях (нулевая гипотеза: предсказания не связаны с эпохой поэта);
  * парное сравнение двух методов на одних и тех же сегментах — бутстреп разности по поэтам
    и перестановочный тест (для каждого поэта ответы двух методов меняются местами с вероятностью 1/2).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = ("balanced_acc", "macro_f1", "acc", "adjacent_acc")


# ---------- калибровка температурой ----------

def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    """softmax(log p / T): T > 1 смягчает переуверенные вероятности; argmax не меняется."""
    lp = np.log(np.clip(np.asarray(probs, float), 1e-12, 1.0)) / t
    lp -= lp.max(axis=1, keepdims=True)
    e = np.exp(lp)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, y, grid=None) -> float:
    """T, минимизирующая среднюю NLL на ВАЛИДАЦИИ (поэты, не участвовавшие в обучении).
    Стандартная пост-калибровка (Guo et al., 2017); перебор по логарифмической сетке.
    По умолчанию T ≥ 1 — только смягчение: если валидация разделяется почти идеально, оптимальная по NLL
    T < 1 ещё сильнее прижала бы вероятности к 0/1, а цель калибровки здесь — обратная."""
    y = np.asarray(y, int)
    fin = np.isfinite(probs).all(axis=1) if probs.size else np.array([], bool)
    probs, y = probs[fin], y[fin]
    if probs.size == 0 or y.size == 0:
        return 1.0
    grid = np.exp(np.linspace(0.0, np.log(20.0), 101)) if grid is None else grid
    nll = [-np.mean(np.log(np.clip(apply_temperature(probs, t)[np.arange(y.size), y], 1e-12, 1)))
           for t in grid]
    return float(grid[int(np.argmin(nll))])


def calibrate_sets(probs: dict[str, np.ndarray], y_val, val_key: str = "val") -> tuple[dict, float]:
    """Подбирает T по probs[val_key] и применяет ко всем наборам. Без валидации — T = 1."""
    if val_key not in probs or len(y_val) == 0:
        return probs, 1.0
    t = fit_temperature(probs[val_key], y_val)
    return {k: apply_temperature(v, t) for k, v in probs.items()}, t


# ---------- матрицы ошибок ----------

def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    return np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes) \
        .reshape(n_classes, n_classes).astype(float)


def from_confusion(cm: np.ndarray, metric: str) -> float:
    cm = np.asarray(cm, float)
    total = cm.sum()
    if total == 0:
        return float("nan")
    support = cm.sum(axis=1)
    present = support > 0
    diag = np.diag(cm)
    if metric == "acc":
        return float(diag.sum() / total)
    if metric == "balanced_acc":
        return float(np.mean(diag[present] / support[present]))
    if metric == "macro_f1":
        denom = support + cm.sum(axis=0)
        f1 = np.where(denom > 0, 2 * diag / np.maximum(denom, 1e-12), 0.0)
        return float(np.mean(f1[present]))
    if metric == "adjacent_acc":
        c = cm.shape[0]
        i, j = np.indices((c, c))
        return float(cm[np.abs(i - j) <= 1].sum() / total)
    raise ValueError(metric)


def all_metrics(cm: np.ndarray) -> dict:
    return {m: from_confusion(cm, m) for m in METRICS}


def per_class_table(cm: np.ndarray, classes: list[str]) -> pd.DataFrame:
    support, predicted, diag = cm.sum(axis=1), cm.sum(axis=0), np.diag(cm)
    rec = np.where(support > 0, diag / np.maximum(support, 1), np.nan)
    prec = np.where(predicted > 0, diag / np.maximum(predicted, 1), np.nan)
    f1 = np.where(support + predicted > 0, 2 * diag / np.maximum(support + predicted, 1), np.nan)
    return pd.DataFrame({"класс": classes, "сегментов": support.astype(int), "предсказано": predicted.astype(int),
                         "полнота": rec.round(3), "точность": prec.round(3), "F1": f1.round(3)})


# ---------- раскладка по поэтам ----------

class PoetBlocks:
    """Предсказания, сгруппированные по поэтам: hist[p, c] — сколько сегментов поэта p отнесено
    к классу c; label[p] — истинный класс поэта."""

    def __init__(self, y_true, y_pred, poets, n_classes: int):
        df = pd.DataFrame({"y": np.asarray(y_true, int), "p": np.asarray(y_pred, int),
                           "g": np.asarray(poets).astype(str)})
        self.poets = np.array(sorted(df["g"].unique()))
        pos = {g: i for i, g in enumerate(self.poets)}
        gi = df["g"].map(pos).to_numpy()
        self.n_classes = n_classes
        self.hist = np.zeros((len(self.poets), n_classes))
        np.add.at(self.hist, (gi, df["p"].to_numpy()), 1)
        lab = df.groupby("g")["y"].agg(lambda s: s.value_counts().index[0])
        self.label = lab.reindex(self.poets).to_numpy().astype(int)

    def cm(self, idx=None, labels=None, hist=None) -> np.ndarray:
        h = self.hist if hist is None else hist
        lab = self.label if labels is None else labels
        if idx is not None:
            h, lab = h[idx], lab[idx]
        onehot = np.eye(self.n_classes)[lab]
        return onehot.T @ h


def poet_votes(probs: np.ndarray, poets) -> tuple[np.ndarray, np.ndarray]:
    """Решение по поэту: argmax средней лог-вероятности его сегментов. Возвращает (поэты, классы)."""
    lp = np.log(np.clip(np.asarray(probs, float), 1e-9, 1.0))
    df = pd.DataFrame(lp)
    df["g"] = np.asarray(poets).astype(str)
    mean = df.groupby("g").mean()
    return mean.index.to_numpy(), mean.to_numpy().argmax(axis=1)


def _strat_resample(labels: np.ndarray, rng) -> np.ndarray:
    idx = []
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        idx.append(rng.choice(members, size=members.size, replace=True))
    return np.concatenate(idx)


def bootstrap_ci(blocks: PoetBlocks, metric: str, n_boot: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = np.array([from_confusion(blocks.cm(_strat_resample(blocks.label, rng)), metric)
                     for _ in range(n_boot)])
    a = (1 - level) / 2
    return float(np.nanquantile(vals, a)), float(np.nanquantile(vals, 1 - a))


def permutation_p(blocks: PoetBlocks, metric: str, n_perm: int = 2000, seed: int = 0) -> float:
    """p(метрика ≥ наблюдаемой | эпохи поэтов перемешаны). Минимум 1/(n_perm+1)."""
    rng = np.random.default_rng(seed)
    obs = from_confusion(blocks.cm(), metric)
    null = np.array([from_confusion(blocks.cm(labels=rng.permutation(blocks.label)), metric)
                     for _ in range(n_perm)])
    return float((1 + np.sum(null >= obs - 1e-12)) / (n_perm + 1))


def summarize(y_true, y_pred, poets, n_classes: int, n_boot: int = 2000, n_perm: int = 2000,
              seed: int = 0) -> dict:
    """Все метрики + интервалы + p-value против случайного (для балансной точности и macro-F1)."""
    b = PoetBlocks(y_true, y_pred, poets, n_classes)
    cm = b.cm()
    out = {"сегментов": int(cm.sum()), "поэтов": len(b.poets)}
    for m in METRICS:
        out[m] = round(from_confusion(cm, m), 4)
    for m in ("balanced_acc", "macro_f1"):
        lo, hi = bootstrap_ci(b, m, n_boot, seed)
        out[f"{m}_ci95"] = f"{lo:.3f}–{hi:.3f}"
        out[f"{m}_p_vs_chance"] = permutation_p(b, m, n_perm, seed)
    out["случайный_уровень_balanced_acc"] = round(1 / max(int((cm.sum(axis=1) > 0).sum()), 1), 4)
    return out


def paired_compare(y_true, pred_a, pred_b, poets, n_classes: int, metric: str = "balanced_acc",
                   n_boot: int = 2000, n_perm: int = 2000, seed: int = 0) -> dict:
    """Метод A против метода B на одних и тех же сегментах. Разность = A − B."""
    a = PoetBlocks(y_true, pred_a, poets, n_classes)
    b = PoetBlocks(y_true, pred_b, poets, n_classes)
    assert (a.poets == b.poets).all()
    obs = from_confusion(a.cm(), metric) - from_confusion(b.cm(), metric)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = _strat_resample(a.label, rng)
        boots.append(from_confusion(a.cm(idx), metric) - from_confusion(b.cm(idx), metric))
    null = []
    for _ in range(n_perm):
        swap = rng.random(len(a.poets)) < 0.5
        ha = np.where(swap[:, None], b.hist, a.hist)
        hb = np.where(swap[:, None], a.hist, b.hist)
        null.append(from_confusion(a.cm(hist=ha), metric) - from_confusion(a.cm(hist=hb), metric))
    null = np.abs(np.array(null))
    return {"метрика": metric, "значение_A": round(from_confusion(a.cm(), metric), 4),
            "значение_B": round(from_confusion(b.cm(), metric), 4), "разность_A_минус_B": round(obs, 4),
            "ci95": f"{np.quantile(boots, 0.025):.3f}–{np.quantile(boots, 0.975):.3f}",
            "p_двусторонний": float((1 + np.sum(null >= abs(obs) - 1e-12)) / (n_perm + 1))}
