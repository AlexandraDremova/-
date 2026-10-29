"""Эталонные распределения из кросс-валидации.  [чистая логика, без torch]

1) Попарная точность — метрика, сопоставимая со статьёй.
   В статье F1 0.97–1.0 получен в постановке «целевой поэт против оппонента(ов)»
   (по тексту статьи это не сказано явно, но обсуждается именно «opposite class»).
   Наш 10-классовый macro-F1 — более строгая задача, сравнивать его с 0.97–1.0 напрямую нельзя.
   Здесь для каждой попарной модели считаем точность на тестовых сегментах ЕЁ ДВУХ авторов.

2) Эталон уверенности в истинном авторе на НЕВИДЕННЫХ поэмах.
   Для каждой тестовой поэмы CV: средняя по её сегментам уверенность в её настоящем авторе.
   Это распределение «как модель оценивает заведомо атрибутированные (по диванам) поэмы,
   которых она не видела». Муаллаку сравниваем с ним: какая доля поэм того же автора
   получила уверенность НЕ ВЫШЕ, чем муаллака (эмпирический перцентиль).
   Это и есть вероятностная оценка из ТЗ — вместо абсолютного порога на чужой величине.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def pairwise_accuracy(preds: pd.DataFrame) -> pd.DataFrame:
    """Точность каждой попарной модели на тестовых сегментах её двух авторов.

    preds: строки с колонками fold, true_author, pair_probs (список (a, b, P(a))).
    Возвращает по строке на (fold, a, b): n, acc, balanced_acc (среднее полноты по двум авторам —
    тест несбалансирован, у малых авторов мало сегментов).
    """
    stats: dict[tuple, dict] = {}
    for row in preds.itertuples(index=False):
        true = str(row.true_author)
        for a, b, p in row.pair_probs:
            if true not in (a, b):
                continue
            key = (getattr(row, "fold", 0), a, b)
            s = stats.setdefault(key, {"n_a": 0, "ok_a": 0, "n_b": 0, "ok_b": 0})
            if true == a:
                s["n_a"] += 1
                s["ok_a"] += int(p >= 0.5)
            else:
                s["n_b"] += 1
                s["ok_b"] += int(p < 0.5)
    rows = []
    for (fold, a, b), s in stats.items():
        n = s["n_a"] + s["n_b"]
        recalls = [s["ok_a"] / s["n_a"] if s["n_a"] else None, s["ok_b"] / s["n_b"] if s["n_b"] else None]
        recalls = [r for r in recalls if r is not None]
        rows.append({"fold": fold, "a": a, "b": b, "n": n, "n_a": s["n_a"], "n_b": s["n_b"],
                     "acc": (s["ok_a"] + s["ok_b"]) / n if n else np.nan,
                     "balanced_acc": float(np.mean(recalls)) if recalls else np.nan})
    return pd.DataFrame(rows)


def poem_true_scores(preds: pd.DataFrame) -> pd.DataFrame:
    """Средняя по сегментам уверенность в НАСТОЯЩЕМ авторе для каждой тестовой поэмы CV."""
    if "true_score" not in preds.columns:
        raise ValueError("В предсказаниях нет true_score — нужен прогон run_cv с раскладкой scores.")
    g = preds.groupby("poem_id")
    return pd.DataFrame({
        "author": g["true_author"].first(),
        "n_segments": g.size(),
        "true_score": g["true_score"].mean(),
    }).reset_index()


def percentile_in(value: float, reference) -> float:
    """Доля эталонных значений, которые НЕ ВЫШЕ value (0..1). NaN, если эталон пуст."""
    ref = np.asarray([x for x in reference if x is not None and np.isfinite(x)], dtype=float)
    if ref.size == 0 or value is None or not np.isfinite(value):
        return float("nan")
    return float((ref <= value).mean())


def summarize_reference(ref: pd.DataFrame) -> pd.DataFrame:
    """Сводка эталона по авторам: число поэм, медиана и 5-й/10-й перцентили уверенности."""
    return (ref.groupby("author")["true_score"]
               .agg(poems="size", median="median",
                    p05=lambda s: s.quantile(0.05), p10=lambda s: s.quantile(0.10))
               .round(3).reset_index())


# ---------- Старый эталон, подобранный по длине (до 28.10; оставлен только для сравнения) ----------
# Муаллаку из k сегментов сравнивали со средними по k сегментам, набранным из невиденных (в CV)
# поэм того же автора (блочный бутстреп). Недостаток: поэма автора в среднем 1.5–3 сегмента,
# поэтому «эталонная поэма» из 10 сегментов на деле склеена из нескольких разных поэм. Сегменты
# одной поэмы похожи друг на друга, и среднее по одной настоящей поэме разбросано сильнее, чем
# среднее по смеси нескольких. Эталон получается слишком узким, p-value — заниженными.
# Исправление — single_poem_reference ниже.

def segment_scores_by_poem(cv_preds: pd.DataFrame) -> dict[str, list[list[float]]]:
    """{автор: [[уверенности сегментов поэмы 1], [поэмы 2], ...]} из cv_predictions.csv."""
    out: dict[str, list[list[float]]] = {}
    for (author, _), g in cv_preds.groupby(["true_author", "poem_id"], sort=False):
        vals = [float(x) for x in g["true_score"] if pd.notna(x)]
        if vals:
            out.setdefault(str(author), []).append(vals)
    return out


def length_matched_reference(poems: list[list[float]], k: int, n_boot: int = 2000,
                             seed: int = 0) -> np.ndarray:
    """n_boot средних по k сегментам: поэмы автора берутся целиком, случайно, с возвращением,
    пока не наберётся k сегментов (лишние в последней поэме отбрасываются)."""
    poems = [np.asarray(p, dtype=float) for p in poems if len(p)]
    if not poems or k <= 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        acc, n = [], 0
        while n < k:
            p = poems[rng.integers(len(poems))]
            acc.append(p)
            n += len(p)
        means[b] = np.concatenate(acc)[:k].mean()
    return means


def author_cv_stats(cv_preds: pd.DataFrame, cv_pairwise: pd.DataFrame | None = None) -> pd.DataFrame:
    """Насколько модель вообще узнаёт автора: recall в 10-классовой CV и средняя
    сбалансированная точность его попарных моделей."""
    st = cv_preds.groupby("true_author")["correct"].mean().rename("recall_CV").to_frame()
    if cv_pairwise is not None and len(cv_pairwise):
        per_pair = cv_pairwise.groupby(["a", "b"])["balanced_acc"].mean().reset_index()
        rows = {}
        for au in st.index:
            m = per_pair[(per_pair["a"] == au) | (per_pair["b"] == au)]["balanced_acc"]
            rows[au] = m.mean() if len(m) else np.nan
        st["попарная_точность"] = pd.Series(rows)
    return st.round(3)


# ---------- Эталон «одна поэма из k сегментов» (с 28.10) ----------
# Модель на логит-шкале уверенности z = logit(y) (y — уверенность в настоящем авторе на сегменте):
#     z_ij = μ + u_i + e_ij,
#   u_i — эффект поэмы i, общий для всех её сегментов (тема, размер, сохранность текста), дисперсия τ²;
#   e_ij — разброс сегментов внутри поэмы, дисперсия σ².
# Среднее по ОДНОЙ поэме из k сегментов разбросано на τ² + σ²/k: часть τ² с длиной не уменьшается.
# Старый эталон брал смесь из нескольких поэм, и эта часть делилась на число поэм в смеси.
#
# Основной способ (method="bayes") — байесовский предсказательный эталон:
#   * μ интегрируется с плоским априорным распределением, на τ — равномерное (τ ≥ 0), на σ — 1/σ;
#   * апостериорное распределение (τ, σ) считается точно на сетке (одномерная модель, это дёшево);
#   * эталонное значение: (τ, σ) из апостериорного, μ | τ, σ, эффект новой поэмы и k отклонений
#     сегментов -> среднее уверенности. ФОРМА эффекта поэмы и отклонений сегментов берётся из данных
#     (стандартизованные остатки), МАСШТАБ — из апостериорного, то есть с учётом того, что τ и σ
#     оценены по конечному числу поэм (у Хариса их 10, у Антары 138).
# Запасной способ (method="bootstrap") — остаточный бутстреп для многоуровневых моделей
#   (Carpenter, Goldstein, Rasbash, 2003) с внешним бутстрепом по поэмам. На синтетике он честен при
#   ~140 поэмах, но при ~25 поэмах даёт ложную тревогу в ~8% случаев вместо 5%: бутстреп по поэмам
#   недооценивает неопределённость τ. Поэтому основной — байесовский. Сравнение — refcheck/ в отчёте.
#
# Допущения и в какую сторону они ошибаются:
#   * разброс эффектов поэм не зависит от длины поэмы (проверяется: reference_fit_table,
#     колонка связь_длины_и_уверенности);
#   * сегменты внутри поэмы похожи одинаково, независимо от расстояния между ними. Если на деле
#     далёкие сегменты длинной поэмы похожи меньше (смена темы), эталон шире нужного — p-value
#     завышены, то есть тест осторожнее, а не смелее;
#   * калибровку на настоящих поэмах проверяет loo_calibration (доля p ≤ 0.05 должна быть ≈ 5%).

SCALES = ("logit", "prob")
MIN_POEMS = 5          # меньше поэм у автора — эталон не строится
MIN_WITHIN_DF = 5      # сумма (n_i − 1) по поэмам: нужно хоть сколько-то поэм из ≥2 сегментов


def _to_scale(y, scale: str, eps: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if scale == "logit":
        y = np.clip(y, eps, 1.0 - eps)
        return np.log(y / (1.0 - y))
    if scale == "prob":
        return y
    raise ValueError(f"scale должен быть одним из {SCALES}, а не {scale!r}")


def _from_scale(z: np.ndarray, scale: str) -> np.ndarray:
    if scale == "logit":
        return 1.0 / (1.0 + np.exp(-z))
    return np.clip(z, 0.0, 1.0)


@dataclass
class PoemModelFit:
    """Оценки модели «поэма + сегменты» для одного автора (на выбранной шкале)."""
    ok: bool
    reason: str
    scale: str
    n_poems: int
    n_segments: int
    multi_segment_poems: int
    within_df: int
    mu: float = float("nan")          # на шкале scale
    tau2: float = float("nan")        # дисперсия эффектов поэм
    sigma2: float = float("nan")      # дисперсия сегментов внутри поэмы
    icc: float = float("nan")         # τ² / (τ² + σ²): насколько сегменты одной поэмы похожи

    def as_dict(self) -> dict:
        return asdict(self)


def _poem_arrays(poems, scale: str, eps: float):
    """n_i, среднее z по поэме, внутрипоэмная сумма квадратов и пул внутрипоэмных отклонений.
    Отклонения домножены на sqrt(n_i/(n_i−1)), чтобы их средний квадрат оценивал σ² без смещения."""
    zs = [_to_scale(p, scale, eps) for p in poems if len(p)]
    zs = [z[np.isfinite(z)] for z in zs]
    zs = [z for z in zs if z.size]
    n = np.array([z.size for z in zs], dtype=float)
    zbar = np.array([z.mean() for z in zs], dtype=float)
    ssw = np.array([((z - z.mean()) ** 2).sum() for z in zs], dtype=float)
    multi = [(z - z.mean()) * math.sqrt(z.size / (z.size - 1)) for z in zs if z.size > 1]
    resid = np.concatenate(multi) if multi else np.array([], dtype=float)
    return n, zbar, ssw, resid


def _anova_moments(n: np.ndarray, zbar: np.ndarray, ssw: np.ndarray, sigma2_fallback: float | None = None):
    """Оценки метода моментов для однофакторной модели со случайным эффектом группы.
    Работает и для одной выборки (форма (m,)), и для пачки бутстреп-повторов (форма (B, m)).
    Возвращает (mu, tau2, sigma2, within_df)."""
    m = n.shape[-1]
    big_n = n.sum(-1)
    within_df = big_n - m
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma2 = np.where(within_df > 0, ssw.sum(-1) / np.maximum(within_df, 1), np.nan)
    if sigma2_fallback is not None:
        sigma2 = np.where(np.isfinite(sigma2), sigma2, sigma2_fallback)
    grand = (n * zbar).sum(-1) / big_n
    msb = (n * (zbar - np.expand_dims(grand, -1)) ** 2).sum(-1) / (m - 1)
    n0 = (big_n - (n ** 2).sum(-1) / big_n) / (m - 1)
    tau2 = np.maximum(0.0, (msb - sigma2) / n0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = 1.0 / (np.expand_dims(tau2, -1) + np.expand_dims(sigma2, -1) / n)
        mu = (w * zbar).sum(-1) / w.sum(-1)
    return mu, tau2, sigma2, within_df


def fit_poem_model(poems, scale: str = "logit", eps: float = 0.01,
                   min_poems: int = MIN_POEMS, min_within_df: int = MIN_WITHIN_DF) -> PoemModelFit:
    """Оценить μ, τ², σ² по поэмам одного автора (poems: [[уверенности сегментов поэмы], ...])."""
    n, zbar, ssw, _ = _poem_arrays(poems, scale, eps)
    m, big_n = len(n), int(n.sum())
    multi = int((n > 1).sum())
    wdf = big_n - m
    base = dict(scale=scale, n_poems=m, n_segments=big_n, multi_segment_poems=multi, within_df=wdf)
    if m < min_poems:
        return PoemModelFit(ok=False, reason=f"мало поэм в эталоне ({m} < {min_poems})", **base)
    if wdf < min_within_df:
        return PoemModelFit(ok=False, reason=f"мало поэм из нескольких сегментов (df={wdf} < {min_within_df})",
                            **base)
    mu, tau2, sigma2, _ = _anova_moments(n, zbar, ssw)
    mu, tau2, sigma2 = float(mu), float(tau2), float(sigma2)
    icc = tau2 / (tau2 + sigma2) if tau2 + sigma2 > 0 else float("nan")
    return PoemModelFit(ok=True, reason="", mu=mu, tau2=tau2, sigma2=sigma2, icc=icc, **base)


def _bootstrap_reference(poems, k: int, n_boot: int = 10000, seed: int = 0, scale: str = "logit",
                          eps: float = 0.01, propagate_uncertainty: bool = True,
                          min_poems: int = MIN_POEMS,
                          min_within_df: int = MIN_WITHIN_DF) -> tuple[np.ndarray, PoemModelFit]:
    """Запасной способ: остаточный бутстреп + внешний бутстреп по поэмам (см. шапку раздела)."""
    fit = fit_poem_model(poems, scale, eps, min_poems, min_within_df)
    if not fit.ok or k <= 0:
        return np.array([]), fit
    rng = np.random.default_rng(seed)
    n, zbar, ssw, resid = _poem_arrays(poems, scale, eps)
    m = len(n)

    # 1) внешний бутстреп по поэмам — неопределённость оценок μ, τ², σ²
    if propagate_uncertainty:
        idx = rng.integers(m, size=(n_boot, m))
    else:
        idx = np.broadcast_to(np.arange(m), (n_boot, m))
    nb, zb, sb = n[idx], zbar[idx], ssw[idx]
    mu_b, tau2_b, sig2_b, _ = _anova_moments(nb, zb, sb, sigma2_fallback=fit.sigma2)

    # 2) эффект поэмы: эмпирические EB-оценки, центрированные и растянутые до дисперсии τ²_b
    lam = np.expand_dims(tau2_b, 1) / (np.expand_dims(tau2_b, 1) + np.expand_dims(sig2_b, 1) / nb)
    u = lam * (zb - np.expand_dims(mu_b, 1))
    u = u - u.mean(axis=1, keepdims=True)
    spread = np.sqrt((u ** 2).mean(axis=1))
    stretch = np.divide(np.sqrt(tau2_b), spread, out=np.zeros_like(spread), where=spread > 1e-12)
    u = u * np.expand_dims(stretch, 1)
    u_star = u[np.arange(n_boot), rng.integers(m, size=n_boot)]

    # 3) k сегментов этой поэмы: внутрипоэмные отклонения (эмпирическая форма, масштаб σ_b)
    r = resid - resid.mean()
    r_scale = math.sqrt(float((r ** 2).mean())) if r.size else 0.0
    if r_scale > 1e-12:
        e = (r / r_scale)[rng.integers(r.size, size=(n_boot, k))] * np.sqrt(np.expand_dims(sig2_b, 1))
    else:
        e = np.zeros((n_boot, k))

    z = np.expand_dims(mu_b + u_star, 1) + e
    return _from_scale(z, scale).mean(axis=1), fit


def _grid_posterior(n: np.ndarray, zbar: np.ndarray, ssw: np.ndarray, n_tau: int = 400, n_sig: int = 120):
    """Апостериорное распределение (τ, σ) на сетке для модели z_ij = μ + u_i + e_ij.

    μ проинтегрирована (плоское априорное), τ ~ равномерное на [0, ∞), σ ~ 1/σ. Поэмы одной длины
    дают одинаковую дисперсию среднего τ² + σ²/n, поэтому считаем по группам длин (быстро и мало памяти).
    Возвращает (taus, sigs, вероятности ячеек, μ̂(τ,σ), Σ весов(τ,σ)).
    """
    lens, inv = np.unique(n, return_inverse=True)
    c = np.bincount(inv).astype(float)                       # поэм каждой длины
    s1 = np.bincount(inv, weights=zbar)                       # Σ z̄ по группе
    s2 = np.bincount(inv, weights=zbar ** 2)                  # Σ z̄² по группе
    m, big_n = len(n), n.sum()
    wdf = big_n - m
    ssw_tot = float(ssw.sum())
    s_hat = math.sqrt(ssw_tot / wdf)
    sd_zbar = float(np.std(zbar, ddof=1)) if m > 1 else 1.0
    tau_max = max(6.0 * sd_zbar, 1.0)
    sig_span = 6.0 / math.sqrt(2.0 * wdf)                    # ±6 стандартных ошибок log σ
    for _ in range(5):                                        # расширяем сетку, если масса упёрлась в край
        taus = np.linspace(0.0, tau_max, n_tau)
        sigs = s_hat * np.exp(np.linspace(-sig_span, sig_span, n_sig))
        t2 = taus[:, None, None] ** 2
        v = t2 + (sigs[None, :, None] ** 2) / lens[None, None, :]          # (n_tau, n_sig, G)
        sw = (c / v).sum(-1)
        sz = (s1 / v).sum(-1)
        szz = (s2 / v).sum(-1)
        mu = sz / sw
        q = szz - sz ** 2 / sw
        logp = -0.5 * (c * np.log(v)).sum(-1) - 0.5 * np.log(sw) - 0.5 * q
        sg = sigs[None, :]
        logp = logp - wdf * np.log(sg) - ssw_tot / (2 * sg ** 2)   # внутрипоэмная часть правдоподобия
        # априорное 1/σ и логарифмический шаг сетки по σ (ширина ячейки ∝ σ) взаимно сокращаются
        logp -= logp.max()
        prob = np.exp(logp)
        prob /= prob.sum()
        tau_edge = prob[-1, :].sum() >= 1e-6
        sig_edge = prob[:, 0].sum() + prob[:, -1].sum() >= 1e-6
        if not (tau_edge or sig_edge):
            break
        tau_max *= 2.0 if tau_edge else 1.0
        sig_span *= 1.5 if sig_edge else 1.0
    return taus, sigs, prob, mu, sw


def _standardized(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    s = math.sqrt(float((x ** 2).mean())) if x.size else 0.0
    return x / s if s > 1e-12 else np.zeros_like(x)


def _bayes_reference(poems, k: int, n_boot: int, seed: int, scale: str, eps: float,
                     fit: PoemModelFit, shapes: str = "empirical") -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, zbar, ssw, resid = _poem_arrays(poems, scale, eps)
    taus, sigs, prob, mu, sw = _grid_posterior(n, zbar, ssw)
    cell = rng.choice(prob.size, size=n_boot, p=prob.ravel())
    it, js = np.unravel_index(cell, prob.shape)
    d_tau = taus[1] - taus[0]
    tau = np.clip(taus[it] + rng.uniform(-d_tau / 2, d_tau / 2, n_boot), 0.0, None)
    d_log = math.log(sigs[1] / sigs[0])
    sig = sigs[js] * np.exp(rng.uniform(-d_log / 2, d_log / 2, n_boot))
    mu_d = mu[it, js] + rng.normal(size=n_boot) / np.sqrt(sw[it, js])

    # форма эффекта поэмы: стандартизованные EB-оценки по поэмам автора (если τ̂ > 0), иначе нормальная
    if shapes == "empirical" and fit.tau2 > 1e-10:
        lam = fit.tau2 / (fit.tau2 + fit.sigma2 / n)
        u_shape = _standardized(lam * (zbar - fit.mu))
        u_std = u_shape[rng.integers(u_shape.size, size=n_boot)]
    else:
        u_std = rng.normal(size=n_boot)
    # форма отклонений сегментов: стандартизованные внутрипоэмные остатки
    r_shape = _standardized(resid)
    if shapes == "empirical" and np.any(r_shape):
        e_std = r_shape[rng.integers(r_shape.size, size=(n_boot, k))]
    else:
        e_std = rng.normal(size=(n_boot, k))
    z = np.expand_dims(mu_d + tau * u_std, 1) + e_std * np.expand_dims(sig, 1)
    return _from_scale(z, scale).mean(axis=1)


def tau_posterior_interval(poems, scale: str = "logit", eps: float = 0.01, level: float = 0.90):
    """Апостериорный интервал для τ (разброс эффектов поэм) — насколько он неопределён."""
    fit = fit_poem_model(poems, scale, eps)
    if not fit.ok:
        return float("nan"), float("nan")
    n, zbar, ssw, _ = _poem_arrays(poems, scale, eps)
    taus, _, prob, _, _ = _grid_posterior(n, zbar, ssw)
    cdf = np.cumsum(prob.sum(axis=1))
    lo = taus[np.searchsorted(cdf, (1 - level) / 2)]
    hi = taus[min(np.searchsorted(cdf, 1 - (1 - level) / 2), taus.size - 1)]
    return float(lo), float(hi)


def single_poem_reference(poems, k: int, n_boot: int = 10000, seed: int = 0, scale: str = "logit",
                          eps: float = 0.01, method: str = "bayes", propagate_uncertainty: bool = True,
                          min_poems: int = MIN_POEMS,
                          min_within_df: int = MIN_WITHIN_DF) -> tuple[np.ndarray, PoemModelFit]:
    """n_boot значений «средняя уверенность ОДНОЙ невиденной поэмы автора из k сегментов».

    poems: [[уверенности сегментов поэмы 1], [поэмы 2], ...] — невиденные в CV поэмы автора.
    method: "bayes" (основной) или "bootstrap" (запасной, для сравнения).
    Возвращает (значения на шкале вероятностей, оценки модели). Если эталон построить нельзя
    (мало поэм), значения — пустой массив, причина — в fit.reason.
    """
    fit = fit_poem_model(poems, scale, eps, min_poems, min_within_df)
    if not fit.ok or k <= 0:
        return np.array([]), fit
    if method == "bayes":
        return _bayes_reference(poems, k, n_boot, seed, scale, eps, fit), fit
    if method == "bootstrap":
        return _bootstrap_reference(poems, k, n_boot, seed, scale, eps, propagate_uncertainty,
                                    min_poems, min_within_df)
    raise ValueError(f"method должен быть 'bayes' или 'bootstrap', а не {method!r}")


def p_lower(value: float, draws) -> float:
    """Односторонний p-value «муаллака похожа на автора НЕ МЕНЬШЕ его типичной поэмы»:
    (1 + число эталонных значений ≤ value) / (n + 1). Поправка +1 не даёт p = 0
    при конечном числе повторов бутстрепа (минимум 1/(n+1))."""
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0 or value is None or not np.isfinite(value):
        return float("nan")
    return float((1 + np.sum(d <= value)) / (d.size + 1))


def combine_seed_pvalues(pvals, r: int | None = None) -> float:
    """Один p-value из p-value нескольких seed (правило Рюгера): r-е по величине p, умноженное на K/r.
    Корректно при ЛЮБОЙ зависимости между seed. По умолчанию r = ⌈K/2⌉ — медиана:
    для 3 seed это 1.5 × медиана, для 1 seed — сам p."""
    p = np.sort(np.asarray([x for x in pvals if x is not None and np.isfinite(x)], dtype=float))
    if p.size == 0:
        return float("nan")
    big_k = p.size
    r = r or math.ceil(big_k / 2)
    return float(min(1.0, big_k / r * p[r - 1]))


def holm(pvals) -> np.ndarray:
    """Поправка Холма на множественные проверки (контроль вероятности хотя бы одной ложной тревоги).
    NaN остаются NaN и в число проверок не входят."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    q = p[ok]
    running = 0.0
    for rank, j in enumerate(np.argsort(q, kind="stable")):
        running = max(running, min(1.0, (q.size - rank) * q[j]))
        out[ok[j]] = running
    return out


def reference_fit_table(reference_segments: dict, scale: str = "logit", eps: float = 0.01) -> pd.DataFrame:
    """Оценки модели по каждому автору + проверка допущения «эффект поэмы не зависит от длины»."""
    try:
        from scipy.stats import spearmanr
    except ImportError:          # scipy приходит с scikit-learn; на всякий случай
        spearmanr = None
    rows = []
    for author in sorted(reference_segments):
        poems = reference_segments[author]
        fit = fit_poem_model(poems, scale, eps)
        lens = [len(p) for p in poems if len(p)]
        means = [float(np.mean(p)) for p in poems if len(p)]
        rho = rho_p = float("nan")
        if spearmanr is not None and len(set(lens)) > 1 and len(lens) >= 5:
            res = spearmanr(lens, means)
            rho, rho_p = float(res[0]), float(res[1])
        rows.append({
            "автор": author, "поэм": fit.n_poems, "сегментов": fit.n_segments,
            "поэм_из_2+_сегментов": fit.multi_segment_poems, "макс_длина_поэмы": max(lens) if lens else 0,
            "типичная_увер": round(float(_from_scale(np.array(fit.mu), scale)), 3) if fit.ok else np.nan,
            "τ_поэм": round(math.sqrt(fit.tau2), 3) if fit.ok else np.nan,
            "τ_ДИ90": "{:.2f}–{:.2f}".format(*tau_posterior_interval(poems, scale, eps)) if fit.ok else "",
            "σ_сегментов": round(math.sqrt(fit.sigma2), 3) if fit.ok else np.nan,
            "ICC": round(fit.icc, 3) if fit.ok else np.nan,
            "связь_длины_и_уверенности": round(rho, 3) if np.isfinite(rho) else np.nan,
            "p_связи": round(rho_p, 4) if np.isfinite(rho_p) else np.nan,
            "эталон": "ok" if fit.ok else fit.reason,
        })
    return pd.DataFrame(rows)


# ---------- Проверка эталона на настоящих поэмах (leave-one-poem-out) ----------
# Если эталон честный, то для заведомо «своих» поэм автора p-value распределены равномерно:
# p ≤ 0.05 у ~5% поэм, p ≤ 0.20 у ~20%. Каждую поэму по очереди считаем «спорной», а эталон
# строим по остальным поэмам того же автора. Это прямая проверка, а не довод «модель правильная».

LENGTH_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 4, "3–4"), (5, 10 ** 9, "5+"))


def _bucket(n: int) -> str:
    for lo, hi, name in LENGTH_BUCKETS:
        if lo <= n <= hi:
            return name
    return "?"


def loo_calibration(reference_segments: dict, n_boot: int = 2000, seed: int = 0, scale: str = "logit",
                    eps: float = 0.01, old_boot: int = 1000, authors=None, progress=None,
                    method: str = "bayes") -> pd.DataFrame:
    """p-value каждой настоящей поэмы против эталона из остальных поэм того же автора:
    p_новый — single_poem_reference, p_старый — прежний эталон (перцентиль length_matched_reference)."""
    rows = []
    authors = sorted(reference_segments) if authors is None else list(authors)
    for a_i, author in enumerate(authors):
        poems = [list(p) for p in reference_segments.get(author, []) if len(p)]
        for i, poem in enumerate(poems):
            others = poems[:i] + poems[i + 1:]
            k, value = len(poem), float(np.mean(poem))
            draws, fit = single_poem_reference(others, k, n_boot=n_boot, seed=seed + i, scale=scale, eps=eps,
                                               method=method)
            old = length_matched_reference(others, k, n_boot=old_boot, seed=seed + i) if old_boot else np.array([])
            rows.append({"автор": author, "поэма": i, "сегментов": k, "длина": _bucket(k),
                         "уверенность": value, "p_новый": p_lower(value, draws),
                         "p_старый": percentile_in(value, old) if old.size else float("nan")})
        if progress is not None:
            progress(a_i + 1, len(authors), author)
    return pd.DataFrame(rows)


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def calibration_summary(cal: pd.DataFrame, alphas=(0.05, 0.20)) -> pd.DataFrame:
    """Доля ложных срабатываний (p ≤ α на настоящих поэмах) по длине поэм, для старого и нового эталона,
    с 95% доверительным интервалом Уилсона. У честного эталона доля ≈ α."""
    rows = []
    groups = [("все", cal)] + [(name, cal[cal["длина"] == name]) for _, _, name in LENGTH_BUCKETS]
    for name, g in groups:
        for method in ("p_новый", "p_старый"):
            p = g[method].dropna() if method in g else pd.Series(dtype=float)
            if p.empty:
                continue
            row = {"длина_поэмы": name, "эталон": "новый" if method == "p_новый" else "старый", "поэм": len(p)}
            for a in alphas:
                hits = int((p <= a).sum())
                lo, hi = _wilson(hits, len(p))
                row[f"доля_p≤{a:g}"] = round(hits / len(p), 3)
                row[f"ДИ95_{a:g}"] = f"{lo:.3f}–{hi:.3f}"
            rows.append(row)
    return pd.DataFrame(rows)
