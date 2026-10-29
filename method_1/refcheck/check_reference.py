"""Проверка эталона A8 на синтетике, где правда известна.

В каждом сценарии генерируется «диван» (короткие поэмы) и одна длинная НАСТОЯЩАЯ поэма из того же процесса.
Её p-value считается тремя эталонами: новым (байесовский, основной), бутстрепом (запасной) и старым
(смесь коротких поэм). Доля p ≤ 0.05 по R повторам — частота ложной тревоги: у честного эталона ≈ 0.05.
Мощность — доля p ≤ 0.05, когда поэма на самом деле сдвинута вниз (−1.5 на логит-шкале).

    python refcheck/check_reference.py 1000 10      # R=1000 повторов, муаллака из 10 сегментов
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aa.eval.reference import (length_matched_reference, p_lower, percentile_in,  # noqa: E402
                               single_poem_reference)

expit = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731


def lengths(rng, m, kind):
    if kind == "short":            # как у Антары/Лабида: в среднем ~1.8 сегмента
        return 1 + rng.poisson(0.8, m)
    if kind == "mixed":            # как у аль-Аши: есть и длинные касыды
        L = 1 + rng.poisson(1.2, m)
        long_ = rng.random(m) < 0.15
        L[long_] = rng.integers(5, 11, long_.sum())
        return L
    raise ValueError(kind)


def poem_normal(rng, n, mu, tau, sigma, shift=0.0):
    return expit(mu + shift + rng.normal(0, tau) + rng.normal(0, sigma, n))


def poem_ar1(rng, n, mu, tau, sigma, phi=0.6, shift=0.0):
    e = np.empty(n)
    e[0] = rng.normal(0, sigma)
    for j in range(1, n):
        e[j] = phi * e[j - 1] + rng.normal(0, sigma * np.sqrt(1 - phi ** 2))
    return expit(mu + shift + rng.normal(0, tau) + e)


def poem_skewed(rng, n, mu, tau, sigma, shift=0.0):
    # эффект поэмы скошен влево: редкие поэмы сильно хуже (гамма, приведённая к дисперсии tau²)
    g = rng.gamma(2.0, 1.0)
    u = -(g - 2.0) / np.sqrt(2.0) * tau
    return expit(mu + shift + u + rng.normal(0, sigma, n))


def poem_beta(rng, n, mu, tau, sigma, shift=0.0):
    # не логит-модель вовсе: средняя поэмы ~ Beta, сегменты ~ Beta вокруг неё
    m = 0.85 if shift == 0 else 0.55
    a = m * 25
    p = rng.beta(a, 25 - a)
    c = 6.0
    return rng.beta(np.clip(p * c, 1e-3, None), np.clip((1 - p) * c, 1e-3, None), n)


SCENARIOS = {
    "модель верна, 140 коротких поэм": dict(gen=poem_normal, m=140, L="short", mu=2.0, tau=0.8, sigma=1.2),
    "модель верна, 25 поэм (как у Тарафы)": dict(gen=poem_normal, m=25, L="short", mu=1.0, tau=0.8, sigma=1.2),
    "есть длинные поэмы (как у аль-Аши)": dict(gen=poem_normal, m=80, L="mixed", mu=1.2, tau=0.9, sigma=1.0),
    "сильная похожесть внутри поэмы (τ>σ)": dict(gen=poem_normal, m=140, L="short", mu=2.0, tau=1.4, sigma=0.8),
    "далёкие сегменты похожи меньше (AR1)": dict(gen=poem_ar1, m=140, L="short", mu=2.0, tau=0.6, sigma=1.2),
    "скошенный эффект поэмы": dict(gen=poem_skewed, m=140, L="short", mu=2.0, tau=0.9, sigma=1.1),
    "данные не из логит-модели (Beta)": dict(gen=poem_beta, m=140, L="short", mu=0, tau=0, sigma=0),
}


def run(cfg, R, k, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    out = {"новый": [], "бутстреп": [], "старый": []}
    for r in range(R):
        L = lengths(rng, cfg["m"], cfg["L"])
        diwan = [cfg["gen"](rng, int(n), cfg["mu"], cfg["tau"], cfg["sigma"]) for n in L]
        T = float(cfg["gen"](rng, k, cfg["mu"], cfg["tau"], cfg["sigma"], shift=shift).mean())
        out["новый"].append(p_lower(T, single_poem_reference(diwan, k, 2000, r)[0]))
        out["бутстреп"].append(p_lower(T, single_poem_reference(diwan, k, 2000, r, method="bootstrap")[0]))
        out["старый"].append(percentile_in(T, length_matched_reference(diwan, k, 1000, r)))
    return {m: np.array(v) for m, v in out.items()}

if __name__ == "__main__":
    R = int(sys.argv[1]); k = int(sys.argv[2])
    print(f"k={k}, R={R}: доля p≤0.05 на настоящих поэмах (честно ≈ 0.05 ± {1.96*np.sqrt(.0475/R):.3f}); "
          f"в скобках доля p≤0.20; мощность — при сдвиге поэмы вниз на 1.5 логита", flush=True)
    print(f"{'сценарий':40s} | {'новый (байес)':>20s} | {'бутстреп':>20s} | {'старый':>20s} | мощность нов/стар")
    for name, cfg in SCENARIOS.items():
        t0 = time.time()
        h0 = run(cfg, R, k)
        h1 = run(cfg, max(100, R // 4), k, shift=-1.5, seed=10**6)
        f = lambda m: f"{(h0[m]<=.05).mean():.3f} ({(h0[m]<=.2).mean():.3f})"
        print(f"{name:40s} | {f('новый'):>20s} | {f('бутстреп'):>20s} | {f('старый'):>20s} | "
              f"{(h1['новый']<=.05).mean():.2f} / {(h1['старый']<=.05).mean():.2f}   ({time.time()-t0:.0f} с)", flush=True)
