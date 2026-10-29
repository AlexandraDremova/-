#!/usr/bin/env python3
"""
Статистическая проверка эксперимента Arud vs MFW (Omer & Oakes, AICCSA 2017).

Кодировщик Arud и матрица признаков берутся из arud_vs_mfw.py без изменений,
поэтому режим shuffled повторяет цифры исходного скрипта (тот же seed, та же нарезка).
Добавлено:
  * p-value «лучше случайного» для ARI: метки авторов переставляются B раз в каждом прогоне;
  * парное сравнение наборов признаков на одних и тех же сэмплах (arud − mfw и др.):
    перестановка знаков разностей + бутстреп-ДИ 95%, поправка Холма;
  * выборка poem: каждая поэма попадает целиком не более чем в один сэмпл
    (в shuffled, как в исходном скрипте, бейты одной поэмы расходятся по разным сэмплам автора);
  * контрольные признаки: vowel_density (доля огласованных букв, то есть признак набора текста,
    а не поэта) и meter (доли метров по полю "meter"), чтобы проверить, не объясняют ли они успех Arud;
  * описательная таблица по поэтам, логирование с ETA, CSV с каждым прогоном.

Про p-value: все прогоны режут один и тот же корпус, поэтому p-value отвечают на вопрос
«устойчив ли результат на этом корпусе при случайной нарезке», а не «верно ли это для
арабской поэзии вообще». Для такого обобщения единица — автор, а их здесь 7–10.

Запуск из папки, где лежит arud_vs_mfw.py:
    python arud_vs_mfw_stats.py                        # 20 прогонов, 1000 перестановок, оба режима выборки
    python arud_vs_mfw_stats.py --runs 50 --perms 2000 # точнее, дольше
    python arud_vs_mfw_stats.py --top-k 30             # без редких Arud-паттернов
    python arud_vs_mfw_stats.py --linkage average      # проверка устойчивости к методу кластеризации
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

import arud_vs_mfw as base

# Кэш кодировщиков: одни и те же слова кодируются тысячи раз. На результат не влияет.
base.encode_word_arud_strict = lru_cache(maxsize=None)(base.encode_word_arud_strict)
base.normalize_for_mfw = lru_cache(maxsize=None)(base.normalize_for_mfw)

SCENARIOS = [(300, 5), (150, 10), (500, 10), (1000, 5)]  # (слов в сэмпле, сэмплов на автора), как в исходном скрипте
FEATURES = ["arud", "arud_2gram", "mfw", "vowel_density", "meter"]
COMPARISONS = [("arud", "mfw"), ("arud_2gram", "mfw"), ("arud", "vowel_density"), ("arud", "meter")]
HARAKAT = set("ًٌٍَُِّْٰ")

log = logging.getLogger("arud_stats")


# ---------------------------------------------------------------- логирование

def setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"arud_stats_{datetime.now():%Y%m%d_%H%M%S}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)

    def _uncaught(exc_type, exc, tb):
        log.critical("Необработанное исключение", exc_info=(exc_type, exc, tb))
    sys.excepthook = _uncaught
    return path


def fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60} мин {sec % 60:02d} с" if sec >= 60 else f"{sec} с"


# ---------------------------------------------------------------- данные

def load_poems(raw_dir: str) -> dict[str, list[dict]]:
    """{поэт: [{'bayts': [str], 'meter': str | None}]} с сохранением границ поэм."""
    files = sorted(glob.glob(f"{raw_dir}/*.json"))
    if not files:
        raise SystemExit(f"В {raw_dir} нет JSON-файлов")
    corpus: dict[str, list[dict]] = {}
    for f in files:
        with open(f, encoding="utf-8") as fp:
            poems = json.load(fp)
        if isinstance(poems, dict):
            poems = [poems]
        for p in poems:
            poet = p.get("poet_slug") or p.get("poet") or Path(f).stem
            bayts = []
            for v in p.get("verses", []):
                line = " ".join(v) if isinstance(v, list) else str(v)
                if line.strip():
                    bayts.append(line.strip())
            if bayts:
                meter = (p.get("meter") or "").strip() or None
                corpus.setdefault(poet, []).append({"bayts": bayts, "meter": meter})
    return corpus


def vowel_density(words: list[str]) -> float:
    """Доля арабских букв, за которыми стоит хотя бы один огласовочный знак."""
    letters = voweled = 0
    for w in words:
        for i, ch in enumerate(w):
            if "ء" <= ch <= "ي" and ch != "ـ":
                letters += 1
                if i + 1 < len(w) and w[i + 1] in HARAKAT:
                    voweled += 1
    return voweled / letters if letters else 0.0


def describe_corpus(corpus: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for poet in sorted(corpus):
        poems = corpus[poet]
        words = [w for p in poems for b in p["bayts"] for w in b.split()]
        meters = pd.Series([p["meter"] for p in poems for _ in p["bayts"]]).dropna()
        top = meters.value_counts(normalize=True).head(3)
        rows.append({
            "поэт": poet,
            "поэм": len(poems),
            "бейтов": sum(len(p["bayts"]) for p in poems),
            "слов": len(words),
            "огласовано_букв": round(vowel_density(words), 3),
            "бейтов_с_метром": round(len(meters) / max(1, sum(len(p["bayts"]) for p in poems)), 2),
            "топ_метры": "; ".join(f"{m.replace('بحر ', '')} {v:.0%}" for m, v in top.items()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- выборка

def sample_shuffled(corpus, chunk, n_samp, rng):
    """Как sample_non_overlapping_chunks в исходном скрипте (тот же порядок вызовов rng)."""
    samples, labels, need = [], [], chunk * n_samp
    for poet in sorted(corpus):
        bayts = [(b, p["meter"]) for p in corpus[poet] for b in p["bayts"]]
        if sum(len(b.split()) for b, _ in bayts) < need:
            continue
        words, meters = [], []
        for i in rng.permutation(len(bayts)):
            b, m = bayts[i]
            w = b.split()
            words += w
            meters += [m] * len(w)
            if len(words) >= need:
                break
        for s in range(n_samp):
            sl = slice(s * chunk, (s + 1) * chunk)
            samples.append({"words": words[sl], "meters": meters[sl]})
            labels.append(poet)
    return samples, labels


def sample_poem_disjoint(corpus, chunk, n_samp, rng, eligible, max_tries=20):
    """Сэмпл набирается из целых поэм подряд; остаток поэмы, не влезший в сэмпл, выбрасывается.
    Одна поэма никогда не попадает в два сэмпла."""
    samples, labels, skipped = [], [], []
    for poet in eligible:
        poems = corpus[poet]
        out = []
        for _ in range(max_tries):
            out, cur_w, cur_m = [], [], []
            for i in rng.permutation(len(poems)):
                p = poems[i]
                for b in p["bayts"]:
                    w = b.split()
                    cur_w += w
                    cur_m += [p["meter"]] * len(w)
                    if len(cur_w) >= chunk:
                        break
                if len(cur_w) >= chunk:
                    out.append({"words": cur_w[:chunk], "meters": cur_m[:chunk]})
                    cur_w, cur_m = [], []
                    if len(out) == n_samp:
                        break
            if len(out) == n_samp:
                break
        if len(out) == n_samp:
            samples += out
            labels += [poet] * n_samp
        else:
            skipped.append(poet)
    return samples, labels, skipped


# ---------------------------------------------------------------- признаки и кластеризация

def feature_matrix(samples, mode, top_k):
    if mode in ("arud", "arud_2gram", "mfw"):
        mat, _ = base.build_feature_matrix([s["words"] for s in samples], mode=mode, top_k=top_k)
        return mat
    if mode == "vowel_density":
        return np.array([[vowel_density(s["words"])] for s in samples])
    if mode == "meter":
        names = sorted({m for s in samples for m in s["meters"] if m})
        if len(names) < 2:
            return None
        idx = {m: i for i, m in enumerate(names)}
        mat = np.zeros((len(samples), len(names)))
        for r, s in enumerate(samples):
            ms = [m for m in s["meters"] if m]
            for m in ms:
                mat[r, idx[m]] += 1
            if ms:
                mat[r] /= len(ms)
        return mat
    raise ValueError(mode)


def delta_clusters(mat, n_clusters, method):
    """Burrows' Delta + иерархическая кластеризация, как burrows_delta_ward_eval в исходном скрипте."""
    sd = mat.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    z = (mat - mat.mean(axis=0)) / sd
    dist = pdist(z, metric="cityblock") / z.shape[1]
    return fcluster(linkage(dist, method=method), t=n_clusters, criterion="maxclust") - 1


def _c2(x):
    x = np.asarray(x, dtype=np.float64)
    return x * (x - 1) / 2


def rand_scores(true, pred, perm_true=None):
    """RI и ARI; при perm_true (B × n) — для каждой перестановки меток сразу (векторно).
    Маргиналы при перестановке не меняются, меняется только сумма C(n_ij, 2)."""
    k = int(pred.max()) + 1
    kt = int(true.max()) + 1
    n2 = _c2(len(true))
    a = _c2(np.bincount(true)).sum()
    b = _c2(np.bincount(pred)).sum()
    expected = a * b / n2
    denom = (a + b) / 2 - expected
    if perm_true is None:
        s = _c2(np.bincount(true * k + pred, minlength=kt * k)).sum()
    else:
        nb = perm_true.shape[0]
        codes = perm_true * k + pred[None, :] + (np.arange(nb)[:, None] * kt * k)
        s = _c2(np.bincount(codes.ravel(), minlength=nb * kt * k).reshape(nb, kt * k)).sum(axis=1)
    ri = (n2 + 2 * s - a - b) / n2
    ari = (s - expected) / denom if denom != 0 else np.ones_like(s, dtype=float)
    return ri, ari


# ---------------------------------------------------------------- статистика

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def sign_flip_test(d, rng, n_perm):
    d = np.asarray(d, dtype=float)
    obs = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).mean(axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + n_perm)
    boot = d[rng.integers(len(d), size=(n_perm, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return obs, lo, hi, p


def fmt_p(p, n_perm):
    return f"<{1 / (n_perm + 1):.3f}" if p <= 1 / (n_perm + 1) + 1e-12 else f"{p:.3f}"


# ---------------------------------------------------------------- основной цикл

def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(Path(args.log_dir))
    log.info("Лог: %s", log_path)
    log.info("Параметры: %s", vars(args))

    corpus = load_poems(args.data)
    desc = describe_corpus(corpus)
    log.info("Корпус:\n%s", desc.to_string(index=False))
    desc.to_csv(out_dir / "corpus_by_poet.csv", index=False)
    vd = desc["огласовано_букв"]
    if vd.max() - vd.min() > 0.15:
        log.warning("Доля огласованных букв у поэтов сильно различается (%.2f–%.2f): кодировщик Arud "
                    "ставит 1 на неогласованные согласные, поэтому Arud-признаки частично измеряют "
                    "огласовку текста. Смотрите строку vowel_density.", vd.min(), vd.max())
    has_meter = desc["бейтов_с_метром"].mean() > 0.5
    features = [f for f in FEATURES if f != "meter" or has_meter]
    if not has_meter:
        log.warning("Поле meter заполнено менее чем у половины бейтов: контроль по метрам пропущен.")

    samplings = ["shuffled", "poem"] if args.sampling == "both" else [args.sampling]
    total_units = len(SCENARIOS) * len(samplings) * args.runs
    done_units, t_start = 0, time.time()
    per_run, nulls = [], {}

    for sc_i, (chunk, n_samp) in enumerate(SCENARIOS):
        need = chunk * n_samp
        eligible = [p for p in sorted(corpus)
                    if sum(len(b.split()) for q in corpus[p] for b in q["bayts"]) >= need]
        for sampling in samplings:
            rng_shuf = np.random.default_rng(args.seed)          # как в исходном run_benchmark
            rng_poem = np.random.default_rng([args.seed, 1, sc_i])
            rng_perm = np.random.default_rng([args.seed, 2, sc_i, {"shuffled": 0, "poem": 1}[sampling]])
            tag = f"{n_samp}×{chunk} слов [{sampling}]"
            skipped_any = set()
            for r in range(args.runs):
                t0 = time.time()
                if sampling == "shuffled":
                    samples, labels = sample_shuffled(corpus, chunk, n_samp, rng_shuf)
                else:
                    samples, labels, skipped = sample_poem_disjoint(corpus, chunk, n_samp, rng_poem, eligible)
                    skipped_any.update(skipped)
                names = sorted(set(labels))
                true = np.array([names.index(x) for x in labels])
                perm_true = rng_perm.permuted(np.tile(true, (args.perms, 1)), axis=1)
                for mode in features:
                    mat = feature_matrix(samples, mode, args.top_k)
                    if mat is None:
                        continue
                    pred = delta_clusters(mat, len(names), args.linkage)
                    ri, ari = rand_scores(true, pred)
                    ri_null, ari_null = rand_scores(true, pred, perm_true)
                    nulls.setdefault((chunk, n_samp, sampling, mode), []).append(ari_null)
                    per_run.append({
                        "слов": chunk, "сэмплов": n_samp, "выборка": sampling, "прогон": r,
                        "признаки": mode, "авторов": len(names), "n": len(labels),
                        "признаков": mat.shape[1], "RI": ri, "ARI": ari,
                        "RI_случайно": ri_null.mean(), "ARI_случайно": ari_null.mean(),
                    })
                done_units += 1
                el = time.time() - t_start
                eta = el / done_units * (total_units - done_units)
                if (r + 1) % max(1, args.runs // 5) == 0 or r == 0:
                    log.info("%s · прогон %d/%d · %.1f с/прогон · прошло %s · осталось ~%s",
                             tag, r + 1, args.runs, time.time() - t0, fmt_dur(el), fmt_dur(eta))
            if skipped_any:
                log.warning("%s: не набрали сэмплы из целых поэм (хотя бы в одном прогоне): %s",
                            tag, ", ".join(sorted(skipped_any)))

    df = pd.DataFrame(per_run)
    df.to_csv(out_dir / "per_run.csv", index=False)

    # --- 1. Лучше ли случайного: суммарная статистика = средний ARI по прогонам
    rows = []
    for (chunk, n_samp, sampling, mode), g in df.groupby(["слов", "сэмплов", "выборка", "признаки"], sort=False):
        null_mat = np.vstack(nulls[(chunk, n_samp, sampling, mode)])   # прогоны × перестановки
        obs = g["ARI"].mean()
        p = (1 + np.sum(null_mat.mean(axis=0) >= obs - 1e-12)) / (1 + args.perms)
        rows.append({"сценарий": f"{n_samp}×{chunk}", "выборка": sampling, "признаки": mode,
                     "авторов": int(g["авторов"].median()), "ARI": obs, "ARI_ст.откл": g["ARI"].std(ddof=1),
                     "ARI_случайно": g["ARI_случайно"].mean(), "RI": g["RI"].mean(),
                     "RI_случайно": g["RI_случайно"].mean(), "p": p})
    chance = pd.DataFrame(rows)
    chance["p_Холм"] = chance.groupby("признаки")["p"].transform(holm)

    # --- 2. Парные сравнения на одних и тех же сэмплах
    rng_pair = np.random.default_rng([args.seed, 3])
    rows = []
    wide = df.pivot_table(index=["слов", "сэмплов", "выборка", "прогон"], columns="признаки", values="ARI")
    for (chunk, n_samp, sampling), g in wide.groupby(level=["слов", "сэмплов", "выборка"], sort=False):
        for a, b in COMPARISONS:
            if a not in g or b not in g:
                continue
            d = (g[a] - g[b]).dropna().to_numpy()
            if len(d) < 2:
                continue
            obs, lo, hi, p = sign_flip_test(d, rng_pair, args.pair_perms)
            rows.append({"сценарий": f"{n_samp}×{chunk}", "выборка": sampling, "сравнение": f"{a} − {b}",
                         "разность_ARI": obs, "ДИ95_низ": lo, "ДИ95_верх": hi,
                         "лучше_в_доле_прогонов": float(np.mean(d > 0)), "p": p})
    pairs = pd.DataFrame(rows)
    if len(pairs):
        pairs["p_Холм"] = pairs.groupby("сравнение")["p"].transform(holm)

    chance.to_csv(out_dir / "summary_vs_chance.csv", index=False)
    pairs.to_csv(out_dir / "summary_pairs.csv", index=False)

    show = chance.copy()
    for c in ("p", "p_Холм"):
        show[c] = show[c].map(lambda x: fmt_p(x, args.perms))
    log.info("\n=== ARI против случайного (перестановка меток, %d раз в каждом из %d прогонов; "
             "Холм внутри каждого набора признаков) ===\n%s",
             args.perms, args.runs, show.round(3).to_string(index=False))
    if len(pairs):
        showp = pairs.copy()
        for c in ("p", "p_Холм"):
            showp[c] = showp[c].map(lambda x: fmt_p(x, args.pair_perms))
        log.info("\n=== Парные сравнения ARI на одних и тех же сэмплах (перестановка знаков, %d раз; "
                 "бутстреп-ДИ 95%%; Холм внутри каждого сравнения) ===\n%s",
                 args.pair_perms, showp.round(3).to_string(index=False))
    log.info("RI не используйте как основную метрику: при 7–10 авторах случайная кластеризация "
             "уже даёт RI около 0.75–0.85 (см. столбец RI_случайно).")
    log.info("Готово за %s. Таблицы: %s", fmt_dur(time.time() - t_start), out_dir.resolve())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw/diwans")
    ap.add_argument("--out", default="results/arud_stats")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--perms", type=int, default=1000, help="перестановок меток на прогон")
    ap.add_argument("--pair-perms", type=int, default=10000, help="перестановок знаков в парном тесте")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--sampling", choices=["shuffled", "poem", "both"], default="both")
    ap.add_argument("--linkage", choices=["ward", "average", "complete"], default="ward")
    ap.add_argument("--seed", type=int, default=42)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
