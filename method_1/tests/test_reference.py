"""Тесты эталона, попарной точности, чистки совпадающих бейтов и выборки по парам. Без torch."""
from __future__ import annotations

import math

import pandas as pd

from aa.data.dupes import bayts_of, drop_shared_bayts, duplicate_report
from aa.eval.reference import pairwise_accuracy, percentile_in, poem_true_scores
from aa.model.run_cases import aggregate_cases, split_and_decontaminate
from aa.model.sampling import pair_frame, pair_seed

B1 = "قفا نبك من ذكرى حبيب ومنزل بسقط اللوى"
B2 = "فتوضح فالمقراة لم يعف رسمها لما نسجتها"
B3 = "وقوفا بها صحبي علي مطيهم يقولون لا تهلك"
B4 = "لخولة أطلال ببرقة ثهمد تلوح كباقي الوشم"


def test_pair_seed_deterministic_and_pair_specific():
    assert pair_seed(42, "a", "b") == pair_seed(42, "a", "b")
    assert pair_seed(42, "a", "b") != pair_seed(42, "a", "c")
    df = pd.DataFrame({"author": ["a"] * 100 + ["b"] * 10, "text": [f"t{i}" for i in range(110)]})
    s1 = set(pair_frame(df, "a", "b", 30, seed=pair_seed(42, "a", "b")).query("label == 1")["text"])
    s2 = set(pair_frame(df, "a", "c", 30, seed=pair_seed(42, "a", "c")).query("label == 1")["text"])
    assert s1 != s2          # разные пары видят разные 30 сегментов крупного автора


def test_pairwise_accuracy_counts_only_own_authors():
    preds = pd.DataFrame([
        {"fold": 0, "true_author": "a", "pair_probs": [("a", "b", 0.9), ("a", "c", 0.4), ("b", "c", 0.7)]},
        {"fold": 0, "true_author": "b", "pair_probs": [("a", "b", 0.2), ("a", "c", 0.5), ("b", "c", 0.6)]},
    ])
    pw = pairwise_accuracy(preds).set_index(["a", "b"])
    assert pw.loc[("a", "b"), "n"] == 2 and pw.loc[("a", "b"), "acc"] == 1.0
    assert pw.loc[("a", "c"), "n"] == 1 and pw.loc[("a", "c"), "acc"] == 0.0   # a получил 0.4
    assert pw.loc[("b", "c"), "acc"] == 1.0


def test_poem_true_scores_and_percentile():
    preds = pd.DataFrame({"poem_id": ["p1", "p1", "p2"], "true_author": ["a", "a", "a"],
                          "true_score": [0.9, 0.7, 0.4]})
    ref = poem_true_scores(preds).set_index("poem_id")
    assert abs(ref.loc["p1", "true_score"] - 0.8) < 1e-9 and ref.loc["p1", "n_segments"] == 2
    assert percentile_in(0.5, [0.4, 0.8]) == 0.5
    assert percentile_in(0.1, [0.4, 0.8]) == 0.0
    assert math.isnan(percentile_in(0.5, []))


def test_drop_shared_bayts_removes_quoted_verses_anywhere():
    mual = pd.DataFrame({"poem_id": ["mualaka::50"], "author": ["imru"], "text": [f"{B1}\n{B2}"]})
    train = pd.DataFrame({
        "poem_id": ["diwan::imru::7", "diwan::tarafa::9", "diwan::imru::8"],
        "author": ["imru", "tarafa", "imru"],
        "text": [f"{B3}\n{B1}", f"{B4}\n{B2}", f"{B3}\n{B4}"],   # B1 и B2 «процитированы»
    })
    cleaned, rep = drop_shared_bayts(train, mual)
    assert list(cleaned["poem_id"]) == ["diwan::imru::8"]
    assert set(rep["автор_в_обучении"]) == {"imru", "tarafa"}       # в т.ч. чужой диван


def test_duplicate_report_finds_cross_author_bayts():
    df = pd.DataFrame({"poem_id": ["p1", "p2", "p3"], "author": ["a", "b", "a"],
                       "text": [f"{B1}\n{B3}", f"{B1}", f"{B4}"]})
    rep = duplicate_report(df)
    assert len(rep) == 1 and rep.iloc[0]["авторов"] == 2
    assert bayts_of("коротко\n" + B1) == [B1]                        # короткие строки игнорируются


def test_decontaminate_ignores_apcd_chunk_ids():
    df = pd.DataFrame({
        "poem_id": ["mualaka::50", "diwan::imru::50", "apcd::x::d::50", "diwan::imru::51"],
        "author": ["imru", "imru", "x", "imru"], "text": ["m", "dup", "apcd", "ok"],
        "period": None, "century": None, "topic": None})
    train, mual = split_and_decontaminate(df)
    assert set(train["poem_id"]) == {"apcd::x::d::50", "diwan::imru::51"}


def test_aggregate_cases_with_reference_percentile():
    mual = pd.DataFrame({"poem_id": ["mualaka::1"] * 2, "author": ["a"] * 2, "text": ["x", "y"]})
    res = [{"pred_author": "a", "confidence": 0.6, "votes": {}, "scores": {"a": 0.6}}] * 2
    ref = pd.DataFrame({"poem_id": [f"p{i}" for i in range(20)], "author": ["a"] * 20,
                        "true_score": [0.7 + i * 0.01 for i in range(20)]})   # все выше 0.6
    row = aggregate_cases(mual, res, reference=ref).iloc[0]
    assert row["перцентиль_у_автора"] == 0.0 and row["поэм_автора_в_эталоне"] == 20
    assert "ниже 95% поэм автора" in row["флаг"]
    small = aggregate_cases(mual, res, reference=ref.head(3)).iloc[0]
    assert "эталон автора мал" in small["флаг"]


def test_calibrate_threshold_never_infinite():
    from aa.eval.calibrate import calibrate_threshold
    flat = pd.DataFrame({"correct": [True, False, True, False], "confidence": [0.8, 0.8, 0.8, 0.8]})
    assert math.isfinite(calibrate_threshold(flat).value)
    sep = pd.DataFrame({"correct": [True, True, False, False], "confidence": [0.95, 0.9, 0.6, 0.55]})
    thr = calibrate_threshold(sep)
    assert 0.6 < thr.value <= 0.9 and thr.auc == 1.0


# ---------- финальная версия A8: варианты бейтов, группы CV, подвыборка, эталон по длине ----------
V1 = "قفا نبك من ذكرى حبيب وعرفان بسقط اللوى"      # B1 с заменой одного слова


def test_fuzzy_detects_variant_but_not_unrelated():
    from aa.data.dupes import similar_bayt_pairs
    hits = {j: jac for _, j, jac in similar_bayt_pairs([B1], [V1, B3, B4], threshold=0.0)}
    assert hits[0] >= 0.5 and hits[1] < 0.2 and hits[2] < 0.2


def test_drop_shared_bayts_fuzzy_removes_variants():
    mual = pd.DataFrame({"poem_id": ["mualaka::50"], "author": ["imru"], "text": [B1]})
    train = pd.DataFrame({"poem_id": ["diwan::imru::7", "diwan::imru::8"], "author": ["imru"] * 2,
                          "text": [f"{B3}\n{V1}", B4]})
    exact, _ = drop_shared_bayts(train, mual)                       # точный режим вариант не видит
    assert len(exact) == 2
    fuzzy, rep = drop_shared_bayts(train, mual, fuzzy_threshold=0.5)
    assert list(fuzzy["poem_id"]) == ["diwan::imru::8"]
    assert rep.iloc[0]["похожих_бейтов"] == 1 and rep.iloc[0]["удалено_сегментов"] == 1


def test_poem_groups_merge_poems_with_shared_variants():
    from aa.data.dupes import poem_groups
    df = pd.DataFrame({"poem_id": ["p1", "p2", "p3"], "author": ["a", "a", "b"],
                       "text": [B1, f"{B3}\n{V1}", B4]})
    g = poem_groups(df, threshold=0.5)
    assert g.iloc[0] == g.iloc[1] and g.iloc[2] != g.iloc[0]


def test_subsample_poems_takes_whole_poems_per_author():
    from aa.data.prepare import subsample_poems
    df = pd.DataFrame({"author": ["a"] * 30 + ["b"] * 9,
                       "poem_id": [f"a{i // 3}" for i in range(30)] + [f"b{i // 3}" for i in range(9)],
                       "text": ["x"] * 39})
    sub = subsample_poems(df, 2 / 3, seed=1)
    assert sub.groupby("poem_id").size().eq(3).all()                 # поэмы целиком
    assert 20 <= (sub.author == "a").sum() <= 22 and (sub.author == "b").sum() == 6
    assert len(subsample_poems(df, 1.0, seed=1)) == 39


def test_length_matched_reference_is_narrower_for_long_poems():
    import numpy as np
    from aa.eval.reference import length_matched_reference
    rng = np.random.default_rng(0)
    poems = [list(rng.uniform(0, 1, size=rng.integers(1, 4))) for _ in range(60)]
    short = length_matched_reference(poems, k=1, seed=3)
    long_ = length_matched_reference(poems, k=10, seed=3)
    assert long_.std() < short.std() * 0.6                           # среднее по 10 сегментам уже
    assert np.allclose(length_matched_reference([[0.7, 0.7]], k=5, seed=0), 0.7)


def test_conclusion_uses_holm_then_raw_p():
    from aa.model.run_cases import VERDICTS, conclusion
    assert conclusion(0.001, 0.008, 100) == VERDICTS[0]          # значимо после поправки
    assert conclusion(0.03, 0.12, 100) == VERDICTS[1]            # только без поправки
    assert conclusion(0.15, 0.6, 100) == VERDICTS[2]
    assert conclusion(0.5, 1.0, 100) == VERDICTS[3]
    assert conclusion(float("nan"), None, 100) == VERDICTS[4]
    assert conclusion(0.001, 0.001, 14) == VERDICTS[5]           # мало данных важнее любого p


def test_combine_seeds_rueger_holm_and_stability():
    from aa.model.run_cases import VERDICTS, combine_seeds
    base = {"сегментов": 8, "доля_сегм_за_приписанного": 0.5, "предсказан": "x", "граница_5%": 0.4,
            "ожидаемая_увер": 0.8, "ICC_автора": 0.3, "перцентиль_старый": 0.01, "эталон": "ok"}
    rows = []
    # 1: устойчиво значимо; 2: значимо без поправки, меняется; 3: норма; 4: мало данных
    for mid, a, n_tr, ps in [("1", "a", 100, [0.0005, 0.0010, 0.0020]), ("2", "b", 100, [0.01, 0.03, 0.30]),
                             ("3", "c", 100, [0.6, 0.7, 0.8]), ("4", "d", 14, [0.0001, 0.0001, 0.0001])]:
        for seed, p in zip((42, 43, 44), ps):
            rows.append({**base, "муаллака_id": mid, "приписан": a, "сегм_автора_в_обучении": n_tr,
                         "seed": seed, "увер_в_приписанном": 0.5, "p": p})
    fin = combine_seeds(pd.DataFrame(rows)).set_index("муаллака_id")
    assert fin.loc["1", "p"] == 0.0015                                # 1.5 × медиана 0.001
    assert fin.loc["2", "p"] == 0.045                                 # 1.5 × 0.03
    assert int(fin["проверок_в_поправке"].iloc[0]) == 3               # «мало данных» не входит в поправку
    assert fin.loc["1", "p_Холм"] == 0.0045                           # 3 × 0.0015
    assert fin.loc["2", "p_Холм"] == 0.09                             # 2 × 0.045
    assert math.isnan(fin.loc["4", "p_Холм"])
    assert fin.loc["1", "вывод"] == VERDICTS[0] and fin.loc["1", "устойчивость"] == "устойчиво"
    assert fin.loc["2", "вывод"] == VERDICTS[1] and fin.loc["2", "устойчивость"].startswith("меняется")
    assert fin.loc["3", "вывод"] == VERDICTS[3]
    assert fin.loc["4", "вывод"] == VERDICTS[5]
    assert list(fin.index[:2]) == ["1", "2"]                          # самые сомнительные — первыми


def test_fit_collect_retries_pair_in_fp32_on_nan_prediction():
    from aa.model.ensemble import fit_collect
    from aa.model.finetune import NonFinitePrediction
    calls = []

    class P:
        def __init__(self, t, o, d, amp=True, **k):
            self.t, self.amp, self.train_info = t, amp, {}
            calls.append(amp)

        def predict_proba_batch(self, texts):
            if self.amp:
                raise NonFinitePrediction("NaN")
            return [0.9] * len(texts)

        def free(self):
            pass

    rep = []
    pp = fit_collect(pd.DataFrame({"author": ["a", "b"], "text": ["x", "y"]}), ["t"],
                     _trainer=P, report=rep, amp=True)
    assert calls == [True, False] and rep[0]["fallback_fp32"] and pp == [[("a", "b", 0.9)]]


def test_rescore_uses_current_reference():
    from aa.model.run_cases import rescore
    saved = pd.DataFrame([{"муаллака_id": 7, "приписан": "a", "увер_в_приписанном": 0.5, "сегментов": 2,
                           "seed": 42, "перцентиль_по_длине": 0.99}])
    ref_segments = {"a": [[0.9, 0.9], [0.95, 0.95]]}             # все эталонные значения выше 0.5
    out = rescore(saved, None, ref_segments)
    assert out.loc[0, "перцентиль_старый"] == 0.0 and out.loc[0, "муаллака_id"] == "7"
    assert "перцентиль_по_длине" not in out.columns              # старая колонка заменена
    assert math.isnan(out.loc[0, "p"]) and "мало поэм" in out.loc[0, "эталон"]   # 2 поэмы — эталона нет


# ---------- эталон «одна поэма из k сегментов» (28.10) ----------
def _synthetic_author(rng, m=120, mu=1.8, tau=1.0, sigma=1.0, mean_len=1.8):
    import numpy as np
    lens = 1 + rng.poisson(mean_len - 1, m)
    return [list(1 / (1 + np.exp(-(mu + rng.normal(0, tau) + rng.normal(0, sigma, n))))) for n in lens]


def test_anova_moments_balanced_case():
    import numpy as np
    from aa.eval.reference import _anova_moments
    z = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [5.0, 5.0, 5.0], [0.0, 1.0, 2.0]])
    n = np.full(4, 3.0)
    zbar, ssw = z.mean(1), ((z - z.mean(1, keepdims=True)) ** 2).sum(1)
    mu, tau2, sigma2, wdf = _anova_moments(n, zbar, ssw)
    msw = ssw.sum() / (12 - 4)
    msb = 3 * ((zbar - zbar.mean()) ** 2).sum() / 3
    assert wdf == 8 and abs(sigma2 - msw) < 1e-12
    assert abs(tau2 - (msb - msw) / 3) < 1e-12 and abs(mu - zbar.mean()) < 1e-12


def test_fit_refuses_too_few_poems():
    from aa.eval.reference import fit_poem_model, single_poem_reference
    assert not fit_poem_model([[0.9, 0.8], [0.7]]).ok
    draws, fit = single_poem_reference([[0.9, 0.8], [0.7], [0.6, 0.5]], k=5)
    assert draws.size == 0 and "мало поэм" in fit.reason
    only_single = [[0.9], [0.8], [0.7], [0.6], [0.95], [0.85]]      # нет поэм из ≥2 сегментов
    assert "нескольких сегментов" in fit_poem_model(only_single).reason


def test_single_poem_reference_wider_than_mixture_for_long_poems():
    import numpy as np
    from aa.eval.reference import length_matched_reference, single_poem_reference
    poems = _synthetic_author(np.random.default_rng(1), tau=1.2, sigma=0.8)
    new, fit = single_poem_reference(poems, k=10, n_boot=8000, seed=0)
    old = length_matched_reference(poems, k=10, n_boot=4000, seed=0)
    assert fit.ok and fit.icc > 0.4
    assert np.quantile(new, 0.05) < np.quantile(old, 0.05) - 0.03    # новый эталон шире внизу
    assert abs(np.median(new) - np.median(old)) < 0.05                # центр тот же
    same, _ = single_poem_reference(poems, k=10, n_boot=8000, seed=0)
    assert np.array_equal(new, same)                                  # воспроизводимо при том же seed


def test_reference_is_calibrated_on_synthetic_long_poems():
    """Настоящая длинная поэма из того же процесса: новый эталон бракует ~5%, старый — заметно больше."""
    import numpy as np
    from aa.eval.reference import (length_matched_reference, p_lower, percentile_in,
                                   single_poem_reference)
    rng = np.random.default_rng(7)
    new_hits = old_hits = 0
    reps = 150
    for r in range(reps):
        poems = _synthetic_author(rng, m=60, tau=1.0, sigma=1.0)
        u = rng.normal(0, 1.0)                                        # эффект этой поэмы
        test = list(1 / (1 + np.exp(-(1.8 + u + rng.normal(0, 1.0, 10)))))   # 10 её сегментов
        value = float(np.mean(test))
        new_hits += p_lower(value, single_poem_reference(poems, 10, n_boot=2000, seed=r)[0]) <= 0.05
        old_hits += percentile_in(value, length_matched_reference(poems, 10, n_boot=1000, seed=r)) <= 0.05
    assert new_hits / reps <= 0.10                                    # честно: ~0.05 (запас на шум 150 повторов)
    assert old_hits / reps >= new_hits / reps + 0.04                  # старый заметно чаще бьёт тревогу


def test_bootstrap_method_runs_and_agrees_roughly():
    import numpy as np
    from aa.eval.reference import single_poem_reference
    poems = _synthetic_author(np.random.default_rng(3))
    b, _ = single_poem_reference(poems, k=8, n_boot=4000, seed=0, method="bayes")
    s, _ = single_poem_reference(poems, k=8, n_boot=4000, seed=0, method="bootstrap")
    assert abs(np.quantile(b, 0.05) - np.quantile(s, 0.05)) < 0.06
    assert abs(np.median(b) - np.median(s)) < 0.03


def test_p_lower_holm_and_rueger():
    import numpy as np
    from aa.eval.reference import combine_seed_pvalues, holm, p_lower
    assert p_lower(0.0, [1, 2, 3]) == 0.25 and p_lower(5.0, [1, 2, 3]) == 1.0      # +1: p никогда не 0
    assert math.isnan(p_lower(0.5, []))
    adj = holm([0.01, 0.04, 0.03, 0.005, float("nan")])
    assert np.allclose(adj[:4], [0.03, 0.06, 0.06, 0.02]) and math.isnan(adj[4])
    assert holm([0.5, 0.9]).tolist() == [1.0, 1.0]
    assert combine_seed_pvalues([0.01, 0.02, 0.5]) == 0.03                         # 1.5 × медиана
    assert combine_seed_pvalues([0.04]) == 0.04
    assert combine_seed_pvalues([0.3, 0.9]) == 0.6                                 # 2 × меньшее
    assert combine_seed_pvalues([0.8, 0.9, 0.9]) == 1.0                            # не больше 1
    assert math.isnan(combine_seed_pvalues([float("nan")]))


def test_loo_calibration_and_summary():
    import numpy as np
    from aa.eval.reference import calibration_summary, loo_calibration
    rng = np.random.default_rng(11)
    segs = {"a": _synthetic_author(rng, m=40), "b": _synthetic_author(rng, m=4)}   # у b эталона нет
    cal = loo_calibration(segs, n_boot=500, old_boot=200)
    assert len(cal) == 44 and cal.loc[cal["автор"] == "b", "p_новый"].isna().all()
    assert cal.loc[cal["автор"] == "a", "p_новый"].between(0, 1).all()
    summ = calibration_summary(cal)
    row = summ[(summ["длина_поэмы"] == "все") & (summ["эталон"] == "новый")].iloc[0]
    assert row["поэм"] == 40 and 0 <= row["доля_p≤0.05"] <= 0.2


def test_reference_fit_table_columns():
    import numpy as np
    from aa.eval.reference import reference_fit_table
    rng = np.random.default_rng(2)
    t = reference_fit_table({"a": _synthetic_author(rng, m=50), "b": [[0.9]]})
    a = t.set_index("автор").loc["a"]
    assert a["эталон"] == "ok" and 0 <= a["ICC"] <= 1 and "–" in a["τ_ДИ90"]
    assert t.set_index("автор").loc["b", "эталон"].startswith("мало поэм")
