"""Тесты классификатора эпохи (aa/era). Без сети и без torch: вместо BERT — заглушка на n-граммах."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest

from aa.era import baselines
from aa.era.apcd import assign_pseudo_poems, build_era_dataset, era_table
from aa.era.labels import (SCHEME_CLASSES, SCHEMES, canonical_era, classes_of, is_project_poet,
                           looks_like_project_poet, normalize_ar, to_class)
from aa.era.metrics import (PoetBlocks, all_metrics, confusion, paired_compare, permutation_p,
                            poet_votes, summarize)
from aa.era.split import apply_scheme, cap_sample, class_weights, poet_folds, split_val_poets

# ---------- синтетический APCD ----------

ERAS_RAW = {"قبل الإسلام": "ابتثجح", "الإسلامي": "خدذرزس", "الأموي": "شصضطظع",
            "العباسي": "غفقكلم", "المملوكي": "نهويءة", "الحديث": "أإآؤئى"}
COMMON = "ابتمنلي"


def _word(rng, letters):
    return "".join(rng.choice(letters + COMMON) for _ in range(rng.randint(3, 6)))


def _bayt(rng, letters):
    return " ".join(_word(rng, letters) for _ in range(8))


def synthetic_apcd(poets_per_era=6, poems_per_poet=3, bayts_per_poem=20, seed=0) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for e, (era, letters) in enumerate(ERAS_RAW.items()):
        for p in range(poets_per_era):
            poet = f"شاعر {e}{p}"
            for k in range(poems_per_poet):
                meter, qaf = ["الطويل", "الكامل", "البسيط"][k % 3], "ر"
                for _ in range(bayts_per_poem):
                    b = _bayt(rng, letters)
                    rows.append({"era": era, "poet": poet, "diwan": "الديوان", "qafiyah": qaf,
                                 "meter": meter, "first_shatr": "", "second_shatr": "", "bayt": b})
    return pd.DataFrame(rows)


def ngram_stub(train_df, val_df, classes, predict_texts, save_to=None, **cfg):
    """Тот же контракт, что у aa.era.model.bert_fit_predict."""
    return baselines.ngram_fit_predict(train_df, classes, predict_texts, seed=cfg.get("seed", 0)), \
        {"stub": True, "n_train": len(train_df)}


# ---------- labels ----------

@pytest.mark.parametrize("raw,key", [
    ("قبل الإسلام", "pre_islamic"), ("العصر الجاهلي", "pre_islamic"), ("المخضرمين", "mukhadram"),
    ("الإسلامي", "islamic"), ("العصر الأموي", "umayyad"), ("بين الدولتين", "between_states"),
    ("العباسي", "abbasid"), ("المغرب والأندلس", "andalus_maghreb"), ("الفاطمي", "fatimid"),
    ("الأيوبي", "ayyubid"), ("المملوكي", "mamluk"), ("العثماني", "ottoman"), ("الحديث", "modern"),
    ("", None), ("nan", None), ("что-то", None),
])
def test_canonical_era(raw, key):
    assert canonical_era(raw) == key


def test_schemes_cover_all_eras_and_classes_ordered():
    for name, mapping in SCHEMES.items():
        assert set(mapping) == set(SCHEMES["full"])
        used = {v for v in mapping.values() if v}
        assert used == set(SCHEME_CLASSES[name])
    assert to_class("mukhadram", "binary") is None
    assert to_class("umayyad", "binary") == "islamic_era"
    assert to_class("between_states", "coarse") == "abbasid"
    assert classes_of("coarse")[0] == "pre_islamic"


def test_project_poet_detection():
    assert normalize_ar("امرُؤُ القَيس") == "امرو القيس"
    for name in ["امرؤ القيس", "عنترة بن شداد", "زهير بن أبي سلمى", "النابغة الذبياني", "الأعشى",
                 "لبيد بن ربيعة العامري", "عبيد بن الأبرص", "طرفة بن العبد", "عمرو بن كلثوم",
                 "الحارث بن حلزة"]:
        assert is_project_poet(name), name
    for name in ["أعشى همدان", "عمرو بن قميئة", "حسان بن ثابت", "المتنبي"]:
        assert not is_project_poet(name), name
    assert looks_like_project_poet("أعشى همدان")


# ---------- apcd ----------

def test_pseudo_poems_split_on_meter_or_rhyme_change():
    df = pd.DataFrame({"poet": ["a"] * 5 + ["b"], "diwan": ["d"] * 6,
                       "meter": ["m1", "m1", "m2", "m2", "m2", "m2"], "qafiyah": ["ر", "ر", "ر", "ل", "ل", "ل"]})
    assert assign_pseudo_poems(df).tolist() == [1, 1, 2, 3, 3, 4]


def test_build_removes_project_poets_duplicates_and_project_bayts():
    apcd = synthetic_apcd(poets_per_era=3, poems_per_poet=2, bayts_per_poem=16)
    rng = random.Random(1)
    proj_bayts = [_bayt(rng, "ابتثجح") for _ in range(16)]
    # поэт проекта под своим именем
    extra = [{"era": "قبل الإسلام", "poet": "امرؤ القيس", "diwan": "الديوان", "qafiyah": "ل", "meter": "الطويل",
              "first_shatr": "", "second_shatr": "", "bayt": b} for b in proj_bayts]
    # те же бейты под чужим именем: точная копия + вариант с одной заменой слова
    variant = proj_bayts[0].split()
    variant[-1] = "قفا"
    extra += [{"era": "قبل الإسلام", "poet": "شاعر 00", "diwan": "الديوان", "qafiyah": "ل", "meter": "الوافر",
               "first_shatr": "", "second_shatr": "", "bayt": b} for b in proj_bayts[1:3] + [" ".join(variant)]]
    # один и тот же бейт у двух разных поэтов APCD
    shared = _bayt(rng, "غفقكلم")
    extra += [{"era": "العباسي", "poet": p, "diwan": "الديوان", "qafiyah": "ن", "meter": "الكامل",
               "first_shatr": "", "second_shatr": "", "bayt": shared} for p in ("شاعر 30", "شاعر 31")]
    apcd = pd.concat([apcd, pd.DataFrame(extra)], ignore_index=True)
    project = pd.DataFrame({"poem_id": ["mualaka::1", "mualaka::1"], "author": ["imru-alqays"] * 2,
                            "text": ["\n".join(proj_bayts[:8]), "\n".join(proj_bayts[8:])]})

    segs, summary = build_era_dataset(apcd, project, fuzzy_eras=["pre_islamic"], threshold=0.5)
    assert "امرؤ القيس" not in set(segs["author"])
    assert summary["исключено_поэтов_проекта"] == 1
    assert summary["точных_совпадений_с_проектом"] == 2
    assert summary["вариантов_бейтов_проекта"] == 1
    assert summary["бейтов_у_нескольких_поэтов"] == 2
    all_bayts = {normalize_ar(b) for t in segs["text"] for b in t.split("\n")}
    assert not any(normalize_ar(b) in all_bayts for b in proj_bayts)
    assert normalize_ar(shared) not in all_bayts
    assert set(segs["period"]) == {"pre_islamic", "islamic", "umayyad", "abbasid", "mamluk", "modern"}
    assert segs["n_bayts"].max() <= 8 and segs["n_bayts"].min() >= 3
    assert set(era_table(segs)["period"]) == set(segs["period"])


# ---------- split ----------

def _era_df():
    segs, _ = build_era_dataset(synthetic_apcd(), None)
    return segs


def test_apply_scheme_and_poet_folds_are_disjoint():
    df, classes = apply_scheme(_era_df(), "coarse", min_poets=3)
    assert classes == ["pre_islamic", "early_islamic", "abbasid", "late_classical", "modern"]
    fold = poet_folds(df, 3, seed=0)
    assert set(fold) == set(df["author"])
    counts = pd.Series(fold).groupby(df.groupby("author")["label"].first()).value_counts()
    assert (counts.groupby(level=0).size() == 3).all()      # каждый класс есть в каждом фолде
    tr, va = split_val_poets(df, 0.2, seed=0)
    assert not set(tr["author"]) & set(va["author"])
    assert set(va["label"]) == set(df["label"])


def test_binary_scheme_drops_modern():
    df, classes = apply_scheme(_era_df(), "binary", min_poets=3)
    assert classes == ["pre_islamic", "islamic_era"]
    assert "modern" not in set(df["period"])


def test_small_classes_dropped():
    df, classes = apply_scheme(_era_df(), "coarse", min_poets=7)   # у каждого класса 6 поэтов (у early — 12)
    assert classes == ["early_islamic"]


def test_cap_sample_limits_and_spreads_over_poets():
    df, _ = apply_scheme(_era_df(), "coarse", min_poets=3)
    s = cap_sample(df, per_poet=2, per_class=None, seed=0)
    assert s.groupby("author").size().max() == 2
    s = cap_sample(df, per_poet=None, per_class=5, seed=0)
    assert s.groupby("label").size().max() == 5
    assert s.groupby("label")["author"].nunique().min() == 5     # «по кругу»: 5 сегментов от 5 поэтов
    w = class_weights(["a", "a", "a", "b"], ["a", "b", "c"])
    assert w[0] < w[1] and w[2] == 0.0


# ---------- metrics ----------

def test_metrics_match_sklearn():
    from sklearn.metrics import balanced_accuracy_score, f1_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 500)
    p = np.where(rng.random(500) < 0.6, y, rng.integers(0, 4, 500))
    m = all_metrics(confusion(y, p, 4))
    assert m["balanced_acc"] == pytest.approx(balanced_accuracy_score(y, p))
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["adjacent_acc"] >= m["acc"]


def _poet_data(signal: float, n_poets=40, per=15, C=3, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(C), n_poets // C + 1)[:n_poets]
    y = np.repeat(lab, per)
    poets = np.repeat([f"p{i}" for i in range(n_poets)], per)
    pred = np.where(rng.random(y.size) < signal, y, rng.integers(0, C, y.size))
    return y, pred, poets


def test_permutation_p_small_with_signal_and_uniformish_without():
    y, p, g = _poet_data(0.7)
    assert permutation_p(PoetBlocks(y, p, g, 3), "balanced_acc", 500) < 0.01
    ps = [permutation_p(PoetBlocks(*_poet_data(0.0, seed=s), 3), "balanced_acc", 200, seed=s) for s in range(30)]
    assert 0.25 < np.mean(ps) < 0.75


def test_summarize_and_paired_compare():
    y, p, g = _poet_data(0.7)
    s = summarize(y, p, g, 3, n_boot=200, n_perm=200)
    lo, hi = map(float, s["balanced_acc_ci95"].split("–"))
    assert lo <= s["balanced_acc"] <= hi
    _, weak, _ = _poet_data(0.2, seed=1)
    c = paired_compare(y, p, weak, g, 3, n_boot=300, n_perm=300)
    assert c["разность_A_минус_B"] > 0 and c["p_двусторонний"] < 0.01
    c0 = paired_compare(y, p, p, g, 3, n_boot=100, n_perm=100)
    assert c0["разность_A_минус_B"] == 0 and c0["p_двусторонний"] == 1.0


def test_poet_votes_average_log_probs():
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.45, 0.55], [0.2, 0.8]])
    names, votes = poet_votes(probs, ["a", "a", "a", "b"])
    assert dict(zip(names, votes)) == {"a": 0, "b": 1}


# ---------- сквозные прогоны с заглушкой ----------

def test_run_era_cv_end_to_end(tmp_path):
    from aa.era import run_era_cv
    from aa.era.apcd import write_table
    data = write_table(_era_df(), tmp_path / "era.csv")
    res = run_era_cv.main(["--data", str(data), "--scheme", "coarse", "--n-splits", "3",
                           "--min-poets-per-class", "3", "--out-dir", str(tmp_path / "out"),
                           "--n-boot", "100"], bert_fn=ngram_stub)
    m = res["metrics"]
    assert set(m["метод"]) == {"bert", "ngram", "meter", "majority"}
    bert_seg = m[(m["метод"] == "bert") & (m["уровень"] == "сегменты")].iloc[0]
    assert bert_seg["balanced_acc"] > 0.8 and bert_seg["balanced_acc_p_vs_chance"] < 0.05
    maj = m[(m["метод"] == "majority") & (m["уровень"] == "сегменты")].iloc[0]
    assert maj["balanced_acc"] == pytest.approx(0.2)
    preds = pd.read_csv(tmp_path / "out" / "era_cv_predictions.csv")
    assert preds.groupby("author")["fold"].nunique().max() == 1     # поэт целиком в одном фолде
    assert (tmp_path / "out" / "era_cv_compare.csv").exists()
    # --resume берёт готовые фолды
    res2 = run_era_cv.main(["--data", str(data), "--scheme", "coarse", "--n-splits", "3",
                            "--min-poets-per-class", "3", "--out-dir", str(tmp_path / "out"),
                            "--n-boot", "100", "--resume"], bert_fn=None)
    assert res2["metrics"]["balanced_acc"].tolist() == m["balanced_acc"].tolist()


def test_run_era_cases_end_to_end(tmp_path):
    from aa.era import run_era_cases
    from aa.era.apcd import write_table
    data = write_table(_era_df(), tmp_path / "era.csv")
    rng = random.Random(5)
    others = [v for k, v in ERAS_RAW.items() if k != "قبل الإسلام"]

    def seg(p_pre):
        """Сегмент, где доля «доисламских» бейтов ~ p_pre — чтобы вероятности не упирались в 0/1."""
        return "\n".join(_bayt(rng, "ابتثجح" if rng.random() < p_pre else rng.choice(others))
                         for _ in range(8))

    rows = []
    for author in ("antar-bin-shaddad", "imru-alqays"):
        for k in range(15):                     # диван: 15 «доисламских» поэм по 1–3 сегмента
            p_poem = rng.uniform(0.6, 0.9)      # у поэм свой уровень -> есть эффект поэмы
            for _ in range(1 + k % 3):
                rows.append({"poem_id": f"diwan::{author}::{k}", "author": author, "text": seg(p_poem)})
    # муаллака Антары «поздняя» (мало доисламских бейтов), муаллака Имру — как его диван
    for i in range(6):
        rows.append({"poem_id": "mualaka::9001", "author": "antar-bin-shaddad", "text": seg(0.1)})
        rows.append({"poem_id": "mualaka::9002", "author": "imru-alqays", "text": seg(0.75)})
    proj = pd.DataFrame(rows).assign(period="pre-islamic", century=pd.NA, topic=pd.NA)
    proj_path = write_table(proj, tmp_path / "project.csv")
    res = run_era_cases.main(["--era-data", str(data), "--data", str(proj_path), "--scheme", "coarse",
                              "--seeds", "2", "--min-poets-per-class", "3", "--n-boot", "2000",
                              "--out-dir", str(tmp_path / "out")], bert_fn=ngram_stub)
    f = res["final"].set_index(["метод", "приписан"])
    assert f.loc[("bert", "antar-bin-shaddad"), "p"] < 0.01
    assert f.loc[("bert", "imru-alqays"), "p"] > 0.2
    assert f.loc[("bert", "antar-bin-shaddad"), "вывод"].startswith("значимо")
    seg = res["segments"]
    assert len(seg) == 2 * 12                   # 12 сегментов муаллак × 2 метода
    assert seg["сегмент_№"].max() == 6
    assert set(res["diwan"]["author"]) == {"antar-bin-shaddad", "imru-alqays"}
    assert (res["diwan"]["доля_сегм_pre"] > 0.8).all()


def test_temperature_softens_overconfident_probs_and_keeps_argmax():
    from aa.era.metrics import apply_temperature, fit_temperature
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    logits = rng.normal(size=(2000, 3))
    logits[np.arange(2000), y] += 1.0                       # честный сигнал
    over = np.exp(5 * logits) / np.exp(5 * logits).sum(1, keepdims=True)   # переуверенная модель (×5)
    t = fit_temperature(over, y)
    assert 3.5 < t < 7
    cal = apply_temperature(over, t)
    assert (cal.argmax(1) == over.argmax(1)).all()
    assert cal.max(1).mean() < over.max(1).mean()


def test_cv_survives_broken_fold_and_nan_predictions(tmp_path):
    from aa.era import run_era_cv
    from aa.era.apcd import write_table
    data = write_table(_era_df(), tmp_path / "era.csv")
    calls = {"n": 0}

    def flaky(train_df, val_df, classes, texts, save_to=None, **cfg):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("сломанный фолд")
        probs, info = ngram_stub(train_df, val_df, classes, texts, **cfg)
        probs["test"][:3] = np.nan                    # пара NaN-строк
        return probs, info

    res = run_era_cv.main(["--data", str(data), "--scheme", "coarse", "--n-splits", "3",
                           "--min-poets-per-class", "3", "--out-dir", str(tmp_path / "out"),
                           "--n-boot", "50"], bert_fn=flaky)
    preds = pd.read_csv(tmp_path / "out" / "era_cv_predictions.csv")
    assert sorted(preds["fold"].unique()) == [0, 2]
    assert preds.filter(like="bert_p_").notna().all().all()
    assert len(res["metrics"])


def test_cases_survive_broken_seed(tmp_path):
    from aa.era import run_era_cases
    from aa.era.apcd import write_table
    data = write_table(_era_df(), tmp_path / "era.csv")
    rng = random.Random(3)
    rows = [{"poem_id": f"diwan::antar-bin-shaddad::{k}", "author": "antar-bin-shaddad",
             "text": "\n".join(_bayt(rng, "ابتثجح") for _ in range(8))} for k in range(8) for _ in range(2)]
    rows += [{"poem_id": "mualaka::9001", "author": "antar-bin-shaddad",
              "text": "\n".join(_bayt(rng, "ابتثجح") for _ in range(8))} for _ in range(4)]
    proj_path = write_table(pd.DataFrame(rows), tmp_path / "project.csv")
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("сломанный seed")
        return ngram_stub(*a, **kw)

    res = run_era_cases.main(["--era-data", str(data), "--data", str(proj_path), "--seeds", "2",
                              "--min-poets-per-class", "3", "--n-boot", "500", "--out-dir", str(tmp_path / "out")],
                             bert_fn=flaky)
    assert set(res["per_seed"]["seed"]) == {43}


def test_cases_resume_reuses_saved_seed(tmp_path):
    from aa.era import run_era_cases
    from aa.era.apcd import write_table
    data = write_table(_era_df(), tmp_path / "era.csv")
    rng = random.Random(4)
    rows = [{"poem_id": f"diwan::antar-bin-shaddad::{k}", "author": "antar-bin-shaddad",
             "text": "\n".join(_bayt(rng, "ابتثجح") for _ in range(8))} for k in range(8) for _ in range(2)]
    rows += [{"poem_id": "mualaka::9001", "author": "antar-bin-shaddad",
              "text": "\n".join(_bayt(rng, "ابتثجح") for _ in range(8))} for _ in range(4)]
    proj = write_table(pd.DataFrame(rows), tmp_path / "project.csv")
    args = ["--era-data", str(data), "--data", str(proj), "--seeds", "2", "--min-poets-per-class", "3",
            "--n-boot", "300", "--out-dir", str(tmp_path / "out")]
    first = run_era_cases.main(args, bert_fn=ngram_stub)
    assert (tmp_path / "out" / "era_cases_seed42.npz").exists()

    def must_not_train(*a, **k):
        raise AssertionError("с --resume обучения быть не должно")
    again = run_era_cases.main(args + ["--resume"], bert_fn=must_not_train)
    pd.testing.assert_frame_equal(first["final"], again["final"])
