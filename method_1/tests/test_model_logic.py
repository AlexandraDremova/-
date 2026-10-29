"""Тесты модельной логики, НЕ требующие torch.

Покрывают: сбалансированный сэмплинг, перечисление пар, majority vote,
и ensemble.predict с подставной функцией вероятностей (без реального CAMeLBERT).
"""
from __future__ import annotations

import pandas as pd

from aa.data.make_toy import make_toy
from aa.model.ensemble import Ensemble, predict
from aa.model.sampling import author_pairs, pair_frame
from aa.model.voting import tally


def test_pair_frame_is_balanced_and_capped():
    df = make_toy(poems_per_author=10, segs_per_poem=4)  # ~40 сегментов на автора
    pf = pair_frame(df, "poet_A", "poet_B", examples_per_poet=15)
    counts = pf["label"].value_counts()
    assert counts[1] == 15 and counts[0] == 15
    assert set(pf["author"]) == {"poet_A", "poet_B"}


def test_pair_frame_handles_scarcity():
    # у автора меньше примеров, чем лимит — берём сколько есть, не падаем
    df = make_toy(poems_per_author=1, segs_per_poem=3)  # 3 сегмента на автора
    pf = pair_frame(df, "poet_A", "poet_B", examples_per_poet=60)
    assert len(pf) == 6  # 3 + 3


def test_author_pairs_round_robin_and_capped():
    authors = ["a", "b", "c", "d"]
    full = author_pairs(authors, max_opponents=None)
    assert len(full) == 6  # C(4,2)
    capped = author_pairs(authors, max_opponents=1)
    assert len(capped) <= 4 and all(x != y for x, y in capped)


def test_tally_majority_vote():
    # a побеждает b и c; confidence = средняя P(a) по моделям с участием a
    pair_probs = [("a", "b", 0.9), ("a", "c", 0.8), ("b", "c", 0.6)]
    res = tally(pair_probs)
    assert res["pred_author"] == "a"
    assert 0.0 <= res["confidence"] <= 1.0
    assert res["votes"]["a"] == 2


def test_a8_decontamination_and_cases():
    import pandas as pd
    from aa.model.run_cases import aggregate_cases, split_and_decontaminate

    df = pd.DataFrame([
        {"poem_id": "mualaka::50", "author": "imru-alqays", "period": "p",
         "century": pd.NA, "topic": pd.NA, "text": "m1"},
        {"poem_id": "diwan::imru-alqays::50", "author": "imru-alqays", "period": "p",
         "century": pd.NA, "topic": pd.NA, "text": "twin"},   # дубль муаллаки -> исключить
        {"poem_id": "diwan::labid::200", "author": "labid", "period": "p",
         "century": pd.NA, "topic": pd.NA, "text": "x"},
    ])
    train, mual = split_and_decontaminate(df)
    assert "diwan::imru-alqays::50" not in set(train["poem_id"])

    def doubt(texts):  # низкая уверенность в приписанном -> сигнал подделки
        return [{"pred_author": "labid", "confidence": 0.8, "votes": {},
                 "scores": {"imru-alqays": 0.4, "labid": 0.8}} for _ in texts]
    tbl = aggregate_cases(mual, doubt, threshold=0.7)
    assert tbl.iloc[0]["вердикт"] == "РАСХОЖДЕНИЕ"
    assert tbl.iloc[0]["флаг"] == "низкая уверенность"


class _FakePair:
    """Подменяет обученную пару: уверенно узнаёт target по токену в тексте."""
    def __init__(self, target, opponent, train_df, **cfg):
        self.target = target

    def predict_proba_batch(self, texts):
        return [0.95 if self.target in t else 0.05 for t in texts]

    def free(self):
        pass


def test_fit_collect_streaming_trains_every_pair_once():
    import pandas as pd
    from aa.model.ensemble import fit_collect, fit_predict_many

    df = pd.DataFrame({"author": ["aa", "bb", "cc"] * 2, "text": ["x"] * 6})
    calls = []

    def trainer(t, o, d, **cfg):
        calls.append((t, o))
        return _FakePair(t, o, d)

    texts = ["<aa>", "<bb>", "<cc>"]
    per_text = fit_collect(df, texts, _trainer=trainer)
    assert len(calls) == 3 and len(per_text) == 3 and all(len(pp) == 3 for pp in per_text)
    preds = fit_predict_many(df, texts, _trainer=_FakePair)
    assert [p["pred_author"] for p in preds] == ["aa", "bb", "cc"]


def test_run_cv_with_fit_predict_many():
    from functools import partial
    from aa.data.make_toy import make_toy
    from aa.eval.harness import run_cv
    from aa.model.ensemble import fit_predict_many

    class ToyPair(_FakePair):  # в toy-данных автор узнаётся по словарю
        VOCAB = {"poet_A": "alif", "poet_B": "jiim", "poet_C": "raa"}

        def predict_proba_batch(self, texts):
            return [0.95 if self.VOCAB[self.target] in t else 0.05 for t in texts]

    report, preds = run_cv(make_toy(), n_splits=3,
                           fit_predict_many=partial(fit_predict_many, _trainer=ToyPair))
    assert report.macro_f1 > 0.9 and len(preds) == 60


def test_ensemble_predict_with_injected_proba():
    # ансамбль из манифеста, вероятности подставляем — torch не нужен
    ens = Ensemble(pairs=[
        {"target": "a", "opponent": "b", "path": "x"},
        {"target": "a", "opponent": "c", "path": "y"},
        {"target": "b", "opponent": "c", "path": "z"},
    ])
    # "a" всегда выигрывает свои пары
    def fake_proba(target, opponent, text, path):
        return 0.95 if target == "a" else 0.4
    out = predict("любой текст", ens, _proba=fake_proba)
    assert out["pred_author"] == "a"
    assert 0.0 <= out["confidence"] <= 1.0


def test_class_weights_balance_small_author():
    from aa.model.finetune import class_weights
    w0, w1 = class_weights(n_neg=60, n_pos=14, balance=True)
    assert abs(w0 * 60 - w1 * 14) < 1e-9          # суммарный вес классов равен
    assert class_weights(60, 14, balance=False) == [1.0, 1.0]
    assert class_weights(60, 60, balance=True) == [1.0, 1.0]


def _expect_raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"ожидалось исключение {exc.__name__}")


def test_voting_rejects_nan_probability():
    # NaN раньше молча уходил в голос за второго автора пары
    _expect_raises(lambda: tally([("a", "b", float("nan")), ("a", "c", 0.9)]), ValueError)
    _expect_raises(lambda: tally([("a", "b", 1.5)]), ValueError)


def test_aggregate_flags_small_author():
    import pandas as pd
    from aa.model.run_cases import aggregate_cases

    mual = pd.DataFrame({"poem_id": ["mualaka::375"] * 2, "author": ["harith"] * 2, "text": ["x", "y"]})
    res = [{"pred_author": "harith", "confidence": 0.9, "votes": {}, "scores": {"harith": 0.9}}] * 2
    tbl = aggregate_cases(mual, res, train_counts={"harith": 14})
    assert tbl.iloc[0]["сегм_автора_в_обучении"] == 14
    assert "мало данных" in tbl.iloc[0]["флаг"]
    assert tbl.iloc[0]["доля_сегм_за_приписанного"] == 1.0
