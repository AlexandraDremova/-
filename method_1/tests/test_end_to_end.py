"""End-to-end на игрушечных данных + стаб-модели.

Проверяет, что труба собрана: данные (контракт) -> fit -> predict (контракт) ->
CV -> F1 -> калибровка порога. Не проверяет качество — только целостность.
"""
from __future__ import annotations

from aa.data.make_toy import make_toy
from aa.data.preprocess import clean
from aa.data.segment import segment_poem
from aa.eval.calibrate import calibrate_threshold
from aa.eval.harness import poem_level_vote, run_cv
from aa.model import stub
from aa import schema


def test_toy_dataset_respects_contract():
    df = schema.validate(make_toy())
    assert list(df.columns) == schema.COLUMNS
    assert df["author"].nunique() == 3


def test_preprocess_and_segment():
    poem = "بيت أول\nبيت ثان\nبيت ثالث\nبيت رابع\nبيت خامس"
    segs = segment_poem(clean(poem), window=3, stride=3, min_bayts=3)
    assert len(segs) >= 1
    assert all(isinstance(s, str) and s for s in segs)


def test_full_pipeline_runs():
    df = make_toy()
    report, preds = run_cv(df, stub.fit, stub.predict, n_splits=3)
    assert 0.0 <= report.macro_f1 <= 1.0
    assert len(report.per_fold_f1) == 3
    assert set(["true_author", "pred_author", "confidence", "correct"]).issubset(preds.columns)
    assert (preds["confidence"].between(0, 1)).all()

    poem = poem_level_vote(preds)
    assert poem["poem_id"].nunique() == df["poem_id"].nunique()

    thr = calibrate_threshold(preds)
    assert 0.0 <= thr.value <= 1.0
