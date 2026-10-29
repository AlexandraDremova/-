"""Тесты stylo_cases (Arud и частые слова на муаллаках): нарезка как у CAMeLBERT, признаки, перестановки,
сводная таблица. arud_vs_mfw.py участника Б для этих тестов не нужен."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from aa.data.preprocess import clean
from aa.data.segment import segment_poem
from aa.model.stylo_cases import (featurize, make_vocab, markdown_table, meter_index, permuted_labels,
                                  reference_from_cv, segment_pairs, vowel_share, wide_table)

FATHA, SUKUN = "َ", "ْ"


def _poem(n_lines: int) -> str:
    """n строк с огласовкой, плюс мусорные строки без арабских букв (их чистка выбрасывает)."""
    lines = []
    for i in range(n_lines):
        lines.append(f"قَ{FATHA}الَ بَيْتٌ {'ب' * (i % 3 + 1)}{SUKUN} رقم")
        if i % 4 == 0:
            lines.append("12345 ---")
    return "\n".join(lines)


def test_segment_pairs_match_camelbert_segmentation():
    for n in [0, 1, 2, 3, 7, 8, 9, 10, 11, 16, 17, 18, 24, 25, 40]:
        text = _poem(n)
        pairs = segment_pairs(text)
        assert [c for c, _ in pairs] == segment_poem(clean(text)), n
        for c, raw in pairs:                       # огласованный текст — те же бейты, что очищенный
            assert len(c.split("\n")) == len(raw.split("\n"))
            assert clean(raw).replace(" \n", "\n") == c.replace(" \n", "\n")
            assert SUKUN in raw


def test_featurize_relative_frequencies_and_train_only_vocab():
    train = [["a", "b", "a"], ["b", "c"]]
    vocab = make_vocab(train, top_k=2)
    assert vocab == ["a", "b"] or vocab == ["b", "a"]
    x = featurize([["a", "a", "z", "b"], []], vocab)
    j = {t: i for i, t in enumerate(vocab)}
    assert math.isclose(x[0, j["a"]], 0.5) and math.isclose(x[0, j["b"]], 0.25)   # делится на все токены
    assert (x[1] == 0).all()


def test_permuted_labels_keep_poems_whole():
    train = pd.DataFrame({"poem_id": ["p1"] * 3 + ["p2"] * 2 + ["p3"] + ["p4"] * 4,
                          "author": ["a"] * 3 + ["a"] * 2 + ["b"] + ["b"] * 4})
    rng = np.random.default_rng(0)
    for _ in range(20):
        lab = permuted_labels(train, rng)
        per_poem = pd.Series(lab).groupby(train["poem_id"]).nunique()
        assert (per_poem == 1).all()
        assert sorted(pd.Series(lab).groupby(train["poem_id"]).first()) == ["a", "a", "b", "b"]


def test_reference_from_cv_groups_by_poem_and_filters():
    cv = pd.DataFrame({"poem_id": ["p1", "p1", "p2", "q1"], "true_author": ["a", "a", "a", "b"],
                       "true_score": [0.1, 0.2, np.nan, 0.5]})
    ref = reference_from_cv(cv)
    assert ref == {"a": [[0.1, 0.2]], "b": [[0.5]]}            # поэма без значений не попадает
    assert reference_from_cv(cv, poem_filter=lambda p: p != "p1") == {"b": [[0.5]]}


def test_vowel_share_ignores_tatweel():
    assert math.isclose(vowel_share(["بَب"]), 0.5)
    assert math.isclose(vowel_share(["بَـب"]), 0.5)


def test_meter_index_takes_mualaka_meter_from_diwan_copy(tmp_path):
    (tmp_path / "diwans").mkdir()
    poems = [{"id": "7", "meter": "بحر الطويل", "verses": [["ا", "ب"]]},
             {"id": "8", "meter": "", "verses": [["ا", "ب"]]}]
    (tmp_path / "diwans" / "poet.json").write_text(json.dumps(poems, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "mualakat.json").write_text(json.dumps([{"id": "7", "verses": [["ا"]]},
                                                        {"id": "9", "meter": "بحر الكامل", "verses": [["ا"]]}],
                                                       ensure_ascii=False), encoding="utf-8")
    idx = meter_index(tmp_path)
    assert idx["diwan::poet::7"] == "الطويل" and "diwan::poet::8" not in idx
    assert idx["mualaka::7"] == "الطويل" and idx["mualaka::9"] == "الكامل"


def test_wide_table_keeps_unevaluable_rows_and_formats(tmp_path):
    cam = pd.DataFrame({"муаллака_id": ["85", "375"], "приписан": ["antar-bin-shaddad", "harith-bin-halzh"],
                        "сегментов": [10, 11], "p": [0.041, np.nan], "p_Холм": [0.33, np.nan]})
    cam.to_csv(tmp_path / "cam.csv", index=False)
    res = pd.DataFrame({"муаллака_id": ["85", "375"] * 2, "приписан": ["antar-bin-shaddad", "harith-bin-halzh"] * 2,
                        "сегментов": [10, 11] * 2, "p": [0.0004, 0.6, 0.2, np.nan],
                        "p_Холм": [0.004, 1.0, 0.4, np.nan], "метод": ["частые слова"] * 2 + ["Arud"] * 2})
    wide = wide_table(res, tmp_path / "cam.csv")
    assert list(wide["муаллака_id"]) == ["85", "375"]                   # Харис не пропал
    assert [c for c in wide.columns if c.endswith(": p")] == ["CAMeLBERT: p", "Arud: p", "частые слова: p"]
    md = markdown_table(wide)
    assert "| Антара | 85 | 10 | **0.041** (0.330) | 0.200 (0.400) | **4.0e-04** (**0.004**) |" in md
    assert "| Харис | 375 | 11 | нельзя оценить | нельзя оценить | 0.600 (1.000) |" in md
    wide2 = wide_table(res, tmp_path / "missing.csv")                    # без CAMeLBERT — только свои методы
    assert "CAMeLBERT: p" not in wide2.columns and len(wide2) == 2
