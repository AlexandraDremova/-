"""Тесты слоя данных (формат aldiwan + группировка APCD). Без сети и без torch."""
from __future__ import annotations

from aa.data.build_dataset import (
    apcd_rows_to_poems,
    as_poem_list,
    load_mualakat,
    verses_to_text,
)


def test_verses_to_text_one_bayt_per_line():
    verses = [["شطر أول", "شطر ثان"], ["بيت", "تاني"]]
    text = verses_to_text(verses)
    assert text.splitlines() == ["شطر أول شطر ثان", "بيت تاني"]


def test_as_poem_list_tolerant():
    assert len(as_poem_list([{"id": 1}, {"id": 2}])) == 2          # список
    assert len(as_poem_list({"poems": [{"id": 1}]})) == 1          # обёртка
    assert len(as_poem_list({"a": {"id": 1}, "b": {"id": 2}})) == 2  # dict id->obj
    assert len(as_poem_list({"id": 1, "verses": []})) == 1          # одиночный объект


def test_mualakat_author_resolution_by_id(tmp_path):
    poem = {"id": "45286", "title": "t", "verses": [["a", "b"], ["c", "d"], ["e", "f"]]}
    f = tmp_path / "mualakat.json"
    f.write_text(__import__("json").dumps([poem]), encoding="utf-8")
    df = load_mualakat(f, id_to_author={"45286": "imru-alqays"})
    assert df["author"].iloc[0] == "imru-alqays"
    assert df["poem_id"].iloc[0] == "mualaka::45286"


def test_mualakat_id_index_beats_object_field(tmp_path):
    # у объекта есть арабское поле автора, но id есть в диванах -> метка = slug дивана
    import json
    poem = {"id": "45286", "الشاعر": "امرؤ القيس", "verses": [["a", "b"], ["c", "d"], ["e", "f"]]}
    f = tmp_path / "m.json"
    f.write_text(json.dumps([poem], ensure_ascii=False), encoding="utf-8")
    df = load_mualakat(f, id_to_author={"45286": "imru-alqays"}, known_authors={"imru-alqays"})
    assert df["author"].iloc[0] == "imru-alqays"


def test_apcd_rows_grouped_into_poems():
    rows = [{"poet": "p", "era": "قبل الإسلام", "diwan": "d", "bayt": f"بيت {i}"} for i in range(90)]
    poems = apcd_rows_to_poems(rows, bayts_per_poem=40)
    assert len(poems) == 3                     # 90 бейтов / 40 -> 3 куска
    assert {p["author"] for p in poems} == {"p"}
    assert all(p["period"] == "قبل الإسلام" for p in poems)


def test_apcd_reconstructs_bayt_from_shatrs():
    rows = [{"poet": "p", "era": "x", "first_shatr": "أول", "second_shatr": "ثان"}]
    poems = apcd_rows_to_poems(rows, bayts_per_poem=40)
    assert poems[0]["text"] == "أول ثان"
