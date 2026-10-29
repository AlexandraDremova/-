"""Сборка готового датасета сегментов из сырых источников aldiwan.  [дорожка данных]

Формат исходных .json (и муалаки, и диваны одинаковы):
    {"id","url","title","topics","meter","num_verses",
     "verses": [["полустишие1","полустишие2"], ...]}   # каждый бейт = пара полустиший

Поток: поэмы -> verses_to_text (бейт на строку) -> clean() -> segment_poem() -> validate() -> parquet.

Источники (положить в data/raw/):
    data/raw/mualakat.json      — 10 муалак (ЦЕЛЕВЫЕ КЕЙСЫ; poem_id начинается с "mualaka::")
    data/raw/diwans/*.json      — по файлу на поэта, ИМЯ ФАЙЛА = автор (обучающий материал)
    [опц.] data/raw/mualakat_authors.json  — {id или title: автор}, если автор муалаки не привязался

Автор муалаки берётся так: поле в объекте -> индекс id->автор из диванов -> ручной маппинг -> warn.

    python -m aa.data.build_dataset --out data/processed/dataset.parquet
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from aa import config, schema
from aa.data.preprocess import clean, using_camel_tools
from aa.data.segment import segment_poem

_AUTHOR_KEYS = ("author", "poet", "poet_name", "poet_slug", "الشاعر")


# ---------- разбор исходного JSON ----------

def as_poem_list(data) -> list[dict]:
    """Терпимо приводит содержимое файла к списку объектов-поэм."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("poems", "data", "items", "results"):
            if isinstance(data.get(k), list):
                return data[k]
        vals = list(data.values())
        if vals and all(isinstance(v, dict) for v in vals):  # dict id->object
            return vals
        return [data]  # одиночный объект
    raise ValueError(f"Неизвестная структура JSON: {type(data)}")


def verses_to_text(verses) -> str:
    """verses -> текст, где каждая строка = один бейт (полустишия через пробел)."""
    lines = []
    for bayt in verses or []:
        if isinstance(bayt, (list, tuple)):
            line = " ".join(str(x).strip() for x in bayt if str(x).strip())
        else:
            line = str(bayt).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _poem_id(poem: dict, i: int) -> str:
    return str(poem.get("id") or poem.get("title") or i)


def _raw_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("period", "century"):
        if col not in df.columns:
            df[col] = pd.NA
    return df


# ---------- загрузчики ----------

def load_diwans(diwans_dir: Path) -> pd.DataFrame:
    """Плоская папка *.json; автор = имя файла (stem). Каждый файл — диван одного поэта."""
    diwans_dir = Path(diwans_dir)
    rows = []
    files = sorted(diwans_dir.glob("*.json"))
    if not files:
        print(f"[warn] load_diwans: в {diwans_dir} нет *.json — проверь, куда распакован архив.")
    for f in files:
        author = f.stem.strip()
        for i, poem in enumerate(as_poem_list(json.loads(f.read_text(encoding="utf-8")))):
            text = verses_to_text(poem.get("verses"))
            if not text:
                continue
            pid = _poem_id(poem, i)
            rows.append({
                "poem_id": f"diwan::{author}::{pid}",
                "author": author,
                "text": text,
                "period": "pre-islamic",
            })
    return _raw_frame(rows)


def index_authors_by_id(diwans_dir: Path) -> dict[str, str]:
    """{aldiwan id -> автор} из диванов — чтобы привязать автора к муалаке по id поэмы."""
    idx = {}
    for f in sorted(Path(diwans_dir).glob("*.json")):
        author = f.stem.strip()
        for poem in as_poem_list(json.loads(f.read_text(encoding="utf-8"))):
            pid = poem.get("id")
            if pid is not None:
                idx[str(pid)] = author
    return idx


def load_mualakat(
    path: Path,
    id_to_author: dict[str, str] | None = None,
    author_map: dict[str, str] | None = None,
    known_authors=None,
) -> pd.DataFrame:
    """10 муалак. Метка автора должна СОВПАДАТЬ с метками обучения (slug дивана), иначе
    атрибуцию для муалаки не оценить. Поэтому приоритет:
      индекс id->автор из диванов  ->  ручной маппинг  ->  поле объекта  ->  UNKNOWN(+warn).
    known_authors (slug'и диванов): если метка им не принадлежит — предупреждаем о несовпадении.
    """
    data = as_poem_list(json.loads(Path(path).read_text(encoding="utf-8")))
    id_to_author, author_map = id_to_author or {}, author_map or {}
    known = set(known_authors or [])
    rows, unresolved, mismatched = [], [], []
    for i, poem in enumerate(data):
        pid = _poem_id(poem, i)
        author = id_to_author.get(str(poem.get("id"))) \
            or author_map.get(pid) or author_map.get(poem.get("title", "")) \
            or next((str(poem[k]).strip() for k in _AUTHOR_KEYS if poem.get(k)), None)
        if not author:
            author = f"UNKNOWN::{pid}"
            unresolved.append(poem.get("title", pid))
        elif known and author not in known:
            mismatched.append((poem.get("title", pid), author))
        text = verses_to_text(poem.get("verses"))
        if not text:
            continue
        rows.append({
            "poem_id": f"mualaka::{pid}",
            "author": author,
            "text": text,
            "period": "pre-islamic",
        })
    if unresolved:
        print(f"[warn] не привязан автор у {len(unresolved)} муалак: {unresolved}\n"
              f"       -> заполни data/raw/mualakat_authors.json вида {{\"<id|title>\": \"<slug>\"}}")
    if mismatched:
        print(f"[warn] у {len(mismatched)} муалак метка не совпадает с диванами "
              f"(атрибуцию для них НЕ оценить):")
        for t, a in mismatched:
            print(f"       '{t}' -> '{a}'  (нет такого дивана)")
        print(f"       сопоставь в mualakat_authors.json с точным slug дивана. Доступные: {sorted(known)}")
    return _raw_frame(rows)


def apcd_rows_to_poems(rows, bayts_per_poem: int = 40, max_poets: int | None = None) -> list[dict]:
    """APCD отдаёт по одному БЕЙТУ на строку (поля poet/era/diwan/bayt/first_shatr/second_shatr).
    Собираем бейты одного (poet, diwan) и режем на псевдо-поэмы по bayts_per_poem.

    В APCD нет границ поэм — псевдо-поэма это осознанное приближение (для контрастного пула
    и калибровки этого достаточно; сегментация потом окном из config всё равно идёт по бейтам).
    Вынесено в чистую функцию, чтобы тестировать без скачивания датасета.
    """
    buckets: dict[tuple[str, str], list[str]] = {}
    poet_era: dict[str, str] = {}
    for ex in rows:
        poet = str(ex.get("poet") or "").strip()
        if not poet:
            continue
        bayt = str(ex.get("bayt") or "").strip()
        if not bayt:  # восстановить из полустиший
            bayt = f"{ex.get('first_shatr') or ''} {ex.get('second_shatr') or ''}".strip()
        if not bayt:
            continue
        buckets.setdefault((poet, str(ex.get("diwan") or "").strip()), []).append(bayt)
        poet_era.setdefault(poet, str(ex.get("era") or "").strip())

    if max_poets:
        keep = list(dict.fromkeys(p for p, _ in buckets))[:max_poets]
        buckets = {k: v for k, v in buckets.items() if k[0] in set(keep)}

    out = []
    for (poet, diwan), bayts in buckets.items():
        for k in range(0, len(bayts), bayts_per_poem):
            out.append({
                "poem_id": f"apcd::{poet}::{diwan}::{k // bayts_per_poem}",
                "author": poet,
                "text": "\n".join(bayts[k:k + bayts_per_poem]),
                "period": poet_era.get(poet) or pd.NA,
            })
    return out


def load_apcd(eras: list[str] | None = None, max_poets: int | None = None,
              bayts_per_poem: int = 40) -> pd.DataFrame:
    """arbml/APCD -> контрастный пул поэтов с метками периода. Требует `pip install datasets`.

    eras: оставить только эти периоды (напр. пост-исламские) — иначе берутся все.
    max_poets: ограничить число поэтов (датасет большой, ~1.83M бейтов).
    Подсказка: распечатать доступные значения era можно через list_apcd_eras().
    """
    try:
        from datasets import load_dataset
    except Exception:
        print("[warn] load_apcd: пакет `datasets` не установлен — пропускаю APCD.")
        return _raw_frame([])
    ds = load_dataset("arbml/APCD", split="train")
    if eras:
        keep = set(eras)
        ds = ds.filter(lambda e: str(e.get("era") or "").strip() in keep)
    return _raw_frame(apcd_rows_to_poems(ds, bayts_per_poem=bayts_per_poem, max_poets=max_poets))


def list_apcd_eras() -> list[str]:
    """Доступные значения era в APCD (чтобы выбрать пост-исламские для контраста)."""
    try:
        from datasets import load_dataset
    except Exception:
        print("[warn] нужен `pip install datasets`.")
        return []
    ds = load_dataset("arbml/APCD", split="train")
    return sorted(set(ds.unique("era")))


# ---------- сборка ----------

def poems_to_segments(poems: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for row in poems.itertuples(index=False):
        cleaned = clean(row.text)
        for seg in segment_poem(cleaned):
            out_rows.append({
                "poem_id": row.poem_id,
                "author": row.author,
                "period": getattr(row, "period", pd.NA),
                "century": getattr(row, "century", pd.NA),
                "topic": pd.NA,          # заполнится на фазе ETM
                "text": seg,
            })
    return schema.validate(pd.DataFrame(out_rows) if out_rows else schema.empty())


def build(out_path: Path, with_apcd: bool = False,
          apcd_eras: list[str] | None = None, apcd_max_poets: int | None = None) -> pd.DataFrame:
    print(f"[info] camel-tools для чистки: {'да' if using_camel_tools() else 'нет (fallback-регулярка)'}")
    diwans_dir = config.DATA_RAW / "diwans"
    frames = []

    # диваны первыми: заодно строим индекс id->автор и множество меток-slug'ов для муалак
    id_index, diwan_authors = {}, None
    if diwans_dir.exists():
        diwans_df = load_diwans(diwans_dir)
        frames.append(diwans_df)
        id_index = index_authors_by_id(diwans_dir)
        diwan_authors = set(diwans_df["author"].unique())
    else:
        print(f"[warn] нет {diwans_dir}/ — распакуй diwans.zip сюда (плоско, *.json).")

    mualakat = config.DATA_RAW / "mualakat.json"
    if mualakat.exists():
        amap_file = config.DATA_RAW / "mualakat_authors.json"
        amap = json.loads(amap_file.read_text(encoding="utf-8")) if amap_file.exists() else None
        frames.append(load_mualakat(mualakat, id_to_author=id_index, author_map=amap,
                                    known_authors=diwan_authors))
    else:
        print(f"[warn] нет {mualakat} — положи файл в data/raw/")

    if with_apcd:
        frames.append(load_apcd(eras=apcd_eras, max_poets=apcd_max_poets))

    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("Нет ни одного источника данных. Заполни data/raw/ и запусти снова.")

    poems = pd.concat(frames, ignore_index=True)
    n_mualaka = poems["poem_id"].str.startswith("mualaka::").sum()
    print(f"[info] сырых поэм: {len(poems)} (из них муалак: {n_mualaka}) | авторов: {poems['author'].nunique()}")

    segs = poems_to_segments(poems)
    if len(segs):
        per_author = segs[~segs["poem_id"].str.startswith("mualaka::")].groupby("author").size()
        print(f"[info] сегментов: {len(segs)} | сегментов/автор без муалак (медиана): "
              f"{int(per_author.median()) if len(per_author) else 0}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segs.to_parquet(out_path, index=False)
    print(f"[ok] записано: {out_path}")
    return segs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.DATA_PROCESSED / "dataset.parquet"))
    ap.add_argument("--with-apcd", action="store_true", help="подмешать контрастных поэтов из APCD")
    ap.add_argument("--apcd-eras", default=None,
                    help="периоды APCD через запятую (напр. пост-исламские); пусто = все")
    ap.add_argument("--apcd-max-poets", type=int, default=None, help="ограничить число поэтов из APCD")
    args = ap.parse_args()
    eras = [e.strip() for e in args.apcd_eras.split(",")] if args.apcd_eras else None
    build(Path(args.out), with_apcd=args.with_apcd, apcd_eras=eras, apcd_max_poets=args.apcd_max_poets)
