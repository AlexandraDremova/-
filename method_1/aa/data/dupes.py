"""Совпадающие и похожие бейты: утечки и дубли в корпусе.  [чистая логика + CLI]

Зачем. Исключение дублей муаллаки по aldiwan-id ловит только ТОЧНУЮ копию поэмы.
Но бейты муаллаки встречаются и в других поэмах: варианты той же касыды (риваяты),
отличающиеся одним-двумя словами, цитаты, фрагменты — в т.ч. в диване другого поэта.
Если такой сегмент попал в обучение, модель видела тестовый текст, и уверенность
завышается (подозрение по прогону 27.10: Амр — 26 обучающих сегментов, recall 0.19
в CV, но сегменты его муаллаки получают 0.98–1.00).

Сходство бейтов — коэффициент Жаккара по множествам символьных триграмм.
Проверено на бейтах Имру аль-Кайса: вариант в одно слово — 0.78, буквенный вариант рави — 0.79,
общее одно полустишие — 0.25–0.34, несвязанные бейты — 0.01–0.03.
Порог «вариант того же бейта» — config.FUZZY_DUP_THRESHOLD (0.5).

Отчёт по всему датасету:
    python -m aa.data.dupes --data data/processed/dataset.parquet
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

MIN_CHARS = 15
PARTIAL_THRESHOLD = 0.2    # в отчёт как «частичное совпадение» (общее полустишие даёт 0.25–0.34), не удаляется


def bayts_of(text: str, min_chars: int = MIN_CHARS) -> list[str]:
    out = []
    for line in str(text).splitlines():
        b = " ".join(line.split())
        if len(b) >= min_chars:
            out.append(b)
    return out


def similar_bayt_pairs(q_bayts: list[str], t_bayts: list[str], threshold: float,
                       chunk: int = 400) -> list[tuple[int, int, float]]:
    """Пары (i, j, J): J = Жаккар по символьным триграммам бейтов q_bayts[i] и t_bayts[j], J ≥ threshold.
    Считается разреженным умножением матриц по кускам — секунды на ~10 тыс. бейтов."""
    if not q_bayts or not t_bayts:
        return []
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(analyzer="char", ngram_range=(3, 3), binary=True, lowercase=False,
                          dtype=np.int32)
    X = vec.fit_transform(list(q_bayts) + list(t_bayts))
    Q, T = X[:len(q_bayts)], X[len(q_bayts):]
    q_len = np.asarray(Q.sum(axis=1)).ravel()
    t_len = np.asarray(T.sum(axis=1)).ravel()
    TT = T.T.tocsc()
    out: list[tuple[int, int, float]] = []
    for s in range(0, Q.shape[0], chunk):
        inter = (Q[s:s + chunk] @ TT).toarray().astype(np.float32)
        union = q_len[s:s + chunk, None] + t_len[None, :] - inter
        jac = inter / np.maximum(union, 1)
        ii, jj = np.nonzero(jac >= threshold)
        out.extend((s + int(i), int(j), float(jac[i, j])) for i, j in zip(ii, jj))
    return out


def _bayt_table(df: pd.DataFrame, min_chars: int) -> tuple[list[str], list]:
    """Плоский список бейтов и для каждого — индекс строки df (сегмента), откуда он взят."""
    bayts, owner = [], []
    for idx, text in zip(df.index, df["text"]):
        for b in bayts_of(text, min_chars):
            bayts.append(b)
            owner.append(idx)
    return bayts, owner


def drop_shared_bayts(train_df: pd.DataFrame, query_df: pd.DataFrame, min_shared: int = 1,
                      min_chars: int = MIN_CHARS, fuzzy_threshold: float | None = None):
    """Убирает из train_df сегменты, где ≥ min_shared бейтов совпадают с бейтами query_df
    (точно, а при fuzzy_threshold — и как варианты: Жаккар триграмм ≥ порога).

    Возвращает (очищенный train_df, отчёт): по строке на (поэма из query, автор обучающего
    сегмента) — число точных, похожих и частичных совпадений и удалённых сегментов.
    """
    q_bayts, q_owner = _bayt_table(query_df, min_chars)
    t_bayts, t_owner = _bayt_table(train_df, min_chars)
    q_poem = {idx: str(p) for idx, p in zip(query_df.index, query_df["poem_id"])}
    t_author = {idx: str(a) for idx, a in zip(train_df.index, train_df["author"])}

    if fuzzy_threshold is None:   # только точные совпадения
        pos = defaultdict(list)
        for j, b in enumerate(t_bayts):
            pos[b].append(j)
        pairs = [(i, j, 1.0) for i, b in enumerate(q_bayts) for j in pos.get(b, [])]
    else:
        pairs = similar_bayt_pairs(q_bayts, t_bayts, min(PARTIAL_THRESHOLD, fuzzy_threshold))

    remove_level = 1.0 if fuzzy_threshold is None else fuzzy_threshold
    hits_per_segment: dict = defaultdict(int)
    rep = defaultdict(lambda: {"точных_бейтов": 0, "похожих_бейтов": 0, "частичных_бейтов": 0,
                               "_segs": set()})
    for i, j, jac in pairs:
        seg = t_owner[j]
        key = (q_poem[q_owner[i]], t_author[seg])
        if jac >= 0.999:
            rep[key]["точных_бейтов"] += 1
        elif jac >= remove_level:
            rep[key]["похожих_бейтов"] += 1
        else:
            rep[key]["частичных_бейтов"] += 1
            continue
        hits_per_segment[seg] += 1
        rep[key]["_segs"].add(seg)

    drop_idx = {seg for seg, n in hits_per_segment.items() if n >= min_shared}
    rows = []
    for (q, a), v in rep.items():
        rows.append({"поэма": q, "автор_в_обучении": a, "точных_бейтов": v["точных_бейтов"],
                     "похожих_бейтов": v["похожих_бейтов"], "частичных_бейтов": v["частичных_бейтов"],
                     "удалено_сегментов": len(v["_segs"] & drop_idx)})
    report = pd.DataFrame(rows, columns=["поэма", "автор_в_обучении", "точных_бейтов", "похожих_бейтов",
                                         "частичных_бейтов", "удалено_сегментов"])
    return train_df.drop(index=sorted(drop_idx)), report


def poem_groups(df: pd.DataFrame, threshold: float, min_chars: int = MIN_CHARS) -> pd.Series:
    """Группы поэм для кросс-валидации: поэмы с общими/похожими бейтами (Жаккар ≥ threshold)
    объединяются в одну группу и всегда попадают в один фолд. Возвращает Series (индекс df)."""
    bayts, owner = _bayt_table(df, min_chars)
    poem_of = dict(zip(df.index, df["poem_id"].astype(str)))
    poems = sorted(set(poem_of.values()))
    parent = {p: p for p in poems}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in similar_bayt_pairs(bayts, bayts, threshold):
        if i < j:
            a, b = find(poem_of[owner[i]]), find(poem_of[owner[j]])
            if a != b:
                parent[b] = a
    return df["poem_id"].astype(str).map(lambda p: find(p))


def duplicate_report(df: pd.DataFrame, min_chars: int = MIN_CHARS) -> pd.DataFrame:
    """Бейты, встречающиеся (точно) в нескольких поэмах: по строке на бейт с авторами и поэмами."""
    where: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in df.itertuples(index=False):
        for b in bayts_of(row.text, min_chars):
            where[b].add((str(row.author), str(row.poem_id)))
    rows = []
    for b, places in where.items():
        poems = {p for _, p in places}
        if len(poems) > 1:
            authors = sorted({a for a, _ in places})
            rows.append({"бейт": b, "поэм": len(poems), "авторов": len(authors),
                         "авторы": ", ".join(authors), "поэмы": ", ".join(sorted(poems))})
    cols = ["бейт", "поэм", "авторов", "авторы", "поэмы"]
    return pd.DataFrame(rows, columns=cols).sort_values(["авторов", "поэм"], ascending=False)


def main() -> None:
    import logging

    from aa import config
    from aa.data.prepare import prepare_training_data
    from aa.log import setup_logging

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_PROCESSED / "dataset.parquet"))
    ap.add_argument("--out", default=str(config.DATA_PROCESSED / "duplicate_bayts.csv"))
    args = ap.parse_args()
    setup_logging("dupes")
    log = logging.getLogger("aa.dupes")

    df = pd.read_parquet(args.data)
    rep = duplicate_report(df)
    rep.to_csv(args.out, index=False)
    total = sum(len(bayts_of(t)) for t in df["text"])
    log.info("бейтов всего: %d; точно повторяются в >1 поэме: %d; из них у разных авторов: %d",
             total, len(rep), int((rep["авторов"] > 1).sum()))
    log.info("полный список точных повторов: %s", args.out)
    # то же, что делают run_cv / run_cases перед обучением
    prepare_training_data(df, log_details=True)


if __name__ == "__main__":
    main()
