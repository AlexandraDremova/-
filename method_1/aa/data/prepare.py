"""Подготовка обучающих данных — ОДИНАКОВАЯ для кросс-валидации (эталон) и A8 (муалаки).

Шаги:
  1. отделить муалаки (poem_id "mualaka::…");
  2. убрать из диванов точные копии муалак (совпадение aldiwan-id у поэм "diwan::…");
  3. убрать сегменты, где есть бейты муалак — точные или варианты (Жаккар ≥ FUZZY_DUP_THRESHOLD).
Плюс:
  * subsample_poems — подвыборка 2/3 диванов (по поэмам, по каждому автору), чтобы модели A8
    обучались на таком же объёме, как модели эталона в CV, и шкалы уверенности были сопоставимы.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aa import config
from aa.data.dupes import drop_shared_bayts

log = logging.getLogger("aa.prepare")


def _poem_tail_id(poem_id: str) -> str:
    """Последний сегмент poem_id — это aldiwan id (и у 'mualaka::50', и у 'diwan::X::50')."""
    return str(poem_id).split("::")[-1]


def split_and_decontaminate(df: pd.DataFrame):
    """Возвращает (train_df без точных копий муалак, mualaka_df)."""
    is_mualaka = df["poem_id"].str.startswith("mualaka::")
    mualaka_df = df[is_mualaka].copy()
    mualaka_ids = set(mualaka_df["poem_id"].map(_poem_tail_id))
    diwan = df[~is_mualaka].copy()
    # Совпадение по aldiwan-id имеет смысл только для поэм из диванов aldiwan ("diwan::<автор>::<id>").
    # У APCD хвост poem_id — номер куска (apcd::…::50), он ложно совпал бы с id муаллаки.
    leak_mask = (diwan["poem_id"].str.startswith("diwan::")
                 & diwan["poem_id"].map(_poem_tail_id).isin(mualaka_ids))
    if leak_mask.any():
        log.info("исключено из обучения точных копий муалак в диванах: %d сегментов", int(leak_mask.sum()))
    return diwan[~leak_mask].reset_index(drop=True), mualaka_df.reset_index(drop=True)


def prepare_training_data(raw: pd.DataFrame, fuzzy_threshold: float | None = config.FUZZY_DUP_THRESHOLD,
                          log_details: bool = True):
    """Возвращает (train_df, mualaka_df, отчёт о совпадениях бейтов муалак с диванами)."""
    train_df, mualaka_df = split_and_decontaminate(raw)
    report = pd.DataFrame()
    if not mualaka_df.empty:
        n0 = len(train_df)
        before = train_df["author"].value_counts()
        train_df, report = drop_shared_bayts(train_df, mualaka_df, fuzzy_threshold=fuzzy_threshold)
        train_df = train_df.reset_index(drop=True)
        lost = (1 - train_df["author"].value_counts().reindex(before.index, fill_value=0) / before)
        for author, share in lost[lost > 0.2].items():
            log.warning("у %s удалено %.0f%% сегментов как совпадающих с муалаками — проверь порог "
                        "FUZZY_DUP_THRESHOLD и отчёт о совпадениях (ожидаются единицы процентов)",
                        author, 100 * share)
        log.info("исключено сегментов с бейтами муалак (точными и вариантами, порог %s): %d",
                 fuzzy_threshold, n0 - len(train_df))
        if log_details and len(report):
            log.info("бейты муалак в оставшихся диванах (после удаления точных копий):\n%s",
                     report.sort_values(["поэма", "похожих_бейтов"], ascending=[True, False])
                           .to_string(index=False))
        elif log_details:
            log.info("бейтов муалак (точных, вариантов, частичных) в оставшихся диванах не найдено")
    log.info("обучающие данные: %d сегментов, %d поэм, %d авторов",
             len(train_df), train_df["poem_id"].nunique(), train_df["author"].nunique())
    return train_df, mualaka_df, report


def subsample_poems(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """По каждому автору: случайные поэмы целиком, пока не наберётся ≥ fraction его сегментов."""
    if fraction >= 1.0:
        return df
    rng = np.random.default_rng(seed)
    keep = []
    for _, g in df.groupby("author"):
        sizes = g.groupby("poem_id").size()
        poems = list(sizes.index)
        rng.shuffle(poems)
        target, acc = fraction * len(g), 0
        for p in poems:
            keep.append(p)
            acc += int(sizes[p])
            if acc >= target:
                break
    return df[df["poem_id"].isin(set(keep))].reset_index(drop=True)
