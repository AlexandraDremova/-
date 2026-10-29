"""Базовые линии для классификатора эпохи.  [sklearn, без torch]

Нужны, чтобы понимать, что именно добавляет BERT:
  * ngram  — символьные n-граммы 2–4 (TF-IDF) + логистическая регрессия (SGD). Сильная дешёвая линия:
             если BERT её не обгоняет значимо, дорогое дообучение не оправдано;
  * meter  — только стихотворный размер сегмента (one-hot) + логистическая регрессия. Показывает,
             сколько эпохи угадывается по одной форме (поздние эпохи — мувашшахи, свободный стих и т.п.).
             Аналог контроля «только метр» в проверке эксперимента Б;
  * majority — всегда самый частый класс обучения (балансная точность = 1/C).
Все возвращают вероятности (n, C) в порядке classes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _align(model, probs: np.ndarray, classes: list[str]) -> np.ndarray:
    out = np.zeros((probs.shape[0], len(classes)))
    for j, c in enumerate(model.classes_):
        out[:, classes.index(c)] = probs[:, j]
    return out


def ngram_fit_predict(train_df: pd.DataFrame, classes: list[str], predict_texts: dict[str, list[str]],
                      seed: int = 0) -> dict[str, np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                          max_features=300_000, lowercase=False)
    X = vec.fit_transform(train_df["text"])
    clf = SGDClassifier(loss="log_loss", alpha=1e-5, class_weight="balanced", max_iter=50, tol=1e-4,
                        random_state=seed)
    clf.fit(X, train_df["label"])
    return {k: _align(clf, clf.predict_proba(vec.transform(t)), classes) for k, t in predict_texts.items()}


def meter_fit_predict(train_df: pd.DataFrame, classes: list[str], predict_meters: dict[str, list[str]],
                      seed: int = 0) -> dict[str, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown="ignore")
    X = enc.fit_transform(train_df[["meter"]].astype(str))
    clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(X, train_df["label"])
    return {k: _align(clf, clf.predict_proba(enc.transform(pd.DataFrame({"meter": list(map(str, m))}))), classes)
            for k, m in predict_meters.items()}


def majority_predict(train_df: pd.DataFrame, classes: list[str], n: int) -> np.ndarray:
    top = train_df["label"].value_counts().index[0]
    out = np.zeros((n, len(classes)))
    out[:, classes.index(top)] = 1.0
    return out
