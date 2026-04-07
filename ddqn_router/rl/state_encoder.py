"""TF-IDF state encoder wrapper with fit/transform/save/load."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class StateEncoder:
    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    @property
    def dim(self) -> int:
        if not self._fitted:
            raise RuntimeError("Encoder not fitted yet")
        return len(self._vectorizer.vocabulary_)

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def transform(self, text: str) -> np.ndarray:
        vec = self._vectorizer.transform([text]).toarray()[0]
        return vec.astype(np.float32)

    def transform_batch(self, texts: list[str]) -> np.ndarray:
        vecs = self._vectorizer.transform(texts).toarray()
        return vecs.astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self._vectorizer, path)

    @classmethod
    def load(cls, path: str | Path) -> StateEncoder:
        encoder = cls.__new__(cls)
        encoder._vectorizer = joblib.load(path)
        encoder._fitted = True
        encoder.max_features = encoder._vectorizer.max_features or 5000
        return encoder
