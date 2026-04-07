"""Supervised baseline: TF-IDF + One-vs-Rest LogisticRegression."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from ddqn_router.agents import AgentRegistry
from ddqn_router.dataset.dataset import Task


class SupervisedRouter:
    def __init__(
        self, registry: AgentRegistry, max_features: int = 5000
    ) -> None:
        self.registry = registry
        self.max_features = max_features
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._classifier = OneVsRestClassifier(
            LogisticRegression(max_iter=1000, solver="lbfgs")
        )

    def _build_labels(self, tasks: list[Task]) -> np.ndarray:
        n = len(tasks)
        k = self.registry.num_agents
        labels = np.zeros((n, k), dtype=int)
        for i, t in enumerate(tasks):
            for a in t["required_agents"]:
                if a < k:
                    labels[i, a] = 1
        return labels

    def fit(self, train_tasks: list[Task]) -> None:
        texts = [t["text"] for t in train_tasks]
        X = self._vectorizer.fit_transform(texts)
        y = self._build_labels(train_tasks)
        self._classifier.fit(X, y)

    def predict(self, tasks: list[Task]) -> list[set[int]]:
        texts = [t["text"] for t in tasks]
        X = self._vectorizer.transform(texts)
        y_pred = self._classifier.predict(X)
        results: list[set[int]] = []
        for row in y_pred:
            selected = set(np.where(row == 1)[0].tolist())
            if not selected:
                probs = self._classifier.predict_proba(
                    self._vectorizer.transform([texts[len(results)]])
                )
                selected = {int(np.argmax(probs))}
            results.append(selected)
        return results

    def save(self, path: str | Path) -> None:
        joblib.dump(
            {"vectorizer": self._vectorizer, "classifier": self._classifier},
            path,
        )

    @classmethod
    def load(cls, path: str | Path, registry: AgentRegistry) -> SupervisedRouter:
        data = joblib.load(path)
        router = cls.__new__(cls)
        router.registry = registry
        router._vectorizer = data["vectorizer"]
        router._classifier = data["classifier"]
        router.max_features = router._vectorizer.max_features or 5000
        return router
