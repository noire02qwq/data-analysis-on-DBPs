"""Utility wrapper for CatBoost models saved per target."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
from catboost import CatBoostRegressor


class CatBoostEnsemble:
    def __init__(self, target_names: Sequence[str]) -> None:
        self.target_names = list(target_names)
        self.models: List[CatBoostRegressor | None] = [None] * len(self.target_names)

    def set_model(self, index: int, model: CatBoostRegressor) -> None:
        self.models[index] = model

    def ensure_all_models(self) -> List[CatBoostRegressor]:
        models: List[CatBoostRegressor] = []
        for name, model in zip(self.target_names, self.models):
            if model is None:
                raise RuntimeError(f"CatBoost model for '{name}' has not been trained.")
            models.append(model)
        return models

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for model, name in zip(self.ensure_all_models(), self.target_names):
            model.save_model(directory / f"{name}.cbm")

    @classmethod
    def load(cls, directory: Path, target_names: Sequence[str]) -> "CatBoostEnsemble":
        instance = cls(target_names)
        models: List[CatBoostRegressor] = []
        for name in target_names:
            path = directory / f"{name}.cbm"
            if not path.exists():
                raise FileNotFoundError(f"CatBoost model missing: {path}")
            model = CatBoostRegressor()
            model.load_model(path)
            models.append(model)
        instance.models = models
        return instance

    def predict(self, features: np.ndarray) -> np.ndarray:
        preds = []
        for model in self.ensure_all_models():
            preds.append(model.predict(features))
        return np.column_stack(preds).astype(np.float32)
