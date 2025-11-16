"""Utility wrapper for LightGBM boosters saved per target."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import lightgbm as lgb


class LightGBMRegressor:
    def __init__(self, target_names: Sequence[str]) -> None:
        self.target_names = list(target_names)
        self.boosters: List[lgb.Booster | None] = [None] * len(self.target_names)

    def set_booster(self, index: int, booster: lgb.Booster) -> None:
        self.boosters[index] = booster

    def ensure_all_boosters(self) -> List[lgb.Booster]:
        boosters: List[lgb.Booster] = []
        for name, booster in zip(self.target_names, self.boosters):
            if booster is None:
                raise RuntimeError(f"Booster for target '{name}' has not been trained.")
            boosters.append(booster)
        return boosters

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for booster, name in zip(self.ensure_all_boosters(), self.target_names):
            booster.save_model(directory / f"{name}.txt")

    @classmethod
    def load(cls, directory: Path, target_names: Sequence[str]) -> "LightGBMRegressor":
        instance = cls(target_names)
        boosters: List[lgb.Booster] = []
        for name in target_names:
            model_path = directory / f"{name}.txt"
            if not model_path.exists():
                raise FileNotFoundError(f"LightGBM model missing: {model_path}")
            boosters.append(lgb.Booster(model_file=str(model_path)))
        instance.boosters = boosters
        return instance

    def predict(self, features: np.ndarray) -> np.ndarray:
        preds = []
        for booster in self.ensure_all_boosters():
            preds.append(booster.predict(features, num_iteration=booster.best_iteration or booster.current_iteration()))
        return np.column_stack(preds).astype(np.float32)
