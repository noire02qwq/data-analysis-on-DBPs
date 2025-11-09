"""
Utility wrapper around multi-target XGBoost boosters used in the DBPS experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import xgboost as xgb


class XGBoostRegressor:
    """Manages one booster per target column and common train/save/load utilities."""

    def __init__(self, target_names: Sequence[str], params: Dict[str, float] | None = None) -> None:
        self.target_names = list(target_names)
        self.params = params or {}
        self.boosters: List[xgb.Booster | None] = [None] * len(self.target_names)

    def train_one_round(self, dtrain_list: Sequence[xgb.DMatrix]) -> List[xgb.Booster]:
        if not self.params:
            raise ValueError("XGBoost parameters must be provided for training.")
        updated: List[xgb.Booster] = []
        for idx, dtrain in enumerate(dtrain_list):
            booster = self.boosters[idx]
            booster = xgb.train(
                self.params,
                dtrain=dtrain,
                num_boost_round=1,
                xgb_model=booster,
                verbose_eval=False,
            )
            self.boosters[idx] = booster
            updated.append(booster)
        return updated

    @staticmethod
    def mean_squared_error(
        boosters: Sequence[xgb.Booster],
        dmatrix_list: Sequence[xgb.DMatrix],
        target_matrix: np.ndarray,
    ) -> float:
        preds = []
        for booster, dm in zip(boosters, dmatrix_list):
            preds.append(booster.predict(dm))
        stacked = np.column_stack(preds)
        return float(np.mean((stacked - target_matrix) ** 2))

    def ensure_all_boosters(self) -> List[xgb.Booster]:
        boosters: List[xgb.Booster] = []
        for name, booster in zip(self.target_names, self.boosters):
            if booster is None:
                raise RuntimeError(f"Booster for target '{name}' has not been trained.")
            boosters.append(booster)
        return boosters

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for booster, name in zip(self.ensure_all_boosters(), self.target_names):
            booster.save_model(directory / f"{name}.json")

    @classmethod
    def load(cls, directory: Path, target_names: Sequence[str]) -> "XGBoostRegressor":
        instance = cls(target_names, params=None)
        boosters = []
        for name in target_names:
            model_path = directory / f"{name}.json"
            if not model_path.exists():
                raise FileNotFoundError(f"Booster file missing: {model_path}")
            booster = xgb.Booster()
            booster.load_model(model_path)
            boosters.append(booster)
        instance.boosters = boosters
        return instance

    def predict(self, features: np.ndarray) -> np.ndarray:
        dmatrix = xgb.DMatrix(features)
        preds = []
        for booster in self.ensure_all_boosters():
            preds.append(booster.predict(dmatrix))
        return np.column_stack(preds).astype(np.float32)
