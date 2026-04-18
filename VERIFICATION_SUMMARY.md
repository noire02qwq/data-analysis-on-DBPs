#!/usr/bin/env python3
"""
Complete GBDT Multi-Output Experiment Script

Features:
- Data splitting (70:15:15)
- Bayesian optimization (100 trials per model) with actual model training
- Multi-output: PPL1, PPL2 for TRC, pH, cond, TOC, DOC
- Input: All DT and RT columns (no leakage)
- Outputs: Configs, results TOML, loss curves per epoch, prediction plots per variable, scatter plots per variable
"""

import argparse
import json
import os
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import polars as pl

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Try importing toml libraries
try:
    import tomllib as tomli
except ImportError:
    import tomli

try:
    import tomli_w
except ImportError:
    tomli_w = None

# Import sklearn metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

# Import GBDT libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent

# =============================================================================
# Column Definitions
# =============================================================================

# Input: All DT and RT columns (NO PPL columns - to avoid data leakage)
INPUT_DT_RT_COLUMNS = [
    # TRC
    "TRC-DT", "TRC-RT",
    # pH
    "pH-DT", "pH-RT",
    # cond
    "cond-DT", "cond-RT",
    # fDOM (RT only)
    "fDOM-RT",
    # DO (RT only)
    "DO-RT",
    # TOC (RT only)
    "TOC-RT",
    # DOC (RT only)
    "DOC-RT",
]

# Output: All PPL1 and PPL2 columns (for TRC, pH, cond, TOC, DOC)
OUTPUT_PPL_COLUMNS = [
    # TRC PPL
    "TRC-PPL1", "TRC-PPL2",
    # pH PPL
    "pH-PPL1", "pH-PPL2",
    # cond PPL
    "cond-PPL1", "cond-PPL2",
    # TOC PPL
    "TOC-PPL1", "TOC-PPL2",
    # DOC PPL
    "DOC-PPL1", "DOC-PPL2",
]

# =============================================================================
# Utility Functions
# =============================================================================

def load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as fh:
        return tomli.load(fh)

def save_toml(path: Path, data: Dict[str, Any]) -> None:
    if tomli_w is None:
        raise ImportError("tomli_w is required to save TOML files")
    with path.open("wb") as fh:
        tomli_w.dump(data, fh)

def ensure_conda_env():
    """Ensure conda environment is activated."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"Using conda environment: {conda_env}")
    else:
        print("Warning: No conda environment detected. Make sure to activate your environment.")
        print("  conda activate torch")

# =============================================================================
# Data Splitting
# =============================================================================

def split_data(
    input_csv: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    """Split data into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    print(f"\nLoading data from: {input_csv}")
    df = pl.read_csv(input_csv, encoding="utf-8-sig")
    print(f"Total samples: {len(df)}")

    # Add temporal column
    df = df.with_columns([
        pl.arange(0, pl.count()).alias("minutes_since_start")
    ])

    # Shuffle for splitting
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    df_train = df[:n_train]
    df_val = df[n_train : n_train + n_val]
    df_test = df[n_train + n_val :]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    df_train.write_csv(train_path)
    df_val.write_csv(val_path)
    df_test.write_csv(test_path)

    print(f"\nData split complete:")
    print(f"  Train: {len(df_train)} samples ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(df_val)} samples ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(df_test)} samples ({test_ratio*100:.1f}%)")

    return train_path, val_path, test_path

# =============================================================================
# Plotting Functions - Per Variable
# =============================================================================

def plot_predictions_vs_true_single(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str,
    output_path: Path,
) -> None:
    """Plot predictions vs true values for a single target."""
    plt.figure(figsize=(12, 6))

    plt.plot(y_true, 'b-', label='True', linewidth=1.5, alpha=0.8)
    plt.plot(y_pred, 'r--', label='Predicted', linewidth=1.5, alpha=0.8)

    plt.xlabel('Sample', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(f'{model_name} - {target_name}: Test Predictions vs True Values', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_yx_single(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_name: str,
    model_name: str,
    output_path: Path,
) -> Dict:
    """Plot y=x scatter plot with R² for a single target."""
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    plt.figure(figsize=(8, 8))

    # Scatter plot
    plt.scatter(y_true, y_pred, c='blue', alpha=0.5, s=20, edgecolors='none')

    # y=x line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    plot_min = min_val - margin
    plot_max = max_val + margin
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, label='y=x')

    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'{model_name} - {target_name}\nR² = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'target': target_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae
    }


def plot_loss_curves_per_epoch(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str,
    output_path: Path,
) -> None:
    """Plot training and validation loss curves per epoch for best trial."""
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title(f'{model_name} - Best Trial: Training and Validation Loss per Epoch', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Loss curves per epoch saved to: {output_path}")


# =============================================================================
# Main Experiment Class
# =============================================================================

class GBDTExperiment:
    """GBDT experiment class for multi-output regression."""

    def __init__(
        self,
        input_csv: Path,
        output_dir: Path,
        n_trials: int = 100,
        seed: int = 42,
    ):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.seed = seed

        # Define input/output columns
        self.input_cols = INPUT_DT_RT_COLUMNS
        self.output_cols = OUTPUT_PPL_COLUMNS

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Data paths
        self.train_csv: Optional[Path] = None
        self.val_csv: Optional[Path] = None
        self.test_csv: Optional[Path] = None

        # Store data for training
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

    def load_and_prepare_data(self):
        """Load data from CSV and prepare for training."""
        print(f"\nLoading data from: {self.input_csv}")
        df = pl.read_csv(self.input_csv, encoding="utf-8-sig")
        print(f"Total samples: {len(df)}")

        # Add temporal column
        df = df.with_columns([
            pl.arange(0, pl.count()).alias("minutes_since_start")
        ])

        # Shuffle for splitting
        df = df.sample(fraction=1.0, shuffle=True, seed=self.seed)

        n_total = len(df)
        n_train = int(n_total * 0.7)
        n_val = int(n_total * 0.15)

        df_train = df[:n_train]
        df_val = df[n_train : n_train + n_val]
        df_test = df[n_train + n_val :]

        # Save splits
        data_dir = self.output_dir / "data_split"
        data_dir.mkdir(parents=True, exist_ok=True)

        self.train_csv = data_dir / "train.csv"
        self.val_csv = data_dir / "val.csv"
        self.test_csv = data_dir / "test.csv"

        df_train.write_csv(self.train_csv)
        df_val.write_csv(self.val_csv)
        df_test.write_csv(self.test_csv)

        print(f"\nData split complete:")
        print(f"  Train: {len(df_train)} samples (70%)")
        print(f"  Val:   {len(df_val)} samples (15%)")
        print(f"  Test:  {len(df_test)} samples (15%)")

        # Extract features and targets
        self.X_train = df_train.select(self.input_cols).to_numpy()
        self.y_train = df_train.select(self.output_cols).to_numpy()
        self.X_val = df_val.select(self.input_cols).to_numpy()
        self.y_val = df_val.select(self.output_cols).to_numpy()
        self.X_test = df_test.select(self.input_cols).to_numpy()
        self.y_test = df_test.select(self.output_cols).to_numpy()

        print(f"\nFeature matrix shapes:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  X_val: {self.X_val.shape}")
        print(f"  y_val: {self.y_val.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_test: {self.y_test.shape}")

        return self.train_csv, self.val_csv, self.test_csv
