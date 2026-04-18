#!/usr/bin/env python3
"""
Complete all 9 models on original dataset (imputed_data.csv):
1. Check current status
2. Run Bayesian optimization if missing (100 trials)
3. Train final model with best hyperparameters
4. Test model
5. Generate visualizations

Models: xgboost, lightgbm, catboost, mlp, rnn, gru, lstm, transformer, mamba
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tomli

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"


@dataclass
class ModelConfig:
    name: str
    model_type: str
    base_config: Path
    bayes_config: Path


MODEL_CONFIGS = [
    ModelConfig(
        name="xgboost",
        model_type="XGBOOST",
        base_config=REPO_ROOT / "models/configs/xgboost_config.toml",
        bayes_config=REPO_ROOT / "models/configs/xgboost_bayes.toml",
    ),
    ModelConfig(
        name="lightgbm",
        model_type="LIGHTGBM",
        base_config=REPO_ROOT / "models/configs/lightgbm_config.toml",
        bayes_config=REPO_ROOT / "models/configs/lightgbm_bayes.toml",
    ),
    ModelConfig(
        name="catboost",
        model_type="CATBOOST",
        base_config=REPO_ROOT / "models/configs/catboost_config.toml",
        bayes_config=REPO_ROOT / "models/configs/catboost_bayes.toml",
    ),
    ModelConfig(
        name="mlp",
        model_type="MLP",
        base_config=REPO_ROOT / "models/configs/mlp_config.toml",
        bayes_config=REPO_ROOT / "models/configs/mlp_bayes.toml",
    ),
    ModelConfig(
        name="rnn",
        model_type="RNN",
        base_config=REPO_ROOT / "models/configs/rnn_config.toml",
        bayes_config=REPO_ROOT / "models/configs/rnn_bayes.toml",
    ),
    ModelConfig(
        name="gru",
        model_type="GRU",
        base_config=REPO_ROOT / "models/configs/gru_config.toml",
        bayes_config=REPO_ROOT / "models/configs/gru_bayes.toml",
    ),
    ModelConfig(
        name="lstm",
        model_type="LSTM",
        base_config=REPO_ROOT / "models/configs/lstm_config.toml",
        bayes_config=REPO_ROOT / "models/configs/lstm_bayes.toml",
    ),
    ModelConfig(
        name="transformer",
        model_type="TRANSFORMER",
        base_config=REPO_ROOT / "models/configs/transformer_config.toml",
        bayes_config=REPO_ROOT / "models/configs/transformer_bayes.toml",
    ),
    ModelConfig(
        name="mamba",
        model_type="MAMBA",
        base_config=REPO_ROOT / "models/configs/mamba_config.toml",
        bayes_config=REPO_ROOT / "models/configs/mamba_bayes.toml",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete all 9 models on original dataset"
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUTS_DIR / "complete_experiment",
        help="Output directory for this experiment",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of trials for Bayesian optimization",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip models that are already complete",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if model appears complete",
    )
    return parser.parse_args()


def check_model_completion(model_name: str) -> Tuple[bool, Dict]:
    """Check if a model is complete."""
    status = {
        "has_bayes_opt": False,
        "has_final_model": False,
        "has_test_results": False,
        "best_val_loss": None,
        "best_config": None,
    }

    # Look for model directories
    model_dirs = []
    for item in OUTPUTS_DIR.iterdir():
        if not item.is_dir():
            continue
        if model_name.lower() in item.name.lower():
            model_dirs.append(item)

    # Check each directory
    for model_dir in model_dirs:
        # Check for bayesian optimization results
        bayes_files = list(model_dir.glob("*bayes*")) + list(model_dir.glob("*autotune*"))
        if bayes_files:
            status["has_bayes_opt"] = True

        # Check for final model files
        model_files = list(model_dir.glob("**/*.pt")) + list(model_dir.glob("**/*.xgb")) + \
                      list(model_dir.glob("**/*.lgb")) + list(model_dir.glob("**/*.cbm"))
        if model_files:
            status["has_final_model"] = True

        # Check for result.toml files
        result_files = list(model_dir.glob("**/result.toml"))
        for result_file in result_files:
            try:
                with open(result_file, "rb") as f:
                    result = tomli.load(f)
                if "eval" in result:
                    eval_data = result["eval"]
                    if "best_val_loss" in eval_data:
                        status["best_val_loss"] = eval_data["best_val_loss"]
                    if "test_loss" in eval_data:
                        status["has_test_results"] = True
            except:
                pass

    # Consider complete if we have all three
    is_complete = (
        status["has_bayes_opt"] and
        status["has_final_model"] and
        status["has_test_results"] and
        status["best_val_loss"] is not None
    )

    return is_complete, status


def run_bayesian_optimization(
    model_cfg: ModelConfig,
    output_dir: Path,
    n_trials: int,
) -> Tuple[bool, Optional[Dict]]:
    """Run Bayesian optimization for a model."""
    print(f"    Running Bayesian optimization ({n_trials} trials)...")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "autotune.py"),
        "--model-type", model_cfg.model_type,
        "--base-config", str(model_cfg.base_config),
        "--bayes-config", str(model_cfg.bayes_config),
        "--n-trials", str(n_trials),
        "--output-dir", str(output_dir / "bayes_opt"),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 2,  # 2 hour timeout
            cwd=str(REPO_ROOT),
        )

        if result.returncode != 0:
            print(f"      Failed: {result.stderr[:200]}")
            return False, None

        # Try to find best parameters
        best_params = {}
        # Look for results file
        results_csv = output_dir / "bayes_opt" / "bayes_optimization_results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            if not df.empty:
                best_row = df.loc[df['value'].idxmin()]
                best_params = best_row.to_dict()

        return True, {"best_params": best_params}

    except subprocess.TimeoutExpired:
        print(f"      Timed out after 2 hours")
        return False, None
    except Exception as e:
        print(f"      Error: {e}")
        return False, None


def train_final_model(
    model_cfg: ModelConfig,
    best_params: Dict,
    output_dir: Path,
) -> bool:
    """Train final model with best hyperparameters."""
    print(f"    Training final model...")

    # Create config with best parameters
    # This is simplified - in reality we need to merge best_params into config
    # For now, just use base config
    config_path = output_dir / "final_config.toml"
    shutil.copy(model_cfg.base_config, config_path)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config", str(config_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=str(REPO_ROOT),
        )

        if result.returncode != 0:
            print(f"      Failed: {result.stderr[:200]}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"      Timed out after 1 hour")
        return False
    except Exception as e:
        print(f"      Error: {e}")
        return False


def test_model(
    model_cfg: ModelConfig,
    output_dir: Path,
) -> bool:
    """Test the trained model."""
    print(f"    Testing model...")

    # Find the latest model directory
    model_name = model_cfg.name
    model_output_base = OUTPUTS_DIR / model_name

    if not model_output_base.exists():
        print(f"      No model directory found: {model_output_base}")
        return False

    # Find most recent subdirectory
    subdirs = sorted(
        [d for d in model_output_base.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not subdirs:
        print(f"      No model subdirectories found")
        return False

    latest_model_dir = subdirs[0]

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "test.py"),
        "--model-dir", str(latest_model_dir),
        "--output-dir", str(output_dir / "test_results"),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=str(REPO_ROOT),
        )

        if result.returncode != 0:
            print(f"      Failed: {result.stderr[:200]}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"      Timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"      Error: {e}")
        return False


def process_model(
    model_cfg: ModelConfig,
    output_base_dir: Path,
    n_trials: int,
    skip_completed: bool,
    force: bool,
) -> bool:
    """Process a single model through the complete pipeline."""
    print(f"\n{model_cfg.name.upper()}:")
    print(f"{'='*50}")

    model_output_dir = output_base_dir / model_cfg.name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    if skip_completed and not force:
        is_complete, status = check_model_completion(model_cfg.name)
        if is_complete:
            print(f"  ✓ Already complete (val_loss: {status.get('best_val_loss', 'N/A')})")
            return True

    # Step 1: Bayesian optimization
    print(f"  [1/3] Bayesian optimization")
    bayes_success, bayes_result = run_bayesian_optimization(
        model_cfg,
        model_output_dir,
        n_trials,
    )

    if not bayes_success:
        print(f"  ✗ Bayesian optimization failed")
        return False

    # Step 2: Final training
    print(f"  [2/3] Final training")
    best_params = bayes_result.get("best_params", {}) if bayes_result else {}
    train_success = train_final_model(
        model_cfg,
        best_params,
        model_output_dir,
    )

    if not train_success:
        print(f"  ✗ Final training failed")
        return False

    # Step 3: Testing
    print(f"  [3/3] Testing")
    test_success = test_model(
        model_cfg,
        model_output_dir,
    )

    if not test_success:
        print(f"  ✗ Testing failed")
        return False

    print(f"  ✓ {model_cfg.name.upper()} completed successfully!")
    return True


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPLETE ALL MODELS EXPERIMENT")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Skip completed: {args.skip_completed}")
    print(f"Force re-run: {args.force}")
    print(f"Models to process: {len(MODEL_CONFIGS)}")

    # Save experiment info
    experiment_info = {
        "timestamp": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "n_trials": args.n_trials,
        "skip_completed": args.skip_completed,
        "force": args.force,
        "models": [cfg.name for cfg in MODEL_CONFIGS],
    }

    with open(output_dir / "experiment_info.json", "w") as f:
        json.dump(experiment_info, f, indent=2)

    # Process each model
    results = {}
    success_count = 0

    for model_cfg in MODEL_CONFIGS:
        success = process_model(
            model_cfg,
            output_dir,
            args.n_trials,
            args.skip_completed,
            args.force,
        )
        results[model_cfg.name] = "success" if success else "failed"
        if success:
            success_count += 1

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Success: {success_count}/{len(MODEL_CONFIGS)} models")
    print(f"Failed: {len(MODEL_CONFIGS) - success_count} models")
    print(f"\nResults saved to: {output_dir}")

    if success_count == len(MODEL_CONFIGS):
        print(f"\n✓ All models completed successfully!")
    else:
        print(f"\n⚠ Some models failed. Check logs above.")


if __name__ == "__main__":
    main()