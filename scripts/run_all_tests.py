#!/usr/bin/env python3
"""Run all model tests with proper visualization."""

import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("XGBoost", "xgboost_final"),
    ("LightGBM", "lightgbm_final"),
    ("CatBoost", "catboost_final"),
    ("MLP", "mlp_final"),
    ("RNN", "rnn_final"),
    ("GRU", "gru_final"),
    ("LSTM", "lstm_final"),
    ("Mamba", "mamba_final"),
    ("Transformer", "transformer_final"),
]

def run_test(model_name, model_dir):
    """Run test for a single model."""
    model_path = REPO_ROOT / "outputs" / model_dir
    if not model_path.exists():
        print(f"  [SKIP] {model_name}: {model_path} does not exist")
        return False

    # Find the timestamp directory
    subdirs = [d for d in model_path.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"  [SKIP] {model_name}: No subdirectory found")
        return False

    latest_dir = sorted(subdirs, key=lambda x: x.stat().st_mtime)[-1]
    model_dir_str = str(latest_dir)

    output_dir = latest_dir / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/test.py",
        "--model-dir", model_dir_str,
        "--output-dir", str(output_dir)
    ]

    print(f"  Running: {model_name}...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"  [ERROR] {model_name}: {result.stderr[:200]}")
        return False
    print(f"  [OK] {model_name}")
    return True

def main():
    print("Running tests for all models...")
    print("=" * 60)

    # Run sequentially to avoid GPU memory issues
    results = {}
    for model_name, model_dir in MODELS:
        results[model_name] = run_test(model_name, model_dir)

    print("\n" + "=" * 60)
    print("Summary:")
    for model_name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {model_name}: {status}")

if __name__ == "__main__":
    main()