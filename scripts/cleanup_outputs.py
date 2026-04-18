#!/usr/bin/env python3
"""
Clean up outputs directory to remove unnecessary files.
Strategy:
1. Keep only the most recent results for each model
2. Remove intermediate trial files
3. Keep configuration files and final results
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomli
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean up outputs directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=3,
        help="Keep the latest N results for each model",
    )
    return parser.parse_args()


def get_model_dirs() -> Dict[str, List[Path]]:
    """Get all model directories grouped by model name."""
    model_dirs = {}

    for item in OUTPUTS_DIR.iterdir():
        if not item.is_dir():
            continue

        # Try to identify model from directory name
        dir_name = item.name.lower()

        # Simple mapping
        if 'xgboost' in dir_name:
            key = 'xgboost'
        elif 'lightgbm' in dir_name:
            key = 'lightgbm'
        elif 'catboost' in dir_name:
            key = 'catboost'
        elif 'mlp' in dir_name:
            key = 'mlp'
        elif 'rnn' in dir_name:
            key = 'rnn'
        elif 'gru' in dir_name:
            key = 'gru'
        elif 'lstm' in dir_name:
            key = 'lstm'
        elif 'transformer' in dir_name:
            key = 'transformer'
        elif 'mamba' in dir_name:
            key = 'mamba'
        else:
            key = 'other'

        if key not in model_dirs:
            model_dirs[key] = []
        model_dirs[key].append(item)

    return model_dirs


def analyze_directory(dir_path: Path) -> Dict:
    """Analyze a directory for completeness."""
    status = {
        'path': dir_path,
        'has_config': False,
        'has_best_model': False,
        'has_result': False,
        'has_test': False,
        'size_mb': 0,
        'files': [],
    }

    # Check for common files
    config_files = list(dir_path.glob("*.toml")) + list(dir_path.glob("*.yaml")) + list(dir_path.glob("*.yml"))
    if config_files:
        status['has_config'] = True

    model_files = list(dir_path.glob("*.pt")) + list(dir_path.glob("*.xgb")) + list(dir_path.glob("*.lgb")) + list(dir_path.glob("*.cbm"))
    if model_files:
        status['has_best_model'] = True

    result_files = list(dir_path.glob("result.toml")) + list(dir_path.glob("*result*.toml"))
    if result_files:
        status['has_result'] = True

    test_files = list(dir_path.glob("*test*.csv")) + list(dir_path.glob("*test*.png")) + list(dir_path.glob("*predict*.csv"))
    if test_files:
        status['has_test'] = True

    # Calculate size
    total_size = 0
    all_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            fp = Path(root) / f
            total_size += fp.stat().st_size
            all_files.append(fp)

    status['size_mb'] = total_size / (1024 * 1024)
    status['files'] = all_files[:50]  # Limit to first 50

    return status


def is_complete(status: Dict) -> bool:
    """Check if a model directory is complete."""
    return (
        status['has_config'] and
        status['has_best_model'] and
        status['has_result'] and
        status['has_test']
    )


def main():
    args = parse_args()

    print("="*80)
    print("CLEANUP OUTPUTS DIRECTORY")
    print("="*80)
    print(f"Outputs directory: {OUTPUTS_DIR}")
    print(f"Keep latest: {args.keep_latest} results per model")
    print(f"Dry run: {args.dry_run}")

    # Get all model directories
    model_dirs = get_model_dirs()
    print(f"\nFound {len(model_dirs)} model types:")
    for model_name, dir_list in model_dirs.items():
        print(f"  {model_name}: {len(dir_list)} directories")

    total_size_before = 0
    for root, dirs, files in os.walk(OUTPUTS_DIR):
        for f in files:
            fp = os.path.join(root, f)
            total_size_before += os.path.getsize(fp)

    print(f"\nTotal size before cleanup: {total_size_before / (1024*1024*1024):.2f} GB")

    # Keep only the latest results for each model
    deleted_count = 0
    deleted_size = 0

    for model_name, dir_list in model_dirs.items():
        # Sort by modification time (newest first)
        sorted_dirs = sorted(dir_list, key=lambda p: p.stat().st_mtime, reverse=True)

        # Keep only the latest N directories
        to_delete = sorted_dirs[args.keep_latest:]

        print(f"\n{model_name.upper()}:")
        print(f"  Found {len(dir_list)} directories")
        print(f"  Will keep {args.keep_latest}, delete {len(to_delete)}")

        for dir_path in to_delete:
            dir_status = analyze_directory(dir_path)
            if not is_complete(dir_status) or len(dir_list) > args.keep_latest:
                size_mb = dir_status['size_mb']
                if not args.dry_run:
                    print(f"    Deleting: {dir_path} ({size_mb:.2f} MB)")
                    shutil.rmtree(dir_path)
                    deleted_count += 1
                    deleted_size += dir_status['size_mb']
                else:
                    print(f"    Would delete: {dir_path} ({size_mb:.2f} MB)")
            else:
                print(f"    Keeping: {dir_path} (complete)")

    print(f"\nTotal to delete: {deleted_count} directories")
    print(f"Total size to free up: {deleted_size:.2f} MB")

    if not args.dry_run and deleted_count > 0:
        print(f"\nCleaned up {deleted_count} directories")
    elif args.dry_run:
        print("\nDry run complete. Run without --dry-run to perform cleanup.")


if __name__ == "__main__":
    main()