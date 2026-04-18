#!/usr/bin/env python3
"""
Quick cleanup of outputs directory.
Keep only the most recent complete result for each model.
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def is_complete_directory(dir_path: Path) -> bool:
    """Check if a directory has complete model results."""
    required_files = [
        'config.toml',
        'result.toml',
    ]

    for req_file in required_files:
        if not (dir_path / req_file).exists():
            return False

    # Check for model files
    model_patterns = ['*.pt', '*.xgb', '*.lgb', '*.cbm']
    has_model = False
    for pattern in model_patterns:
        if list(dir_path.glob(pattern)):
            has_model = True
            break

    if not has_model:
        return False

    return True


def get_model_type(dir_name: str) -> str:
    """Extract model type from directory name."""
    dir_lower = dir_name.lower()

    if 'xgboost' in dir_lower or 'xgb' in dir_lower:
        return 'xgboost'
    elif 'lightgbm' in dir_lower or 'lgb' in dir_lower:
        return 'lightgbm'
    elif 'catboost' in dir_lower or 'cbm' in dir_lower:
        return 'catboost'
    elif 'mlp' in dir_lower:
        return 'mlp'
    elif 'rnn' in dir_lower:
        return 'rnn'
    elif 'gru' in dir_lower:
        return 'gru'
    elif 'lstm' in dir_lower:
        return 'lstm'
    elif 'transformer' in dir_lower:
        return 'transformer'
    elif 'mamba' in dir_lower:
        return 'mamba'
    else:
        return 'other'


def main():
    print("="*80)
    print("QUICK CLEANUP OF OUTPUTS DIRECTORY")
    print("="*80)

    print(f"Outputs directory: {OUTPUTS_DIR}")

    if not OUTPUTS_DIR.exists():
        print("Outputs directory does not exist.")
        return

    # Get all directories grouped by model type
    model_groups: Dict[str, List[Path]] = {}

    for item in OUTPUTS_DIR.iterdir():
        if not item.is_dir():
            continue

        model_type = get_model_type(item.name)
        if model_type not in model_groups:
            model_groups[model_type] = []
        model_groups[model_type].append(item)

    print(f"\nFound {len(model_groups)} model types:")
    for model_type, dir_list in model_groups.items():
        print(f"  {model_type}: {len(dir_list)} directories")

    # Keep only the most recent complete directory for each model
    deleted_count = 0
    deleted_size = 0

    for model_type, dir_list in model_groups.items():
        # Sort by modification time (newest first)
        dir_list.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Keep only the most recent directory that is complete
        kept_one = False
        for dir_path in dir_list:
            if is_complete_directory(dir_path) and not kept_one:
                print(f"  Keeping: {dir_path} (complete)")
                kept_one = True
            else:
                size_mb = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file()) / (1024 * 1024)
                print(f"  Deleting: {dir_path} ({size_mb:.2f} MB)")

                # In dry run mode, just show
                # Actually do it
                shutil.rmtree(dir_path)
                deleted_count += 1
                deleted_size += size_mb

    print(f"\nTotal deleted: {deleted_count} directories")
    print(f"Total size freed: {deleted_size:.2f} MB")


if __name__ == "__main__":
    main()