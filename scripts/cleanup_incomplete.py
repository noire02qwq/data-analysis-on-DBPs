#!/usr/bin/env python3
"""
Clean up incomplete model results.
Keep only the most recent complete result for each model type.
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def is_complete_model_directory(dir_path: Path) -> bool:
    """Check if a directory contains a complete model training result."""
    required_files = [
        'config.toml',
        'result.toml',
    ]

    for req_file in required_files:
        if not (dir_path / req_file).exists():
            return False

    # Check for model files
    model_patterns = ['*.pt', '*.xgb', '*.lgb', '*.cbm']
    for pattern in model_patterns:
        if list(dir_path.glob(pattern)):
            return True

    return False


def get_model_type_from_name(name: str) -> str:
    """Extract model type from directory name."""
    lower_name = name.lower()

    # Check for specific model types
    if 'xgboost' in lower_name:
        return 'xgboost'
    elif 'lightgbm' in lower_name:
        return 'lightgbm'
    elif 'catboost' in lower_name:
        return 'catboost'
    elif 'mlp' in lower_name:
        return 'mlp'
    elif 'rnn' in lower_name:
        return 'rnn'
    elif 'gru' in lower_name:
        return 'gru'
    elif 'lstm' in lower_name:
        return 'lstm'
    elif 'transformer' in lower_name:
        return 'transformer'
    elif 'mamba' in lower_name:
        return 'mamba'
    else:
        return 'other'


def get_directory_size(dir_path: Path) -> float:
    """Get directory size in megabytes."""
    total_size = 0
    for item in dir_path.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size / (1024 * 1024)


def analyze_outputs():
    """Analyze the outputs directory and clean up."""
    print("="*80)
    print("CLEANUP INCOMPLETE MODEL RESULTS")
    print("="*80)

    if not OUTPUTS_DIR.exists():
        print("Outputs directory does not exist.")
        return

    # Get all directories
    all_dirs = []
    for item in OUTPUTS_DIR.iterdir():
        if not item.is_dir():
            continue

        size_mb = get_directory_size(item)
        is_complete = is_complete_model_directory(item)
        mtime = item.stat().st_mtime
        model_type = get_model_type_from_name(item.name)

        all_dirs.append({
            'path': item,
            'name': item.name,
            'model_type': model_type,
            'size_mb': size_mb,
            'is_complete': is_complete,
            'mtime': mtime,
        })

    # Group by model type
    grouped: Dict[str, List] = {}
    for d in all_dirs:
        model_type = d['model_type']
        if model_type not in grouped:
            grouped[model_type] = []
        grouped[model_type].append(d)

    # Sort each group by modification time (newest first)
    for model_type in grouped:
        grouped[model_type].sort(key=lambda x: x['mtime'], reverse=True)

    # Statistics
    total_dirs = len(all_dirs)
    complete_dirs = sum(1 for d in all_dirs if d['is_complete'])
    incomplete_dirs = total_dirs - complete_dirs
    total_size = sum(d['size_mb'] for d in all_dirs)

    print(f"\nTotal directories: {total_dirs}")
    print(f"Complete: {complete_dirs}")
    print(f"Incomplete: {incomplete_dirs}")
    print(f"Total size: {total_size:.2f} MB")

    # Find incomplete directories to delete
    to_delete: List[Path] = []

    print("\nDirectory analysis:")
    for model_type, dirs in grouped.items():
        complete_in_group = sum(1 for d in dirs if d['is_complete'])
        print(f"\n  {model_type}: {len(dirs)} total, {complete_in_group} complete")
        for d in dirs:
            status = "✓ complete" if d['is_complete'] else "✗ incomplete"
            print(f"    {d['name']}: {d['size_mb']:.2f} MB - {status}")
            if not d['is_complete']:
                to_delete.append(d['path'])

    if to_delete:
        print(f"\nDeleting {len(to_delete)} incomplete directories...")
        total_freed = 0
        for path in to_delete:
            size_before = get_directory_size(path)
            total_freed += size_before
            try:
                shutil.rmtree(path)
                print(f"  Deleted: {path.relative_to(OUTPUTS_DIR)} ({size_before:.2f} MB)")
            except Exception as e:
                print(f"  Failed to delete {path}: {e}")
        print(f"\nFreed {total_freed:.2f} MB of disk space.")
    else:
        print("\nNo incomplete directories to delete.")

    # Keep only the most recent complete result for each model type (optional cleanup)
    print("\nChecking for multiple complete results per model...")
    multiple_found = False
    for model_type, dirs in grouped.items():
        complete_dirs_in_group = [d for d in dirs if d['is_complete']]
        if len(complete_dirs_in_group) > 1:
            multiple_found = True
            print(f"\n  {model_type} has {len(complete_dirs_in_group)} complete results:")
            # Keep the newest one, delete the rest
            for i, d in enumerate(complete_dirs_in_group):
                if i == 0:
                    print(f"    Keeping: {d['name']} ({d['size_mb']:.2f} MB) - newest")
                else:
                    print(f"    Will delete: {d['name']} ({d['size_mb']:.2f} MB) - older")
                    to_delete.append(d['path'])

    if multiple_found and to_delete:
        print(f"\nDeleting {len(to_delete) - len(to_delete)} old complete directories...")
        # Wait, actually ask for confirmation
        print("\nWould you like to delete the older complete directories to save space?")
        response = input("Enter 'y' to confirm deletion: ").lower().strip()
        if response == 'y':
            total_freed = 0
            for path in to_delete:
                if path.exists():
                    size_before = get_directory_size(path)
                    total_freed += size_before
                    try:
                        shutil.rmtree(path)
                        print(f"  Deleted: {path.relative_to(OUTPUTS_DIR)} ({size_before:.2f} MB)")
                    except Exception as e:
                        print(f"  Failed to delete {path}: {e}")
            print(f"\nTotal freed: {total_freed:.2f} MB")
        else:
            print("Skipping deletion of older complete directories.")

    print("\nCleanup complete!")


def main():
    parser = argparse.ArgumentParser(description='Clean up incomplete model outputs')
    parser.add_argument('--dry-run', action='store_true', help='Only show what would be deleted')
    args = parser.parse_args()

    analyze_outputs()


if __name__ == "__main__":
    main()
