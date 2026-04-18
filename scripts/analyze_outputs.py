#!/usr/bin/env python3
"""
Analyze outputs directory to see what's taking space.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "outputs"


def get_directory_size(path: Path) -> float:
    """Get directory size in MB."""
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            total_size += fp.stat().st_size
    return total_size / (1024 * 1024)


def check_completeness(path: Path) -> Tuple[bool, int]:
    """Check if directory has complete results.
    Returns (is_complete, score) where score is number of criteria met.
    """
    criteria = {
        'has_config': False,
        'has_model': False,
        'has_result': False,
        'has_test': False,
    }

    # Check recursively
    for root, dirs, files in os.walk(path):
        for f in files:
            fpath = Path(root) / f
            if 'config' in f.lower() and f.endswith(('.toml', '.yaml', '.yml')):
                criteria['has_config'] = True
            elif f.endswith(('.pt', '.xgb', '.lgb', '.cbm')):
                criteria['has_model'] = True
            elif 'result' in f.lower() and f.endswith(('.toml', '.json', '.csv')):
                criteria['has_result'] = True
            elif 'test' in f.lower() or 'predict' in f.lower():
                criteria['has_test'] = True

    score = sum(1 for v in criteria.values() if v)
    is_complete = all(criteria.values())
    return is_complete, score


def main():
    print("="*80)
    print("ANALYZE OUTPUTS DIRECTORY")
    print("="*80)

    if not OUTPUTS_DIR.exists():
        print("Outputs directory does not exist.")
        return

    # List all directories
    all_items = []
    for item in OUTPUTS_DIR.iterdir():
        if not item.is_dir():
            continue
        size_mb = get_directory_size(item)
        is_complete, score = check_completeness(item)
        all_items.append({
            'name': item.name,
            'path': item,
            'size_mb': size_mb,
            'is_complete': is_complete,
            'score': score,
            'mtime': item.stat().st_mtime,
        })

    # Sort by size (largest first)
    all_items.sort(key=lambda x: x['size_mb'], reverse=True)

    print(f"\nTotal directories: {len(all_items)}")
    total_size = sum(item['size_mb'] for item in all_items)
    print(f"Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    complete_count = sum(1 for item in all_items if item['is_complete'])
    print(f"Complete directories: {complete_count}")

    print(f"\nTop 20 largest directories:")
    print(f"{'Name':<40} {'Size (MB)':>12} {'Complete':<10} {'Score':<6} {'Last Modified'}")
    print("-" * 90)

    for item in all_items[:20]:
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(item['mtime']).strftime('%Y-%m-%d %H:%M')
        print(f"{item['name'][:38]:<40} {item['size_mb']:>12.2f} {str(item['is_complete']):<10} {item['score']:<6} {mtime_str}")

    # Group by model type
    model_types = {}
    for item in all_items:
        name_lower = item['name'].lower()

        if 'xgboost' in name_lower:
            key = 'xgboost'
        elif 'lightgbm' in name_lower:
            key = 'lightgbm'
        elif 'catboost' in name_lower:
            key = 'catboost'
        elif 'mlp' in name_lower:
            key = 'mlp'
        elif 'rnn' in name_lower:
            key = 'rnn'
        elif 'gru' in name_lower:
            key = 'gru'
        elif 'lstm' in name_lower:
            key = 'lstm'
        elif 'transformer' in name_lower:
            key = 'transformer'
        elif 'mamba' in name_lower:
            key = 'mamba'
        else:
            key = 'other'

        if key not in model_types:
            model_types[key] = []
        model_types[key].append(item)

    print(f"\n\nSummary by model type:")
    print(f"{'Model':<15} {'Count':<8} {'Total Size (MB)':<18} {'Avg Size':<12} {'Complete'}")
    print("-" * 65)

    for model_type, items in sorted(model_types.items()):
        count = len(items)
        total_size = sum(item['size_mb'] for item in items)
        avg_size = total_size / count if count > 0 else 0
        complete_count = sum(1 for item in items if item['is_complete'])
        print(f"{model_type:<15} {count:<8} {total_size:<18.2f} {avg_size:<12.2f} {complete_count}/{count}")


if __name__ == "__main__":
    main()