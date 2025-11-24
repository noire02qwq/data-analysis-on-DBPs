#!/usr/bin/env python3
"""
Analyze results from all model trials and generate cross-model comparison CSVs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent


def read_bayes_results(model_dir: Path) -> List[Dict[str, object]]:
    """Read bayes optimization results and return trials sorted by validation loss"""
    results_file = model_dir / "bayes_optimization_results.csv"
    if not results_file.exists():
        raise FileNotFoundError(f"Bayes optimization results not found: {results_file}")
    
    trials = []
    with results_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert value to float for sorting
            try:
                row["value"] = float(row["value"])
                trials.append(row)
            except ValueError:
                # Skip rows with invalid values
                continue
    
    # Sort by validation loss (value) ascending
    trials.sort(key=lambda x: x["value"])
    return trials


def read_trial_metrics(trial_dir: Path) -> Dict[str, Dict[str, float]]:
    """Read metrics from a trial directory"""
    metrics_file = trial_dir / "test_metrics.csv"
    if not metrics_file.exists():
        return {}
    
    metrics = {}
    with metrics_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row["target"]
            metrics[target] = {
                "r2": float(row["r2"]),
                "mse": float(row["mse"]),
                "rmse": float(row["rmse"]),
                "mae": float(row["mae"])
            }
    return metrics


def get_top_trials(model_dir: Path, n: int = 3) -> List[Dict[str, object]]:
    """Get top n trials based on validation loss"""
    trials = read_bayes_results(model_dir)
    return trials[:n]


def find_trial_dir(model_dir: Path, trial_identifier: str) -> Path | None:
    """Find the actual trial directory based on trial identifier"""
    # 如果trial_identifier看起来像一个目录名(包含下划线)，直接使用它
    if "_" in trial_identifier:
        trial_dir = model_dir / trial_identifier
        if trial_dir.exists():
            return trial_dir
        return None
    
    # 否则，它是一个数字，需要在目录中查找匹配的trial目录
    for trial_dir in model_dir.iterdir():
        if trial_dir.is_dir() and trial_dir.name.startswith("trial_"):
            # 从目录名中提取trial编号 (trial_0_73952c -> 0)
            parts = trial_dir.name.split("_")
            if len(parts) >= 2 and parts[1] == trial_identifier:
                return trial_dir
    return None


def generate_comparison_csvs() -> None:
    """Generate three comparison CSVs for TRC-PPL2, pH-PPL2, and TOC-PPL2"""
    
    # Define models and modes to analyze
    models = ["mlp", "mlp_with_history", "lstm", "rnn"]
    modes = ["trc-value", "trc-rate", "other-value", "other-rate"]
    
    # Target columns for each comparison
    trc_target = "TRC-PPL2"
    ph_target = "pH-PPL2"
    toc_target = "TOC-PPL2"
    
    # Rate target columns for rate regression comparisons
    trc_rate_target = "TRC-PPL2"
    ph_rate_target = "pH-PPL2"
    toc_rate_target = "TOC-PPL2"
    
    # Prepare data structures for each comparison
    trc_data = []
    ph_data = []
    toc_data = []
    
    # Rate regression comparison data
    trc_rate_data = []
    ph_rate_data = []
    toc_rate_data = []
    
    # Process each model and mode
    for model in models:
        for mode in modes:
            model_dir_name = f"{model}-{mode}"
            model_dir = REPO_ROOT / "scripts" / "outputs" / model_dir_name
            
            if not model_dir.exists():
                print(f"Warning: Model directory not found: {model_dir}")
                continue
                
            try:
                top_trials = get_top_trials(model_dir, 3)
                print(f"Processing {model}-{mode}: found {len(top_trials)} top trials")
                
                for trial in top_trials:
                    trial_identifier = str(trial["trial_number"])  # Convert to string
                    
                    # 根据trial标识符查找实际的trial目录
                    trial_dir = find_trial_dir(model_dir, trial_identifier)
                    
                    if trial_dir is None:
                        print(f"Warning: Trial directory not found for trial identifier: {trial_identifier}")
                        continue
                    
                    metrics = read_trial_metrics(trial_dir)
                    if not metrics:
                        print(f"Warning: No metrics found in {trial_dir}")
                        continue
                    
                    # Create entry name: model-mode-trial_identifier
                    entry_name = f"{model}-{mode}-{trial_identifier.split('_')[1] if '_' in trial_identifier else trial_identifier}"
                    val_loss = trial["value"]
                    
                    # Add to TRC comparison if TRC-PPL2 exists
                    if trc_target in metrics:
                        trc_metrics = metrics[trc_target]
                        trc_data.append({
                            "model": entry_name,
                            "val_loss": val_loss,
                            "r2": trc_metrics["r2"],
                            "mse": trc_metrics["mse"],
                            "rmse": trc_metrics["rmse"],
                            "mae": trc_metrics["mae"]
                        })
                    
                    # Add to pH comparison if pH-PPL2 exists
                    if ph_target in metrics:
                        ph_metrics = metrics[ph_target]
                        ph_data.append({
                            "model": entry_name,
                            "val_loss": val_loss,
                            "r2": ph_metrics["r2"],
                            "mse": ph_metrics["mse"],
                            "rmse": ph_metrics["rmse"],
                            "mae": ph_metrics["mae"]
                        })
                    
                    # Add to TOC comparison if TOC-PPL2 exists
                    if toc_target in metrics:
                        toc_metrics = metrics[toc_target]
                        toc_data.append({
                            "model": entry_name,
                            "val_loss": val_loss,
                            "r2": toc_metrics["r2"],
                            "mse": toc_metrics["mse"],
                            "rmse": toc_metrics["rmse"],
                            "mae": toc_metrics["mae"]
                        })
                        
                    # Add to rate regression comparisons if in rate mode
                    if "rate" in mode:
                        # TRC rate comparison
                        if trc_rate_target in metrics:
                            trc_rate_metrics = metrics[trc_rate_target]
                            trc_rate_data.append({
                                "model": entry_name,
                                "val_loss": val_loss,
                                "r2": trc_rate_metrics["r2"],
                                "mse": trc_rate_metrics["mse"],
                                "rmse": trc_rate_metrics["rmse"],
                                "mae": trc_rate_metrics["mae"]
                            })
                        
                        # pH rate comparison
                        if ph_rate_target in metrics:
                            ph_rate_metrics = metrics[ph_rate_target]
                            ph_rate_data.append({
                                "model": entry_name,
                                "val_loss": val_loss,
                                "r2": ph_rate_metrics["r2"],
                                "mse": ph_rate_metrics["mse"],
                                "rmse": ph_rate_metrics["rmse"],
                                "mae": ph_rate_metrics["mae"]
                            })
                        
                        # TOC rate comparison
                        if toc_rate_target in metrics:
                            toc_rate_metrics = metrics[toc_rate_target]
                            toc_rate_data.append({
                                "model": entry_name,
                                "val_loss": val_loss,
                                "r2": toc_rate_metrics["r2"],
                                "mse": toc_rate_metrics["mse"],
                                "rmse": toc_rate_metrics["rmse"],
                                "mae": toc_rate_metrics["mae"]
                            })
                        
            except Exception as e:
                print(f"Error processing {model}-{mode}: {e}")
                continue
    
    # Sort data by validation loss
    trc_data.sort(key=lambda x: float(x["val_loss"]))
    ph_data.sort(key=lambda x: float(x["val_loss"]))
    toc_data.sort(key=lambda x: float(x["val_loss"]))
    
    trc_rate_data.sort(key=lambda x: float(x["val_loss"]))
    ph_rate_data.sort(key=lambda x: float(x["val_loss"]))
    toc_rate_data.sort(key=lambda x: float(x["val_loss"]))
    
    # Write TRC comparison CSV
    trc_csv_path = REPO_ROOT / "trc_ppl2_comparison.csv"
    with trc_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "r2", "mse", "rmse", "mae"])
        for entry in trc_data:
            writer.writerow([
                entry["model"],
                f"{entry['r2']:.6f}",
                f"{entry['mse']:.6f}",
                f"{entry['rmse']:.6f}",
                f"{entry['mae']:.6f}"
            ])
    
    # Write pH comparison CSV
    ph_csv_path = REPO_ROOT / "ph_ppl2_comparison.csv"
    with ph_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "r2", "mse", "rmse", "mae"])
        for entry in ph_data:
            writer.writerow([
                entry["model"],
                f"{entry['r2']:.6f}",
                f"{entry['mse']:.6f}",
                f"{entry['rmse']:.6f}",
                f"{entry['mae']:.6f}"
            ])
    
    # Write TOC comparison CSV
    toc_csv_path = REPO_ROOT / "toc_ppl2_comparison.csv"
    with toc_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "r2", "mse", "rmse", "mae"])
        for entry in toc_data:
            writer.writerow([
                entry["model"],
                f"{entry['r2']:.6f}",
                f"{entry['mse']:.6f}",
                f"{entry['rmse']:.6f}",
                f"{entry['mae']:.6f}"
            ])
    
    # Write rate regression comparison CSVs
    trc_rate_csv_path = REPO_ROOT / "trc_ppl2_rate_comparison.csv"
    with trc_rate_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "r2", "mse", "rmse", "mae"])
        for entry in trc_rate_data:
            writer.writerow([
                entry["model"],
                f"{entry['r2']:.6f}",
                f"{entry['mse']:.6f}",
                f"{entry['rmse']:.6f}",
                f"{entry['mae']:.6f}"
            ])
    
    ph_rate_csv_path = REPO_ROOT / "ph_ppl2_rate_comparison.csv"
    with ph_rate_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "r2", "mse", "rmse", "mae"])
        for entry in ph_rate_data:
            writer.writerow([
                entry["model"],
                f"{entry['r2']:.6f}",
                f"{entry['mse']:.6f}",
                f"{entry['rmse']:.6f}",
                f"{entry['mae']:.6f}"
            ])
    
    toc_rate_csv_path = REPO_ROOT / "toc_ppl2_rate_comparison.csv"
    with toc_rate_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "r2", "mse", "rmse", "mae"])
        for entry in toc_rate_data:
            writer.writerow([
                entry["model"],
                f"{entry['r2']:.6f}",
                f"{entry['mse']:.6f}",
                f"{entry['rmse']:.6f}",
                f"{entry['mae']:.6f}"
            ])
    
    print(f"Generated comparison CSVs:")
    print(f"  - {trc_csv_path}")
    print(f"  - {ph_csv_path}")
    print(f"  - {toc_csv_path}")
    print(f"  - {trc_rate_csv_path}")
    print(f"  - {ph_rate_csv_path}")
    print(f"  - {toc_rate_csv_path}")
    print(f"\nSummary:")
    print(f"  TRC-PPL2 comparisons: {len(trc_data)} entries")
    print(f"  pH-PPL2 comparisons: {len(ph_data)} entries")
    print(f"  TOC-PPL2 comparisons: {len(toc_data)} entries")
    print(f"  TRC-PPL2 rate comparisons: {len(trc_rate_data)} entries")
    print(f"  pH-PPL2 rate comparisons: {len(ph_rate_data)} entries")
    print(f"  TOC-PPL2 rate comparisons: {len(toc_rate_data)} entries")


if __name__ == "__main__":
    generate_comparison_csvs()