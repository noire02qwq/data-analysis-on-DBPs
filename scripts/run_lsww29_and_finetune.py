#!/usr/bin/env python3
"""
Run LSWW29 Transformer training and CAWW35/LSWW35 finetuning with 3 modes
"""
import os
import sys
import subprocess
import json
from pathlib import Path

BASE_DIR = Path("/home/amoris/dbps/data-analysis-on-DBPs")
OUTPUTS_DIR = BASE_DIR / "outputs"

def run_command(cmd, env=None):
    """Run command and return result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    else:
        print(f"Success: {result.stdout[:500]}")
    return result.returncode == 0

def train_lsww29_transformer():
    """Train Transformer on LSWW29 dataset"""
    print("="*80)
    print("Training LSWW29 Transformer...")
    print("="*80)

    # Use CAWW29 transformer best config but update data paths
    config_content = '''[model]
type = "TRANSFORMER"
name = "transformer_regressor"
history_length = 90
d_model = 240
nhead = 4
num_encoder_layers = 4
dim_feedforward = 429
dropout = 0.06349433443149866

[training]
max_epochs = 150
batch_size = 65
learning_rate = 0.0016818818084880578
weight_decay = 0.00036337451245327287
patience = 15
seed = 42

[data]
train_csv = "data/lsww_29c_split/train.csv"
val_csv = "data/lsww_29c_split/val.csv"
test_csv = "data/lsww_29c_split/test.csv"
input_columns = [
    "TRC-DT", "pH-DT", "cond-DT",
    "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT",
    "TOC-RT", "DOC-RT"
]
output_columns = [
    "TRC-PPL1", "TRC-PPL2",
    "pH-PPL1", "pH-PPL2",
    "cond-PPL1", "cond-PPL2"
]
'''

    config_path = OUTPUTS_DIR / "lsww29_transformer_config.toml"
    with open(config_path, 'w') as f:
        f.write(config_content)

    # Run training
    cmd = f"conda run -n torch python {BASE_DIR}/scripts/train.py --config {config_path}"
    return run_command(cmd)

def finetune_caww35():
    """Finetune on CAWW35 with 3 modes: full, partial, frozen"""
    print("="*80)
    print("Finetuning CAWW35...")
    print("="*80)

    # Load CAWW29 transformer best config
    base_config = BASE_DIR / "outputs/caww29_unified/final_models/transformer/config.toml"
    checkpoint = BASE_DIR / "outputs/caww29_unified/final_models/transformer/best_model.pt"

    modes = {
        'full': {'freeze': [], 'lr': 0.0001},
        'partial': {'freeze': ['encoder.layers.0', 'encoder.layers.1'], 'lr': 0.0005},
        'frozen': {'freeze': ['encoder'], 'lr': 0.001}
    }

    results = {}

    for mode, config in modes.items():
        print(f"\nFinetuning CAWW35 with mode: {mode}")

        output_dir = OUTPUTS_DIR / f"finetune_caww35_{mode}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create config
        finetune_config = f'''[model]
type = "TRANSFORMER"
name = "transformer_regressor"
history_length = 90
d_model = 240
nhead = 4
num_encoder_layers = 4
dim_feedforward = 429
dropout = 0.06349433443149866
checkpoint = "{checkpoint}"
freeze_layers = {config['freeze']}

[training]
max_epochs = 100
batch_size = 65
learning_rate = {config['lr']}
weight_decay = 0.00036337451245327287
patience = 10
seed = 42

[data]
train_csv = "data/caww_35c_split/train.csv"
val_csv = "data/caww_35c_split/val.csv"
test_csv = "data/caww_35c_split/test.csv"
input_columns = [
    "TRC-DT", "pH-DT", "cond-DT",
    "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT",
    "TOC-RT", "DOC-RT"
]
output_columns = [
    "TRC-PPL1", "TRC-PPL2",
    "pH-PPL1", "pH-PPL2",
    "cond-PPL1", "cond-PPL2"
]
'''

        config_path = output_dir / "config.toml"
        with open(config_path, 'w') as f:
            f.write(finetune_config)

        # Run training
        cmd = f"conda run -n torch python {BASE_DIR}/scripts/train.py --config {config_path}"
        success = run_command(cmd)
        results[f'caww35_{mode}'] = success

    return results

def finetune_lsww35():
    """Finetune on LSWW35 with 3 modes"""
    print("="*80)
    print("Finetuning LSWW35...")
    print("="*80)

    checkpoint = BASE_DIR / "outputs/caww29_unified/final_models/transformer/best_model.pt"

    modes = {
        'full': {'freeze': [], 'lr': 0.0001},
        'partial': {'freeze': ['encoder.layers.0', 'encoder.layers.1'], 'lr': 0.0005},
        'frozen': {'freeze': ['encoder'], 'lr': 0.001}
    }

    results = {}

    for mode, config in modes.items():
        print(f"\nFinetuning LSWW35 with mode: {mode}")

        output_dir = OUTPUTS_DIR / f"finetune_lsww35_{mode}"
        output_dir.mkdir(parents=True, exist_ok=True)

        finetune_config = f'''[model]
type = "TRANSFORMER"
name = "transformer_regressor"
history_length = 90
d_model = 240
nhead = 4
num_encoder_layers = 4
dim_feedforward = 429
dropout = 0.06349433443149866
checkpoint = "{checkpoint}"
freeze_layers = {config['freeze']}

[training]
max_epochs = 100
batch_size = 65
learning_rate = {config['lr']}
weight_decay = 0.00036337451245327287
patience = 10
seed = 42

[data]
train_csv = "data/lsww_35c_split/train.csv"
val_csv = "data/lsww_35c_split/val.csv"
test_csv = "data/lsww_35c_split/test.csv"
input_columns = [
    "TRC-DT", "pH-DT", "cond-DT",
    "TRC-RT", "pH-RT", "cond-RT", "fDOM-RT", "DO-RT",
    "TOC-RT", "DOC-RT"
]
output_columns = [
    "TRC-PPL1", "TRC-PPL2",
    "pH-PPL1", "pH-PPL2",
    "cond-PPL1", "cond-PPL2"
]
'''

        config_path = output_dir / "config.toml"
        with open(config_path, 'w') as f:
            f.write(finetune_config)

        cmd = f"conda run -n torch python {BASE_DIR}/scripts/train.py --config {config_path}"
        success = run_command(cmd)
        results[f'lsww35_{mode}'] = success

    return results

def create_summary():
    """Create summary of all results"""
    print("="*80)
    print("Creating summary...")
    print("="*80)

    summary = {
        'lsww29_transformer': str(OUTPUTS_DIR / 'transformer_regressor'),
        'caww35_finetune': {},
        'lsww35_finetune': {}
    }

    # Collect CAWW35 results
    for mode in ['full', 'partial', 'frozen']:
        result_file = OUTPUTS_DIR / f"finetune_caww35_{mode}" / "transformer_regressor" / "result.toml"
        if result_file.exists():
            summary['caww35_finetune'][mode] = str(result_file)

    # Collect LSWW35 results
    for mode in ['full', 'partial', 'frozen']:
        result_file = OUTPUTS_DIR / f"finetune_lsww35_{mode}" / "transformer_regressor" / "result.toml"
        if result_file.exists():
            summary['lsww35_finetune'][mode] = str(result_file)

    # Save summary
    summary_file = OUTPUTS_DIR / "lsww29_and_finetune_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_file}")
    return summary

if __name__ == "__main__":
    print("="*80)
    print("LSWW29 Transformer + CAWW35/LSWW35 Finetune Pipeline")
    print("="*80)

    # Check if LSWW29 transformer exists
    lsww29_model = OUTPUTS_DIR / "lsww29_transformer" / "best_model.pt"
    if not lsww29_model.exists():
        print("\n[1/4] Training LSWW29 Transformer...")
        train_lsww29_transformer()
    else:
        print("\n[1/4] LSWW29 Transformer already exists, skipping...")

    # Finetune CAWW35
    print("\n[2/4] Finetuning CAWW35...")
    finetune_caww35()

    # Finetune LSWW35
    print("\n[3/4] Finetuning LSWW35...")
    finetune_lsww35()

    # Create summary
    print("\n[4/4] Creating summary...")
    create_summary()

    print("\n" + "="*80)
    print("Pipeline completed!")
    print("="*80)
