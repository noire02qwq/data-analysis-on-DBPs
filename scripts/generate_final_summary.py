#!/usr/bin/env python3
"""
Generate final summary of all experiments
"""
import os
import re
import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path("/home/amoris/dbps/data-analysis-on-DBPs")
OUTPUTS_DIR = BASE_DIR / "outputs"

def parse_result_toml(filepath):
    """Parse result.toml file"""
    if not filepath.exists():
        return None

    result = {}
    with open(filepath, 'r') as f:
        content = f.read()

    # Parse best_val_loss
    match = re.search(r'best_val_loss\s*=\s*([\d.]+)', content)
    if match:
        result['best_val_loss'] = float(match.group(1))

    # Parse test_loss
    match = re.search(r'test_loss\s*=\s*([\d.]+)', content)
    if match:
        result['test_loss'] = float(match.group(1))

    # Parse best_epoch
    match = re.search(r'best_epoch\s*=\s*(\d+)', content)
    if match:
        result['best_epoch'] = int(match.group(1))

    return result

def collect_caww29_results():
    """Collect CAWW29 9-model results"""
    print("Collecting CAWW29 results...")

    models = ['mlp', 'lstm', 'rnn', 'gru', 'transformer', 'mamba', 'xgboost', 'lightgbm', 'catboost']
    results = []

    unified_dir = OUTPUTS_DIR / 'caww29_unified' / 'final_models'

    for model in models:
        model_dir = unified_dir / model
        if not model_dir.exists():
            continue

        result_file = model_dir / 'result.toml'
        result = parse_result_toml(result_file)

        if result:
            results.append({
                'model': model.upper(),
                'val_loss': result.get('best_val_loss', 0),
                'test_loss': result.get('test_loss', 0),
                'best_epoch': result.get('best_epoch', 'N/A')
            })

    # Sort by val_loss
    results.sort(key=lambda x: x['val_loss'])

    # Save as CSV and JSON
    df = pd.DataFrame(results)
    summary_dir = OUTPUTS_DIR / 'FINAL_SUMMARY'
    summary_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(summary_dir / 'caww29_9models_summary.csv', index=False)
    with open(summary_dir / 'caww29_9models_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"CAWW29 summary saved ({len(results)} models)")
    return results

def collect_finetune_results():
    """Collect finetune results for CAWW35 and LSWW35"""
    print("Collecting finetune results...")

    datasets = ['caww35', 'lsww35']
    modes = ['full', 'partial', 'frozen']
    results = []

    for dataset in datasets:
        for mode in modes:
            result_file = OUTPUTS_DIR / f"finetune_{dataset}_{mode}" / "transformer_regressor" / "result.toml"
            result = parse_result_toml(result_file)

            if result:
                results.append({
                    'dataset': dataset.upper(),
                    'mode': mode,
                    'val_loss': result.get('best_val_loss', 0),
                    'test_loss': result.get('test_loss', 0),
                    'best_epoch': result.get('best_epoch', 'N/A')
                })

    # Sort by dataset and val_loss
    results.sort(key=lambda x: (x['dataset'], x['val_loss']))

    # Save
    df = pd.DataFrame(results)
    summary_dir = OUTPUTS_DIR / 'FINAL_SUMMARY'
    df.to_csv(summary_dir / 'finetune_summary.csv', index=False)
    with open(summary_dir / 'finetune_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Fintune summary saved ({len(results)} experiments)")
    return results

def generate_markdown_report(caww29_results, finetune_results):
    """Generate final markdown report"""
    print("Generating markdown report...")

    report = f"""# Final Summary Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. CAWW29 9-Model Results

| Rank | Model | Val Loss | Test Loss | Best Epoch |
|------|-------|----------|-----------|------------|
"""

    for i, r in enumerate(caww29_results, 1):
        epoch = r.get('best_epoch', 'N/A')
        if isinstance(epoch, float):
            epoch = int(epoch)
        report += f"| {i} | {r['model']} | {r['val_loss']:.6f} | {r['test_loss']:.6f} | {epoch} |\n"

    report += f"""

## 2. Finetune Results (CAWW35 & LSWW35)

| Dataset | Mode | Val Loss | Test Loss | Best Epoch |
|---------|------|----------|-----------|------------|
"""

    for r in finetune_results:
        epoch = r.get('best_epoch', 'N/A')
        if isinstance(epoch, float):
            epoch = int(epoch)
        report += f"| {r['dataset']} | {r['mode']} | {r['val_loss']:.6f} | {r['test_loss']:.6f} | {epoch} |\n"

    report += f"""

## 3. Directory Structure

```
outputs/
├── caww29_unified/          # CAWW29 9模型统一目录
│   ├── final_models/        # 最终模型
│   └── test_results/        # 测试结果
├── lsww29_transformer/       # LSWW29 Transformer训练
├── finetune_caww35_{full,partial,frozen}/  # CAWW35 Finetune
├── finetune_lsww35_{full,partial,frozen}/  # LSWW35 Finetune
└── FINAL_SUMMARY/           # 最终汇总
    ├── caww29_9models_summary.csv
    ├── finetune_summary.csv
    └── final_report.md
```

## 4. Notes

- All experiments use the same transformer architecture from CAWW29 best model
- Finetune modes:
  - `full`: No freezing, full fine-tuning
  - `partial`: Freeze first 2 encoder layers
  - `frozen`: Freeze entire encoder, only train head

---

*Generated by final summary script*
"""

    # Save report
    summary_dir = OUTPUTS_DIR / 'FINAL_SUMMARY'
    with open(summary_dir / 'final_report.md', 'w') as f:
        f.write(report)

    print(f"Report saved to {summary_dir / 'final_report.md'}")
    return report

def main():
    """Main function"""
    print("="*80)
    print("Generating Final Summary")
    print("="*80)

    # Collect results
    caww29_results = collect_caww29_results()
    finetune_results = collect_finetune_results()

    # Generate report
    report = generate_markdown_report(caww29_results, finetune_results)

    print("\n" + "="*80)
    print("Summary generation complete!")
    print("="*80)

    # Print summary
    print(f"\nCAWW29 models: {len(caww29_results)}")
    print(f"Fintune experiments: {len(finetune_results)}")
    print(f"\nOutput directory: {OUTPUTS_DIR / 'FINAL_SUMMARY'}")

if __name__ == "__main__":
    main()
