# data-analysis-on-DBPs

End-to-end workflow for preparing dissolved by-product sensor data, aligning time steps, and training PyTorch regressors to predict DO/TOC/DOC signals at PPL1/PPL2.

## Repository Layout

| Path | Description |
| --- | --- |
| `data/` | Raw and processed CSV files (`raw_data.csv`, `imputed_data.csv`, `time_aligned_data.csv`). |
| `models/` | Trainable model code (`mlp_regressor.py`, `lstm_regressor.py`) and YAML configs under `models/configs/`. |
| `prompt-draft/` | Vibe-coding task briefs that describe each experiment in natural language. |
| `scripts/` | All runnable scripts (`fill_missing.py`, `align_time.py`, `train_regression.py`, `test_regression.py`, shared regression utilities, plus a script-level README) and generated artifacts in `scripts/outputs/<model_name>/`. |
| `requirements.txt` | Python dependencies for running every step. |

## Installing Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Typical Workflow

1. **Fill missing values**  
   `python scripts/fill_missing.py`
2. **Align time steps**  
   `python scripts/align_time.py`
3. **Train a model** (pick a YAML config)  
   `python scripts/train_regression.py --config models/configs/mlp_config.yaml`  
   `python scripts/train_regression.py --config models/configs/lstm_config.yaml`
4. **Evaluate / plot predictions**  
   `python scripts/test_regression.py --model-dir scripts/outputs/<model_name>`

Detailed script options and outputs are documented in `scripts/README.md`.

## Notes

- All configs now use YAML (`models/configs/*.yaml`). Duplicate and edit them to explore different history windows or architectures.
- `prompt-draft/` files are reference documents for the coding tasks and are not executed directly.
- Each training run writes checkpoints, scalers, plots, and metadata under `scripts/outputs/<model_name>/` so testing can reuse the exact scalers and splits.
