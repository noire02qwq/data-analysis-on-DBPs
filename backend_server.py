#!/usr/bin/env python3
"""
Flask Backend Server for DBPs Deep Learning Platform
Provides REST API for time series analysis
Runs on port 5555
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import uuid
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import TOML libraries
try:
    import tomllib as tomli
except ImportError:
    import tomli

try:
    import tomli_w
except ImportError:
    tomli_w = None

# Flask app initialization
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    },
    r"/health": {
        "origins": ["*"],
        "methods": ["GET", "OPTIONS"]
    }
})

# In-memory storage
jobs: Dict[str, Dict[str, Any]] = {}
uploaded_datasets: Dict[str, Dict[str, Any]] = {}
models: Dict[str, Dict[str, Any]] = {}

# Job management lock
job_lock = threading.Lock()

# Directories
UPLOAD_DIR = REPO_ROOT / "uploads"
OUTPUT_DIR = REPO_ROOT / "outputs"
CONFIG_DIR = REPO_ROOT / "server_configs"

def ensure_directories():
    """Ensure required directories exist."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    CONFIG_DIR.mkdir(exist_ok=True)

def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())

def create_response(success: bool, data: Any = None, error: Dict = None) -> tuple:
    """Create a standardized API response."""
    response = {"success": success}
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = error
    return jsonify(response)


# ==================== Health Check ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return create_response(True, {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


# ==================== Data Management ====================

@app.route('/api/v1/data/upload', methods=['POST'])
def upload_data():
    """Upload a dataset file (CSV or Excel)."""
    try:
        if 'file' not in request.files:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "No file provided"
            }), 400

        file = request.files['file']
        if file.filename == '':
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "Empty filename"
            }), 400

        # Generate dataset ID
        dataset_id = generate_id()

        # Save file
        file_ext = Path(file.filename).suffix.lower()
        saved_path = UPLOAD_DIR / f"{dataset_id}{file_ext}"
        file.save(str(saved_path))

        # Parse file to get info
        try:
            import polars as pl

            total_rows = 0
            df = None
            if file_ext == '.csv':
                # Count total rows first (streaming, doesn't load data)
                try:
                    # utf-8-sig not supported by scan_csv, use utf8 and strip BOM
                    total_rows = pl.scan_csv(saved_path, encoding="utf8").select(pl.len()).collect().item()
                except Exception:
                    total_rows = 0
                df = pl.read_csv(saved_path, encoding="utf-8-sig", n_rows=1000)
            elif file_ext in ['.xlsx', '.xls']:
                # For Excel, we need to read to count rows
                df = pl.read_excel(saved_path, n_rows=1000)
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(saved_path, read_only=True, data_only=True)
                    total_rows = wb.active.max_row or 0
                    wb.close()
                except Exception:
                    total_rows = df.shape[0]
            else:
                df = None

            columns = []
            if df is not None:
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    if "Float" in dtype or "Int" in dtype:
                        col_type = "float"
                    elif "Date" in dtype or "Time" in dtype:
                        col_type = "datetime"
                    else:
                        col_type = "string"
                    columns.append({"name": col, "type": col_type})

                row_count = total_rows if total_rows > 0 else df.shape[0]
            else:
                columns = []
                row_count = 0

        except Exception as e:
            logger.warning(f"Failed to parse uploaded file: {e}")
            columns = []
            row_count = 0

        # Store dataset info
        uploaded_datasets[dataset_id] = {
            "id": dataset_id,
            "filename": file.filename,
            "path": str(saved_path),
            "size": saved_path.stat().st_size,
            "columns": columns,
            "row_count": row_count,
            "uploaded_at": datetime.now().isoformat()
        }

        return create_response(True, {
            "datasetId": dataset_id,
            "filename": file.filename,
            "size": saved_path.stat().st_size,
            "rowCount": row_count,
            "columns": columns,
            "message": "File uploaded successfully"
        }), 201

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.route('/api/v1/data/info/<dataset_id>', methods=['GET'])
def get_dataset_info(dataset_id: str):
    """Get information about an uploaded dataset."""
    if dataset_id not in uploaded_datasets:
        return create_response(False, error={
            "code": "NOT_FOUND",
            "message": f"Dataset {dataset_id} not found"
        }), 404

    dataset = uploaded_datasets[dataset_id]
    return create_response(True, dataset)


@app.route('/api/v1/data/preview/<dataset_id>', methods=['GET'])
def get_dataset_preview(dataset_id: str):
    """Get a preview of an uploaded dataset (first N rows as JSON)."""
    if dataset_id not in uploaded_datasets:
        return create_response(False, error={
            "code": "NOT_FOUND",
            "message": f"Dataset {dataset_id} not found"
        }), 404

    dataset = uploaded_datasets[dataset_id]
    try:
        import polars as pl
        limit = request.args.get('limit', 500, type=int)
        input_path = Path(dataset['path'])
        if input_path.suffix == '.csv':
            df = pl.read_csv(input_path, encoding="utf-8-sig", n_rows=limit)
        elif input_path.suffix in ['.xlsx', '.xls']:
            df = pl.read_excel(input_path, n_rows=limit)
        else:
            return create_response(False, error={"code": "UNSUPPORTED", "message": "Unsupported file format"}), 400

        rows = df.to_dicts()
        # Convert values to JSON-serializable types
        for row in rows:
            for key, val in row.items():
                if val is None:
                    row[key] = None
                elif hasattr(val, 'item'):
                    row[key] = val.item()
                else:
                    row[key] = str(val) if not isinstance(val, (int, float, str, bool)) else val

        return create_response(True, {
            "rows": rows,
            "columns": df.columns,
            "totalRows": dataset.get('row_count', len(rows))
        })
    except Exception as e:
        logger.error(f"Preview error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.route('/api/v1/data/split', methods=['POST'])
def split_data():
    """Split a dataset into train/val/test sets."""
    try:
        data = request.get_json()
        if not data:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "No JSON data provided"
            }), 400

        dataset_id = data.get('datasetId')
        if not dataset_id or dataset_id not in uploaded_datasets:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": "Dataset not found"
            }), 404

        dataset = uploaded_datasets[dataset_id]
        input_csv = Path(dataset['path'])

        # Get split parameters
        train_ratio = data.get('trainRatio', 0.7)
        val_ratio = data.get('valRatio', 0.15)
        test_ratio = data.get('testRatio', 0.15)
        shuffle = data.get('shuffle', False)
        seed = data.get('seed', 42)

        # Read the CSV to get row count and calculate split rows
        try:
            import polars as pl
            df_full = pl.read_csv(input_csv)
            total_rows = len(df_full)
        except Exception as e:
            logger.warning(f"Failed to read full CSV for row count: {e}")
            # Try with the original approach
            total_rows = dataset.get('row_count', 0)

        # Calculate row counts from ratios
        train_rows = int(total_rows * train_ratio)
        val_rows = int(total_rows * val_ratio)
        test_rows = int(total_rows * test_ratio)

        # Create output directory
        split_id = generate_id()
        output_dir = UPLOAD_DIR / "splits" / split_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Call split_data.py script with row counts
        cmd = [
            _get_python_executable(),
            str(REPO_ROOT / "scripts" / "split_data.py"),
            "--input", str(input_csv),
            "--train-rows", str(train_rows),
            "--val-rows", str(val_rows),
            "--test-rows", str(test_rows),
            "--output-dir", str(output_dir),
            "--seed", str(seed)
        ]
        if shuffle:
            cmd.append("--shuffle")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            # Find the output files
            train_csv = output_dir / "train.csv"
            val_csv = output_dir / "val.csv"
            test_csv = output_dir / "test.csv"

            return create_response(True, {
                "splitId": split_id,
                "datasetId": dataset_id,
                "trainPath": str(train_csv) if train_csv.exists() else None,
                "valPath": str(val_csv) if val_csv.exists() else None,
                "testPath": str(test_csv) if test_csv.exists() else None,
                "message": "Data split completed successfully"
            })
        else:
            return create_response(False, error={
                "code": "INTERNAL_ERROR",
                "message": result.stderr
            }), 500

    except Exception as e:
        logger.error(f"Split data error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


# ==================== Training ====================

def create_training_config(config: Dict) -> Path:
    """Create a TOML config file from training configuration."""
    # Normalize model type to uppercase
    raw_model_type = config.get('modelType', 'LSTM')
    model_type = raw_model_type.upper()
    # Frontend sends modelParams; backend may also receive modelConfig
    model_config = config.get('modelParams', config.get('modelConfig', {}))
    # Frontend sends trainingParams; backend may also receive trainingConfig
    training_config = config.get('trainingParams', config.get('trainingConfig', {}))
    data_config = config.get('dataConfig', {})

    # Build TOML content
    toml_content = f"""[model]
type = "{model_type}"
name = "{raw_model_type.lower()}_regressor"
"""

    # Add model-specific params
    if model_type == 'MLP':
        toml_content += f"""
hidden_layers = {model_config.get('mid_layer_count', 3) * [model_config.get('mid_layer_size', 256)]}
dropout = {model_config.get('dropout', 0.15)}
"""
    elif model_type in ['LSTM', 'RNN', 'GRU']:
        toml_content += f"""
history_length = {model_config.get('history_length', 64)}
units = {model_config.get('units', 192)}
num_layers = {model_config.get('num_layers', 8)}
dropout = {model_config.get('dropout', 0.35)}
"""
    elif model_type == 'TRANSFORMER':
        toml_content += f"""
history_length = {model_config.get('history_length', 64)}
d_model = {model_config.get('d_model', 128)}
nhead = {model_config.get('nhead', 8)}
num_encoder_layers = {model_config.get('num_encoder_layers', 4)}
dim_feedforward = {model_config.get('dim_feedforward', 512)}
dropout = {model_config.get('dropout', 0.1)}
"""
    elif model_type in ['XGBOOST', 'LIGHTGBM', 'CATBOOST']:
        toml_content += f"""
max_depth = {model_config.get('max_depth', 8)}
learning_rate = {model_config.get('learning_rate', 0.05)}
subsample = {model_config.get('subsample', 0.9)}
colsample_bytree = {model_config.get('colsample_bytree', 0.8)}
gamma = {model_config.get('gamma', 0.0)}
reg_lambda = {model_config.get('reg_lambda', 1.0)}
min_child_weight = {model_config.get('min_child_weight', 1.0)}
"""

    # Training config
    toml_content += f"""
[training]
max_epochs = {training_config.get('max_epochs', training_config.get('maxEpochs', 100))}
batch_size = {training_config.get('batch_size', training_config.get('batchSize', 128))}
learning_rate = {training_config.get('learning_rate', training_config.get('learningRate', 0.001))}
weight_decay = {training_config.get('weight_decay', training_config.get('weightDecay', 0.0))}
patience = {training_config.get('patience', 50)}
seed = {training_config.get('seed', 42)}
"""

    # Data config
    input_cols = data_config.get('inputColumns', [])
    output_cols = data_config.get('outputColumns', [])

    toml_content += f"""
[data]
train_csv = "{data_config.get('trainPath', data_config.get('trainCsv', ''))}"
val_csv = "{data_config.get('valPath', data_config.get('valCsv', ''))}"
test_csv = "{data_config.get('testPath', data_config.get('testCsv', ''))}"
input_columns = {json.dumps(input_cols)}
output_columns = {json.dumps(output_cols)}
"""

    # Save config file
    config_id = generate_id()
    config_path = CONFIG_DIR / f"{config_id}.toml"

    with open(config_path, 'w') as f:
        f.write(toml_content)

    return config_path


@app.route('/api/v1/train', methods=['POST'])
def start_training():
    """Start a training job."""
    try:
        data = request.get_json()
        if not data:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "No JSON data provided"
            }), 400

        # Validate required fields (accept both snake_case and camelCase)
        required_fields = ['datasetId', 'modelType', 'dataConfig']
        has_model_config = 'modelConfig' in data or 'modelParams' in data
        has_training_config = 'trainingConfig' in data or 'trainingParams' in data

        for field in required_fields:
            if field not in data:
                return create_response(False, error={
                    "code": "VALIDATION_ERROR",
                    "message": f"Missing required field: {field}"
                }), 400

        if not has_model_config:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "Missing required field: modelConfig or modelParams"
            }), 400

        if not has_training_config:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "Missing required field: trainingConfig or trainingParams"
            }), 400

        # Create TOML config file
        try:
            config_path = create_training_config(data)
        except Exception as e:
            logger.error(f"Failed to create config: {e}")
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": f"Failed to create training config: {str(e)}"
            }), 400

        # Pre-compute model output directory so loss-history is available during training
        raw_model_type = data.get('modelType', 'LSTM')
        model_name = f"{raw_model_type.lower()}_regressor"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = str(REPO_ROOT / "outputs" / model_name / timestamp)

        # Create job
        job_id = generate_id()
        with job_lock:
            jobs[job_id] = {
                "jobId": job_id,
                "type": "train",
                "status": "pending",
                "progress": 0.0,
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat(),
                "config": data,
                "configPath": str(config_path),
                "result": None,
                "modelDir": model_dir,
                "error": None,
                "logs": []
            }

        # Start training in background thread
        thread = threading.Thread(target=run_training_job, args=(job_id, config_path))
        thread.daemon = True
        thread.start()

        return create_response(True, {
            "jobId": job_id,
            "status": "pending",
            "message": "Training job started"
        }), 201

    except Exception as e:
        logger.error(f"Start training error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


def _get_python_executable():
    """Get the correct Python executable, preferring uv venv."""
    venv_path = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_path.exists():
        return str(venv_path)
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if conda_prefix:
        conda_python = Path(conda_prefix) / "bin" / "python"
        if conda_python.exists():
            return str(conda_python)
    return sys.executable


def run_training_job(job_id: str, config_path: Path):
    """Run training job in background thread."""
    try:
        with job_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "running"
            jobs[job_id]["updatedAt"] = datetime.now().isoformat()
            jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] Starting training job...")

        python_exec = _get_python_executable()
        logger.info(f"Using Python: {python_exec}")

        # Run the actual training script, passing the pre-computed output directory
        # so loss_history.csv is written to the expected path
        with job_lock:
            model_dir = jobs[job_id].get("modelDir") if job_id in jobs else None

        cmd = [
            python_exec,
            str(REPO_ROOT / "scripts" / "train.py"),
            "--config", str(config_path)
        ]
        if model_dir:
            cmd.extend(["--output-dir", model_dir])
        logger.info(f"Training command: {' '.join(cmd)}")

        # Use Popen to stream output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output to job logs
        for line in process.stdout:
            line = line.strip()
            if line:
                with job_lock:
                    if job_id in jobs:
                        jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] {line}")
                        # Parse progress from output if available
                        if "Epoch" in line and "/" in line:
                            try:
                                parts = line.split("Epoch")[1].split("/")
                                current = int(parts[0].strip())
                                total = int(parts[1].split()[0].strip())
                                jobs[job_id]["progress"] = (current / total) * 100
                            except:
                                pass

        process.wait()

        if process.returncode == 0:
            with job_lock:
                if job_id in jobs:
                    # Find model output directory from logs
                    model_dir = None
                    for log_line in jobs[job_id].get("logs", []):
                        if "Output saved to:" in log_line:
                            model_dir = log_line.split("Output saved to:")[-1].strip()
                            break

                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["progress"] = 100.0
                    jobs[job_id]["updatedAt"] = datetime.now().isoformat()
                    result = {
                        "message": "Training completed successfully",
                        "configPath": str(config_path)
                    }
                    if model_dir:
                        result["modelDir"] = model_dir
                    elif jobs[job_id].get("modelDir"):
                        result["modelDir"] = jobs[job_id]["modelDir"]
                    jobs[job_id]["result"] = result
                    jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] Training completed successfully!")
        else:
            raise Exception(f"Training script exited with code {process.returncode}")

    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        with job_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["updatedAt"] = datetime.now().isoformat()
                jobs[job_id]["error"] = str(e)
                jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] ERROR: {str(e)}")


@app.route('/api/v1/train/<job_id>/status', methods=['GET'])
def get_training_status(job_id: str):
    """Get the status of a training job."""
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        return create_response(True, {
            "jobId": job["jobId"],
            "type": job["type"],
            "status": job["status"],
            "progress": job["progress"],
            "createdAt": job["createdAt"],
            "updatedAt": job["updatedAt"],
            "result": job["result"],
            "error": job["error"],
            "logs": job.get("logs", [])[-100:]  # Return last 100 logs
        })


@app.route('/api/v1/train/<job_id>/stop', methods=['POST'])
def stop_training(job_id: str):
    """Stop a running training job."""
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        if job["status"] not in ["pending", "running"]:
            return create_response(False, error={
                "code": "CONFLICT",
                "message": f"Job {job_id} is not running"
            }), 409

        job["status"] = "stopped"
        job["updatedAt"] = datetime.now().isoformat()

        return create_response(True, {
            "jobId": job_id,
            "status": "stopped",
            "message": "Job stopped"
        })


# ==================== Tuning (Bayesian Optimization) ====================

@app.route('/api/v1/tune', methods=['POST'])
def start_tuning():
    """Start a hyperparameter tuning job using Bayesian optimization."""
    try:
        data = request.get_json()
        if not data:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "No JSON data provided"
            }), 400

        # Validate required fields
        has_model = 'modelConfig' in data or 'modelParams' in data
        has_training = 'trainingConfig' in data or 'trainingParams' in data
        has_search = 'searchSpace' in data or 'tuningConfig' in data

        if 'datasetId' not in data:
            return create_response(False, error={"code": "VALIDATION_ERROR", "message": "Missing required field: datasetId"}), 400
        if 'modelType' not in data:
            return create_response(False, error={"code": "VALIDATION_ERROR", "message": "Missing required field: modelType"}), 400
        if not has_model:
            return create_response(False, error={"code": "VALIDATION_ERROR", "message": "Missing required field: modelConfig or modelParams"}), 400
        if not has_training:
            return create_response(False, error={"code": "VALIDATION_ERROR", "message": "Missing required field: trainingConfig or trainingParams"}), 400
        if not has_search:
            return create_response(False, error={"code": "VALIDATION_ERROR", "message": "Missing required field: searchSpace or tuningConfig"}), 400
        if 'dataConfig' not in data:
            return create_response(False, error={"code": "VALIDATION_ERROR", "message": "Missing required field: dataConfig"}), 400

        # Create job
        job_id = generate_id()
        with job_lock:
            jobs[job_id] = {
                "jobId": job_id,
                "type": "tune",
                "status": "pending",
                "progress": 0.0,
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat(),
                "config": data,
                "result": None,
                "error": None,
                "logs": []
            }

        # Start tuning in background thread
        thread = threading.Thread(target=run_tuning_job, args=(job_id, data))
        thread.daemon = True
        thread.start()

        return create_response(True, {
            "jobId": job_id,
            "status": "pending",
            "message": "Hyperparameter tuning started"
        }), 201

    except Exception as e:
        logger.error(f"Start tuning error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


# Default Bayesian search spaces by model type
DEFAULT_BAYES_CONFIGS = {
    "LSTM": {
        "history_length": {"min": 32, "max": 128, "step": 8, "log": False},
        "units": {"min": 64, "max": 256, "step": 8, "log": False},
        "num_layers": {"min": 1, "max": 4, "step": 1, "log": False},
        "dropout": {"min": 0.1, "max": 0.5, "log": True},
        "batch_size": {"min": 64, "max": 256, "step": 16, "log": False},
        "learning_rate": {"min": 0.0002, "max": 0.002, "log": True},
        "weight_decay": {"min": 0.0001, "max": 0.01, "log": True},
    },
    "RNN": {
        "history_length": {"min": 32, "max": 128, "step": 8, "log": False},
        "units": {"min": 64, "max": 256, "step": 8, "log": False},
        "num_layers": {"min": 1, "max": 4, "step": 1, "log": False},
        "dropout": {"min": 0.1, "max": 0.5, "log": True},
        "batch_size": {"min": 64, "max": 256, "step": 16, "log": False},
        "learning_rate": {"min": 0.0002, "max": 0.002, "log": True},
        "weight_decay": {"min": 0.0001, "max": 0.01, "log": True},
    },
    "GRU": {
        "history_length": {"min": 32, "max": 128, "step": 8, "log": False},
        "units": {"min": 64, "max": 256, "step": 8, "log": False},
        "num_layers": {"min": 1, "max": 4, "step": 1, "log": False},
        "dropout": {"min": 0.1, "max": 0.5, "log": True},
        "batch_size": {"min": 64, "max": 256, "step": 16, "log": False},
        "learning_rate": {"min": 0.0002, "max": 0.002, "log": True},
        "weight_decay": {"min": 0.0001, "max": 0.01, "log": True},
    },
    "TRANSFORMER": {
        "history_length": {"min": 32, "max": 128, "step": 8, "log": False},
        "d_model": {"min": 64, "max": 256, "step": 8, "log": False},
        "nhead": {"min": 2, "max": 8, "step": 2, "log": False},
        "num_encoder_layers": {"min": 2, "max": 6, "step": 1, "log": False},
        "dim_feedforward": {"min": 128, "max": 512, "step": 16, "log": False},
        "dropout": {"min": 0.1, "max": 0.5, "log": True},
        "batch_size": {"min": 64, "max": 256, "step": 16, "log": False},
        "learning_rate": {"min": 0.0002, "max": 0.002, "log": True},
        "weight_decay": {"min": 0.0001, "max": 0.01, "log": True},
    },
    "MLP": {
        "mid_layer_count": {"min": 1, "max": 5, "step": 1, "log": False},
        "mid_layer_size": {"min": 64, "max": 512, "step": 16, "log": False},
        "dropout": {"min": 0.1, "max": 0.5, "log": True},
        "batch_size": {"min": 64, "max": 256, "step": 16, "log": False},
        "learning_rate": {"min": 0.0002, "max": 0.002, "log": True},
        "weight_decay": {"min": 0.0001, "max": 0.01, "log": True},
    },
    "XGBOOST": {
        "max_depth": {"min": 4, "max": 12, "step": 1, "log": False},
        "learning_rate": {"min": 0.01, "max": 0.3, "log": True},
        "n_estimators": {"min": 50, "max": 300, "step": 10, "log": False},
        "subsample": {"min": 0.6, "max": 1.0, "log": False},
        "colsample_bytree": {"min": 0.6, "max": 1.0, "log": False},
    },
    "LIGHTGBM": {
        "max_depth": {"min": 4, "max": 12, "step": 1, "log": False},
        "learning_rate": {"min": 0.01, "max": 0.3, "log": True},
        "n_estimators": {"min": 50, "max": 300, "step": 10, "log": False},
        "subsample": {"min": 0.6, "max": 1.0, "log": False},
        "colsample_bytree": {"min": 0.6, "max": 1.0, "log": False},
    },
    "CATBOOST": {
        "max_depth": {"min": 4, "max": 12, "step": 1, "log": False},
        "learning_rate": {"min": 0.01, "max": 0.3, "log": True},
        "n_estimators": {"min": 50, "max": 300, "step": 10, "log": False},
        "subsample": {"min": 0.6, "max": 1.0, "log": False},
        "colsample_bylevel": {"min": 0.6, "max": 1.0, "log": False},
    },
}


def create_bayes_config(search_space: Dict, model_type: str) -> Path:
    """Create a Bayesian search space config TOML file."""
    # Use provided search space or fall back to defaults
    params = search_space if search_space else DEFAULT_BAYES_CONFIGS.get(model_type, {})
    if not params:
        # Generic fallback
        params = {
            "learning_rate": {"min": 0.0001, "max": 0.01, "log": True},
            "batch_size": {"min": 64, "max": 256, "step": 16, "log": False},
        }

    bayes_content = {}
    for param_name, spec in params.items():
        bayes_content[param_name] = spec

    config_id = generate_id()
    config_path = CONFIG_DIR / f"{config_id}_bayes.toml"

    if tomli_w:
        with open(config_path, "wb") as f:
            tomli_w.dump({"parameters": bayes_content}, f)
    else:
        # Fallback: write TOML manually
        with open(config_path, "w") as f:
            f.write("[parameters]\n")
            for param_name, spec in params.items():
                f.write(f"\n[parameters.{param_name}]\n")
                for key, val in spec.items():
                    if isinstance(val, bool):
                        f.write(f"{key} = {str(val).lower()}\n")
                    elif isinstance(val, (int, float)):
                        f.write(f"{key} = {val}\n")
                    else:
                        f.write(f'{key} = "{val}"\n')

    return config_path


def run_tuning_job(job_id: str, config: Dict):
    """Run tuning job in background thread using autotune.py."""
    try:
        with job_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "running"
            jobs[job_id]["updatedAt"] = datetime.now().isoformat()
            jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] Starting Bayesian optimization...")

        # Get configuration
        model_type = config.get('modelType', 'LSTM').upper()
        search_space = config.get('searchSpace', config.get('tuningConfig', {}))
        tuning_config = config.get('tuningConfig', config.get('searchSpace', {}))
        n_trials = tuning_config.get('trials', 20)

        # Create base config (same as training)
        base_config_path = create_training_config(config)

        # Create bayes config
        bayes_config_path = create_bayes_config(search_space, model_type)

        # Pre-compute output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(REPO_ROOT / "outputs" / f"{model_type.lower()}_autotune" / timestamp)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Store modelDir in job for loss-history endpoint
        with job_lock:
            if job_id in jobs:
                jobs[job_id]["modelDir"] = output_dir

        # Run autotune.py
        python_exec = _get_python_executable()
        cmd = [
            python_exec,
            str(REPO_ROOT / "scripts" / "autotune.py"),
            "--model-type", model_type,
            "--base-config", str(base_config_path),
            "--bayes-config", str(bayes_config_path),
            "--n-trials", str(n_trials),
            "--output-dir", output_dir,
        ]

        logger.info(f"Starting autotune: {' '.join(cmd)}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output to job logs
        for line in process.stdout:
            line = line.strip()
            if line:
                with job_lock:
                    if job_id in jobs:
                        jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] {line}")
                        # Parse progress from trial completion
                        if "Trial" in line and "completed" in line:
                            try:
                                trial_num = int(line.split("Trial")[1].split()[0])
                                jobs[job_id]["progress"] = (trial_num / n_trials) * 100
                            except:
                                pass

        process.wait()

        if process.returncode == 0:
            # Parse results
            trials = []
            best_params = {}
            best_value = None
            best_model_dir = None

            # Read optimization results CSV
            results_csv = Path(output_dir) / "bayes_optimization_results.csv"
            if results_csv.exists():
                import csv
                with open(results_csv, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trial_num = int(row.get("trial_number", 0))
                        val_loss = float(row.get("value", 0))
                        trial_params = {k: float(v) if v != "" else v
                                        for k, v in row.items()
                                        if k not in ("trial_number", "value") and v != ""}
                        trials.append({
                            "id": trial_num,
                            "valLoss": val_loss,
                            "params": trial_params
                        })

                if trials:
                    best_trial = min(trials, key=lambda t: t["valLoss"])
                    best_params = best_trial["params"]
                    best_value = best_trial["valLoss"]

                    # Find best model directory from trial_info.toml
                    best_trial_dir = None
                    for d in Path(output_dir).iterdir():
                        if d.is_dir() and d.name.startswith("trial_"):
                            trial_info = d / "trial_info.toml"
                            if trial_info.exists():
                                import tomli
                                with open(trial_info, "rb") as f:
                                    info = tomli.load(f)
                                # Check if this is the best trial
                                trial_num = int(d.name.split("_")[1])
                                if trial_num == best_trial["id"]:
                                    best_model_dir = info.get("model_dir")
                                    break

            with job_lock:
                if job_id in jobs:
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["progress"] = 100.0
                    jobs[job_id]["updatedAt"] = datetime.now().isoformat()
                    result = {
                        "message": "Hyperparameter tuning completed successfully",
                        "bestParams": best_params,
                        "bestValue": best_value,
                        "trials": trials,
                    }
                    if best_model_dir:
                        result["modelDir"] = best_model_dir
                    elif output_dir:
                        result["modelDir"] = output_dir
                    jobs[job_id]["result"] = result
                    jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] Tuning completed! Best val loss: {best_value}")
        else:
            raise Exception(f"autotune.py exited with code {process.returncode}")

    except Exception as e:
        logger.error(f"Tuning job {job_id} failed: {e}")
        with job_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["updatedAt"] = datetime.now().isoformat()
                jobs[job_id]["error"] = str(e)
                jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] ERROR: {str(e)}")


@app.route('/api/v1/tune/<job_id>/status', methods=['GET'])
def get_tuning_status(job_id: str):
    """Get the status of a tuning job."""
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        return create_response(True, {
            "jobId": job["jobId"],
            "type": job["type"],
            "status": job["status"],
            "progress": job["progress"],
            "createdAt": job["createdAt"],
            "updatedAt": job["updatedAt"],
            "result": job["result"],
            "error": job["error"],
            "logs": job.get("logs", [])[-100:]
        })


@app.route('/api/v1/tune/<job_id>/stop', methods=['POST'])
def stop_tuning(job_id: str):
    """Stop a running tuning job."""
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        if job["status"] not in ["pending", "running"]:
            return create_response(False, error={
                "code": "CONFLICT",
                "message": f"Job {job_id} is not running"
            }), 409

        job["status"] = "stopped"
        job["updatedAt"] = datetime.now().isoformat()

        return create_response(True, {
            "jobId": job_id,
            "status": "stopped",
            "message": "Job stopped"
        })


# ==================== Testing ====================

@app.route('/api/v1/test', methods=['POST'])
def start_testing():
    """Start a testing job."""
    try:
        data = request.get_json()
        if not data:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "No JSON data provided"
            }), 400

        # Create job
        job_id = generate_id()
        with job_lock:
            jobs[job_id] = {
                "jobId": job_id,
                "type": "test",
                "status": "pending",
                "progress": 0.0,
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat(),
                "config": data,
                "result": None,
                "error": None,
                "logs": []
            }

        # Start testing in background thread
        thread = threading.Thread(target=run_testing_job, args=(job_id, data))
        thread.daemon = True
        thread.start()

        return create_response(True, {
            "jobId": job_id,
            "status": "pending",
            "message": "Testing job started"
        }), 201

    except Exception as e:
        logger.error(f"Start testing error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


def run_testing_job(job_id: str, config: Dict):
    """Run testing job in background thread."""
    try:
        with job_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "running"
            jobs[job_id]["updatedAt"] = datetime.now().isoformat()
            jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] Starting testing...")

        # Get model directory
        model_dir = config.get('modelDir')
        test_csv = config.get('testCsv')

        if not model_dir:
            raise ValueError("Model directory not specified")

        # Call test.py script
        cmd = [
            _get_python_executable(),
            str(REPO_ROOT / "scripts" / "test.py"),
            "--model-dir", str(model_dir)
        ]
        if test_csv:
            cmd.extend(["--test-csv", str(test_csv)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        with job_lock:
            if job_id in jobs:
                if result.returncode == 0:
                    # Parse metrics from test output
                    metrics = {}
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        if ':' not in line:
                            continue
                        parts = line.split(':')
                        key = parts[0].strip().lower()
                        try:
                            value = float(parts[-1].strip())
                            if key == 'mse':
                                metrics['mse'] = value
                            elif key == 'rmse':
                                metrics['rmse'] = value
                            elif key == 'mae':
                                metrics['mae'] = value
                            elif key == 'r2':
                                metrics['r2'] = value
                        except (ValueError, IndexError):
                            pass

                    test_result = {
                        "message": "Testing completed successfully",
                        "output": result.stdout
                    }
                    if metrics:
                        test_result["metrics"] = metrics
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["progress"] = 100.0
                    jobs[job_id]["result"] = test_result
                    jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] Testing completed!")
                else:
                    raise Exception(result.stderr)

    except Exception as e:
        logger.error(f"Testing job {job_id} failed: {e}")
        with job_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
                jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] ERROR: {str(e)}")


@app.route('/api/v1/test/<job_id>/status', methods=['GET'])
def get_test_status(job_id: str):
    """Get the status of a testing job."""
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        return create_response(True, {
            "jobId": job["jobId"],
            "type": job["type"],
            "status": job["status"],
            "progress": job["progress"],
            "createdAt": job["createdAt"],
            "updatedAt": job["updatedAt"],
            "result": job["result"],
            "error": job["error"],
            "logs": job.get("logs", [])[-100:]
        })


# ==================== Prediction ====================

@app.route('/api/v1/predict', methods=['POST'])
def start_prediction():
    """Start a prediction job."""
    try:
        data = request.get_json()
        if not data:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "No JSON data provided"
            }), 400

        # Create job
        job_id = generate_id()
        with job_lock:
            jobs[job_id] = {
                "jobId": job_id,
                "type": "predict",
                "status": "pending",
                "progress": 0.0,
                "createdAt": datetime.now().isoformat(),
                "updatedAt": datetime.now().isoformat(),
                "config": data,
                "result": None,
                "error": None,
                "logs": []
            }

        # Start prediction in background thread
        thread = threading.Thread(target=run_prediction_job, args=(job_id, data))
        thread.daemon = True
        thread.start()

        return create_response(True, {
            "jobId": job_id,
            "status": "pending",
            "message": "Prediction job started"
        }), 201

    except Exception as e:
        logger.error(f"Start prediction error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


def run_prediction_job(job_id: str, config: Dict):
    """Run prediction job in background thread."""
    try:
        with job_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "running"
            jobs[job_id]["updatedAt"] = datetime.now().isoformat()
            jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] Starting prediction...")

        # Get model directory
        model_dir = config.get('modelDir')
        dataset_id = config.get('datasetId')

        if not model_dir:
            raise ValueError("Model directory not specified")

        # Find test CSV from dataset if provided
        test_csv = None
        if dataset_id and dataset_id in uploaded_datasets:
            test_csv = uploaded_datasets[dataset_id].get('path')

        # Call test.py script for prediction
        cmd = [
            _get_python_executable(),
            str(REPO_ROOT / "scripts" / "test.py"),
            "--model-dir", str(model_dir)
        ]
        if test_csv:
            cmd.extend(["--test-csv", str(test_csv)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        with job_lock:
            if job_id in jobs:
                if result.returncode == 0:
                    # Parse metrics from test output
                    metrics = {}
                    for line in result.stdout.split('\n'):
                        line = line.strip()
                        if ':' not in line:
                            continue
                        parts = line.split(':')
                        key = parts[0].strip().lower()
                        try:
                            value = float(parts[-1].strip())
                            if key == 'mse':
                                metrics['mse'] = value
                            elif key == 'rmse':
                                metrics['rmse'] = value
                            elif key == 'mae':
                                metrics['mae'] = value
                            elif key == 'r2':
                                metrics['r2'] = value
                        except (ValueError, IndexError):
                            pass

                    pred_result = {
                        "message": "Prediction completed successfully",
                        "output": result.stdout
                    }
                    if metrics:
                        pred_result["metrics"] = metrics
                    jobs[job_id]["status"] = "completed"
                    jobs[job_id]["progress"] = 100.0
                    jobs[job_id]["result"] = pred_result
                    jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] Prediction completed!")
                else:
                    raise Exception(result.stderr)

    except Exception as e:
        logger.error(f"Prediction job {job_id} failed: {e}")
        with job_lock:
            if job_id in jobs:
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["error"] = str(e)
                jobs[job_id]["logs"].append(f"[{datetime.now().isoformat()}] ERROR: {str(e)}")


@app.route('/api/v1/predict/<job_id>/status', methods=['GET'])
def get_predict_status(job_id: str):
    """Get the status of a prediction job."""
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        return create_response(True, {
            "jobId": job["jobId"],
            "type": job["type"],
            "status": job["status"],
            "progress": job["progress"],
            "createdAt": job["createdAt"],
            "updatedAt": job["updatedAt"],
            "result": job["result"],
            "error": job["error"],
            "logs": job.get("logs", [])[-100:]
        })


# ==================== Training Output Files ====================

@app.route('/api/v1/output/<path:file_path>', methods=['GET'])
def get_output_file(file_path: str):
    """Serve a file from the training output directory."""
    full_path = OUTPUT_DIR / file_path
    if not full_path.exists():
        return create_response(False, error={
            "code": "NOT_FOUND",
            "message": f"File not found: {file_path}"
        }), 404

    if full_path.is_dir():
        return create_response(False, error={
            "code": "BAD_REQUEST",
            "message": "Path is a directory, not a file"
        }), 400

    return send_file(str(full_path))


@app.route('/api/v1/jobs/<job_id>/loss-history', methods=['GET'])
def get_loss_history(job_id: str):
    """Get loss history CSV data for a training job.

    For PyTorch models, reads ``loss_history.csv``.
    For GBDT models (XGBoost, LightGBM, CatBoost), merges per-target
    files ``loss_history_{target}.csv`` into a single dataset.
    """
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        # Prefer result.modelDir (actual output from training/autotune), then pre-computed modelDir
        result = job.get("result") or {}
        model_dir = result.get("modelDir") or job.get("modelDir")

    if not model_dir:
        return create_response(False, error={
            "code": "NOT_FOUND",
            "message": "Model directory not available yet"
        }), 404

    model_path = Path(model_dir)
    loss_file = model_path / "loss_history.csv"

    try:
        import polars as pl

        if loss_file.exists():
            # Standard PyTorch loss history
            df = pl.read_csv(loss_file)
        else:
            # GBDT models save per-target: loss_history_{target}.csv
            target_files = sorted(model_path.glob("loss_history_*.csv"))
            if not target_files:
                return create_response(False, error={
                    "code": "NOT_FOUND",
                    "message": "loss_history.csv not found"
                }), 404

            # Merge per-target files: average train_loss and val_loss across targets
            dfs = []
            for tf in target_files:
                target_name = tf.stem.replace("loss_history_", "")
                tdf = pl.read_csv(tf).rename({
                    "train_loss": f"train_loss_{target_name}",
                    "val_loss": f"val_loss_{target_name}",
                })
                dfs.append(tdf)

            # Join on epoch column
            merged = dfs[0]
            for tdf in dfs[1:]:
                merged = merged.join(tdf, on="epoch", how="full")

            # Compute average train_loss and val_loss across targets
            train_cols = [c for c in merged.columns if c.startswith("train_loss_")]
            val_cols = [c for c in merged.columns if c.startswith("val_loss_")]
            merged = merged.with_columns([
                pl.mean_horizontal(train_cols).alias("train_loss"),
                pl.mean_horizontal(val_cols).alias("val_loss"),
            ])
            df = merged.select(["epoch", "train_loss", "val_loss"])

        rows = df.to_dicts()
        for row in rows:
            for key, val in row.items():
                if hasattr(val, 'item'):
                    row[key] = val.item()
                elif val is None:
                    row[key] = None
        return create_response(True, {
            "lossHistory": rows,
            "columns": df.columns
        })
    except Exception as e:
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.route('/api/v1/jobs/<job_id>/test-comparison', methods=['GET'])
def get_test_comparison(job_id: str):
    """Get test comparison data (true vs predicted) for a completed test job."""
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        # Check multiple sources for modelDir (prefer actual result, then config, then pre-computed)
        result = job.get("result") or {}
        config = job.get("config") or {}
        model_dir = result.get("modelDir") or config.get("modelDir") or job.get("modelDir")

    if not model_dir:
        # Try getting from any completed training job
        with job_lock:
            for j in jobs.values():
                if j.get("type") == "train" and j.get("status") == "completed":
                    model_dir = (j.get("result") or {}).get("modelDir") or j.get("modelDir")
                    if model_dir:
                        break

    if not model_dir:
        return create_response(False, error={
            "code": "NOT_FOUND",
            "message": "Model directory not available"
        }), 404

    comparison_file = Path(model_dir) / "test_comparison.csv"
    if not comparison_file.exists():
        return create_response(False, error={
            "code": "NOT_FOUND",
            "message": "test_comparison.csv not found"
        }), 404

    try:
        import polars as pl
        df = pl.read_csv(comparison_file)

        # Detect output columns from column names (pattern: X_true, X_pred)
        output_columns = []
        for col in df.columns:
            if col.endswith("_true"):
                output_columns.append(col[:-5])

        rows = df.to_dicts()
        for row in rows:
            for key, val in row.items():
                if hasattr(val, 'item'):
                    row[key] = val.item()
                elif val is None:
                    row[key] = None

        return create_response(True, {
            "comparison": rows,
            "outputColumns": output_columns,
            "columns": df.columns
        })
    except Exception as e:
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


# ==================== Model Management ====================

@app.route('/api/v1/models', methods=['GET'])
def list_models():
    """List all saved models."""
    try:
        # Scan outputs directory for trained models
        models_list = []
        if OUTPUT_DIR.exists():
            for model_dir in OUTPUT_DIR.iterdir():
                if model_dir.is_dir():
                    for timestamp_dir in model_dir.iterdir():
                        if timestamp_dir.is_dir():
                            # Check for result.toml or config.toml
                            result_file = timestamp_dir / "result.toml"
                            config_file = timestamp_dir / "config.toml"

                            model_info = {
                                "id": f"{model_dir.name}/{timestamp_dir.name}",
                                "name": model_dir.name,
                                "type": "unknown",
                                "createdAt": datetime.fromtimestamp(timestamp_dir.stat().st_mtime).isoformat(),
                                "size": sum(f.stat().st_size for f in timestamp_dir.rglob('*') if f.is_file()),
                                "path": str(timestamp_dir),
                                "metrics": {}
                            }

                            # Parse config if available
                            if config_file.exists():
                                try:
                                    with open(config_file, 'rb') as f:
                                        config = tomli.load(f)
                                        if 'model' in config:
                                            model_info['type'] = config['model'].get('type', 'unknown')
                                except:
                                    pass

                            models_list.append(model_info)

        return create_response(True, {
            "models": models_list,
            "total": len(models_list)
        })

    except Exception as e:
        logger.error(f"List models error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.route('/api/v1/models/<model_id>', methods=['GET'])
def get_model(model_id: str):
    """Get model details."""
    try:
        # Decode model_id (it might contain /)
        model_path = OUTPUT_DIR / model_id.replace('/', os.sep)

        if not model_path.exists():
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Model {model_id} not found"
            }), 404

        # Load result and config
        result_file = model_path / "result.toml"
        config_file = model_path / "config.toml"

        model_info = {
            "id": model_id,
            "path": str(model_path),
            "createdAt": datetime.fromtimestamp(model_path.stat().st_mtime).isoformat(),
            "files": [f.name for f in model_path.iterdir() if f.is_file()]
        }

        if config_file.exists():
            try:
                with open(config_file, 'rb') as f:
                    config = tomli.load(f)
                    model_info['config'] = config
            except Exception as e:
                logger.warning(f"Failed to parse config: {e}")

        if result_file.exists():
            try:
                with open(result_file, 'rb') as f:
                    result = tomli.load(f)
                    model_info['result'] = result
            except Exception as e:
                logger.warning(f"Failed to parse result: {e}")

        return create_response(True, model_info)

    except Exception as e:
        logger.error(f"Get model error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.route('/api/v1/models/<model_id>', methods=['DELETE'])
def delete_model(model_id: str):
    """Delete a model."""
    try:
        model_path = OUTPUT_DIR / model_id.replace('/', os.sep)

        if not model_path.exists():
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Model {model_id} not found"
            }), 404

        # Delete directory
        shutil.rmtree(model_path)

        return create_response(True, {"message": "Model deleted successfully"})

    except Exception as e:
        logger.error(f"Delete model error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.route('/api/v1/models/<model_id>/download', methods=['GET'])
def download_model(model_id: str):
    """Download a model as a zip file."""
    try:
        model_path = OUTPUT_DIR / model_id.replace('/', os.sep)

        if not model_path.exists():
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Model {model_id} not found"
            }), 404

        # Create zip file
        import zipfile
        zip_path = REPO_ROOT / "temp" / f"{model_id.replace('/', '_')}.zip"
        zip_path.parent.mkdir(exist_ok=True)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in model_path.rglob('*'):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(model_path))

        return send_file(zip_path, as_attachment=True, download_name=f"{model_id.replace('/', '_')}.zip")

    except Exception as e:
        logger.error(f"Download model error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


# ==================== Configuration ====================

@app.route('/api/v1/config/save', methods=['POST'])
def save_config():
    """Save configuration to TOML file."""
    try:
        data = request.get_json()
        if not data:
            return create_response(False, error={
                "code": "VALIDATION_ERROR",
                "message": "No JSON data provided"
            }), 400

        # Generate config ID
        config_id = generate_id()

        # Save to TOML file
        config_path = CONFIG_DIR / f"{config_id}.toml"

        if tomli_w:
            with open(config_path, "wb") as f:
                tomli_w.dump(data, f)
        else:
            # Fallback: write as JSON
            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)

        return create_response(True, {
            "configId": config_id,
            "message": "Configuration saved successfully"
        }), 201

    except Exception as e:
        logger.error(f"Save config error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


@app.route('/api/v1/config/load/<config_id>', methods=['GET'])
def load_config(config_id: str):
    """Load configuration from TOML file."""
    try:
        config_path = CONFIG_DIR / f"{config_id}.toml"
        if not config_path.exists():
            # Try JSON fallback
            config_path = config_path.with_suffix('.json')
            if not config_path.exists():
                return create_response(False, error={
                    "code": "NOT_FOUND",
                    "message": f"Configuration {config_id} not found"
                }), 404

        with open(config_path, "rb") as f:
            config = tomli.load(f)

        return create_response(True, config)

    except Exception as e:
        logger.error(f"Load config error: {e}")
        return create_response(False, error={
            "code": "INTERNAL_ERROR",
            "message": str(e)
        }), 500


# ==================== Logs ====================

@app.route('/api/v1/jobs/<job_id>/logs', methods=['GET'])
def get_job_logs(job_id: str):
    """Get logs for a job."""
    with job_lock:
        if job_id not in jobs:
            return create_response(False, error={
                "code": "NOT_FOUND",
                "message": f"Job {job_id} not found"
            }), 404

        job = jobs[job_id]
        # Get optional limit parameter
        limit = request.args.get('limit', 1000, type=int)

        return create_response(True, {
            "jobId": job_id,
            "logs": job.get("logs", [])[-limit:]
        })


# ==================== Main Entry Point ====================

def main():
    """Run the Flask server."""
    import argparse
    parser = argparse.ArgumentParser(description="Run the Flask backend API server")
    parser.add_argument("--port", type=int, default=5555, help="Port to run server on (default: 5555)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    ensure_directories()
    logger.info(f"Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
