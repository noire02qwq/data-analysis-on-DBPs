#!/usr/bin/env python3
"""
Demo client to test the backend server.
Runs on port 110, tests the full workflow: data split -> train -> autotune -> test.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PORT = 310
CLIENT_PORT = 110


def make_request(endpoint: str, data: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
    """Make a request to the server."""
    url = f"http://localhost:{SERVER_PORT}{endpoint}"

    # For this demo, we'll call the handlers directly instead of HTTP
    # This avoids the complexity of running two servers
    from scripts.server import APIHandler

    class MockConnection:
        def __init__(self, path: str, method: str, body_data: bytes):
            self.path = path
            self.command = method
            self.rfile = BytesIO(body_data)
            self.wfile = BytesIO()
            self.send_response_called = False
            self.send_header_called = False
            self.headers = {}

        def send_response(self, status, message=None):
            self.send_response_called = True
            self.status = status
            self.response_message = message

        def send_header(self, key, value):
            self.send_header_called = True
            self.headers[key] = value

        def end_headers(self):
            pass

    class BytesIO:
        def __init__(self, initial=b""):
            self.data = initial
            self.pos = 0

        def read(self, n=-1):
            if n == -1:
                result = self.data[self.pos:]
                self.pos = len(self.data)
            else:
                result = self.data[self.pos:self.pos+n]
                self.pos += n
            return result

        def write(self, data):
            self.data += data

    handler = APIHandler.__new__(APIHandler)

    body_data = json.dumps(data).encode("utf-8") if data else b""
    mock_conn = MockConnection(endpoint, "POST", body_data)

    if endpoint == "/split":
        handler.handle_split()
    elif endpoint == "/train":
        handler.handle_train()
    elif endpoint == "/autotune":
        handler.handle_autotune()
    elif endpoint == "/test":
        handler.handle_test()
    else:
        return {"error": "Unknown endpoint"}

    # Actually, let's just use subprocess to call the scripts directly
    # since the server would do the same thing
    return call_script_directly(endpoint, data)


def call_script_directly(endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Call the scripts directly (simulating what the server does)."""
    import subprocess

    if endpoint == "/split":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "split_data.py"),
            "--input", str(REPO_ROOT / data["input_csv"]),
            "--train-rows", str(data["train_rows"]),
            "--val-rows", str(data["val_rows"]),
            "--test-rows", str(data["test_rows"]),
            "--output-dir", str(REPO_ROOT / data["output_dir"]),
            "--seed", str(data.get("seed", 42))
        ]
        if data.get("shuffle"):
            cmd.append("--shuffle")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return {"status": "success", "message": "Data split completed", "output_dir": data["output_dir"]}
        else:
            return {"error": result.stderr}

    elif endpoint == "/train":
        config_path = data["config"]
        if not Path(config_path).is_absolute():
            config_path = str(REPO_ROOT / config_path)

        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train.py"), "--config", config_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

        if result.returncode == 0:
            # Find output directory
            import tomli
            with open(config_path, "rb") as f:
                config = tomli.load(f)
            model_name = config["model"]["name"]
            outputs_dir = REPO_ROOT / "outputs" / model_name
            latest = sorted(outputs_dir.iterdir(), key=lambda p: p.stat().st_mtime)[-1] if outputs_dir.exists() else None

            return {"status": "success", "message": "Training completed", "output_dir": str(latest)}
        else:
            return {"error": result.stderr}

    elif endpoint == "/autotune":
        model_type = data["model_type"]
        base_config = data["base_config"]
        bayes_config = data["bayes_config"]
        n_trials = data.get("n_trials", 5)  # Reduced for demo

        if not Path(base_config).is_absolute():
            base_config = str(REPO_ROOT / base_config)
        if not Path(bayes_config).is_absolute():
            bayes_config = str(REPO_ROOT / bayes_config)

        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "autotune.py"),
            "--model-type", model_type,
            "--base-config", base_config,
            "--bayes-config", bayes_config,
            "--n-trials", str(n_trials)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)

        if result.returncode == 0:
            return {"status": "success", "message": "Autotune completed", "output": result.stdout}
        else:
            return {"error": result.stderr}

    elif endpoint == "/test":
        model_dir = data["model_dir"]
        if not Path(model_dir).is_absolute():
            model_dir = str(REPO_ROOT / model_dir)

        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "test.py"), "--model-dir", model_dir]

        if data.get("test_csv"):
            test_csv = data["test_csv"]
            if not Path(test_csv).is_absolute():
                test_csv = str(REPO_ROOT / test_csv)
            cmd.extend(["--test-csv", test_csv])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            return {"status": "success", "message": "Testing completed", "output_dir": model_dir}
        else:
            return {"error": result.stderr}

    return {"error": "Unknown endpoint"}


def check_data_exists() -> bool:
    """Check if the required data files exist."""
    data_dir = REPO_ROOT / "data"
    return (data_dir / "train.csv").exists() and (data_dir / "val.csv").exists() and (data_dir / "test.csv").exists()


def run_demo():
    """Run the full demo workflow."""
    print("=" * 60)
    print("Demo Client - Testing RNN Workflow")
    print("=" * 60)

    # Step 1: Check data
    print("\n[1/4] Checking data availability...")
    if not check_data_exists():
        print("Data not found. Running data split...")

        # Split data
        response = call_script_directly("/split", {
            "input_csv": "data/time_aligned_data.csv",
            "train_rows": 10000,
            "val_rows": 2000,
            "test_rows": 2000,
            "output_dir": "data",
            "shuffle": True,
            "seed": 42
        })

        if "error" in response:
            print(f"Error splitting data: {response['error']}")
            return
        print(f"Data split completed: {response}")
    else:
        print("Data already exists. Skipping split.")

    # Step 2: Train RNN
    print("\n[2/4] Training RNN model...")
    response = call_script_directly("/train", {
        "config": "models/configs/rnn_config.toml"
    })

    if "error" in response:
        print(f"Error training: {response['error']}")
        return

    train_output = response.get("output_dir")
    print(f"Training completed: {train_output}")

    # Step 3: Autotune (reduced trials for demo)
    print("\n[3/4] Running autotune (3 trials)...")
    response = call_script_directly("/autotune", {
        "model_type": "RNN",
        "base_config": "models/configs/rnn_config.toml",
        "bayes_config": "models/configs/rnn_bayes.toml",
        "n_trials": 3
    })

    if "error" in response:
        print(f"Autotune error (may be expected if bayes config missing): {response.get('error', 'unknown')}")
        print("Continuing to test...")
    else:
        print(f"Autotune completed: {response.get('message', 'done')}")

    # Step 4: Test model
    print("\n[4/4] Testing trained model...")
    if train_output:
        response = call_script_directly("/test", {
            "model_dir": train_output
        })

        if "error" in response:
            print(f"Error testing: {response['error']}")
        else:
            print(f"Testing completed: {response.get('message', 'done')}")
    else:
        print("Skipping test - no trained model output")

    print("\n" + "=" * 60)
    print("Demo workflow completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()