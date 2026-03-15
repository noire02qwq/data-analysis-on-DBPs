#!/usr/bin/env python3
"""
Backend server for data analysis API.
Runs on port 310, handles requests for data splitting, training, autotuning, and testing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional

import tomli

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class APIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the API."""

    def log_message(self, format, *args):
        """Override to customize logging."""
        print(f"[{self.address_string()}] {format % args}")

    def send_json_response(self, status: int, data: Any) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def read_json_body(self) -> Dict[str, Any]:
        """Read JSON body from request."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self.send_json_response(200, {"status": "ok"})
        elif self.path == "/models":
            # List available model configs
            configs = list((REPO_ROOT / "models" / "configs").glob("*_config.toml"))
            model_names = [c.stem.replace("_config", "") for c in configs]
            self.send_json_response(200, {"models": model_names})
        else:
            self.send_json_response(404, {"error": "Not found"})

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/split":
            self.handle_split()
        elif self.path == "/train":
            self.handle_train()
        elif self.path == "/autotune":
            self.handle_autotune()
        elif self.path == "/test":
            self.handle_test()
        else:
            self.send_json_response(404, {"error": "Not found"})

    def handle_split(self) -> None:
        """Handle data splitting request."""
        try:
            body = self.read_json_body()
            input_csv = body.get("input_csv")
            train_rows = body.get("train_rows")
            val_rows = body.get("val_rows")
            test_rows = body.get("test_rows")
            output_dir = body.get("output_dir", "data")
            shuffle = body.get("shuffle", False)
            seed = body.get("seed", 42)

            if not input_csv or train_rows is None or val_rows is None or test_rows is None:
                self.send_json_response(400, {"error": "Missing required parameters"})
                return

            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "split_data.py"),
                "--input", str(REPO_ROOT / input_csv),
                "--train-rows", str(train_rows),
                "--val-rows", str(val_rows),
                "--test-rows", str(test_rows),
                "--output-dir", str(REPO_ROOT / output_dir),
                "--seed", str(seed)
            ]
            if shuffle:
                cmd.append("--shuffle")

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                self.send_json_response(200, {
                    "status": "success",
                    "message": f"Data split into {output_dir}/train.csv, val.csv, test.csv",
                    "output_dir": output_dir
                })
            else:
                self.send_json_response(500, {"error": result.stderr})

        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def handle_train(self) -> None:
        """Handle training request."""
        try:
            body = self.read_json_body()
            config_path = body.get("config")

            if not config_path:
                self.send_json_response(400, {"error": "Missing config parameter"})
                return

            # Resolve config path
            if not Path(config_path).is_absolute():
                config_path = REPO_ROOT / config_path

            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "train.py"),
                "--config", str(config_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                # Find the output directory (most recent in outputs/<model_name>)
                model_name = tomli.loads(config_path.read_text())["model"]["name"]
                outputs_dir = REPO_ROOT / "outputs" / model_name
                if outputs_dir.exists():
                    subdirs = sorted(outputs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
                    latest_output = subdirs[-1] if subdirs else None
                else:
                    latest_output = None

                self.send_json_response(200, {
                    "status": "success",
                    "message": "Training completed",
                    "output_dir": str(latest_output) if latest_output else None
                })
            else:
                self.send_json_response(500, {"error": result.stderr})

        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def handle_autotune(self) -> None:
        """Handle autotune request."""
        try:
            body = self.read_json_body()
            model_type = body.get("model_type")
            base_config = body.get("base_config")
            bayes_config = body.get("bayes_config")
            n_trials = body.get("n_trials", 20)

            if not model_type or not base_config or not bayes_config:
                self.send_json_response(400, {"error": "Missing required parameters"})
                return

            # Resolve paths
            if not Path(base_config).is_absolute():
                base_config = REPO_ROOT / base_config
            if not Path(bayes_config).is_absolute():
                bayes_config = REPO_ROOT / bayes_config

            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "autotune.py"),
                "--model-type", model_type,
                "--base-config", str(base_config),
                "--bayes-config", str(bayes_config),
                "--n-trials", str(n_trials)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)

            if result.returncode == 0:
                self.send_json_response(200, {
                    "status": "success",
                    "message": "Autotune completed",
                    "output": result.stdout
                })
            else:
                self.send_json_response(500, {"error": result.stderr})

        except Exception as e:
            self.send_json_response(500, {"error": str(e)})

    def handle_test(self) -> None:
        """Handle testing request."""
        try:
            body = self.read_json_body()
            model_dir = body.get("model_dir")
            test_csv = body.get("test_csv")

            if not model_dir:
                self.send_json_response(400, {"error": "Missing model_dir parameter"})
                return

            # Resolve paths
            if not Path(model_dir).is_absolute():
                model_dir = REPO_ROOT / model_dir

            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "test.py"),
                "--model-dir", str(model_dir)
            ]
            if test_csv:
                if not Path(test_csv).is_absolute():
                    test_csv = REPO_ROOT / test_csv
                cmd.extend(["--test-csv", str(test_csv)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                self.send_json_response(200, {
                    "status": "success",
                    "message": "Testing completed",
                    "output_dir": str(model_dir)
                })
            else:
                self.send_json_response(500, {"error": result.stderr})

        except Exception as e:
            self.send_json_response(500, {"error": str(e)})


def run_server(port: int = 310) -> None:
    """Run the API server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, APIHandler)
    print(f"Server running on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the backend API server")
    parser.add_argument("--port", type=int, default=310, help="Port to run server on")
    args = parser.parse_args()
    run_server(args.port)