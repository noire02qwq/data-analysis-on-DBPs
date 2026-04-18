#!/usr/bin/env python3
"""
测试所有模型的脚本
为每个模型生成:
1. 真实值预测值对比图像
2. y=x散点图
3. 预测值真实值对比表格
4. 测试结果统计数据(mae, mse, rmse, r2)
"""

import subprocess
import sys
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def test_model(model_dir: Path, output_dir: Path, model_name: str) -> bool:
    """测试单个模型"""
    print(f"\nTesting {model_name}...")

    # 查找最新的模型输出目录
    if not model_dir.exists():
        print(f"  Model directory not found: {model_dir}")
        return False

    subdirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not subdirs:
        print(f"  No model subdirectories found in {model_dir}")
        return False

    latest_model_dir = subdirs[0]
    print(f"  Using model from: {latest_model_dir.name}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行测试
    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "test.py"),
        "--model-dir", str(latest_model_dir),
        "--test-csv", str(REPO_ROOT / "data/test.csv"),
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Test failed: {result.stderr}")
        return False

    print(f"  Test completed successfully!")
    return True


def main():
    print("="*60)
    print("Testing all models")
    print("="*60)

    # 模型列表
    models = [
        ("mlp", "MLP"),
        ("rnn", "RNN"),
        ("gru", "GRU"),
        ("lstm", "LSTM"),
        ("transformer", "TRANSFORMER"),
        ("mamba", "MAMBA"),
        ("xgboost", "XGBOOST"),
        ("lightgbm", "LIGHTGBM"),
        ("catboost", "CATBOOST"),
    ]

    # 对于每个模型，从final目录或bayes目录获取最佳模型
    for model_key, model_name in models:
        model_dir = REPO_ROOT / "outputs" / f"{model_key}_final"
        output_dir = REPO_ROOT / "outputs" / f"{model_key}_final" / "test_results"

        if model_dir.exists():
            success = test_model(model_dir, output_dir, model_name)
            if not success:
                # 尝试从bayes目录获取最佳模型
                bayes_dir = REPO_ROOT / "outputs" / f"{model_key}_bayes_v3"
                if bayes_dir.exists():
                    # 找到best trial
                    trials = list(bayes_dir.glob("trial_*"))
                    if trials:
                        # 选择result.toml存在的trial中val_loss最小的
                        best_trial = None
                        best_val_loss = float('inf')

                        for trial in trials:
                            result_file = trial / "result.toml"
                            if result_file.exists():
                                import tomli
                                with open(result_file, "rb") as f:
                                    result = tomli.load(f)
                                    val_loss = result.get("eval", {}).get("best_val_loss", float('inf'))
                                    if val_loss < best_val_loss:
                                        best_val_loss = val_loss
                                        best_trial = trial

                        if best_trial:
                            print(f"  Using best trial from bayes: {best_trial.name}")
                            # 复制best_model.pt和其他文件到final目录
                            final_dir = REPO_ROOT / "outputs" / f"{model_key}_final" / best_trial.name
                            final_dir.mkdir(parents=True, exist_ok=True)

                            for f in ["best_model.pt", "config.toml", "result.toml", "scalers.npz"]:
                                src = best_trial / f
                                if src.exists():
                                    shutil.copy(src, final_dir / f)

                            # 运行测试
                            test_model(final_dir, output_dir, model_name)

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()