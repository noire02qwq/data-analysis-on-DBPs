#!/usr/bin/env python3
"""
LSWW数据集处理脚本
由于LSWW和CAWW的列组成不同，需要单独处理
"""

import sys
import polars as pl
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def load_lsww_excel(excel_path: Path) -> pl.DataFrame:
    """加载LSWW Excel文件"""
    print(f"Loading {excel_path}...")
    df = pl.read_excel(excel_path)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns}")
    return df


def impute_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    """缺失值插补 - 使用线性插值然后前向/后向填充"""
    print("Imputing missing values...")

    # 识别数值列
    numeric_cols = []
    for col in df.columns:
        if col.lower() not in ["date, time", "date", "time"]:
            numeric_cols.append(col)

    # 对数值列进行插值
    df_imputed = df.clone()
    for col in numeric_cols:
        if col in df.columns:
            # 转换为Float64以便插值
            series = df[col].cast(pl.Float64)

            # 线性插值
            series = series.interpolate()

            # 前向填充
            series = series.forward_fill()

            # 后向填充
            series = series.backward_fill()

            df_imputed = df_imputed.with_columns(series.alias(col))

    # 删除仍有空值的行
    df_imputed = df_imputed.drop_nulls()
    print(f"  After imputation: {len(df_imputed)} rows")

    return df_imputed


def split_data(df: pl.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42):
    """拆分数据集"""
    print("Splitting data...")

    # 随机打乱
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    df_train = df[:n_train]
    df_val = df[n_train:n_train + n_val]
    df_test = df[n_train + n_val:]

    print(f"  Train: {len(df_train)} ({train_ratio*100:.0f}%)")
    print(f"  Val: {len(df_val)} ({val_ratio*100:.0f}%)")
    print(f"  Test: {len(df_test)} ({test_ratio*100:.0f}%)")

    return df_train, df_val, df_test


def process_lsww_dataset(prefix: str):
    """处理单个LSWW数据集"""
    excel_path = DATA_DIR / f"{prefix}_DT_full.xlsx"

    if not excel_path.exists():
        print(f"Excel file not found: {excel_path}")
        return

    # 加载数据
    df = load_lsww_excel(excel_path)

    # 插补
    df_imputed = impute_missing_values(df)

    # 保存插补后的数据
    imputed_path = DATA_DIR / f"{prefix.lower()}_imputed_data.csv"
    df_imputed.write_csv(imputed_path)
    print(f"Saved imputed data to: {imputed_path}")

    # 拆分数据
    df_train, df_val, df_test = split_data(df_imputed, seed=42)

    # 保存拆分后的数据
    split_dir = DATA_DIR / f"{prefix.lower()}_split"
    split_dir.mkdir(exist_ok=True)

    df_train.write_csv(split_dir / "train.csv")
    df_val.write_csv(split_dir / "val.csv")
    df_test.write_csv(split_dir / "test.csv")

    print(f"Saved split data to: {split_dir}")


def main():
    # 处理LSWW_29C和LSWW_35C
    for prefix in ["LSWW_29C", "LSWW_35C"]:
        process_lsww_dataset(prefix)
        print()


if __name__ == "__main__":
    main()