#!/usr/bin/env python3
"""
为Archived/before-midterm目录中的每个结果生成test_metrics.csv文件
"""

import numpy as np
import os
import csv

def calculate_metrics(y_true, y_pred):
    """Calculate R2, MSE, RMSE, MAE for predictions"""
    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    r2 = 1 - (ss_res / ss_tot)
    
    # MSE
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    
    # RMSE
    rmse = np.sqrt(mse)
    
    # MAE
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    
    return {
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }

def process_directory(dir_path):
    """处理单个目录，生成test_metrics.csv文件"""
    test_predictions_path = os.path.join(dir_path, "test_predictions.csv")
    
    if not os.path.exists(test_predictions_path):
        print(f"警告: {test_predictions_path} 不存在，跳过")
        return False
    
    # 读取预测数据
    data = np.genfromtxt(test_predictions_path, delimiter=',', skip_header=1)
    
    # 获取列名
    with open(test_predictions_path, 'r') as f:
        header = f.readline().strip()
        columns = header.split(',')
    
    # 提取目标变量名称（从列名中获取）
    target_names = []
    for i in range(0, len(columns), 2):
        if columns[i].endswith('_true'):
            target_name = columns[i][:-5]  # 去掉'_true'后缀
            target_names.append(target_name)
    
    # 分离真实值和预测值
    y_true_list = []
    y_pred_list = []
    
    for i in range(len(target_names)):
        y_true_list.append(data[:, 2*i])      # 真实值列
        y_pred_list.append(data[:, 2*i + 1])  # 预测值列
    
    y_true = np.column_stack(y_true_list)
    y_pred = np.column_stack(y_pred_list)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 写入metrics CSV文件
    metrics_path = os.path.join(dir_path, "test_metrics.csv")
    with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["target", "r2", "mse", "rmse", "mae"])
        for idx, name in enumerate(target_names):
            writer.writerow([
                name,
                f"{metrics['r2'][idx]:.6f}",
                f"{metrics['mse'][idx]:.6f}",
                f"{metrics['rmse'][idx]:.6f}",
                f"{metrics['mae'][idx]:.6f}"
            ])
    
    print(f"已生成: {metrics_path}")
    return True

def main():
    base_dir = "Archived/before-midterm"
    
    if not os.path.exists(base_dir):
        print(f"错误: 目录 {base_dir} 不存在")
        return
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(base_dir) 
               if os.path.isdir(os.path.join(base_dir, d))]
    
    print(f"找到 {len(subdirs)} 个目录需要处理")
    
    success_count = 0
    for subdir in subdirs:
        dir_path = os.path.join(base_dir, subdir)
        print(f"处理目录: {subdir}")
        if process_directory(dir_path):
            success_count += 1
    
    print(f"\n处理完成! 成功处理 {success_count}/{len(subdirs)} 个目录")

if __name__ == "__main__":
    main()