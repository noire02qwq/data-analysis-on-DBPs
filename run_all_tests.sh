#!/bin/bash

# Run all test scripts for all models and modes
# This script will test the top 3 trials for each model-mode combination

echo "开始运行所有模型的测试..."

# 设置工作目录
cd /home/amoris/data-analysis-on-dbps

# 定义模型和模式
models=("mlp" "mlp_with_history" "lstm" "rnn")
modes=("trc-value" "trc-rate" "other-value" "other-rate")

# 获取前3个最佳trial的函数
get_top_trials() {
    local model_dir=$1
    # Skip the header line and get top 3 trials by validation loss (value)
    tail -n +2 "$model_dir/bayes_optimization_results.csv" | sort -t',' -k2 -n | head -n 3 | cut -d',' -f1
}

# 计数器
count=1
total=16

# 遍历每个模型和模式
for model in "${models[@]}"; do
    for mode in "${modes[@]}"; do
        model_dir="scripts/outputs/${model}-${mode}"
        
        # 检查目录是否存在
        if [ ! -d "$model_dir" ]; then
            echo "警告: 目录不存在: $model_dir"
            continue
        fi
        
        # 检查结果文件是否存在
        if [ ! -f "$model_dir/bayes_optimization_results.csv" ]; then
            echo "警告: 结果文件不存在: $model_dir/bayes_optimization_results.csv"
            continue
        fi
        
        echo "[$count/$total] 处理 $model-$mode..."
        
        # 获取前3个最佳trial
        trials=$(get_top_trials "$model_dir")
        
        trial_count=1
        echo "$trials" | while read trial; do
            # 跳过空行
            if [ -z "$trial" ]; then
                continue
            fi
            
            trial_dir="$model_dir/$trial"
            
            # 检查trial目录是否存在
            if [ ! -d "$trial_dir" ]; then
                echo "  警告: Trial目录不存在: $trial_dir"
                continue
            fi
            
            echo "  测试 $model-$mode trial $trial_count ($trial)..."
            
            # 根据模式选择测试脚本
            if [[ $mode == *"rate"* ]]; then
                if [[ $mode == *"trc"* ]]; then
                    python scripts/test_trc_rate.py --model-dir "$trial_dir"
                else
                    python scripts/test_other_rate.py --model-dir "$trial_dir"
                fi
            else
                if [[ $mode == *"trc"* ]]; then
                    python scripts/test_trc_value.py --model-dir "$trial_dir"
                else
                    python scripts/test_other_value.py --model-dir "$trial_dir"
                fi
            fi
            
            ((trial_count++))
        done
        
        ((count++))
    done
done

echo "所有测试已完成！"
echo "现在生成横向对比CSV文件..."
python analyze_results.py
echo "所有任务完成！"