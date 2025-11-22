#!/bin/bash

# LSTM自动调参脚本（并行版本）
# 同时运行所有四个调参程序

echo "开始并行运行LSTM自动调参程序..."

# 设置工作目录
cd /home/amoris/data-analysis-on-dbps

# 同时启动所有四个调参程序
echo "启动TRC值回归调参..."
nohup python scripts/bayes_autotune_trc.py --model-type LSTM --n-trials 100 > lstm_trc_value.log 2>&1 &
PID1=$!

echo "启动TRC比率回归调参..."
nohup python scripts/bayes_autotune_trc_rate.py --model-type LSTM --n-trials 100 > lstm_trc_rate.log 2>&1 &
PID2=$!

echo "启动OTHER值回归调参..."
nohup python scripts/bayes_autotune_other.py --model-type LSTM --n-trials 100 > lstm_other_value.log 2>&1 &
PID3=$!

echo "启动OTHER比率回归调参..."
nohup python scripts/bayes_autotune_other_rate.py --model-type LSTM --n-trials 100 > lstm_other_rate.log 2>&1 &
PID4=$!

echo "所有调参程序已在后台启动:"
echo "TRC值回归调参 PID: $PID1"
echo "TRC比率回归调参 PID: $PID2"
echo "OTHER值回归调参 PID: $PID3"
echo "OTHER比率回归调参 PID: $PID4"

echo "您可以使用以下命令查看运行状态:"
echo "jobs -l"
echo "或者查看日志文件:"
echo "tail -f lstm_*.log"

# 等待所有任务完成
echo "等待所有任务完成..."
wait $PID1 $PID2 $PID3 $PID4

echo "所有LSTM自动调参程序均已运行完成！"