#!/bin/bash

# LSTM自动调参脚本
# 按照指定顺序依次运行四个调参程序

echo "开始运行LSTM自动调参程序..."

# 设置工作目录
cd /home/amoris/data-analysis-on-dbps

# 1. 运行TRC值回归调参
echo "1/4: 运行TRC值回归调参..."
nohup python scripts/bayes_autotune_trc.py --model-type LSTM --n-trials 100 > lstm_trc_value.log 2>&1 &
PID1=$!
echo "TRC值回归调参已在后台运行，PID: $PID1"

# 等待第一个任务完成
wait $PID1
echo "TRC值回归调参已完成"

# 2. 运行TRC比率回归调参
echo "2/4: 运行TRC比率回归调参..."
nohup python scripts/bayes_autotune_trc_rate.py --model-type LSTM --n-trials 100 > lstm_trc_rate.log 2>&1 &
PID2=$!
echo "TRC比率回归调参已在后台运行，PID: $PID2"

# 等待第二个任务完成
wait $PID2
echo "TRC比率回归调参已完成"

# 3. 运行OTHER值回归调参
echo "3/4: 运行OTHER值回归调参..."
nohup python scripts/bayes_autotune_other.py --model-type LSTM --n-trials 100 > lstm_other_value.log 2>&1 &
PID3=$!
echo "OTHER值回归调参已在后台运行，PID: $PID3"

# 等待第三个任务完成
wait $PID3
echo "OTHER值回归调参已完成"

# 4. 运行OTHER比率回归调参
echo "4/4: 运行OTHER比率回归调参..."
nohup python scripts/bayes_autotune_other_rate.py --model-type LSTM --n-trials 100 > lstm_other_rate.log 2>&1 &
PID4=$!
echo "OTHER比率回归调参已在后台运行，PID: $PID4"

# 等待第四个任务完成
wait $PID4
echo "OTHER比率回归调参已完成"

echo "所有LSTM自动调参程序均已运行完成！"
echo "请查看以下日志文件获取详细信息："
echo "- lstm_trc_value.log"
echo "- lstm_trc_rate.log" 
echo "- lstm_other_value.log"
echo "- lstm_other_rate.log"