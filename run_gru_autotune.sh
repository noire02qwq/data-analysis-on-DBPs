#!/bin/bash

# 运行GRU的四个模式自动调参任务
# 所有模型的trial数量为64

echo "开始运行GRU的自动调参任务..."

# 设置工作目录
cd /home/amoris/data-analysis-on-dbps

# 计数器
count=1
total=4

# 1. 运行GRU TRC值回归调参
echo "$count/$total: 运行GRU TRC值回归调参..."
nohup python scripts/bayes_autotune_trc.py --model-type GRU --n-trials 64 \
  --base-config models/configs/gru_config.yaml \
  --grid-config models/configs/gru_bayes.yaml > gru_trc_value.log 2>&1 &
PID1=$!
echo "GRU TRC值回归调参已在后台运行，PID: $PID1"

# 等待第一个任务完成
wait $PID1
echo "GRU TRC值回归调参已完成"
((count++))

# 2. 运行GRU TRC比率回归调参
echo "$count/$total: 运行GRU TRC比率回归调参..."
nohup python scripts/bayes_autotune_trc_rate.py --model-type GRU --n-trials 64 \
  --base-config models/configs/gru_config.yaml \
  --grid-config models/configs/gru_bayes.yaml > gru_trc_rate.log 2>&1 &
PID2=$!
echo "GRU TRC比率回归调参已在后台运行，PID: $PID2"

# 等待第二个任务完成
wait $PID2
echo "GRU TRC比率回归调参已完成"
((count++))

# 3. 运行GRU OTHER值回归调参
echo "$count/$total: 运行GRU OTHER值回归调参..."
nohup python scripts/bayes_autotune_other.py --model-type GRU --n-trials 64 \
  --base-config models/configs/gru_config.yaml \
  --grid-config models/configs/gru_bayes.yaml > gru_other_value.log 2>&1 &
PID3=$!
echo "GRU OTHER值回归调参已在后台运行，PID: $PID3"

# 等待第三个任务完成
wait $PID3
echo "GRU OTHER值回归调参已完成"
((count++))

# 4. 运行GRU OTHER比率回归调参
echo "$count/$total: 运行GRU OTHER比率回归调参..."
nohup python scripts/bayes_autotune_other_rate.py --model-type GRU --n-trials 64 \
  --base-config models/configs/gru_config.yaml \
  --grid-config models/configs/gru_bayes.yaml > gru_other_rate.log 2>&1 &
PID4=$!
echo "GRU OTHER比率回归调参已在后台运行，PID: $PID4"

# 等待第四个任务完成
wait $PID4
echo "GRU OTHER比率回归调参已完成"

echo "所有GRU的自动调参任务均已运行完成！"
echo "请查看以下日志文件获取详细信息："
echo "- gru_trc_value.log"
echo "- gru_trc_rate.log"
echo "- gru_other_value.log"
echo "- gru_other_rate.log"