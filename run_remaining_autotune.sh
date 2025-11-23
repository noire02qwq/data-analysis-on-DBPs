#!/bin/bash

# 继续运行剩余的自动调参任务
# 从LSTM的两个OTHER模式开始，然后依次调参MLP、MLPHIS、RNN和GRU的四种模式
# 所有模型的trial数量从100减少到64

echo "开始运行剩余的自动调参任务..."

# 设置工作目录
cd /home/amoris/data-analysis-on-dbps

# 计数器
count=1
total=17

# 1. 运行MLP TRC值回归调参
echo "$count/$total: 运行MLP TRC值回归调参..."
nohup python scripts/bayes_autotune_trc.py --model-type MLP --n-trials 64 \
  --base-config models/configs/mlp_config.yaml \
  --grid-config models/configs/mlp_bayes.yaml > mlp_trc_value.log 2>&1 &
PID1=$!
echo "MLP TRC值回归调参已在后台运行，PID: $PID1"

# 等待第一个任务完成
wait $PID1
echo "MLP TRC值回归调参已完成"
((count++))

# 2. 运行MLP TRC比率回归调参
echo "$count/$total: 运行MLP TRC比率回归调参..."
nohup python scripts/bayes_autotune_trc_rate.py --model-type MLP --n-trials 64 \
  --base-config models/configs/mlp_config.yaml \
  --grid-config models/configs/mlp_bayes.yaml > mlp_trc_rate.log 2>&1 &
PID2=$!
echo "MLP TRC比率回归调参已在后台运行，PID: $PID2"

# 等待第二个任务完成
wait $PID2
echo "MLP TRC比率回归调参已完成"
((count++))

# 3. 运行MLP OTHER值回归调参
echo "$count/$total: 运行MLP OTHER值回归调参..."
nohup python scripts/bayes_autotune_other.py --model-type MLP --n-trials 64 \
  --base-config models/configs/mlp_config.yaml \
  --grid-config models/configs/mlp_bayes.yaml > mlp_other_value.log 2>&1 &
PID3=$!
echo "MLP OTHER值回归调参已在后台运行，PID: $PID3"

# 等待第三个任务完成
wait $PID3
echo "MLP OTHER值回归调参已完成"
((count++))

# 4. 运行MLP OTHER比率回归调参
echo "$count/$total: 运行MLP OTHER比率回归调参..."
nohup python scripts/bayes_autotune_other_rate.py --model-type MLP --n-trials 64 \
  --base-config models/configs/mlp_config.yaml \
  --grid-config models/configs/mlp_bayes.yaml > mlp_other_rate.log 2>&1 &
PID4=$!
echo "MLP OTHER比率回归调参已在后台运行，PID: $PID4"

# 等待第四个任务完成
wait $PID4
echo "MLP OTHER比率回归调参已完成"
((count++))

# 5. 运行MLP HIS TRC值回归调参
echo "$count/$total: 运行MLP HIS TRC值回归调参..."
nohup python scripts/bayes_autotune_trc.py --model-type MLP_WITH_HISTORY --n-trials 64 \
  --base-config models/configs/mlp_with_history_config.yaml \
  --grid-config models/configs/mlp_with_history_grid.yaml > mlphis_trc_value.log 2>&1 &
PID5=$!
echo "MLP HIS TRC值回归调参已在后台运行，PID: $PID5"

# 等待第五个任务完成
wait $PID5
echo "MLP HIS TRC值回归调参已完成"
((count++))

# 6. 运行MLP HIS TRC比率回归调参
echo "$count/$total: 运行MLP HIS TRC比率回归调参..."
nohup python scripts/bayes_autotune_trc_rate.py --model-type MLP_WITH_HISTORY --n-trials 64 \
  --base-config models/configs/mlp_with_history_config.yaml \
  --grid-config models/configs/mlp_with_history_grid.yaml > mlphis_trc_rate.log 2>&1 &
PID6=$!
echo "MLP HIS TRC比率回归调参已在后台运行，PID: $PID6"

# 等待第六个任务完成
wait $PID6
echo "MLP HIS TRC比率回归调参已完成"
((count++))

# 7. 运行MLP HIS OTHER值回归调参
echo "$count/$total: 运行MLP HIS OTHER值回归调参..."
nohup python scripts/bayes_autotune_other.py --model-type MLP_WITH_HISTORY --n-trials 64 \
  --base-config models/configs/mlp_with_history_config.yaml \
  --grid-config models/configs/mlp_with_history_grid.yaml > mlphis_other_value.log 2>&1 &
PID7=$!
echo "MLP HIS OTHER值回归调参已在后台运行，PID: $PID7"

# 等待第七个任务完成
wait $PID7
echo "MLP HIS OTHER值回归调参已完成"
((count++))

# 8. 运行MLP HIS OTHER比率回归调参
echo "$count/$total: 运行MLP HIS OTHER比率回归调参..."
nohup python scripts/bayes_autotune_other_rate.py --model-type MLP_WITH_HISTORY --n-trials 64 \
  --base-config models/configs/mlp_with_history_config.yaml \
  --grid-config models/configs/mlp_with_history_grid.yaml > mlphis_other_rate.log 2>&1 &
PID8=$!
echo "MLP HIS OTHER比率回归调参已在后台运行，PID: $PID8"

# 等待第八个任务完成
wait $PID8
echo "MLP HIS OTHER比率回归调参已完成"
((count++))

# 9. 运行RNN TRC值回归调参
echo "$count/$total: 运行RNN TRC值回归调参..."
nohup python scripts/bayes_autotune_trc.py --model-type RNN --n-trials 64 \
  --base-config models/configs/rnn_config.yaml \
  --grid-config models/configs/rnn_bayes.yaml > rnn_trc_value.log 2>&1 &
PID9=$!
echo "RNN TRC值回归调参已在后台运行，PID: $PID9"

# 等待第九个任务完成
wait $PID9
echo "RNN TRC值回归调参已完成"
((count++))

# 10. 运行RNN TRC比率回归调参
echo "$count/$total: 运行RNN TRC比率回归调参..."
nohup python scripts/bayes_autotune_trc_rate.py --model-type RNN --n-trials 64 \
  --base-config models/configs/rnn_config.yaml \
  --grid-config models/configs/rnn_bayes.yaml > rnn_trc_rate.log 2>&1 &
PID10=$!
echo "RNN TRC比率回归调参已在后台运行，PID: $PID10"

# 等待第十个任务完成
wait $PID10
echo "RNN TRC比率回归调参已完成"
((count++))

# 11. 运行RNN OTHER值回归调参
echo "$count/$total: 运行RNN OTHER值回归调参..."
nohup python scripts/bayes_autotune_other.py --model-type RNN --n-trials 64 \
  --base-config models/configs/rnn_config.yaml \
  --grid-config models/configs/rnn_bayes.yaml > rnn_other_value.log 2>&1 &
PID11=$!
echo "RNN OTHER值回归调参已在后台运行，PID: $PID11"

# 等待第十一个任务完成
wait $PID11
echo "RNN OTHER值回归调参已完成"
((count++))

# 12. 运行RNN OTHER比率回归调参
echo "$count/$total: 运行RNN OTHER比率回归调参..."
nohup python scripts/bayes_autotune_other_rate.py --model-type RNN --n-trials 64 \
  --base-config models/configs/rnn_config.yaml \
  --grid-config models/configs/rnn_bayes.yaml > rnn_other_rate.log 2>&1 &
PID12=$!
echo "RNN OTHER比率回归调参已在后台运行，PID: $PID12"

# 等待第十二个任务完成
wait $PID12
echo "RNN OTHER比率回归调参已完成"
((count++))

# 13. 运行GRU TRC值回归调参
echo "$count/$total: 运行GRU TRC值回归调参..."
nohup python scripts/bayes_autotune_trc.py --model-type GRU --n-trials 64 \
  --base-config models/configs/rnn_config.yaml \
  --grid-config models/configs/gru_bayes.yaml > gru_trc_value.log 2>&1 &
PID13=$!
echo "GRU TRC值回归调参已在后台运行，PID: $PID13"

# 等待第十三个任务完成
wait $PID13
echo "GRU TRC值回归调参已完成"
((count++))

# 14. 运行GRU TRC比率回归调参
echo "$count/$total: 运行GRU TRC比率回归调参..."
nohup python scripts/bayes_autotune_trc_rate.py --model-type GRU --n-trials 64 \
  --base-config models/configs/rnn_config.yaml \
  --grid-config models/configs/gru_bayes.yaml > gru_trc_rate.log 2>&1 &
PID14=$!
echo "GRU TRC比率回归调参已在后台运行，PID: $PID14"

# 等待第十四个任务完成
wait $PID14
echo "GRU TRC比率回归调参已完成"
((count++))

# 15. 运行GRU OTHER值回归调参
echo "$count/$total: 运行GRU OTHER值回归调参..."
nohup python scripts/bayes_autotune_other.py --model-type GRU --n-trials 64 \
  --base-config models/configs/rnn_config.yaml \
  --grid-config models/configs/gru_bayes.yaml > gru_other_value.log 2>&1 &
PID15=$!
echo "GRU OTHER值回归调参已在后台运行，PID: $PID15"

# 等待第十五个任务完成
wait $PID15
echo "GRU OTHER值回归调参已完成"
((count++))

# 16. 运行GRU OTHER比率回归调参
echo "$count/$total: 运行GRU OTHER比率回归调参..."
nohup python scripts/bayes_autotune_other_rate.py --model-type GRU --n-trials 64 \
  --base-config models/configs/rnn_config.yaml \
  --grid-config models/configs/gru_bayes.yaml > gru_other_rate.log 2>&1 &
PID16=$!
echo "GRU OTHER比率回归调参已在后台运行，PID: $PID16"

# 等待第十六个任务完成
wait $PID16
echo "GRU OTHER比率回归调参已完成"
((count++))

# 17. 运行LSTM OTHER比率回归调参（放在最后）
echo "$count/$total: 运行LSTM OTHER比率回归调参..."
nohup python scripts/bayes_autotune_other_rate.py --model-type LSTM --n-trials 64 > lstm_other_rate.log 2>&1 &
PID17=$!
echo "LSTM OTHER比率回归调参已在后台运行，PID: $PID17"

# 等待第十七个任务完成
wait $PID17
echo "LSTM OTHER比率回归调参已完成"

echo "所有剩余的自动调参任务均已运行完成！"
echo "请查看以下日志文件获取详细信息："
echo "- mlp_trc_value.log"
echo "- mlp_trc_rate.log"
echo "- mlp_other_value.log"
echo "- mlp_other_rate.log"
echo "- mlphis_trc_value.log"
echo "- mlphis_trc_rate.log"
echo "- mlphis_other_value.log"
echo "- mlphis_other_rate.log"
echo "- rnn_trc_value.log"
echo "- rnn_trc_rate.log"
echo "- rnn_other_value.log"
echo "- rnn_other_rate.log"
echo "- gru_trc_value.log"
echo "- gru_trc_rate.log"
echo "- gru_other_value.log"
echo "- gru_other_rate.log"
echo "- lstm_other_rate.log"