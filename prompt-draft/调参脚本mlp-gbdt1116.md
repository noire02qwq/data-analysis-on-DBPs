# 调参脚本及新的树模型

## MLP, MLP_with_history的调参脚本

基于1109 1111 1115的回归实验、调参脚本、变化率回归，完成MLP MLP_with_history的实验

可调参数

- hidden layers: 以{中间层层数、规模}的方式定义
    - 首层大小为中间层*2，末层大小为中间层/2
    - 举例说明：若{中间层层数为4、规模为512}，则对应的原隐藏层为[1024, 512, 512, 512, 512, 256]
    - 中间层层数: 2~16 起始4间隔2
    - 中间层规模：128~1024 起始256间隔32
- dropout:
    min: 0.1
    max: 0.5
    step: 0.05
    start: 0.3
- batch_size:
    min: 128
    max: 512
    step: 16
    start: 256
- learning_rate:
    min: 0.0004
    max: 0.0020
    step: 0.0002
    start: 0.0008
- weight_decay:
    min: 0.0
    max: 0.01
    step: 0.002
    start: 0.0
- 只针对MLP_with_history：history_length:
    min: 16
    max: 128
    step: 8
    start: 80

基于并核查原来的训练、测试、调参脚本，并编写配置文件，让我可以进行MLP, MLP_with_history的调参，包括直接回归数值以及回归rate的两种方式

## xgboost调参脚本

目前实验中xgboost的参数量较小表达力较差，所以需要扩大模型参数量，甚至扩大数量级。查询xgboost相关参考资料和网站，与其他深度学习模型的工作流类似，根据time_aligned_date的行列规模，为我设置好调参的范围和网格大小来对可调整的超参数进行自动调参。把通用的训练、测试程序复核适配好，自动调参程序和配置文件都准备好，包括直接回归数值以及回归rate的两种方式。爬山调参时，判断是否发生improvement使用TRC-PPL1的val loss当做模型的val loss即可。

## 其他gbdt的测试

跑通xgboost的回归和变化率回归自动调参之后，再尝试lightgbm和catboost，准备好模型本体、自动调参程序、配置文件，并让之前的训练、测试程序进行适配，包括直接回归数值以及回归rate的两种方式。