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