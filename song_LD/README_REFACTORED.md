# TEA数据集训练脚本 - 重构版本使用说明

## 概述

这是TEA数据集训练脚本的重构版本,在原有功能基础上进行了以下改进:

### 主要改进点

1. **使用argparse管理超参数** - 所有配置可通过命令行参数传递
2. **集成TensorBoard** - 实时可视化训练过程中的loss、accuracy、F1-score等指标
3. **模块化设计** - 代码按功能模块划分,提高可读性和可维护性
4. **类型注解** - 使用Python类型提示,增强代码可读性
5. **面向对象设计** - 使用类封装配置和指标跟踪逻辑

### 代码结构

```
tea_train_refactor.py
├── 配置管理模块
│   ├── TrainingConfig      # 训练配置类
│   ├── parse_arguments()   # 命令行参数解析
│   └── set_seed()          # 随机种子设置
│
├── 数据加载与处理模块
│   ├── load_mat_var()              # MAT文件读取
│   └── load_and_preprocess_data()  # 数据加载和预处理
│
├── 数据集类
│   └── GaitDataset         # PyTorch Dataset封装
│
├── 模型定义
│   └── ResNetModel         # MSDNN + 分类头
│
├── 训练与评估模块
│   ├── MetricsTracker      # 指标跟踪器
│   ├── train_one_epoch()   # 单epoch训练
│   ├── evaluate()          # 模型评估
│   └── train_fold()        # 单fold训练
│
├── 结果分析模块
│   ├── compute_cross_validation_summary()  # 交叉验证汇总
│   ├── plot_learning_curves()              # 学习曲线绘制
│   └── save_training_summary()             # 训练摘要保存
│
└── 主函数
    └── main()             # 主训练流程
```

## 使用方法

### 基本使用

使用默认参数运行:

```bash
python tea_train_refactor.py
```

### 自定义参数

可以通过命令行参数自定义训练配置:

```bash
# 示例1: 修改学习率和batch size
python tea_train_refactor.py --lr 0.001 --batch_size 128

# 示例2: 修改训练轮数和早停耐心值
python tea_train_refactor.py --num_epochs 100 --early_patience 10

# 示例3: 指定数据路径和保存路径
python tea_train_refactor.py --data_dir "path/to/data" --save_dir "path/to/save"
```

### 所有可用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_dir` | str | `D:\...\data` | 数据文件所在目录 |
| `--save_dir` | str | `D:\...\save_model\refactored` | 模型和日志保存目录 |
| `--num_epochs` | int | 50 | 训练轮数 |
| `--batch_size` | int | 64 | 批次大小 |
| `--lr` | float | 1e-4 | 初始学习率 |
| `--weight_decay` | float | 3e-4 | L2正则化系数 |
| `--dropout_p` | float | 0.4 | Dropout概率 |
| `--n_splits` | int | 5 | 交叉验证折数 |
| `--use_zscore` | bool | True | 是否使用Z-score归一化 |
| `--zscore_by_channel` | bool | True | 是否按通道归一化 |
| `--use_amp` | bool | False | 是否使用混合精度训练 |
| `--max_norm` | float | 1.0 | 梯度裁剪阈值 |
| `--lr_sched_patience` | int | 3 | 学习率调度器耐心值 |
| `--early_patience` | int | 7 | 早停耐心值 |
| `--min_lr` | float | 1e-6 | 最小学习率 |
| `--seed` | int | 42 | 随机种子 |
| `--use_tensorboard` | bool | True | 是否使用TensorBoard |

### 查看帮助信息

```bash
python tea_train_refactor.py --help
```

## TensorBoard使用

### 启动TensorBoard

训练开始后,使用以下命令启动TensorBoard:

```bash
tensorboard --logdir="D:\workspace\researches\情绪\workspace\song_LD\save_model\refactored\tensorboard_logs"
```

或在训练输出中查看显示的TensorBoard启动命令。

### 访问TensorBoard

在浏览器中打开: `http://localhost:6006`

### 可视化内容

TensorBoard会记录以下指标:

- **Loss**: 训练和验证损失曲线
- **Accuracy**: 训练和验证准确率曲线
- **F1-score**: 训练和验证F1分数曲线
- **Learning Rate**: 学习率变化曲线

每个fold的指标会分别记录在 `Fold1`, `Fold2`, ... 目录下。

## 输出文件

训练完成后,会在保存目录下生成以下文件:

```
save_model/refactored/
├── tensorboard_logs/           # TensorBoard日志
│   ├── events.out.tfevents...
│
├── best_model_fold1.pth        # Fold1最佳模型
├── best_model_fold2.pth        # Fold2最佳模型
├── ...
├── best_model_fold5.pth        # Fold5最佳模型
│
├── learning_curves.png         # 学习曲线图
└── training_summary.json       # 训练摘要(包含配置、结果统计等)
```

### training_summary.json说明

包含以下信息:

```json
{
  "timestamp": "训练时间",
  "config": {
    "num_epochs": 训练轮数,
    "batch_size": 批次大小,
    "learning_rate": 学习率,
    ...
  },
  "data_info": {
    "num_classes": 类别数,
    "classes_values": 原始类别值,
    "value2id": 类别映射
  },
  "epochs_per_fold": [每折实际训练的epoch数],
  "cv_summary": {
    "train_loss_mean": 训练损失均值,
    "train_loss_std": 训练损失标准差,
    "test_loss_mean": 测试损失均值,
    "test_loss_std": 测试损失标准差,
    "train_f1_mean": 训练F1均值,
    "train_f1_std": 训练F1标准差,
    "test_f1_mean": 测试F1均值,
    "test_f1_std": 测试F1标准差,
    "train_acc_mean": 训练准确率均值,
    "train_acc_std": 训练准确率标准差,
    "test_acc_mean": 测试准确率均值,
    "test_acc_std": 测试准确率标准差
  }
}
```

## 代码设计原则

### SOLID原则应用

1. **单一职责原则(SRP)**
   - 每个类和函数只负责一个明确的功能
   - 例如: `TrainingConfig`只负责配置管理, `MetricsTracker`只负责指标跟踪

2. **开闭原则(OCP)**
   - 通过配置类扩展功能,无需修改核心训练逻辑
   - 新的指标或可视化可以通过继承实现

3. **依赖倒置原则(DIP)**
   - 使用`TrainingConfig`抽象配置,依赖抽象而非具体实现

### KISS原则

- 保持函数简洁,每个函数控制在合理长度内
- 避免过度设计,使用直观的解决方案

### DRY原则

- 提取公共逻辑为独立函数(如`train_one_epoch`, `evaluate`)
- 使用`MetricsTracker`统一指标计算逻辑

## 与原版本对比

| 特性 | 原版本 | 重构版本 |
|------|--------|----------|
| 超参数管理 | 硬编码全局变量 | argparse + TrainingConfig类 |
| 日志记录 | 控制台输出 + JSON文件 | TensorBoard + 控制台 + JSON |
| 代码组织 | 单一脚本 | 模块化设计 |
| 类型提示 | 无 | 完整类型注解 |
| 可扩展性 | 较低 | 高(易于添加新功能) |
| 可读性 | 一般 | 优秀(清晰的模块划分) |
| 实验管理 | 需修改代码 | 命令行参数控制 |

## 常见问题

### Q1: 如何禁用TensorBoard?

```bash
python tea_train_refactor.py --use_tensorboard False
```

### Q2: 如何使用GPU加速?

确保安装了CUDA版本的PyTorch,脚本会自动检测并使用GPU。

### Q3: 如何恢复训练?

当前版本不支持断点续训。如需此功能,可扩展保存checkpoint的逻辑。

### Q4: 如何调整类别权重?

类别权重根据训练集自动计算(balanced模式)。如需自定义,可在`train_fold`函数中修改`cls_weights`计算逻辑。

## 依赖项

```bash
pip install torch torchvision torch.utils.tensorboard scipy scikit-learn matplotlib numpy h5py
```

## 许可

与原项目保持一致。
