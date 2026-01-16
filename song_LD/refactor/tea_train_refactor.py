# -*- coding: utf-8 -*-
"""
TEA数据集训练脚本 - 重构版本
使用TensorBoard记录训练指标,argparse管理超参数,模块化设计
"""

import os
import json
import random
import argparse
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

# ====== 模型导入 ======
from models.MSDNN import MSDNN
from models.resnet import ResNet18
from models.Conformer import Conformer

# ====== 数据导入 ======
from datasets.amigos import AmigosDataset, load_and_preprocess_data

# ====== utils导入 ======
from utils.metrics import MetricsTracker

# ============================================================================
# 配置管理模块
# ============================================================================

class TrainingConfig:
    """训练配置类,集中管理所有超参数"""

    def __init__(self, args: argparse.Namespace):
        # 数据路径
        self.data_dir = Path(args.data_dir)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data_name = args.data_name

        # 训练超参数
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.dropout_p = args.dropout_p
        self.model_type = args.model_type

        # 数据处理
        self.n_splits = args.n_splits
        self.use_zscore = args.use_zscore
        self.zscore_by_channel = args.zscore_by_channel

        # 训练策略
        self.use_amp = args.use_amp
        self.max_norm = args.max_norm
        self.lr_sched_patience = args.lr_sched_patience
        self.early_patience = args.early_patience
        self.min_lr = args.min_lr

        # 随机种子
        self.seed = args.seed

        # TensorBoard
        self.use_tensorboard = args.use_tensorboard
        self.tensorboard_dir = self.save_dir / "tensorboard_logs"

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式用于保存"""
        return {
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "dropout_p": self.dropout_p,
            "n_splits": self.n_splits,
            "use_zscore": self.use_zscore,
            "zscore_by_channel": self.zscore_by_channel,
            "use_amp": self.use_amp,
            "max_norm": self.max_norm,
            "lr_sched_patience": self.lr_sched_patience,
            "early_patience": self.early_patience,
            "min_lr": self.min_lr,
            "seed": self.seed,
            "device": str(self.device),
        }


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AMIGOS 数据集进行情绪识别任务",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据路径
    parser.add_argument("--data_dir", type=str, default=Path(os.path.dirname(os.path.abspath(__file__))) / "datasets", help="数据文件所在目录")
    parser.add_argument("--save_dir", type=str, default=Path(os.path.dirname(os.path.abspath(__file__))) / "save_models", help="模型和日志保存目录")
    parser.add_argument("--data_name", type=str, default="Arousal", help="任务名称, Arousal or Valence")

    # 训练超参数
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="L2正则化系数")
    parser.add_argument("--dropout_p", type=float, default=0.4, help="Dropout概率")
    parser.add_argument("--model_type", type=str, default="Conformer", help="模型类型, 可以选MSDNN,ResNet,EEGNet,Conformer")

    # 数据处理
    parser.add_argument("--n_splits", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--use_zscore", type=bool, default=True, help="是否使用Z-score归一化")
    parser.add_argument("--zscore_by_channel", type=bool, default=True, help="是否按通道进行归一化")

    # 训练策略
    parser.add_argument("--use_amp", type=bool, default=True, help="是否使用混合精度训练")
    parser.add_argument("--max_norm", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--lr_sched_patience", type=int, default=3, help="学习率调度器耐心值")
    parser.add_argument("--early_patience", type=int, default=70, help="早停耐心值")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="最小学习率")
    parser.add_argument("--use_StratifiedKFold", type=bool, default=False, help="是否使用StratifiedKFold交叉验证")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="是否使用TensorBoard记录")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================================
# 模型定义
# ============================================================================

class ResNetModel(nn.Module):
    """ResNet模型包装类,包含特征提取器和分类头"""

    def __init__(self, base_model: nn.Module, num_classes: int, dropout_p: float = 0.5):
        """
        Args:
            base_model: 基础特征提取模型
            num_classes: 分类类别数
            dropout_p: Dropout概率
        """
        super().__init__()
        self.encoder = base_model
        if hasattr(self.encoder, "fc"):
            self.encoder.fc = nn.Identity()

        feat_dim = getattr(self.encoder, "num_channels", 64)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits



def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    config: TrainingConfig,
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    tracker = MetricsTracker()

    for i, (xb, yb) in enumerate(dataloader):
        # xb = xb.reshape(xb.shape[0] ,1, 54, 128*20 )
        xb = xb.to(config.device, dtype=torch.float32)
        yb = yb.to(config.device, dtype=torch.long)

        if not torch.isfinite(xb).all():
            continue

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=(config.device.type == 'cuda' and config.use_amp)):
            logits = model(xb)
            if not torch.isfinite(logits).all():
                continue
            loss = criterion(logits, yb)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_norm)
        scaler.step(optimizer)
        scaler.update()

        tracker.update(loss.item(), logits, yb)
        # break

    return tracker.compute()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: TrainingConfig,
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    tracker = MetricsTracker()

    with torch.no_grad():
        for xb, yb in dataloader:
            # xb = xb.reshape(xb.shape[0] ,1, 54, 128*20 )
            xb = xb.to(config.device, dtype=torch.float32)
            yb = yb.to(config.device, dtype=torch.long)

            if not torch.isfinite(xb).all():
                continue

            with torch.amp.autocast(device_type='cuda', enabled=(config.device.type == 'cuda' and config.use_amp)):
                logits = model(xb)
                if not torch.isfinite(logits).all():
                    continue
                loss = criterion(logits, yb)

            if not torch.isfinite(loss):
                continue

            tracker.update(loss.item(), logits, yb)

    return tracker.compute()


def train_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    config: TrainingConfig,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, Any]:
    """
    训练单个fold

    Args:
        fold_idx: fold索引
        train_idx: 训练集索引
        test_idx: 测试集索引
        X: 特征数组
        y: 标签数组
        num_classes: 类别数
        config: 训练配置
        writer: TensorBoard writer

    Returns:
        fold训练结果字典
    """
    print(f"\n{'='*50} Fold {fold_idx}/{config.n_splits} {'='*50}")

    # 创建数据加载器
    train_dataset = AmigosDataset(
        X[train_idx], y[train_idx],
        use_zscore=config.use_zscore,
        by_channel=config.zscore_by_channel
    )
    test_dataset = AmigosDataset(
        X[test_idx], y[test_idx],
        use_zscore=config.use_zscore,
        by_channel=config.zscore_by_channel
    )

    pin_memory = config.device.type == 'cuda'
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0, pin_memory=pin_memory
    )

    # 计算类别权重
    y_train_fold = y[train_idx]
    classes = np.arange(num_classes)
    cls_weights = compute_class_weight('balanced', classes=classes, y=y_train_fold)
    cls_weights_t = torch.tensor(cls_weights, dtype=torch.float32, device=config.device)
    print(f"[Info][Fold{fold_idx}] 类别权重: {cls_weights.tolist()}")

    # 创建模型、优化器、损失函数
    if config.model_type == 'ResNet':
        encoder = ResNet18()
    elif config.model_type == 'Conformer':
        encoder = Conformer(emb_size=40, depth=6, n_classes=4)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")
    model = ResNetModel(encoder, num_classes=num_classes, dropout_p=config.dropout_p).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=cls_weights_t)

    scaler = torch.amp.GradScaler(enabled=(config.device.type == 'cuda' and config.use_amp))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config.lr_sched_patience,
        min_lr=config.min_lr
    )

    # 训练历史
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_f1": [],
        "test_f1": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }

    best_loss = float("inf")
    no_improve = 0
    best_path = config.save_dir / f"best_model_fold{fold_idx}.pth"

    for epoch in range(config.num_epochs):
        # 训练
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, config)

        # 评估
        test_metrics = evaluate(model, test_loader, criterion, config)

        # 更新学习率
        if np.isfinite(test_metrics["loss"]):
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(test_metrics["loss"])
            new_lr = optimizer.param_groups[0]['lr']
            if old_lr != new_lr:
                print(f"[LR Scheduler] Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # 保存最佳模型
        improved = (np.isfinite(test_metrics["loss"]) and test_metrics["loss"] < best_loss - 1e-6)
        if improved:
            best_loss = test_metrics["loss"]
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1

        # 记录历史
        for key in history.keys():
            metric_name = key.replace("train_", "").replace("test_", "")
            if "train" in key:
                history[key].append(train_metrics[metric_name])
            else:
                history[key].append(test_metrics[metric_name])

        # TensorBoard记录
        if writer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            global_step = fold_idx * 1000 + epoch

            writer.add_scalar(f"Fold{fold_idx}/Loss/train", train_metrics["loss"], global_step)
            writer.add_scalar(f"Fold{fold_idx}/Loss/test", test_metrics["loss"], global_step)
            writer.add_scalar(f"Fold{fold_idx}/Accuracy/train", train_metrics["accuracy"], global_step)
            writer.add_scalar(f"Fold{fold_idx}/Accuracy/test", test_metrics["accuracy"], global_step)
            writer.add_scalar(f"Fold{fold_idx}/F1/train", train_metrics["f1"], global_step)
            writer.add_scalar(f"Fold{fold_idx}/F1/test", test_metrics["f1"], global_step)
            writer.add_scalar(f"Fold{fold_idx}/Learning_Rate", current_lr, global_step)

        # 打印进度
        print(
            f"Fold {fold_idx} | Epoch {epoch+1:02d}/{config.num_epochs} | "
            f"Train: Loss={train_metrics['loss']:.4f} F1={train_metrics['f1']:.4f} Acc={train_metrics['accuracy']:.4f} | "
            f"Test: Loss={test_metrics['loss']:.4f} F1={test_metrics['f1']:.4f} Acc={test_metrics['accuracy']:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"{'*BEST*' if improved else ''}"
        )

        # 早停
        if no_improve >= config.early_patience:
            print(f"[EarlyStop] Fold {fold_idx}: {config.early_patience}个epoch无提升, 提前停止")
            break

    return {
        "history": history,
        "best_loss": best_loss,
        "epochs_trained": len(history["train_loss"]),
    }

# ============================================================================
# 主函数
# ============================================================================

def main():
    """主训练流程"""
    # 解析参数
    args = parse_arguments()

    # 创建配置对象
    config = TrainingConfig(args)

    # 设置随机种子
    set_seed(config.seed)

    print(f"{'='*60}")
    print(f"训练配置:")
    print(f"  设备: {config.device}")
    print(f"  保存目录: {config.save_dir}")
    print(f"  TensorBoard: {'启用' if config.use_tensorboard else '禁用'}")
    print(f"{'='*60}\n")

    # 初始化TensorBoard
    writer = None
    if config.use_tensorboard:
        # 按时间创建子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_log_dir = config.tensorboard_dir / f"run_{timestamp}"
        tb_log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(tb_log_dir)
        print(f"[Info] TensorBoard日志目录: {tb_log_dir}")
        print(f"[Info] 启动TensorBoard: tensorboard --logdir={config.tensorboard_dir}\n")
        print(f"{'='*60}\n")

    # 加载数据
    data_path, label_path = config.data_dir/(config.data_name+'_X.dat'), config.data_dir/(config.data_name+'_y.npy')
    X, y, meta = load_and_preprocess_data(data_path, label_path)
    
    # 交叉验证训练
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        result = train_fold(
            fold_idx=fold_idx,
            train_idx=train_idx,
            test_idx=test_idx,
            X=X,
            y=y,
            num_classes=meta["num_classes"],
            config=config,
            writer=writer,
        )
        
        fold_results.append(result)
        if args.use_StratifiedKFold == False:
            break

    # 计算交叉验证汇总
    # cv_summary = compute_cross_validation_summary(fold_results)

    # 绘制学习曲线
    # curves_path = config.save_dir / "learning_curves.png"
    # plot_learning_curves(fold_results, curves_path)
    # print(f"\n[Info] 学习曲线已保存至: {curves_path}")

    # 保存训练摘要
    # save_training_summary(config, fold_results, cv_summary, meta)

        
    # 关闭TensorBoard writer
    if writer is not None:
        writer.close()
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)

if __name__ == "__main__":
    main()
