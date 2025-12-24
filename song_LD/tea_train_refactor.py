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
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# ====== 模型导入 ======
try:
    from models.MSDNN import MSDNN
except ImportError:
    from MSDNN import MSDNN


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

        # 训练超参数
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.weight_decay = args.weight_decay
        self.dropout_p = args.dropout_p

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
        description="TEA数据集训练脚本 - 使用MSDNN模型进行分类",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 数据路径
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\workspace\researches\情绪\workspace\song_LD\data",
        help="数据文件所在目录"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"D:\workspace\researches\情绪\workspace\song_LD\save_model\refactored",
        help="模型和日志保存目录"
    )

    # 训练超参数
    parser.add_argument("--num_epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=3e-4, help="L2正则化系数")
    parser.add_argument("--dropout_p", type=float, default=0.4, help="Dropout概率")

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
# 数据加载与处理模块
# ============================================================================

def load_mat_var(path: str, candidates: List[str], squeeze: bool = True) -> np.ndarray:
    """
    通用MAT文件读取函数,支持v7.3/cell/ASCII格式

    Args:
        path: MAT文件路径
        candidates: 候选变量名列表
        squeeze: 是否压缩维度

    Returns:
        numpy数组
    """
    arr = None
    need_h5py = False

    # 尝试使用scipy.io读取
    try:
        m = sio.loadmat(path)
        for k in candidates:
            if k in m:
                arr = m[k]
                break
        if arr is None:
            for k, v in m.items():
                if not k.startswith('__'):
                    arr = v
                    break
    except NotImplementedError:
        need_h5py = True
    except Exception as e:
        if any(s in str(e).lower() for s in ["v7.3", "hdf"]):
            need_h5py = True
        else:
            raise

    # 使用h5py读取v7.3格式
    if arr is None and need_h5py:
        import h5py

        def _deref_any(obj, f):
            """递归解析h5py对象"""
            if isinstance(obj, (h5py.Reference, h5py.h5r.Reference)):
                if not obj:
                    return None
                return _deref_any(f[obj], f)

            if isinstance(obj, h5py.Group):
                for name in obj.keys():
                    got = _deref_any(obj[name], f)
                    if got is not None:
                        return got
                return None

            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                if isinstance(data, np.ndarray) and data.dtype.kind == 'O':
                    flat = []
                    for el in data.flat:
                        if isinstance(el, (h5py.Reference, h5py.h5r.Reference)):
                            val = _deref_any(el, f)
                        else:
                            val = el

                        if isinstance(val, np.ndarray):
                            if val.dtype.kind in ('U', 'S') or getattr(val.dtype, "char", '') == 'S':
                                try:
                                    s = ''.join(val.reshape(-1).astype(str).tolist())
                                except Exception:
                                    s = str(val)
                                flat.append(s)
                            elif np.issubdtype(val.dtype, np.integer) and val.size <= 16:
                                try:
                                    vv = val.reshape(-1)
                                    if np.all((vv >= 32) & (vv <= 126)):
                                        s = ''.join(chr(int(c)) for c in vv)
                                        flat.append(s)
                                    else:
                                        flat.append(vv[0] if vv.size >= 1 else np.nan)
                                except Exception:
                                    vv = val.reshape(-1)
                                    flat.append(vv[0] if vv.size >= 1 else np.nan)
                            else:
                                vv = val.reshape(-1)
                                flat.append(vv[0] if vv.size >= 1 else np.nan)
                        else:
                            flat.append(val)

                    out = np.array(flat, dtype=object).reshape(data.shape)
                    return out
                else:
                    return data
            return None

        with h5py.File(path, 'r') as f:
            for k in candidates:
                if k in f:
                    arr = _deref_any(f[k], f)
                    break
            if arr is None:
                for name in f.keys():
                    arr = _deref_any(f[name], f)
                    if arr is not None:
                        break
            if arr is None:
                raise KeyError(f"HDF5中未找到候选变量: {candidates}; 顶层键={list(f.keys())}")

    if arr is None:
        raise KeyError(f"MAT文件中未找到候选变量: {candidates}")

    arr = np.asarray(arr)
    if squeeze:
        arr = np.squeeze(arr)
    return arr


def load_and_preprocess_data(config: TrainingConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    加载和预处理数据

    Args:
        config: 训练配置对象

    Returns:
        (X, y, meta): 特征数组, 标签数组, 元数据字典
    """
    data_path = config.data_dir / "tea_train_data.mat"
    label_path = config.data_dir / "tea_train_valencelabel.mat"

    # 读取特征数据
    X = load_mat_var(str(data_path), ["tea_train_data", "data", "X"])
    X = np.asarray(X)

    if X.ndim != 3:
        raise ValueError(f"tea_train_data期望3维, 实际{X.shape}")

    print(f"[Debug] 原始X形状: {X.shape}")

    # 重新排列维度为 [N, C, T]
    shape = X.shape
    try:
        idx_ch = shape.index(54)
        idx_t = shape.index(2560)
    except ValueError:
        raise ValueError(f"在X.shape={shape}中找不到54或2560, 请检查数据")

    idx_all = {0, 1, 2}
    idx_b_list = list(idx_all - {idx_ch, idx_t})
    if len(idx_b_list) != 1:
        raise ValueError(f"维度无法唯一确定batch维: shape={shape}, idx_ch={idx_ch}, idx_t={idx_t}")
    idx_b = idx_b_list[0]

    X = np.transpose(X, (idx_b, idx_ch, idx_t))
    print(f"[Info] 重排后X形状: {X.shape} (应为[N,54,2560])")

    # 读取标签
    y_raw = load_mat_var(str(label_path), ["tea_train_valencelabel", "label", "y"])
    y_raw = np.asarray(y_raw)
    y_raw = np.squeeze(y_raw)
    if y_raw.ndim != 1:
        y_raw = y_raw.reshape(-1)

    # 转换标签: 1/2/3 -> 0/1/2
    y_values = y_raw.astype(np.int64)
    classes_values = np.unique(y_values)
    num_classes = int(classes_values.size)

    value2id = {v: i for i, v in enumerate(classes_values)}
    y = np.array([value2id[v] for v in y_values], dtype=np.int64)

    # 数据清洗
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"样本数不一致: X={X.shape[0]}, y={y.shape[0]}")
    if not (X.shape[1] == 54 and X.shape[2] == 2560):
        raise ValueError(f"数据维度应为(N,54,2560), 实际{X.shape}")

    print(f"[Check] 清理前: X finite={np.isfinite(X).all()}, y finite={np.isfinite(y).all()}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0).astype(np.int64)

    print(f"[Check] 清理后: X finite={np.isfinite(X).all()}, y finite={np.isfinite(y).all()}")
    print(f"[Info] 加载完成: X={X.shape}, y={y.shape}, 类别值={classes_values.tolist()} (K={num_classes})")
    print(f"[Info] 标签映射: {value2id}")

    meta = {
        "num_classes": num_classes,
        "classes_values": classes_values,
        "value2id": value2id,
    }

    return X, y, meta


# ============================================================================
# 数据集类
# ============================================================================

class GaitDataset(Dataset):
    """步态数据集类"""

    def __init__(self, data_array: np.ndarray, labels: np.ndarray,
                 use_zscore: bool = True, by_channel: bool = True):
        """
        Args:
            data_array: 特征数组, shape [N, C, T]
            labels: 标签数组, shape [N]
            use_zscore: 是否使用Z-score归一化
            by_channel: 是否按通道归一化
        """
        self.data_array = data_array
        self.labels = labels
        self.use_zscore = use_zscore
        self.by_channel = by_channel

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data_array[idx], dtype=torch.float32)

        if self.use_zscore:
            if self.by_channel:
                mean = x.mean(dim=1, keepdim=True)
                std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            else:
                mean = x.mean()
                std = x.std().clamp_min(1e-6)
            x = (x - mean) / std

        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


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

        feat_dim = getattr(self.encoder, "num_channels", 176)
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


# ============================================================================
# 训练与评估模块
# ============================================================================

class MetricsTracker:
    """指标跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有指标"""
        self.loss = 0.0
        self.correct = 0
        self.total = 0
        self.y_true = []
        self.y_pred = []
        self.valid_batches = 0

    def update(self, loss: float, logits: torch.Tensor, labels: torch.Tensor):
        """更新指标"""
        self.loss += loss
        self.valid_batches += 1

        preds = logits.argmax(1)
        self.y_true.extend(labels.detach().cpu().numpy().tolist())
        self.y_pred.extend(preds.detach().cpu().numpy().tolist())
        self.correct += (preds == labels).sum().item()
        self.total += labels.size(0)

    def compute(self) -> Dict[str, float]:
        """计算并返回所有指标"""
        loss = self.loss / self.valid_batches if self.valid_batches > 0 else float("nan")
        acc = self.correct / self.total if self.total > 0 else 0.0
        f1 = f1_score(self.y_true, self.y_pred, average='macro') if len(self.y_true) > 0 else 0.0

        return {
            "loss": loss,
            "accuracy": acc,
            "f1": f1,
        }


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
    train_dataset = GaitDataset(
        X[train_idx], y[train_idx],
        use_zscore=config.use_zscore,
        by_channel=config.zscore_by_channel
    )
    test_dataset = GaitDataset(
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
    model = ResNetModel(MSDNN(), num_classes=num_classes, dropout_p=config.dropout_p).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=cls_weights_t)

    scaler = torch.amp.GradScaler(enabled=(config.device.type == 'cuda' and config.use_amp))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config.lr_sched_patience,
        min_lr=config.min_lr, verbose=True
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
# 结果分析模块
# ============================================================================

def compute_cross_validation_summary(fold_results: List[Dict]) -> Dict[str, float]:
    """计算交叉验证汇总统计"""
    summaries = []

    for result in fold_results:
        hist = result["history"]
        summaries.append({
            "train_loss": hist["train_loss"][-1],
            "test_loss": hist["test_loss"][-1],
            "train_f1": hist["train_f1"][-1],
            "test_f1": hist["test_f1"][-1],
            "train_acc": hist["train_accuracy"][-1],
            "test_acc": hist["test_accuracy"][-1],
        })

    metrics = ["train_loss", "test_loss", "train_f1", "test_f1", "train_acc", "test_acc"]
    summary = {}

    for metric in metrics:
        values = [s[metric] for s in summaries]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))

    return summary


def plot_learning_curves(
    fold_results: List[Dict],
    save_path: Path,
    dpi: int = 150,
) -> None:
    """绘制学习曲线"""
    def pad_to_same_length(list_of_lists, pad_value=np.nan):
        max_len = max(len(lst) for lst in list_of_lists)
        out = np.full((len(list_of_lists), max_len), pad_value, dtype=np.float64)
        for i, lst in enumerate(list_of_lists):
            out[i, :len(lst)] = lst
        return out

    # 提取所有fold的历史数据
    fold_train_losses = [r["history"]["train_loss"] for r in fold_results]
    fold_test_losses = [r["history"]["test_loss"] for r in fold_results]
    fold_train_f1s = [r["history"]["train_f1"] for r in fold_results]
    fold_test_f1s = [r["history"]["test_f1"] for r in fold_results]
    fold_train_accs = [r["history"]["train_accuracy"] for r in fold_results]
    fold_test_accs = [r["history"]["test_accuracy"] for r in fold_results]

    # 填充到相同长度
    train_loss_mx = pad_to_same_length(fold_train_losses)
    test_loss_mx = pad_to_same_length(fold_test_losses)
    train_f1_mx = pad_to_same_length(fold_train_f1s)
    test_f1_mx = pad_to_same_length(fold_test_f1s)
    train_acc_mx = pad_to_same_length(fold_train_accs)
    test_acc_mx = pad_to_same_length(fold_test_accs)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(np.nanmean(train_loss_mx, axis=0), label="Train Loss (mean)", linewidth=2)
    axes[0, 0].plot(np.nanmean(test_loss_mx, axis=0), label="Test Loss (mean)", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # F1-score
    axes[0, 1].plot(np.nanmean(train_f1_mx, axis=0), label="Train F1 (macro, mean)", linewidth=2)
    axes[0, 1].plot(np.nanmean(test_f1_mx, axis=0), label="Test F1 (macro, mean)", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("F1-score (macro)")
    axes[0, 1].set_title("F1-score Curve")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 0].plot(np.nanmean(train_acc_mx, axis=0), label="Train Acc (mean)", linewidth=2)
    axes[1, 0].plot(np.nanmean(test_acc_mx, axis=0), label="Test Acc (mean)", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy Curve")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 隐藏第四个子图
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_training_summary(
    config: TrainingConfig,
    fold_results: List[Dict],
    cv_summary: Dict[str, float],
    meta: Dict[str, Any],
) -> None:
    """保存训练摘要"""
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "data_info": {
            "num_classes": meta["num_classes"],
            "classes_values": meta["classes_values"].tolist(),
            "value2id": {int(k): int(v) for k, v in meta["value2id"].items()},
        },
        "epochs_per_fold": [r["epochs_trained"] for r in fold_results],
        "cv_summary": cv_summary,
    }

    summary_path = config.save_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50} 交叉验证汇总 {'='*50}")
    for key, value in cv_summary.items():
        print(f"{key}: {value:.4f}")
    print(f"\n训练摘要已保存至: {summary_path}")


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
        config.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(config.tensorboard_dir)
        print(f"[Info] TensorBoard日志目录: {config.tensorboard_dir}")
        print(f"[Info] 启动TensorBoard: tensorboard --logdir={config.tensorboard_dir}\n")

    # 加载数据
    X, y, meta = load_and_preprocess_data(config)

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

    # 关闭TensorBoard writer
    if writer is not None:
        writer.close()

    # 计算交叉验证汇总
    cv_summary = compute_cross_validation_summary(fold_results)

    # 绘制学习曲线
    curves_path = config.save_dir / "learning_curves.png"
    plot_learning_curves(fold_results, curves_path)
    print(f"\n[Info] 学习曲线已保存至: {curves_path}")

    # 保存训练摘要
    save_training_summary(config, fold_results, cv_summary, meta)

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)


if __name__ == "__main__":
    main()
