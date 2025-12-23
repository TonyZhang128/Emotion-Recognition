# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# ====== 导入 MSDNN（两种路径都试一下）======
try:
    from models.MSDNN import MSDNN   # 若项目结构为 models/MSDNN.py
except ImportError:
    from MSDNN import MSDNN          # 若 tea_train.py 与 MSDNN.py 同目录

# ====== 全局通用配置（数据路径等）======
DATA_DIR   = r"D:\song_LD\data"      # ← 改成你的 .mat 所在目录
ROOT_SAVE_DIR = r"D:\song_LD\save_model\multi_try"   # 所有组合的总目录
os.makedirs(ROOT_SAVE_DIR, exist_ok=True)

# 通用训练设置（每个组合可以覆盖）
GLOBAL_NUM_EPOCHS = 50
GLOBAL_N_SPLITS   = 5
BATCH_SIZE        = 64
USE_ZSCORE        = True
ZSCORE_BY_CHANNEL = True
SEED              = 42

USE_AMP    = False           # 为稳妥起见先关掉 AMP
MAX_NORM   = 1.0
LR_SCHED_PATIENCE = 3
EARLY_PATIENCE    = 7
MIN_LR            = 1e-6

# 是否使用快速模式（快速粗筛组合，可自己改成 True/False）
FAST_MODE = False   # True：比如 n_splits=1, num_epochs=15 先大致看一下趋势


# ====== 随机种子 ======
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ====== 通用 .mat 读取（支持 v7.3 / cell / ASCII）======
def load_mat_var(path: str, candidates: list, squeeze=True):
    import numpy as _np

    arr = None
    need_h5py = False
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

    if arr is None and need_h5py:
        import h5py

        def _deref_any(obj, f):
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
                if isinstance(data, _np.ndarray) and data.dtype.kind == 'O':
                    flat = []
                    for el in data.flat:
                        if isinstance(el, (h5py.Reference, h5py.h5r.Reference)):
                            val = _deref_any(el, f)
                        else:
                            val = el

                        if isinstance(val, _np.ndarray):
                            if val.dtype.kind in ('U', 'S') or getattr(val.dtype, "char", '') == 'S':
                                try:
                                    s = ''.join(val.reshape(-1).astype(str).tolist())
                                except Exception:
                                    s = str(val)
                                flat.append(s)
                            elif _np.issubdtype(val.dtype, _np.integer) and val.size <= 16:
                                try:
                                    vv = val.reshape(-1)
                                    if _np.all((vv >= 32) & (vv <= 126)):
                                        s = ''.join(chr(int(c)) for c in vv)
                                        flat.append(s)
                                    else:
                                        flat.append(vv[0] if vv.size >= 1 else _np.nan)
                                except Exception:
                                    vv = val.reshape(-1)
                                    flat.append(vv[0] if vv.size >= 1 else _np.nan)
                            else:
                                vv = val.reshape(-1)
                                flat.append(vv[0] if vv.size >= 1 else _np.nan)
                        else:
                            flat.append(val)

                    out = _np.array(flat, dtype=object).reshape(data.shape)
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
                raise KeyError(f"HDF5 中未找到候选变量：{candidates}；顶层键={list(f.keys())}")

    if arr is None:
        raise KeyError(f"MAT 文件中未找到候选变量：{candidates}")

    arr = np.asarray(arr)
    if squeeze:
        arr = np.squeeze(arr)
    return arr


# ====== 数据集 ======
class GaitDataset(Dataset):
    def __init__(self, data_array: np.ndarray, labels: np.ndarray,
                 use_zscore: bool = True, by_channel: bool = True):
        self.data_array = data_array
        self.labels = labels
        self.use_zscore = use_zscore
        self.by_channel = by_channel

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        x = torch.tensor(self.data_array[idx], dtype=torch.float32)  # [C, T]
        if self.use_zscore:
            if self.by_channel:
                mean = x.mean(dim=1, keepdim=True)
                std  = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            else:
                mean = x.mean()
                std  = x.std().clamp_min(1e-6)
            x = (x - mean) / std

        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


# ====== 一次性读入 X / y（所有组合共用）======
data_path  = os.path.join(DATA_DIR, "tea_train_data.mat")
label_path = os.path.join(DATA_DIR, "tea_train_valencelabel.mat")  # 1×5421 double, 数值 1/2/3

# --- 读数据 X ---
X = load_mat_var(data_path,  ["tea_train_data", "data", "X"])
X = np.asarray(X)
if X.ndim != 3:
    raise ValueError(f"tea_train_data 期望 3 维，实际 {X.shape}")

print(f"[Debug] Raw X shape from MAT: {X.shape}")

shape = X.shape
try:
    idx_ch = shape.index(54)
    idx_t  = shape.index(2560)
except ValueError:
    raise ValueError(f"在 X.shape={shape} 中找不到 54 或 2560，请检查数据")

idx_all = {0, 1, 2}
idx_b_list = list(idx_all - {idx_ch, idx_t})
if len(idx_b_list) != 1:
    raise ValueError(f"维度无法唯一确定 batch 维：shape={shape}, idx_ch={idx_ch}, idx_t={idx_t}")
idx_b = idx_b_list[0]

X = np.transpose(X, (idx_b, idx_ch, idx_t))  # [N,54,2560]
print(f"[Info] Reordered X shape: {X.shape}  (should be [N,54,2560])")

# --- 读标签 y（1/2/3 double）---
y_raw = load_mat_var(label_path, ["tea_train_valencelabel", "label", "y"])
y_raw = np.asarray(y_raw)
y_raw = np.squeeze(y_raw)
if y_raw.ndim != 1:
    y_raw = y_raw.reshape(-1)

y_values = y_raw.astype(np.int64)         # [1,2,3,...]
classes_values = np.unique(y_values)      # e.g. [1,2,3]
num_classes = int(classes_values.size)

# 1/2/3 → 0/1/2
value2id = {v: i for i, v in enumerate(classes_values)}
y = np.array([value2id[v] for v in y_values], dtype=np.int64)

# --- 基本检查 + 清理 NaN/Inf ---
X = X.astype(np.float32)
y = y.astype(np.int64)

if X.shape[0] != y.shape[0]:
    raise ValueError(f"样本数不一致：X={X.shape[0]}, y={y.shape[0]}")
if not (X.shape[1] == 54 and X.shape[2] == 2560):
    raise ValueError(f"数据维度应为 (N,54,2560)，实际 {X.shape}")

print(f"[Check] Before nan_to_num: "
      f"X finite={np.isfinite(X).all()}, "
      f"y finite={np.isfinite(y).all()}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0, posinf=0, neginf=0).astype(np.int64)

print(f"[Check] After nan_to_num: "
      f"X finite={np.isfinite(X).all()}, "
      f"y finite={np.isfinite(y).all()}")

print(f"[Info] Loaded: X={X.shape}, y={y.shape}, classes_values={classes_values.tolist()} (K={num_classes})")
print(f"[Info] Label value -> id mapping: {value2id}")

# ====== 简单的一维卷积编码器，用于单模态 ======
class TemporalEncoder1D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int = 32):
        """
        in_channels: 输入通道数（EEG=24, ECG=2, GSR=1）
        base_channels: 第一层的通道数，可根据模态大小调整
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),          # T: 2560 -> 1280

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),          # T: 1280 -> 640

            nn.AdaptiveAvgPool1d(1)   # 聚成一个时间点
        )
        self.out_dim = base_channels * 2

    def forward(self, x):
        # x: [B, C, T]
        x = self.net(x)          # [B, C_out, 1]
        return x.squeeze(-1)     # [B, C_out]


# ====== 多模态老师模型：EEG/ECG/GSR 分支 + 融合 ======
class MultiModalTeacher(nn.Module):
    def __init__(self, num_classes: int, dropout_p: float = 0.5):
        """
        输入: x [B, 54, 2560]
        54 = 2 * 27,  27 = 24(EEG) + 2(ECG) + 1(GSR)
        """
        super().__init__()

        # EEG / ECG / GSR 各自的编码器（两个人共享同一个编码器）
        self.eeg_encoder = TemporalEncoder1D(in_channels=24, base_channels=64)
        self.ecg_encoder = TemporalEncoder1D(in_channels=2,  base_channels=16)
        self.gsr_encoder = TemporalEncoder1D(in_channels=1,  base_channels=8)

        eeg_dim = self.eeg_encoder.out_dim    # 64*2 = 128
        ecg_dim = self.ecg_encoder.out_dim    # 16*2 = 32
        gsr_dim = self.gsr_encoder.out_dim    # 8*2  = 16
        fusion_dim = eeg_dim + ecg_dim + gsr_dim

        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, x):
        """
        x: [B, 54, 2560]
        通道顺序假定为：
        - 人1: 0~23 EEG1, 24~25 ECG1, 26 GSR1
        - 人2: 27~50 EEG2, 51~52 ECG2, 53 GSR2
        """
        B, C, T = x.shape
        assert C == 54, f"Expect 54 channels, got {C}"

        # 拆成两个人
        p1 = x[:, :27, :]   # [B,27,T]
        p2 = x[:, 27:, :]   # [B,27,T]

        eeg1 = p1[:, :24, :]       # [B,24,T]
        ecg1 = p1[:, 24:26, :]     # [B,2,T]
        gsr1 = p1[:, 26:27, :]     # [B,1,T]

        eeg2 = p2[:, :24, :]
        ecg2 = p2[:, 24:26, :]
        gsr2 = p2[:, 26:27, :]

        # 两个人分别过同一个模态编码器
        f_eeg1 = self.eeg_encoder(eeg1)
        f_eeg2 = self.eeg_encoder(eeg2)
        f_ecg1 = self.ecg_encoder(ecg1)
        f_ecg2 = self.ecg_encoder(ecg2)
        f_gsr1 = self.gsr_encoder(gsr1)
        f_gsr2 = self.gsr_encoder(gsr2)

        # 在“模态内部”做人物融合：这里用平均，也可以改成 concat
        f_eeg = 0.5 * (f_eeg1 + f_eeg2)
        f_ecg = 0.5 * (f_ecg1 + f_ecg2)
        f_gsr = 0.5 * (f_gsr1 + f_gsr2)

        # 多模态拼接
        f_all = torch.cat([f_eeg, f_ecg, f_gsr], dim=1)   # [B, fusion_dim]

        f_all = self.dropout(f_all)
        logits = self.classifier(f_all)                  # [B, num_classes]
        return logits




# ====== 工具函数：pad 曲线到同一长度 ======
def pad_to_same_length(list_of_lists, pad_value=np.nan):
    max_len = max(len(lst) for lst in list_of_lists)
    out = np.full((len(list_of_lists), max_len), pad_value, dtype=np.float64)
    for i, lst in enumerate(list_of_lists):
        out[i, :len(lst)] = lst
    return out


# ====== 核心：跑一个超参组合（可以 5 折 / 1 折） ======
def run_one_config(config_name: str,
                   lr: float,
                   weight_decay: float,
                   dropout_p: float,
                   num_epochs: int = GLOBAL_NUM_EPOCHS,
                   n_splits: int = GLOBAL_N_SPLITS,
                   fast_mode: bool = False):
    """
    单个组合：
      - 指定 lr / weight_decay / dropout_p
      - 进行 n_splits 折交叉验证，训练 num_epochs 以内（有早停）
      - 保存 best_model_foldX、summary 和 曲线图
      - 返回一个 summary dict（里面有 test_f1_mean / test_acc_mean 等）
    """

    # 快速模式：粗略搜索，用更少的 epoch & 折数
    if fast_mode:
        n_splits = 1 if n_splits > 1 else n_splits
        num_epochs = min(num_epochs, 15)
        print(f"[FastMode] {config_name}: n_splits={n_splits}, num_epochs={num_epochs}")

    # 每个组合用单独目录
    save_dir = os.path.join(
        ROOT_SAVE_DIR,
        f"{config_name}_lr{lr}_wd{weight_decay}_dp{dropout_p}"
    )
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n================ Run config {config_name} ================")
    print(f"lr={lr}, weight_decay={weight_decay}, dropout_p={dropout_p}, "
          f"num_epochs={num_epochs}, n_splits={n_splits}")
    print(f"Save to: {save_dir}")

    # 为了可比较，每个组合重新设一次随机种子
    set_seed(SEED)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_train_losses, fold_test_losses = [], []
    fold_train_f1s,  fold_test_f1s   = [], []
    fold_train_accs, fold_test_accs  = [], []
    fold_train_last_f1, fold_test_last_f1 = [], []
    fold_train_last_acc, fold_test_last_acc = [], []
    fold_train_last_loss, fold_test_last_loss = [], []
    epochs_per_fold = []

    pin = torch.cuda.is_available()

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold}/{n_splits} =====")

        train_dataset = GaitDataset(X[train_idx], y[train_idx],
                                    use_zscore=USE_ZSCORE, by_channel=ZSCORE_BY_CHANNEL)
        test_dataset  = GaitDataset(X[test_idx],  y[test_idx],
                                    use_zscore=USE_ZSCORE, by_channel=ZSCORE_BY_CHANNEL)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=pin)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=pin)

        # 类别权重，缓解不平衡
        y_train_fold = y[train_idx]
        classes0 = np.arange(num_classes)
        cls_weights = compute_class_weight('balanced', classes=classes0, y=y_train_fold)
        cls_weights_t = torch.tensor(cls_weights, dtype=torch.float32, device=device)
        print(f"[Info][Fold{fold}] Class weights (balanced): {cls_weights.tolist()}")

        model = MultiModalTeacher(num_classes=num_classes, dropout_p=dropout_p).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=cls_weights_t)

        scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda' and USE_AMP))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=LR_SCHED_PATIENCE,
            min_lr=MIN_LR, verbose=True
        )
        best_loss = float("inf")
        no_improve = 0
        best_path = os.path.join(save_dir, f"best_model_fold{fold}.pth")

        epoch_train_losses, epoch_test_losses = [], []
        epoch_train_f1s,  epoch_test_f1s  = [], []
        epoch_train_accs, epoch_test_accs = [], []

        for epoch in range(num_epochs):
            # ----- Train -----
            model.train()
            run_loss = 0.0
            valid_train_batches = 0
            y_true_train, y_pred_train = [], []
            correct_train, total_train = 0, 0

            for b_idx, (xb, yb) in enumerate(train_loader):
                xb = xb.to(device, dtype=torch.float32)
                yb = yb.to(device, dtype=torch.long)

                if not torch.isfinite(xb).all():
                    continue

                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device == 'cuda' and USE_AMP)):
                    logits = model(xb)
                    if not torch.isfinite(logits).all():
                        continue
                    loss = criterion(logits, yb)

                if not torch.isfinite(loss):
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
                scaler.step(optimizer)
                scaler.update()

                run_loss += loss.item()
                valid_train_batches += 1

                pred = logits.argmax(1)
                y_true_train.extend(yb.detach().cpu().numpy().tolist())
                y_pred_train.extend(pred.detach().cpu().numpy().tolist())
                correct_train += (pred == yb).sum().item()
                total_train   += yb.size(0)

            train_loss = run_loss / valid_train_batches if valid_train_batches > 0 else float("nan")
            train_f1   = f1_score(y_true_train, y_pred_train, average='macro') if len(y_true_train) > 0 else 0.0
            train_acc  = correct_train / total_train if total_train > 0 else 0.0

            # ----- Eval -----
            model.eval()
            run_loss = 0.0
            valid_test_batches = 0
            y_true_test, y_pred_test = [], []
            correct_test, total_test = 0, 0

            with torch.no_grad():
                for b_idx, (xb, yb) in enumerate(test_loader):
                    xb = xb.to(device, dtype=torch.float32)
                    yb = yb.to(device, dtype=torch.long)

                    if not torch.isfinite(xb).all():
                        continue

                    with torch.cuda.amp.autocast(enabled=(device == 'cuda' and USE_AMP)):
                        logits = model(xb)
                        if not torch.isfinite(logits).all():
                            continue
                        loss = criterion(logits, yb)

                    if not torch.isfinite(loss):
                        continue

                    run_loss += loss.item()
                    valid_test_batches += 1

                    pred = logits.argmax(1)
                    y_true_test.extend(yb.detach().cpu().numpy().tolist())
                    y_pred_test.extend(pred.detach().cpu().numpy().tolist())
                    correct_test += (pred == yb).sum().item()
                    total_test   += yb.size(0)

            test_loss = run_loss / valid_test_batches if valid_test_batches > 0 else float("nan")
            test_f1   = f1_score(y_true_test, y_pred_test, average='macro') if len(y_true_test) > 0 else 0.0
            test_acc  = correct_test / total_test if total_test > 0 else 0.0

            if np.isfinite(test_loss):
                scheduler.step(test_loss)

            improved = (np.isfinite(test_loss) and test_loss < best_loss - 1e-6)
            if improved:
                best_loss = test_loss
                no_improve = 0
                torch.save(model.state_dict(), best_path)
            else:
                no_improve += 1

            epoch_train_losses.append(train_loss)
            epoch_test_losses.append(test_loss)
            epoch_train_f1s.append(train_f1)
            epoch_test_f1s.append(test_f1)
            epoch_train_accs.append(train_acc)
            epoch_test_accs.append(test_acc)

            print(f"Fold {fold} | Epoch {epoch+1:02d}/{num_epochs} | "
                  f"TrainLoss={train_loss:.4f} TrainF1={train_f1:.4f} TrainAcc={train_acc:.4f} | "
                  f"TestLoss={test_loss:.4f} TestF1={test_f1:.4f} TestAcc={test_acc:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e} | "
                  f"{'*BEST*' if improved else ''}")

            if no_improve >= EARLY_PATIENCE:
                print(f"[EarlyStop] Fold {fold}: no improve {EARLY_PATIENCE} epochs. Stop early.")
                break

        fold_train_losses.append(epoch_train_losses)
        fold_test_losses.append(epoch_test_losses)
        fold_train_f1s.append(epoch_train_f1s)
        fold_test_f1s.append(epoch_test_f1s)
        fold_train_accs.append(epoch_train_accs)
        fold_test_accs.append(epoch_test_accs)

        fold_train_last_loss.append(epoch_train_losses[-1])
        fold_test_last_loss.append(epoch_test_losses[-1])
        fold_train_last_f1.append(epoch_train_f1s[-1])
        fold_test_last_f1.append(epoch_test_f1s[-1])
        fold_train_last_acc.append(epoch_train_accs[-1])
        fold_test_last_acc.append(epoch_test_accs[-1])
        epochs_per_fold.append(len(epoch_train_losses))

    # ====== 折间统计与保存（F1 + Acc） ======
    mean_train_f1 = float(np.mean(fold_train_last_f1))
    std_train_f1  = float(np.std(fold_train_last_f1))
    mean_test_f1  = float(np.mean(fold_test_last_f1))
    std_test_f1   = float(np.std(fold_test_last_f1))

    mean_train_acc = float(np.mean(fold_train_last_acc))
    std_train_acc  = float(np.std(fold_train_last_acc))
    mean_test_acc  = float(np.mean(fold_test_last_acc))
    std_test_acc   = float(np.std(fold_test_last_acc))

    mean_train_los = float(np.mean(fold_train_last_loss))
    std_train_los  = float(np.std(fold_train_last_loss))
    mean_test_los  = float(np.mean(fold_test_last_loss))
    std_test_los   = float(np.std(fold_test_last_loss))

    summary = {
        "config_name": config_name,
        "fast_mode": fast_mode,
        "num_epochs_target": num_epochs,
        "epochs_per_fold": epochs_per_fold,
        "batch_size": BATCH_SIZE,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout_p": dropout_p,
        "n_splits": n_splits,
        "use_zscore": USE_ZSCORE,
        "zscore_by_channel": ZSCORE_BY_CHANNEL,
        "seed": SEED,
        "classes_values": classes_values.tolist(),
        "value2id": {int(k): int(v) for k, v in value2id.items()},
        "train_f1_mean": mean_train_f1,
        "train_f1_std":  std_train_f1,
        "test_f1_mean":  mean_test_f1,
        "test_f1_std":   std_test_f1,
        "train_acc_mean": mean_train_acc,
        "train_acc_std":  std_train_acc,
        "test_acc_mean":  mean_test_acc,
        "test_acc_std":   std_test_acc,
        "train_loss_mean": mean_train_los,
        "train_loss_std":  std_train_los,
        "test_loss_mean":  mean_test_los,
        "test_loss_std":   std_test_los,
    }
    print("\n===== SUMMARY for config", config_name, "=====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(os.path.join(save_dir, "teacher_summary_f1_acc.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ====== 绘图（平均 Loss / F1 / Acc 曲线）======
    train_loss_mx = pad_to_same_length(fold_train_losses)
    test_loss_mx  = pad_to_same_length(fold_test_losses)
    train_f1_mx   = pad_to_same_length(fold_train_f1s)
    test_f1_mx    = pad_to_same_length(fold_test_f1s)
    train_acc_mx  = pad_to_same_length(fold_train_accs)
    test_acc_mx   = pad_to_same_length(fold_test_accs)

    plt.figure(figsize=(14, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(np.nanmean(train_loss_mx, axis=0), label="Train Loss (mean)")
    plt.plot(np.nanmean(test_loss_mx,  axis=0), label="Test Loss (mean)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    # F1
    plt.subplot(2, 2, 2)
    plt.plot(np.nanmean(train_f1_mx, axis=0), label="Train F1 (macro, mean)")
    plt.plot(np.nanmean(test_f1_mx,  axis=0), label="Test  F1 (macro, mean)")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score (macro)")
    plt.title("F1-score")
    plt.legend()

    # Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(np.nanmean(train_acc_mx, axis=0), label="Train Acc (mean)")
    plt.plot(np.nanmean(test_acc_mx,  axis=0), label="Test  Acc (mean)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "teacher_learning_curves_f1_acc.png"), dpi=150)
    plt.close()

    return summary


# ====== 要批量尝试的组合列表（你可以随便改/加）======
CONFIGS = [
    # name,      lr,     weight_decay, dropout
    ("A", 1e-4, 1e-4, 0.5),
    ("C", 1e-4, 3e-4, 0.4),
    ("D", 1e-4, 1e-4, 0.4),
    ("E", 5e-5, 5e-4, 0.5),
    ("F", 7e-5, 1e-4, 0.5),
    ("G", 1e-4, 1e-4, 0.6),
    ("H", 1e-4, 3e-4, 0.4),
]

# ====== 主程序：依次跑每个组合，并按 F1 排序展示 ======
if __name__ == "__main__":
    all_summaries = []
    for name, lr, wd, dp in CONFIGS:
        summary = run_one_config(
            config_name=name,
            lr=lr,
            weight_decay=wd,
            dropout_p=dp,
            num_epochs=GLOBAL_NUM_EPOCHS,
            n_splits=GLOBAL_N_SPLITS,
            fast_mode=FAST_MODE
        )
        all_summaries.append(summary)

    # 按 test_f1_mean 排个序看一下
    all_summaries.sort(key=lambda d: d["test_f1_mean"], reverse=True)
    print("\n=========== All configs sorted by test_f1_mean (desc) ===========")
    for s in all_summaries:
        print(f"{s['config_name']}: "
              f"test_f1_mean={s['test_f1_mean']:.4f}, "
              f"test_acc_mean={s['test_acc_mean']:.4f}, "
              f"lr={s['lr']}, wd={s['weight_decay']}, dp={s['dropout_p']}")
