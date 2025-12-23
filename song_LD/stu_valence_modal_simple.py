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

# ====== 导入 MSDNN（和老师模型一样）======
try:
    from models.MSDNN import MSDNN   # 若项目结构为 models/MSDNN.py
except ImportError:
    from MSDNN import MSDNN          # 若 stu_xxx.py 与 MSDNN.py 同目录

# ====== 全局配置 ======
DATA_DIR   = r"D:\song_LD\data"                       # 你的学生数据 .mat 所在目录
SAVE_ROOT  = r"D:\song_LD\stu_models_valence"         # 每个模态会建子文件夹
os.makedirs(SAVE_ROOT, exist_ok=True)

NUM_EPOCHS = 50
BATCH_SIZE = 64
LR          = 1e-4          # 用你老师模型那组较好的超参
WEIGHT_DECAY = 3e-4
DROPOUT_P    = 0.5
N_SPLITS   = 5
USE_ZSCORE = True
ZSCORE_BY_CHANNEL = True
SEED       = 42

USE_AMP    = False           # 学生模型先关掉 AMP，更稳定
MAX_NORM   = 1.0
LR_SCHED_PATIENCE = 3
EARLY_PATIENCE    = 7
MIN_LR            = 1e-6

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

# ====== 读 mat 中某个变量的简单函数（学生版不需要 H/L 解码那么复杂）======
def load_mat_var_simple(path: str, candidates, squeeze=True):
    """
    尝试：
      1) 用 scipy.io.loadmat 读取 (非 v7.3)
      2) 如果报 NotImplementedError / v7.3，则用 h5py 读取
    返回 np.ndarray，并根据 squeeze 参数决定是否压缩维度
    """
    import numpy as _np

    arr = None
    need_h5py = False

    # ---------- 先试 scipy.io.loadmat ----------
    try:
        m = sio.loadmat(path)
        for k in candidates:
            if k in m:
                arr = m[k]
                break
        if arr is None:
            # 退而求其次：拿一个非 "__" 开头的变量
            for k, v in m.items():
                if not k.startswith("__"):
                    arr = v
                    break
    except NotImplementedError:
        # v7.3 / HDF5 的 mat 文件
        need_h5py = True
    except Exception as e:
        # 有些版本会在报错信息里带 v7.3 / hdf 字样
        if any(s in str(e).lower() for s in ["v7.3", "hdf"]):
            need_h5py = True
        else:
            raise

    # ---------- 若需要，用 h5py 读取 v7.3 ----------
    if arr is None and need_h5py:
        import h5py
        with h5py.File(path, "r") as f:
            # 优先找 candidates 里的变量名
            for k in candidates:
                if k in f:
                    arr = f[k][()]      # 直接取 dataset 内容
                    break
            # 找不到就拿第一个顶层变量
            if arr is None:
                for name in f.keys():
                    arr = f[name][()]
                    break
        if arr is None:
            raise KeyError(f"{path} (v7.3) 中未找到候选变量 {candidates}")

    if arr is None:
        raise KeyError(f"{path} 中未找到候选变量 {candidates}")

    arr = _np.asarray(arr)
    if squeeze:
        arr = _np.squeeze(arr)
    return arr


# ====== 通用：把 X 重排成 [N, C, T]，T=2560 ======
def reorder_to_NCT(X, T_expected=2560):
    """
    输入 X 可能是:
      (N, C, T) 或 (T, C, N) 或 (C, T, N) 等
    只要里面有一个维度=通道数C，一个维度=T_expected，一个维度=N，我们就能通过 index 定位。
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"期望 3 维数据, 实际 {X.shape}")

    shape = list(X.shape)
    # 找时间维
    try:
        idx_t = shape.index(T_expected)
    except ValueError:
        raise ValueError(f"在 X.shape={shape} 中找不到时间维 {T_expected}")

    # 剩下两维分别是 N 和 C，这个在 ECG/EEG/GSR 时我们会提前知道“C期望值”，
    # 但为了通用，这里先不判断 C ，直接把剩下两维一个当 N 一个当 C。
    idx_all = {0, 1, 2}
    others = list(idx_all - {idx_t})
    # others[0], others[1] 两个位置中，一个是 N，一个是 C
    # 我们后面知道 C_expected 时再校验。
    return X, idx_t, others

# ====== 数据集 ======
class StudentDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 use_zscore=True, by_channel=True):
        self.X = X       # [N, C, T]
        self.y = y       # [N]
        self.use_zscore = use_zscore
        self.by_channel = by_channel

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # [C, T]
        if self.use_zscore:
            if self.by_channel:
                mean = x.mean(dim=1, keepdim=True)
                std  = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            else:
                mean = x.mean()
                std  = x.std().clamp_min(1e-6)
            x = (x - mean) / std

        y = torch.tensor(int(self.y[idx]), dtype=torch.long)  # 0/1
        return x, y

# ====== 学生模型：MSDNN(in_channels=C) + Dropout + Linear ======
# ====== 学生网络：前面加一个通道适配器，再接老师的 MSDNN ======
class StudentNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout_p: float = 0.5):
        """
        in_channels: 学生模态的通道数
            - ECG: 2
            - EEG: 24
            - GSR: 1
        num_classes: 分类类别数（现在是 2：0/1）
        dropout_p:  分类头前的 dropout 概率
        """
        super().__init__()

        # 1) 通道适配器：把 C_in → 54（老师 MSDNN 的输入通道）
        self.adapter = nn.Conv1d(
            in_channels=in_channels,
            out_channels=54,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # 2) 老师的 MSDNN（保持原样，只能吃 54 通道）
        self.encoder = MSDNN()
        # 去掉老师模型最后的 fc，全当“特征提取器”
        if hasattr(self.encoder, "fc"):
            self.encoder.fc = nn.Identity()

        # 3) 读出 MSDNN 最终特征维度（你之前用过 num_channels=176）
        feat_dim = getattr(self.encoder, "num_channels", 176)

        # 4) Dropout + 线性分类头
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear  = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        """
        x: [B, C_in, 2560]  比如 ECG: [B, 2, 2560]
        """
        # 先把通道对齐到 54
        x = self.adapter(x)          # → [B, 54, 2560]

        # 再喂给老师 MSDNN
        x = self.encoder(x)          # → [B, feat_dim] 或 [B, feat_dim, 1]

        x = x.view(x.size(0), -1)    # 压成 [B, feat_dim]
        x = self.dropout(x)
        logits = self.linear(x)      # [B, num_classes]
        return logits


# ====== 训练单个模态（ECG / EEG / GSR 共用）======
def train_student_modal(modal_name: str,
                        data_mat_name: str,
                        label_mat_name: str,
                        C_expected: int,
                        num_epochs: int = NUM_EPOCHS,
                        lr: float = LR,
                        weight_decay: float = WEIGHT_DECAY,
                        dropout_p: float = DROPOUT_P,
                        n_splits: int = N_SPLITS):
    """
    modal_name: 'ECG' / 'EEG' / 'GSR'
    data_mat_name: 例如 'stu_test_valence_ECGdata.mat'
    label_mat_name: 例如 'stu_test_valence_ECGlabel.mat'
    C_expected: 通道数（ECG=2, EEG=24, GSR=1）
    """

    save_dir = os.path.join(
        SAVE_ROOT,
        f"{modal_name}_lr{lr}_wd{weight_decay}_dp{dropout_p}"
    )
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n========== [{modal_name}] 开始训练 ==========")

    # 1) 读数据
    data_path  = os.path.join(DATA_DIR, data_mat_name)
    label_path = os.path.join(DATA_DIR, label_mat_name)

    # X_raw = load_mat_var_simple(data_path, ["data", "X", "stu_test_data", "stu_test_valence"])
    # print(f"[Debug][{modal_name}] raw X shape from MAT: {X_raw.shape}")
    #
    # # 通用 reorder
    # X_raw = np.asarray(X_raw)
    # X_raw, idx_t, others = reorder_to_NCT(X_raw, T_expected=2560)
    # shape = list(X_raw.shape)
    #
    # # 确认 C/N 位置
    # # others 里两个 index，其中一个是 C_expected
    # idx_c = None
    # idx_b = None
    # for i in others:
    #     if shape[i] == C_expected:
    #         idx_c = i
    #     else:
    #         idx_b = i
    # if idx_c is None or idx_b is None:
    #     raise ValueError(f"[{modal_name}] 无法根据 C_expected={C_expected} 在 {shape} 中确定通道维/样本维")
    #
    # # 转成 [N, C, T]
    # X = np.transpose(X_raw, (idx_b, idx_c, idx_t))
    # print(f"[Info][{modal_name}] reorder_to_NCT: shape {shape} -> {X.shape} (N,C,T) = ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")
    X_raw = load_mat_var_simple(data_path, ["data", "X", "stu_test_data", "stu_test_valence"])
    print(f"[Debug][{modal_name}] raw X shape from MAT: {X_raw.shape}")

    X_raw = np.asarray(X_raw)

    # ===== 特例：GSR / 单通道，数据被 squeeze 成 2 维 =====
    if X_raw.ndim == 2 and C_expected == 1:
        # 形状可能是 (2560, N) 或 (N, 2560)
        if X_raw.shape[0] == 2560:
            # [T, N] -> [N, T]
            X_nt = X_raw.T
        elif X_raw.shape[1] == 2560:
            # 已经是 [N, T]
            X_nt = X_raw
        else:
            raise ValueError(f"[{modal_name}] GSR 数据形状异常：{X_raw.shape}，找不到长度为 2560 的时间维")

        # 加上通道维 C=1，得到 [N, 1, T]
        X = X_nt[:, np.newaxis, :]   # [N, 1, 2560]
        print(f"[Info][{modal_name}] GSR special case: shape {X_raw.shape} -> {X.shape} (N= {X.shape[0]}, C=1, T=2560)")

    else:
        # ===== 通用 3 维情况：ECG / EEG 等 =====
        X_raw, idx_t, others = reorder_to_NCT(X_raw, T_expected=2560)
        shape = list(X_raw.shape)

        # others 里两个 index，其中一个是 C_expected
        idx_c = None
        idx_b = None
        for i in others:
            if shape[i] == C_expected:
                idx_c = i
            else:
                idx_b = i
        if idx_c is None or idx_b is None:
            raise ValueError(f"[{modal_name}] 无法根据 C_expected={C_expected} 在 {shape} 中确定通道维/样本维")

        # 转成 [N, C, T]
        X = np.transpose(X_raw, (idx_b, idx_c, idx_t))
        print(f"[Info][{modal_name}] reorder_to_NCT: shape {shape} -> {X.shape} (N,C,T) = ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")

    # 2) 读标签（2206×1 double, 0/1）
    y_raw = load_mat_var_simple(label_path, ["label", "valence_label", "stu_test_label", "y"])
    y_raw = np.asarray(y_raw)
    y_raw = np.squeeze(y_raw)
    if y_raw.ndim != 1:
        y_raw = y_raw.reshape(-1)

    y = y_raw.astype(np.int64)   # 0 or 1
    print(f"[Debug][{modal_name}] y shape from MAT: {y.shape}, unique={np.unique(y).tolist()}")

    # 3) 基本检查 + 清理 NaN/Inf
    X = X.astype(np.float32)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"[{modal_name}] 样本数不一致：X={X.shape[0]}, y={y.shape[0]}")

    print(f"[Check][{modal_name}] Before nan_to_num: X finite={np.isfinite(X).all()}, y finite={np.isfinite(y).all()}")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0).astype(np.int64)
    print(f"[Check][{modal_name}] After nan_to_num:  X finite={np.isfinite(X).all()}, y finite={np.isfinite(y).all()}")

    num_classes = len(np.unique(y))
    assert num_classes == 2, f"[{modal_name}] 目前脚本假定二分类，实际类别数={num_classes}"

    print(f"[Info][{modal_name}] Loaded: X={X.shape}, y={y.shape}, num_classes={num_classes}")

    # ====== 5 折 CV ======
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_train_losses, fold_test_losses = [], []
    fold_train_f1s,  fold_test_f1s   = [], []
    fold_train_accs, fold_test_accs  = [], []
    fold_train_last_f1, fold_test_last_f1 = [], []
    fold_train_last_acc, fold_test_last_acc = [], []
    fold_train_last_loss, fold_test_last_loss = [], []
    epochs_per_fold = []

    pin = torch.cuda.is_available()

    for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== [{modal_name}] Fold {fold_id}/{n_splits} =====")

        train_dataset = StudentDataset(X[train_idx], y[train_idx],
                                       use_zscore=USE_ZSCORE,
                                       by_channel=ZSCORE_BY_CHANNEL)
        test_dataset  = StudentDataset(X[test_idx],  y[test_idx],
                                       use_zscore=USE_ZSCORE,
                                       by_channel=ZSCORE_BY_CHANNEL)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=pin)
        test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=pin)

        # 类别权重
        y_train_fold = y[train_idx]
        classes0 = np.arange(num_classes)
        cls_weights = compute_class_weight('balanced', classes=classes0, y=y_train_fold)
        cls_weights_t = torch.tensor(cls_weights, dtype=torch.float32, device=device)
        print(f"[Info][{modal_name}][Fold{fold_id}] Class weights (balanced): {cls_weights.tolist()}")

        model = StudentNet(in_channels=C_expected,
                           num_classes=num_classes,
                           dropout_p=dropout_p).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=cls_weights_t)

        scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda' and USE_AMP))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=LR_SCHED_PATIENCE,
            min_lr=MIN_LR, verbose=True
        )
        best_loss = float("inf")
        no_improve = 0
        best_path = os.path.join(save_dir, f"{modal_name}_best_model_fold{fold_id}.pth")

        epoch_train_losses, epoch_test_losses = [], []
        epoch_train_f1s,  epoch_test_f1s  = [], []
        epoch_train_accs, epoch_test_accs = [], []

        for epoch in range(num_epochs):
            # ---- Train ----
            model.train()
            run_loss = 0.0
            valid_train_batches = 0
            y_true_train, y_pred_train = [], []
            correct_train, total_train = 0, 0

            for xb, yb in train_loader:
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

            # ---- Eval ----
            model.eval()
            run_loss = 0.0
            valid_test_batches = 0
            y_true_test, y_pred_test = [], []
            correct_test, total_test = 0, 0

            with torch.no_grad():
                for xb, yb in test_loader:
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

            print(f"[{modal_name}] Fold {fold_id} | Epoch {epoch+1:02d}/{num_epochs} | "
                  f"TrainLoss={train_loss:.4f} TrainF1={train_f1:.4f} TrainAcc={train_acc:.4f} | "
                  f"TestLoss={test_loss:.4f} TestF1={test_f1:.4f} TestAcc={test_acc:.4f} | "
                  f"LR={optimizer.param_groups[0]['lr']:.2e} | "
                  f"{'*BEST*' if improved else ''}")

            if no_improve >= EARLY_PATIENCE:
                print(f"[EarlyStop][{modal_name}] Fold {fold_id}: no improve {EARLY_PATIENCE} epochs. Stop.")
                break

        # 折内汇总
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

    # ====== 折间统计 ======
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
        "modal_name": modal_name,
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
    print(f"\n===== [{modal_name}] 5 折汇总（最后一轮指标） =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(os.path.join(save_dir, f"{modal_name}_summary_f1_acc.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ====== 画平均曲线 ======
    def pad_to_same_length(list_of_lists, pad_value=np.nan):
        max_len = max(len(lst) for lst in list_of_lists)
        out = np.full((len(list_of_lists), max_len), pad_value, dtype=np.float64)
        for i, lst in enumerate(list_of_lists):
            out[i, :len(lst)] = lst
        return out

    train_loss_mx = pad_to_same_length(fold_train_losses)
    test_loss_mx  = pad_to_same_length(fold_test_losses)
    train_f1_mx   = pad_to_same_length(fold_train_f1s)
    test_f1_mx    = pad_to_same_length(fold_test_f1s)
    train_acc_mx  = pad_to_same_length(fold_train_accs)
    test_acc_mx   = pad_to_same_length(fold_test_accs)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(np.nanmean(train_loss_mx, axis=0), label="Train Loss (mean)")
    plt.plot(np.nanmean(test_loss_mx,  axis=0), label="Test Loss (mean)")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{modal_name} - Loss"); plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(np.nanmean(train_f1_mx, axis=0), label="Train F1 (macro, mean)")
    plt.plot(np.nanmean(test_f1_mx,  axis=0), label="Test F1 (macro, mean)")
    plt.xlabel("Epoch"); plt.ylabel("F1-score (macro)"); plt.title(f"{modal_name} - F1"); plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(np.nanmean(train_acc_mx, axis=0), label="Train Acc (mean)")
    plt.plot(np.nanmean(test_acc_mx,  axis=0), label="Test Acc (mean)")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{modal_name} - Accuracy"); plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{modal_name}_learning_curves_f1_acc.png"), dpi=150)
    plt.close()

    return summary

# ====== 主函数：按需注释/启用不同模态 ======
if __name__ == "__main__":
    # 先只跑 ECG（valence）
    # train_student_modal(
    #     modal_name="ECG_valence",
    #     data_mat_name="stu_test_valence_ECGdata.mat",
    #     label_mat_name="stu_test_valence_ECGlabel.mat",
    #     C_expected=2,
    # )

    # train_student_modal(
    #     modal_name="EEG_valence",
    #     data_mat_name="stu_test_valence_EEGdata.mat",
    #     label_mat_name="stu_test_valence_EEGlabel.mat",
    #     C_expected=24,
    # )


    train_student_modal(
        modal_name="GSR_valence",
        data_mat_name="stu_test_valence_GSRdata.mat",
        label_mat_name="stu_test_valence_GSRlabel.mat",
        C_expected=1,
    )
