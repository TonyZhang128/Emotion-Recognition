# -*- coding: utf-8 -*-
import os, json, random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# ===================== 0) 全局配置 =====================
DATA_DIR = r"D:\song_LD\data"
SAVE_DIR = os.path.join(DATA_DIR, "teacher_pair2bin_py")
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_MAT  = os.path.join(DATA_DIR, "tea_train_data.mat")
LABEL_MAT = os.path.join(DATA_DIR, "tea_train_arousallabel_num.mat")

NUM_EPOCHS   = 40
BATCH_SIZE   = 64
LR           = 1e-4
WEIGHT_DECAY = 3e-4
DROPOUT_P    = 0.5
N_SPLITS     = 5
SEED         = 42

USE_ZSCORE   = True
ZSCORE_BY_CH = True

USE_AMP      = False
MAX_NORM     = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 1) 工具函数 =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

def load_first_var(mat_path: str):
    """
    读取 mat 文件第一个“顶层变量”
    - 普通 mat：scipy.io.loadmat
    - v7.3 mat：h5py 读取 HDF5
    """
    # 1) 先试普通 mat
    try:
        m = sio.loadmat(mat_path)
        for k, v in m.items():
            if not k.startswith("__"):
                return np.asarray(v)
        raise KeyError(f"No valid variable found in {mat_path}")
    except NotImplementedError:
        pass
    except Exception as e:
        # 有些版本会用别的异常提示 v7.3/hdf5
        if not any(s in str(e).lower() for s in ["v7.3", "hdf", "hdf5"]):
            raise

    # 2) v7.3：用 h5py
    import h5py
    with h5py.File(mat_path, "r") as f:
        # 找第一个顶层 key
        keys = list(f.keys())
        if len(keys) == 0:
            raise KeyError(f"No datasets in v7.3 mat: {mat_path}")
        k0 = keys[0]
        arr = f[k0][()]   # 直接取数据
        return np.asarray(arr)


def reorder_to_NCT(X: np.ndarray, C_expected=54, T_expected=2560):
    """
    把任意 (N,C,T)/(C,T,N)/(T,N,C)... 变成 [N,C,T]
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"期望 3 维数据, 实际 {X.shape}")

    shape = list(X.shape)
    try:
        idx_c = shape.index(C_expected)
        idx_t = shape.index(T_expected)
    except ValueError:
        raise ValueError(f"在 shape={shape} 里找不到 C={C_expected} 或 T={T_expected}")

    idx_all = {0, 1, 2}
    idx_n = list(idx_all - {idx_c, idx_t})
    if len(idx_n) != 1:
        raise ValueError(f"无法唯一确定 N 维: shape={shape}")
    idx_n = idx_n[0]

    X = np.transpose(X, (idx_n, idx_c, idx_t))
    return X

def zscore_tensor(x: torch.Tensor, by_channel=True):
    # x: [C,T]
    if by_channel:
        mean = x.mean(dim=1, keepdim=True)
        std  = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    else:
        mean = x.mean()
        std  = x.std().clamp_min(1e-6)
    return (x - mean) / std

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

# ===================== 2) 数据：4类 -> 两个2类 =====================
def map_4class_to_twoheads(y4: np.ndarray):
    """
    y4: 1/2/3/4 = LL/LH/HL/HH
    输出：
      y_p1: 0/1 (L/H)
      y_p2: 0/1 (L/H)
    """
    y4 = y4.astype(np.int64).reshape(-1)
    if not np.all(np.isin(np.unique(y4), [1,2,3,4])):
        raise ValueError(f"y4 必须是 1/2/3/4，实际 unique={np.unique(y4)}")

    # Person-1: LL/LH -> L(0), HL/HH -> H(1)
    y_p1 = np.zeros_like(y4)
    y_p1[np.isin(y4, [1,2])] = 0
    y_p1[np.isin(y4, [3,4])] = 1

    # Person-2: LL/HL -> L(0), LH/HH -> H(1)
    y_p2 = np.zeros_like(y4)
    y_p2[np.isin(y4, [1,3])] = 0
    y_p2[np.isin(y4, [2,4])] = 1

    return y_p1, y_p2

# ===================== 3) Dataset =====================
class TeaPair2BinDataset(Dataset):
    def __init__(self, X_54, y_p1, y_p2, use_zscore=True, by_channel=True):
        """
        X_54: [N,54,2560]
        y_p1/y_p2: [N] 0/1
        """
        self.X = X_54.astype(np.float32)
        self.y1 = y_p1.astype(np.int64)
        self.y2 = y_p2.astype(np.int64)
        self.use_z = use_zscore
        self.by_ch = by_channel

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # [54,2560]
        if self.use_z:
            x = zscore_tensor(x, by_channel=self.by_ch)
        y1 = torch.tensor(self.y1[idx], dtype=torch.long)
        y2 = torch.tensor(self.y2[idx], dtype=torch.long)
        return x, y1, y2

# ===================== 4) 老师模型：分模态分支 + 融合 + 两个head =====================
class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, c1, c2, k1=7, k2=5, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, c1, kernel_size=k1, padding=k1//2, bias=True),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),

            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2//2, bias=True),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TeacherPair2BinNet(nn.Module):
    """
    输入: [B,54,2560]
    结构:
      - EEG 分支: 24ch（每个人），但这里我们把两个人 EEG 合并为 48ch？（不建议）
      更合理：每个模态用“全体 54ch 对应的子集”做编码，然后融合
    我们这里按你的定义：54 = P1(24+2+1) + P2(24+2+1)
    所以：
      EEG: 24+24=48
      ECG: 2+2=4
      GSR: 1+1=2
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        # 通道切片
        # P1: 0-23 EEG1, 24-25 ECG1, 26 GSR1
        # P2: 27-50 EEG2, 51-52 ECG2, 53 GSR2
        self.eeg_encoder = ConvBlock1D(48, 64, 128, k1=7, k2=5, dropout=dropout)
        self.ecg_encoder = ConvBlock1D(4,  16,  32,  k1=7, k2=5, dropout=dropout)
        self.gsr_encoder = ConvBlock1D(2,   8,  16,  k1=7, k2=5, dropout=dropout)

        feat_dim = 128 + 32 + 16
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # 两个 head：各2类
        self.head_p1 = nn.Linear(feat_dim, 2)
        self.head_p2 = nn.Linear(feat_dim, 2)

    def forward(self, x):
        # x: [B,54,T]
        eeg1 = x[:, 0:24, :]
        ecg1 = x[:, 24:26, :]
        gsr1 = x[:, 26:27, :]

        eeg2 = x[:, 27:51, :]
        ecg2 = x[:, 51:53, :]
        gsr2 = x[:, 53:54, :]

        eeg = torch.cat([eeg1, eeg2], dim=1)  # [B,48,T]
        ecg = torch.cat([ecg1, ecg2], dim=1)  # [B,4,T]
        gsr = torch.cat([gsr1, gsr2], dim=1)  # [B,2,T]

        feeg = self.eeg_encoder(eeg)  # [B,128]
        fecg = self.ecg_encoder(ecg)  # [B,32]
        fgsr = self.gsr_encoder(gsr)  # [B,16]

        f = torch.cat([feeg, fecg, fgsr], dim=1)   # [B,176]
        f = self.fuse(f)

        logit1 = self.head_p1(f)  # [B,2]
        logit2 = self.head_p2(f)  # [B,2]
        return logit1, logit2

# ===================== 5) 训练/评估 =====================
def run_one_epoch(net, loader, optimizer=None, scaler=None):
    is_train = optimizer is not None
    net.train(is_train)

    losses = []
    y1_true, y1_pred = [], []
    y2_true, y2_pred = [], []

    for x, y1, y2 in loader:
        x  = x.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device=="cuda" and USE_AMP)):
            logit1, logit2 = net(x)
            loss1 = F.cross_entropy(logit1, y1)
            loss2 = F.cross_entropy(logit2, y2)
            loss  = 0.5 * (loss1 + loss2)

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())

        p1 = torch.argmax(logit1, dim=1).detach().cpu().numpy()
        p2 = torch.argmax(logit2, dim=1).detach().cpu().numpy()
        y1_true.extend(y1.detach().cpu().numpy().tolist())
        y2_true.extend(y2.detach().cpu().numpy().tolist())
        y1_pred.extend(p1.tolist())
        y2_pred.extend(p2.tolist())

    loss_mean = float(np.mean(losses)) if len(losses)>0 else np.nan

    acc1 = accuracy_score(y1_true, y1_pred) if len(y1_true)>0 else 0.0
    acc2 = accuracy_score(y2_true, y2_pred) if len(y2_true)>0 else 0.0
    f11  = macro_f1(y1_true, y1_pred) if len(y1_true)>0 else 0.0
    f12  = macro_f1(y2_true, y2_pred) if len(y2_true)>0 else 0.0

    # 你要的“avg of two heads”
    acc_avg = 0.5*(acc1+acc2)
    f1_avg  = 0.5*(f11+f12)

    return loss_mean, acc_avg, f1_avg, (acc1, acc2, f11, f12)

def pad_to_same_length(list_of_lists, pad_value=np.nan):
    max_len = max(len(lst) for lst in list_of_lists)
    out = np.full((len(list_of_lists), max_len), pad_value, dtype=np.float64)
    for i, lst in enumerate(list_of_lists):
        out[i, :len(lst)] = lst
    return out

# ===================== 6) 主训练入口 =====================
def main():
    print("========== [Teacher Pair2Bin] Loading data ==========")
    X_raw = load_first_var(DATA_MAT)
    y4 = load_first_var(LABEL_MAT)

    print("[Debug] raw X_raw shape:", np.asarray(X_raw).shape)
    print("[Debug] raw y4 shape:", np.asarray(y4).shape)

    X = reorder_to_NCT(X_raw, 54, 2560).astype(np.float32)  # [N,54,2560]
    y4 = np.asarray(y4).reshape(-1).astype(np.int64)

    assert X.shape[0] == y4.shape[0], f"N不一致: X={X.shape[0]}, y={y4.shape[0]}"
    N = X.shape[0]
    print("[Debug] X:", X.shape, " y4:", y4.shape, " unique:", np.unique(y4))

    y_p1, y_p2 = map_4class_to_twoheads(y4)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_train_loss, fold_test_loss = [], []
    fold_train_f1,   fold_test_f1   = [], []
    fold_train_acc,  fold_test_acc  = [], []

    last_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y4), 1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

        ds_tr = TeaPair2BinDataset(X[tr_idx], y_p1[tr_idx], y_p2[tr_idx],
                                  use_zscore=USE_ZSCORE, by_channel=ZSCORE_BY_CH)
        ds_te = TeaPair2BinDataset(X[te_idx], y_p1[te_idx], y_p2[te_idx],
                                  use_zscore=USE_ZSCORE, by_channel=ZSCORE_BY_CH)

        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        net = TeacherPair2BinNet(dropout=DROPOUT_P).to(device)
        opt = optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda" and USE_AMP))

        best_f1 = -1.0
        best_path = os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth")

        tr_losses, te_losses = [], []
        tr_f1s, te_f1s = [], []
        tr_accs, te_accs = [], []

        for epoch in range(1, NUM_EPOCHS+1):
            tr_loss, tr_acc, tr_f1, tr_detail = run_one_epoch(net, dl_tr, optimizer=opt, scaler=scaler)
            te_loss, te_acc, te_f1, te_detail = run_one_epoch(net, dl_te, optimizer=None, scaler=None)

            tr_losses.append(tr_loss); te_losses.append(te_loss)
            tr_f1s.append(tr_f1);     te_f1s.append(te_f1)
            tr_accs.append(tr_acc);   te_accs.append(te_acc)

            improved = te_f1 > best_f1 + 1e-6
            if improved:
                best_f1 = te_f1
                torch.save({
                    "model": net.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "cfg": {
                        "dropout": DROPOUT_P, "lr": LR, "wd": WEIGHT_DECAY,
                        "zscore": USE_ZSCORE, "zscore_by_ch": ZSCORE_BY_CH
                    }
                }, best_path)

            (tr_acc1, tr_acc2, tr_f11, tr_f12) = tr_detail
            (te_acc1, te_acc2, te_f11, te_f12) = te_detail

            print(f"[Fold{fold}] Ep{epoch:02d}/{NUM_EPOCHS} | "
                  f"Train Loss={tr_loss:.4f} Acc(avg)={tr_acc:.4f} F1(avg)={tr_f1:.4f} "
                  f"(P1 Acc={tr_acc1:.3f} F1={tr_f11:.3f} | P2 Acc={tr_acc2:.3f} F1={tr_f12:.3f}) || "
                  f"Test Loss={te_loss:.4f} Acc(avg)={te_acc:.4f} F1(avg)={te_f1:.4f} "
                  f"(P1 Acc={te_acc1:.3f} F1={te_f11:.3f} | P2 Acc={te_acc2:.3f} F1={te_f12:.3f}) "
                  f"{'*BEST*' if improved else ''}")

        fold_train_loss.append(tr_losses); fold_test_loss.append(te_losses)
        fold_train_f1.append(tr_f1s);     fold_test_f1.append(te_f1s)
        fold_train_acc.append(tr_accs);   fold_test_acc.append(te_accs)

        last_metrics.append({
            "fold": fold,
            "best_f1": best_f1,
            "last_train_f1": tr_f1s[-1],
            "last_test_f1": te_f1s[-1],
            "ckpt": best_path
        })

        # 曲线
        fig = plt.figure(figsize=(14,5))
        plt.subplot(1,3,1)
        plt.plot(tr_losses, label="Train")
        plt.plot(te_losses, label="Test")
        plt.title("Pair2Bin Teacher - Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

        plt.subplot(1,3,2)
        plt.plot(tr_f1s, label="Train")
        plt.plot(te_f1s, label="Test")
        plt.title("Pair2Bin Teacher - MacroF1(avg of two heads)"); plt.xlabel("Epoch"); plt.ylabel("Macro-F1"); plt.legend()

        plt.subplot(1,3,3)
        plt.plot(tr_accs, label="Train")
        plt.plot(te_accs, label="Test")
        plt.title("Pair2Bin Teacher - Acc(avg of two heads)"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"teacher_pair2bin_curves_fold{fold}.png"), dpi=150)
        plt.close(fig)

    # 汇总
    summary = {
        "data_mat": DATA_MAT,
        "label_mat": LABEL_MAT,
        "N": int(N),
        "splits": N_SPLITS,
        "epochs": NUM_EPOCHS,
        "batch": BATCH_SIZE,
        "lr": LR,
        "wd": WEIGHT_DECAY,
        "dropout": DROPOUT_P,
        "zscore": USE_ZSCORE,
        "zscore_by_channel": ZSCORE_BY_CH,
        "fold_metrics": last_metrics,
        "mean_best_f1": float(np.mean([m["best_f1"] for m in last_metrics])),
        "std_best_f1":  float(np.std([m["best_f1"] for m in last_metrics])),
    }
    with open(os.path.join(SAVE_DIR, "teacher_pair2bin_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Done. Saved to:", SAVE_DIR)
    print("mean_best_f1:", summary["mean_best_f1"], "std_best_f1:", summary["std_best_f1"])

if __name__ == "__main__":
    main()
