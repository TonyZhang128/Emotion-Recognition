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
from sklearn.metrics import f1_score

# ===================== 0) 配置区：你只需要改这里 =====================
DATA_DIR = r"D:\song_LD\data"

# ===== 选择蒸馏哪个模态（改这里即可）=====
MODAL = "ECG"   # "EEG" / "ECG" / "GSR"
TASK  = "arousal"  # 以后想跑 arousal，把 valence 改成 arousal，同时换文件名即可

# ===== 学生数据文件名（你现成的 stu_test_xxx）=====
STU_DATA_MAT  = os.path.join(DATA_DIR, f"stu_test_{TASK}_{MODAL}data.mat")
STU_LABEL_MAT = os.path.join(DATA_DIR, f"stu_test_{TASK}_{MODAL}label.mat")

# ===== 老师模型 ckpt（你刚训练的 teacher_pair2bin_py）=====
TEACHER_DIR   = os.path.join(DATA_DIR, "teacher_pair2bin_py")
TEACHER_FOLD  = 1
TEACHER_CKPT  = os.path.join(TEACHER_DIR, f"best_model_fold{TEACHER_FOLD}.pth")

# ===== 输出目录 =====
SAVE_DIR = os.path.join(DATA_DIR, f"stu_kd_{TASK}_{MODAL}_from_teacherFold{TEACHER_FOLD}")
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 训练超参 =====
NUM_EPOCHS   = 50
BATCH_SIZE   = 64
LR           = 1e-4
WEIGHT_DECAY = 3e-4
DROPOUT_P    = 0.5
N_SPLITS     = 5
SEED         = 42

# ===== KD 超参 =====
KD_T     = 4.0
KD_ALPHA = 0.5   # total_loss = alpha*KD + (1-alpha)*CE

# ===== 稳定性 =====
USE_ZSCORE   = True
ZSCORE_BY_CH = True
USE_AMP      = False
MAX_NORM     = 1.0
EARLY_PATIENCE = 7
LR_SCHED_PATIENCE = 3
MIN_LR = 1e-6

# ===== 选择 teacher 用哪个 head 当软标签 =====
# 你说“学第一个人/第二个人都可以”，这里二选一
TEACHER_USE_PERSON = 1   # 1 => 用 head_p1, 2 => 用 head_p2

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


def load_mat_firstvar(path: str, squeeze=True):
    """
    读取 .mat 第一个顶层变量：
      - 普通 mat：scipy.io.loadmat
      - v7.3：h5py
    """
    arr = None
    need_h5 = False
    try:
        m = sio.loadmat(path)
        for k, v in m.items():
            if not k.startswith("__"):
                arr = v
                break
    except NotImplementedError:
        need_h5 = True
    except Exception as e:
        if any(s in str(e).lower() for s in ["v7.3", "hdf", "hdf5"]):
            need_h5 = True
        else:
            raise

    if arr is None and need_h5:
        import h5py
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            if len(keys) == 0:
                raise KeyError(f"no dataset in {path}")
            arr = f[keys[0]][()]

    if arr is None:
        raise KeyError(f"no valid var found in {path}")

    arr = np.asarray(arr)
    if squeeze:
        arr = np.squeeze(arr)
    return arr


def reorder_student_to_NCT(X, T_expected=2560):
    """
    学生数据可能是：
      - (2560, C, N) 例如 ECG: (2560,2,2206)
      - (N, C, 2560)
      - (2560, N) 或 (N,2560) (GSR 有时变成2D)
    目标输出： [N, C, 2560]
    """
    X = np.asarray(X)

    # --- 2D: (T,N) or (N,T) -> 认为 C=1 ---
    if X.ndim == 2:
        if X.shape[0] == T_expected:
            X = X.T  # (N,T)
        elif X.shape[1] == T_expected:
            pass     # already (N,T)
        else:
            raise ValueError(f"2D 学生数据找不到 T={T_expected}: shape={X.shape}")
        X = X[:, None, :]  # (N,1,T)
        return X

    if X.ndim != 3:
        raise ValueError(f"期望 2D/3D, 实际 {X.shape}")

    shape = list(X.shape)
    # 找 time 维
    if T_expected not in shape:
        raise ValueError(f"在 shape={shape} 中找不到 T={T_expected}")
    idx_t = shape.index(T_expected)

    # 其余两个维：N 和 C
    idx_other = [i for i in [0,1,2] if i != idx_t]
    # 经验：N通常较大(2206)，C较小(24/2/1)
    a, b = idx_other
    if shape[a] > shape[b]:
        idx_n, idx_c = a, b
    else:
        idx_n, idx_c = b, a

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


def pad_to_same_length(list_of_lists, pad_value=np.nan):
    max_len = max(len(lst) for lst in list_of_lists)
    out = np.full((len(list_of_lists), max_len), pad_value, dtype=np.float64)
    for i, lst in enumerate(list_of_lists):
        out[i, :len(lst)] = lst
    return out


# ===================== 2) 导入 MSDNN =====================
try:
    from models.MSDNN import MSDNN
except ImportError:
    from MSDNN import MSDNN


# ===================== 3) 老师模型结构（必须和你训练时完全一致） =====================
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
    切片规则：
      P1: EEG 0:24, ECG 24:26, GSR 26:27
      P2: EEG 27:51, ECG 51:53, GSR 53:54
    三分支编码(48/4/2)，融合后两个 head 输出各2类。
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.eeg_encoder = ConvBlock1D(48, 64, 128, k1=7, k2=5, dropout=dropout)
        self.ecg_encoder = ConvBlock1D(4,  16,  32,  k1=7, k2=5, dropout=dropout)
        self.gsr_encoder = ConvBlock1D(2,   8,  16,  k1=7, k2=5, dropout=dropout)

        feat_dim = 128 + 32 + 16  # 176
        self.fuse = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head_p1 = nn.Linear(feat_dim, 2)
        self.head_p2 = nn.Linear(feat_dim, 2)

    def forward(self, x):
        eeg1 = x[:, 0:24, :]
        ecg1 = x[:, 24:26, :]
        gsr1 = x[:, 26:27, :]

        eeg2 = x[:, 27:51, :]
        ecg2 = x[:, 51:53, :]
        gsr2 = x[:, 53:54, :]

        eeg = torch.cat([eeg1, eeg2], dim=1)  # [B,48,T]
        ecg = torch.cat([ecg1, ecg2], dim=1)  # [B,4,T]
        gsr = torch.cat([gsr1, gsr2], dim=1)  # [B,2,T]

        feeg = self.eeg_encoder(eeg)
        fecg = self.ecg_encoder(ecg)
        fgsr = self.gsr_encoder(gsr)

        f = torch.cat([feeg, fecg, fgsr], dim=1)  # [B,176]
        f = self.fuse(f)

        return self.head_p1(f), self.head_p2(f)


def load_teacher(ckpt_path: str) -> TeacherPair2BinNet:
    assert os.path.exists(ckpt_path), f"Teacher ckpt not found: {ckpt_path}"
    net = TeacherPair2BinNet(dropout=DROPOUT_P).to(device)
    obj = torch.load(ckpt_path, map_location=device)

    # 你保存的是 dict: {"model": state_dict, ...}
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


# ===================== 4) 学生模型（单模态2分类，和你之前一样） =====================
class StudentNet(nn.Module):
    """
    单模态输入: [B,C_in,2560] -> adapter(->54) -> MSDNN -> 2类
    """
    def __init__(self, in_channels: int, dropout_p: float = 0.5):
        super().__init__()
        self.adapter = nn.Conv1d(in_channels, 54, kernel_size=1, bias=True)
        self.encoder = MSDNN()
        if hasattr(self.encoder, "fc"):
            self.encoder.fc = nn.Identity()
        feat_dim = getattr(self.encoder, "num_channels", 176)
        self.dropout = nn.Dropout(dropout_p)
        self.head = nn.Linear(feat_dim, 2)

    def forward(self, x):
        x = self.adapter(x)      # [B,54,T]
        x = self.encoder(x)      # [B,feat] or [B,feat,1]
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.head(x)      # [B,2]


# ===================== 5) 构造 teacher 输入（补零策略） =====================
def build_teacher_input_from_student(X_student: np.ndarray, modal: str, person: int = 1):
    """
    X_student: [N,C,T]  单人单模态
    返回：X_teacher_in: [N,54,T]，把对应通道填上，其余为0。
    modal: EEG/ECG/GSR
    person: 1 or 2
    """
    N, C, T = X_student.shape
    X54 = np.zeros((N, 54, T), dtype=np.float32)

    if person not in [1, 2]:
        raise ValueError("person must be 1 or 2")

    if modal.upper() == "EEG":
        assert C == 24, f"EEG C should be 24, got {C}"
        if person == 1:
            X54[:, 0:24, :] = X_student
        else:
            X54[:, 27:51, :] = X_student

    elif modal.upper() == "ECG":
        assert C == 2, f"ECG C should be 2, got {C}"
        if person == 1:
            X54[:, 24:26, :] = X_student
        else:
            X54[:, 51:53, :] = X_student

    elif modal.upper() == "GSR":
        assert C == 1, f"GSR C should be 1, got {C}"
        if person == 1:
            X54[:, 26:27, :] = X_student
        else:
            X54[:, 53:54, :] = X_student

    else:
        raise ValueError(f"Unknown modal: {modal}")

    return X54


@torch.no_grad()
def make_soft_labels_from_teacher(teacher: TeacherPair2BinNet, X54: np.ndarray, T=4.0, batch_size=64):
    """
    用 teacher 在补零后的 [N,54,T] 上输出 soft label（2类），返回 [N,2]
    """
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X54))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    outs = []
    for (xb,) in dl:
        xb = xb.to(device, dtype=torch.float32)
        log1, log2 = teacher(xb)
        logits = log1 if TEACHER_USE_PERSON == 1 else log2
        p = F.softmax(logits / T, dim=1)  # [B,2]
        outs.append(p.cpu().numpy())
    soft = np.concatenate(outs, axis=0)
    return soft


# ===================== 6) Dataset（硬标签 + 软标签） =====================
class KDDataset(Dataset):
    def __init__(self, X, y, soft, use_zscore=True, by_channel=True):
        self.X = X.astype(np.float32)       # [N,C,T]
        self.y = y.astype(np.int64).reshape(-1)  # [N]
        self.soft = soft.astype(np.float32) # [N,2]
        self.use_z = use_zscore
        self.by_ch = by_channel

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)  # [C,T]
        if self.use_z:
            x = zscore_tensor(x, by_channel=self.by_ch)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        s = torch.tensor(self.soft[idx], dtype=torch.float32)
        return x, y, s


# ===================== 7) 训练一折 =====================
def run_epoch(student, loader, optimizer=None, scaler=None):
    train = optimizer is not None
    student.train(train)

    losses = []
    y_true, y_pred = [], []

    for xb, yb, sb in loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.long)
        sb = sb.to(device, dtype=torch.float32)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device == "cuda" and USE_AMP)):
            logits = student(xb)  # [B,2]

            # KD: KL( log_softmax(student/T), soft_teacher )
            log_ps = F.log_softmax(logits / KD_T, dim=1)
            loss_kd = F.kl_div(log_ps, sb, reduction="batchmean") * (KD_T * KD_T)

            # CE: hard label
            loss_ce = F.cross_entropy(logits, yb)

            loss = KD_ALPHA * loss_kd + (1.0 - KD_ALPHA) * loss_ce

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())
        pred = logits.argmax(1)
        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())

    loss_mean = float(np.mean(losses)) if len(losses) else float("nan")
    f1 = float(f1_score(y_true, y_pred, average="macro")) if len(y_true) else 0.0
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if len(y_true) else 0.0
    return loss_mean, f1, acc


# ===================== 8) 主流程 =====================
def main():
    print("========== [Student KD] ==========")
    print("[Info] MODAL =", MODAL, "TASK =", TASK)
    print("[Info] Student data:", STU_DATA_MAT)
    print("[Info] Student label:", STU_LABEL_MAT)
    print("[Info] Teacher ckpt :", TEACHER_CKPT)
    print("[Info] Teacher head :", "P1" if TEACHER_USE_PERSON == 1 else "P2")

    # ---- load student data ----
    X_raw = load_mat_firstvar(STU_DATA_MAT)
    y_raw = load_mat_firstvar(STU_LABEL_MAT)

    X = reorder_student_to_NCT(X_raw, T_expected=2560)  # [N,C,T]
    y = np.asarray(y_raw).reshape(-1).astype(np.int64)

    assert X.shape[0] == y.shape[0], f"样本数不一致：X={X.shape[0]} y={y.shape[0]}"
    print("[Debug] X:", X.shape, " y:", y.shape, " unique y:", np.unique(y).tolist())

    # ---- infer C_expected ----
    C_expected = X.shape[1]
    if MODAL.upper() == "EEG" and C_expected != 24:
        raise ValueError(f"EEG 应该 C=24，但读到 C={C_expected}，检查 mat 维度/重排")
    if MODAL.upper() == "ECG" and C_expected != 2:
        raise ValueError(f"ECG 应该 C=2，但读到 C={C_expected}")
    if MODAL.upper() == "GSR" and C_expected != 1:
        raise ValueError(f"GSR 应该 C=1，但读到 C={C_expected}")

    # ---- build teacher input & soft labels ----
    teacher = load_teacher(TEACHER_CKPT)
    X54 = build_teacher_input_from_student(X, modal=MODAL, person=TEACHER_USE_PERSON)  # [N,54,T]
    soft = make_soft_labels_from_teacher(teacher, X54, T=KD_T, batch_size=BATCH_SIZE)  # [N,2]
    print("[Debug] soft:", soft.shape, " soft row sum mean:", float(np.mean(soft.sum(axis=1))))

    # ---- CV train ----
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_tr_losses, fold_te_losses = [], []
    fold_tr_f1s, fold_te_f1s = [], []
    fold_tr_accs, fold_te_accs = [], []
    last_te_f1 = []

    pin = torch.cuda.is_available()

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")

        ds_tr = KDDataset(X[tr_idx], y[tr_idx], soft[tr_idx], USE_ZSCORE, ZSCORE_BY_CH)
        ds_te = KDDataset(X[te_idx], y[te_idx], soft[te_idx], USE_ZSCORE, ZSCORE_BY_CH)
        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
        dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

        student = StudentNet(in_channels=C_expected, dropout_p=DROPOUT_P).to(device)
        opt = optim.Adam(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and USE_AMP))
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=LR_SCHED_PATIENCE, min_lr=MIN_LR, verbose=True
        )

        best_f1 = -1.0
        best_path = os.path.join(SAVE_DIR, f"student_kd_best_fold{fold}.pth")
        no_improve = 0

        tr_losses, te_losses = [], []
        tr_f1s, te_f1s = [], []
        tr_accs, te_accs = [], []

        for ep in range(1, NUM_EPOCHS + 1):
            tr_loss, tr_f1, tr_acc = run_epoch(student, dl_tr, optimizer=opt, scaler=scaler)
            te_loss, te_f1, te_acc = run_epoch(student, dl_te, optimizer=None, scaler=None)

            tr_losses.append(tr_loss); te_losses.append(te_loss)
            tr_f1s.append(tr_f1);     te_f1s.append(te_f1)
            tr_accs.append(tr_acc);   te_accs.append(te_acc)

            if np.isfinite(te_loss):
                sched.step(te_loss)

            improved = te_f1 > best_f1 + 1e-6
            if improved:
                best_f1 = te_f1
                no_improve = 0
                torch.save(student.state_dict(), best_path)
            else:
                no_improve += 1

            print(f"[Fold{fold}] Ep{ep:02d}/{NUM_EPOCHS} | "
                  f"Train loss={tr_loss:.4f} f1={tr_f1:.4f} acc={tr_acc:.4f} || "
                  f"Test loss={te_loss:.4f} f1={te_f1:.4f} acc={te_acc:.4f} | "
                  f"LR={opt.param_groups[0]['lr']:.2e} | {'*BEST*' if improved else ''}")

            if no_improve >= EARLY_PATIENCE:
                print(f"[EarlyStop] Fold{fold} no improve {EARLY_PATIENCE} epochs.")
                break

        fold_tr_losses.append(tr_losses); fold_te_losses.append(te_losses)
        fold_tr_f1s.append(tr_f1s);       fold_te_f1s.append(te_f1s)
        fold_tr_accs.append(tr_accs);     fold_te_accs.append(te_accs)
        last_te_f1.append(best_f1)

    # ---- summary ----
    summary = {
        "modal": MODAL,
        "task": TASK,
        "teacher_ckpt": TEACHER_CKPT,
        "teacher_person_head": TEACHER_USE_PERSON,
        "N": int(X.shape[0]),
        "C": int(X.shape[1]),
        "T": int(X.shape[2]),
        "splits": N_SPLITS,
        "epochs_target": NUM_EPOCHS,
        "batch": BATCH_SIZE,
        "lr": LR,
        "wd": WEIGHT_DECAY,
        "dropout": DROPOUT_P,
        "KD_T": KD_T,
        "KD_ALPHA": KD_ALPHA,
        "test_best_f1_mean": float(np.mean(last_te_f1)),
        "test_best_f1_std":  float(np.std(last_te_f1)),
    }
    with open(os.path.join(SAVE_DIR, f"summary_KD_{MODAL}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ Done. Saved to:", SAVE_DIR)
    print("test_best_f1_mean:", summary["test_best_f1_mean"], "std:", summary["test_best_f1_std"])

    # ---- plot mean curves ----
    tr_loss_mx = pad_to_same_length(fold_tr_losses)
    te_loss_mx = pad_to_same_length(fold_te_losses)
    tr_f1_mx   = pad_to_same_length(fold_tr_f1s)
    te_f1_mx   = pad_to_same_length(fold_te_f1s)
    tr_acc_mx  = pad_to_same_length(fold_tr_accs)
    te_acc_mx  = pad_to_same_length(fold_te_accs)

    plt.figure(figsize=(14, 10))
    plt.subplot(2,2,1)
    plt.plot(np.nanmean(tr_loss_mx, axis=0), label="Train")
    plt.plot(np.nanmean(te_loss_mx, axis=0), label="Test")
    plt.title(f"KD-{MODAL} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(2,2,2)
    plt.plot(np.nanmean(tr_f1_mx, axis=0), label="Train")
    plt.plot(np.nanmean(te_f1_mx, axis=0), label="Test")
    plt.title(f"KD-{MODAL} Macro-F1"); plt.xlabel("Epoch"); plt.ylabel("F1"); plt.legend()

    plt.subplot(2,2,3)
    plt.plot(np.nanmean(tr_acc_mx, axis=0), label="Train")
    plt.plot(np.nanmean(te_acc_mx, axis=0), label="Test")
    plt.title(f"KD-{MODAL} Acc"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()

    plt.tight_layout()
    out_png = os.path.join(SAVE_DIR, f"learning_curves_KD_{MODAL}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
