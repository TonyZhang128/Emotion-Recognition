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

# ===================== 0) 配置 =====================
DATA_DIR = r"D:\song_LD\data"

# ====== 老师模型 ckpt：改成你真实的 best_model_fold?.pth ======
TEACHER_CKPT = r"D:\song_LD\data\teacher_pair2bin_py\best_model_fold1.pth"  # <<< 改这里
assert os.path.exists(TEACHER_CKPT), f"TEACHER_CKPT 不存在：{TEACHER_CKPT}"

# ====== 学生数据（ECG）======
STU_DATA_MAT  = os.path.join(DATA_DIR, "stu_test_valence_ECGdata.mat")
STU_LABEL_MAT = os.path.join(DATA_DIR, "stu_test_valence_ECGlabel.mat")
assert os.path.exists(STU_DATA_MAT), STU_DATA_MAT
assert os.path.exists(STU_LABEL_MAT), STU_LABEL_MAT

# 你的学生数据每个样本代表“某个人的 ECG”
# 需要决定：用老师的 head_p1 还是 head_p2 来蒸馏（以及填充到 54 通道哪个位置）
STUDENT_PERSON = 1   # <<< 1=Person-1(head_p1 + 填到24:26)；2=Person-2(head_p2 + 填到51:53)

SAVE_DIR = os.path.join(DATA_DIR, "stu_kd_ECG_valence")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_EPOCHS   = 50
BATCH_SIZE   = 64
LR           = 1e-4
WEIGHT_DECAY = 3e-4
DROPOUT_P    = 0.5
N_SPLITS     = 5
SEED         = 42

USE_ZSCORE   = True
ZSCORE_BY_CH = True

KD_T     = 4.0
KD_ALPHA = 0.5

USE_AMP  = False
MAX_NORM = 1.0
LR_SCHED_PATIENCE = 3
EARLY_PATIENCE    = 7
MIN_LR            = 1e-6

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

def load_mat_var(path: str, candidates, squeeze=True):
    """
    先 scipy.io.loadmat（普通 mat）
    如果遇 v7.3，则用 h5py
    """
    arr = None
    need_h5 = False
    try:
        m = sio.loadmat(path)
        for k in candidates:
            if k in m:
                arr = m[k]
                break
        if arr is None:
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
            for k in candidates:
                if k in f:
                    arr = f[k][()]
                    break
            if arr is None:
                keys = list(f.keys())
                if len(keys) == 0:
                    raise KeyError(f"v7.3 mat 无变量：{path}")
                arr = f[keys[0]][()]
    if arr is None:
        raise KeyError(f"未找到变量：{path} candidates={candidates}")

    arr = np.asarray(arr)
    if squeeze:
        arr = np.squeeze(arr)
    return arr

def reorder_student_to_NCT(X, C_expected: int, T_expected=2560):
    """
    学生数据可能是：
      - 3D: (T,C,N) / (N,C,T) / (T,N,C) ...
      - 2D(GSR常见): (T,N) 视为 (T,1,N)
    输出固定 [N,C,T]
    """
    X = np.asarray(X)
    if X.ndim == 2:
        # (T,N) -> (T,1,N)
        if X.shape[0] == T_expected:
            X = X[:, None, :]
        elif X.shape[1] == T_expected:
            X = X[None, :, :]  # 兜底（很少见）
        else:
            raise ValueError(f"2D数据但找不到T=2560：shape={X.shape}")

    if X.ndim != 3:
        raise ValueError(f"期望 3D 或可转 3D，实际 {X.shape}")

    shape = list(X.shape)
    if T_expected not in shape:
        raise ValueError(f"找不到时间维 T={T_expected} in shape={shape}")
    idx_t = shape.index(T_expected)

    # 其余两维：一个是 C_expected，一个是 N
    idxs = [0,1,2]
    idxs.remove(idx_t)
    idx_a, idx_b = idxs[0], idxs[1]

    if shape[idx_a] == C_expected:
        idx_c, idx_n = idx_a, idx_b
    elif shape[idx_b] == C_expected:
        idx_c, idx_n = idx_b, idx_a
    else:
        raise ValueError(f"找不到通道维 C={C_expected} in shape={shape}")

    X = np.transpose(X, (idx_n, idx_c, idx_t))  # [N,C,T]
    return X

def zscore_tensor(x: torch.Tensor, by_channel=True):
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

# ===================== 2) 老师模型结构（必须和你 teacher_pair2bin_train.py 一致）=====================
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
    按你定义切片：
      P1: EEG 0:24, ECG 24:26, GSR 26:27
      P2: EEG 27:51, ECG 51:53, GSR 53:54
    输出: 两个head，各2类
    """
    def __init__(self, dropout=0.5):
        super().__init__()
        self.eeg_encoder = ConvBlock1D(48, 64, 128, k1=7, k2=5, dropout=dropout)
        self.ecg_encoder = ConvBlock1D(4,  16,  32,  k1=7, k2=5, dropout=dropout)
        self.gsr_encoder = ConvBlock1D(2,   8,  16,  k1=7, k2=5, dropout=dropout)

        feat_dim = 128 + 32 + 16
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

        eeg = torch.cat([eeg1, eeg2], dim=1)  # 48
        ecg = torch.cat([ecg1, ecg2], dim=1)  # 4
        gsr = torch.cat([gsr1, gsr2], dim=1)  # 2

        feeg = self.eeg_encoder(eeg)
        fecg = self.ecg_encoder(ecg)
        fgsr = self.gsr_encoder(gsr)

        f = torch.cat([feeg, fecg, fgsr], dim=1)
        f = self.fuse(f)

        return self.head_p1(f), self.head_p2(f)

def load_teacher(ckpt_path: str) -> TeacherPair2BinNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net = TeacherPair2BinNet(dropout=DROPOUT_P).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net

# ===================== 3) 学生模型（ECG 2通道）=====================
class ECGStudentNet(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.cls = nn.Linear(64, 2)

    def forward(self, x):
        return self.cls(self.encoder(x))

# ===================== 4) teacher输入构造：只放ECG到54通道 =====================
def build_teacher_input_from_single_ecg(X_ecg: np.ndarray, person: int):
    # X_ecg: [N,2,2560]
    N, C, T = X_ecg.shape
    X54 = np.zeros((N, 54, T), dtype=np.float32)

    if person == 1:
        X54[:, 24:26, :] = X_ecg  # P1 ECG
    else:
        X54[:, 51:53, :] = X_ecg  # P2 ECG
    return X54

# ===================== 5) 软标签生成（取对应head）=====================
@torch.no_grad()
def make_soft_labels_from_teacher(teacher: TeacherPair2BinNet, X54: np.ndarray, T=4.0, batch_size=64, person=1):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X54))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    outs = []
    for (xb,) in dl:
        xb = xb.to(device, dtype=torch.float32)
        logit1, logit2 = teacher(xb)
        logits = logit1 if person == 1 else logit2
        p = F.softmax(logits / T, dim=1)     # [B,2]
        outs.append(p.cpu().numpy())
    return np.concatenate(outs, axis=0)      # [N,2]

# ===================== 6) Dataset（硬标签 + 软标签）=====================
class KDDataset(Dataset):
    def __init__(self, X, y, soft2, use_z=True, by_ch=True):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64).reshape(-1)   # 0/1
        self.soft = soft2.astype(np.float32)      # [N,2]
        self.use_z = use_z
        self.by_ch = by_ch

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        if self.use_z:
            x = zscore_tensor(x, by_channel=self.by_ch)
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        s = torch.tensor(self.soft[idx], dtype=torch.float32)
        return x, y, s

# ===================== 7) 训练/评估 =====================
def run_epoch(model, loader, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)

    losses = []
    y_true, y_pred = [], []

    for xb, yb, sb in loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device, dtype=torch.long)
        sb = sb.to(device, dtype=torch.float32)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device=="cuda" and USE_AMP)):
            logits = model(xb)  # [B,2]

            # KD
            log_p = F.log_softmax(logits / KD_T, dim=1)
            loss_kd = F.kl_div(log_p, sb, reduction="batchmean") * (KD_T * KD_T)

            # CE
            loss_ce = F.cross_entropy(logits, yb)

            loss = KD_ALPHA * loss_kd + (1.0 - KD_ALPHA) * loss_ce

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())
        pred = logits.argmax(1).detach().cpu().numpy()
        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(pred.tolist())

    loss_mean = float(np.mean(losses)) if len(losses) else np.nan
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if len(y_true) else 0.0
    f1  = f1_score(y_true, y_pred, average="macro") if len(y_true) else 0.0
    return loss_mean, acc, f1

# ===================== 8) 主函数 =====================
def main():
    print("========== [ECG-KD] Load student data ==========")
    X_raw = load_mat_var(STU_DATA_MAT, ["data","X","stu_test_valence_ECGdata","stu_test_valence_ECGdata"])
    y_raw = load_mat_var(STU_LABEL_MAT, ["label","y","stu_test_valence_ECGlabel","stu_test_valence_ECGlabel"])

    X = reorder_student_to_NCT(X_raw, C_expected=2, T_expected=2560)  # [N,2,2560]
    y = np.asarray(y_raw).reshape(-1).astype(np.int64)               # [N]

    print("[Debug] X:", X.shape, " y:", y.shape, " y uniq:", np.unique(y))
    assert X.shape[0] == y.shape[0], f"N不一致：X={X.shape[0]} y={y.shape[0]}"
    assert set(np.unique(y)).issubset({0,1}), f"标签必须0/1，实际：{np.unique(y)}"

    # teacher soft
    print("========== [ECG-KD] Load teacher & make soft labels ==========")
    teacher = load_teacher(TEACHER_CKPT)
    X54 = build_teacher_input_from_single_ecg(X, person=STUDENT_PERSON)
    soft2 = make_soft_labels_from_teacher(teacher, X54, T=KD_T, batch_size=BATCH_SIZE, person=STUDENT_PERSON)
    print("[Debug] soft2:", soft2.shape, " row sum range:", soft2.sum(1).min(), soft2.sum(1).max())

    # CV
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    fold_tr_losses, fold_te_losses = [], []
    fold_tr_f1s, fold_te_f1s = [], []
    fold_tr_accs, fold_te_accs = [], []

    last_te_f1 = []

    pin = torch.cuda.is_available()
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        print(f"\n========== Fold {fold}/{N_SPLITS} ==========")

        ds_tr = KDDataset(X[tr], y[tr], soft2[tr], use_z=USE_ZSCORE, by_ch=ZSCORE_BY_CH)
        ds_te = KDDataset(X[te], y[te], soft2[te], use_z=USE_ZSCORE, by_ch=ZSCORE_BY_CH)

        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin)
        dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

        model = ECGStudentNet(dropout=DROPOUT_P).to(device)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda" and USE_AMP))
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=LR_SCHED_PATIENCE, min_lr=MIN_LR, verbose=True)

        best_f1 = -1.0
        no_imp = 0
        best_path = os.path.join(SAVE_DIR, f"best_student_ECG_KD_fold{fold}.pth")

        tr_losses, te_losses = [], []
        tr_f1s, te_f1s = [], []
        tr_accs, te_accs = [], []

        for ep in range(1, NUM_EPOCHS+1):
            tr_loss, tr_acc, tr_f1 = run_epoch(model, dl_tr, optimizer=opt, scaler=scaler)
            te_loss, te_acc, te_f1 = run_epoch(model, dl_te, optimizer=None, scaler=None)

            tr_losses.append(tr_loss); te_losses.append(te_loss)
            tr_f1s.append(tr_f1);     te_f1s.append(te_f1)
            tr_accs.append(tr_acc);   te_accs.append(te_acc)

            if np.isfinite(te_loss):
                sched.step(te_loss)

            improved = te_f1 > best_f1 + 1e-6
            if improved:
                best_f1 = te_f1
                no_imp = 0
                torch.save({"model": model.state_dict(), "fold": fold, "best_f1": best_f1}, best_path)
            else:
                no_imp += 1

            print(f"[ECG-KD][Fold{fold}] Ep{ep:02d}/{NUM_EPOCHS} | "
                  f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} || "
                  f"Test loss={te_loss:.4f} acc={te_acc:.4f} f1={te_f1:.4f} | "
                  f"LR={opt.param_groups[0]['lr']:.2e} {'*BEST*' if improved else ''}")

            if no_imp >= EARLY_PATIENCE:
                print(f"[EarlyStop] Fold{fold} no improve {EARLY_PATIENCE} epochs.")
                break

        fold_tr_losses.append(tr_losses); fold_te_losses.append(te_losses)
        fold_tr_f1s.append(tr_f1s);       fold_te_f1s.append(te_f1s)
        fold_tr_accs.append(tr_accs);     fold_te_accs.append(te_accs)
        last_te_f1.append(best_f1)

        # 每折曲线
        fig = plt.figure(figsize=(14,5))
        plt.subplot(1,3,1); plt.plot(tr_losses,label="Train"); plt.plot(te_losses,label="Test"); plt.title("Loss"); plt.legend()
        plt.subplot(1,3,2); plt.plot(tr_f1s,label="Train"); plt.plot(te_f1s,label="Test"); plt.title("MacroF1"); plt.legend()
        plt.subplot(1,3,3); plt.plot(tr_accs,label="Train"); plt.plot(te_accs,label="Test"); plt.title("Acc"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"curves_ECG_KD_fold{fold}.png"), dpi=150)
        plt.close(fig)

    summary = {
        "modal": "ECG",
        "student_person": int(STUDENT_PERSON),
        "teacher_ckpt": TEACHER_CKPT,
        "N": int(X.shape[0]),
        "splits": N_SPLITS,
        "epochs": NUM_EPOCHS,
        "batch": BATCH_SIZE,
        "lr": LR,
        "wd": WEIGHT_DECAY,
        "dropout": DROPOUT_P,
        "KD_T": KD_T,
        "KD_ALPHA": KD_ALPHA,
        "best_f1_each_fold": last_te_f1,
        "best_f1_mean": float(np.mean(last_te_f1)),
        "best_f1_std":  float(np.std(last_te_f1)),
    }
    with open(os.path.join(SAVE_DIR, "summary_KD_ECG.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ ECG-KD Done. Saved to:", SAVE_DIR)
    print("best_f1_mean:", summary["best_f1_mean"], "best_f1_std:", summary["best_f1_std"])

if __name__ == "__main__":
    main()
