# -*- coding: utf-8 -*-
import os, json, random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

# ===================== 配置 =====================
DATA_DIR = r"D:\song_LD\data"
SAVE_DIR = r"D:\song_LD\save_model\teacher_pair2bin_strong"
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
N_SPLITS = 5
NUM_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-4
WEIGHT_DECAY = 3e-4
DROPOUT = 0.5
MODAL_DROP_P = 0.25   # modality dropout 概率：0.2~0.4 都可以试

device = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(SEED)

def load_mat_any(path, candidates):
    arr = None
    try:
        m = sio.loadmat(path)
        for k in candidates:
            if k in m:
                arr = m[k]; break
        if arr is None:
            for k,v in m.items():
                if not k.startswith("__"):
                    arr = v; break
    except NotImplementedError:
        arr = None
    if arr is None:
        import h5py
        with h5py.File(path, "r") as f:
            for k in candidates:
                if k in f:
                    arr = f[k][()]; break
            if arr is None:
                arr = f[list(f.keys())[0]][()]
    return np.squeeze(np.asarray(arr))

def to_NCT_54(X):
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"tea_train_data 必须3维，实际 {X.shape}")
    sh = list(X.shape)
    i_ch = sh.index(54)
    i_t  = sh.index(2560)
    i_n  = [i for i in [0,1,2] if i not in [i_ch, i_t]][0]
    return np.transpose(X, (i_n, i_ch, i_t)).astype(np.float32)

def parse_cell_HHHL(y_cell):
    y_cell = np.asarray(y_cell).reshape(-1)
    y = []
    for it in y_cell:
        if isinstance(it, bytes):
            s = it.decode("utf-8")
        else:
            s = str(it)
        s = s.strip().replace("'", "").replace('"', "")
        y.append(s)
    y1 = np.array([1 if s[0].upper()=="H" else 0 for s in y], dtype=np.int64)
    y2 = np.array([1 if s[1].upper()=="H" else 0 for s in y], dtype=np.int64)
    return y1, y2, y

# ===================== 模型 =====================
class TemporalEncoder1D(nn.Module):
    def __init__(self, in_ch, c1, c2, k1=7, k2=5, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, c1, k1, padding=k1//2, bias=True),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, k2, padding=k2//2, bias=True),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class TeacherPair2BinStrong(nn.Module):
    """
    输入: [B,54,T]
    输出: logits1 [B,2], logits2 [B,2]
    """
    def __init__(self, dropout=0.5, modal_drop_p=0.25):
        super().__init__()
        self.modal_drop_p = modal_drop_p

        self.eeg_encoder = TemporalEncoder1D(24, 64, 128, dropout=dropout)  # 与你ckpt形状一致
        self.ecg_encoder = TemporalEncoder1D(2,  16, 32,  dropout=dropout)
        self.gsr_encoder = TemporalEncoder1D(1,  8,  16,  dropout=dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)

        # person feature dim = 128+32+16=176
        # interaction dim = concat(f1,f2,|f1-f2|,f1*f2) => 704
        self.fuse = nn.Sequential(
            nn.Linear(176*4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head1 = nn.Linear(256, 2)
        self.head2 = nn.Linear(256, 2)

    def _encode_person(self, x27):
        eeg = x27[:, 0:24, :]
        ecg = x27[:, 24:26, :]
        gsr = x27[:, 26:27, :]

        # ===== Modality Dropout：训练时随机丢掉某个模态 =====
        if self.training and self.modal_drop_p > 0:
            r = torch.rand(3, device=x27.device)
            if r[0] < self.modal_drop_p: eeg = torch.zeros_like(eeg)
            if r[1] < self.modal_drop_p: ecg = torch.zeros_like(ecg)
            if r[2] < self.modal_drop_p: gsr = torch.zeros_like(gsr)

        fe = self.pool(self.eeg_encoder(eeg)).squeeze(-1)
        fc = self.pool(self.ecg_encoder(ecg)).squeeze(-1)
        fg = self.pool(self.gsr_encoder(gsr)).squeeze(-1)
        return torch.cat([fe, fc, fg], dim=1)  # [B,176]

    def forward(self, x):
        x1 = x[:, 0:27, :]
        x2 = x[:, 27:54, :]
        f1 = self._encode_person(x1)
        f2 = self._encode_person(x2)

        inter = torch.cat([f1, f2, torch.abs(f1-f2), f1*f2], dim=1)  # [B,704]
        g = self.fuse(inter)  # [B,256]
        return self.head1(g), self.head2(g)

# ===================== 数据集 =====================
class DS(Dataset):
    def __init__(self, X, y1, y2, zscore=True):
        self.X = X
        self.y1 = y1
        self.y2 = y2
        self.z = zscore
    def __len__(self): return len(self.y1)
    def __getitem__(self, i):
        x = torch.tensor(self.X[i], dtype=torch.float32)
        if self.z:
            mean = x.mean(dim=1, keepdim=True)
            std  = x.std(dim=1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std
        return x, int(self.y1[i]), int(self.y2[i])

# ===================== 训练 =====================
def class_weights(y):
    # y: 0/1
    c0 = (y==0).sum()
    c1 = (y==1).sum()
    w0 = (c0+c1) / max(c0,1)
    w1 = (c0+c1) / max(c1,1)
    w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    w = w / w.mean()  # 归一化一下，避免太大
    return w

def main():
    X_raw = load_mat_any(os.path.join(DATA_DIR, "tea_train_data.mat"), ["tea_train_data","X","data"])
    X = to_NCT_54(X_raw)

    y_cell = load_mat_any(os.path.join(DATA_DIR, "tea_train_valencelabel.mat"),
                          ["tea_train_valencelabel","label","y"])
    y1, y2, ytxt = parse_cell_HHHL(y_cell)

    # 分层：用 4类字符串更稳（防止 fold 某类缺失）
    # 这里把 HH/HL/LH/LL 映射到 0..3
    map4 = {"LL":0,"LH":1,"HL":2,"HH":3}
    y4 = np.array([map4[s.upper()] for s in ytxt], dtype=np.int64)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    all_best = []
    for fold, (tr, te) in enumerate(skf.split(X, y4), 1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")
        ds_tr = DS(X[tr], y1[tr], y2[tr], zscore=True)
        ds_te = DS(X[te], y1[te], y2[te], zscore=True)

        dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True)
        dl_te = DataLoader(ds_te, batch_size=BATCH_SIZE, shuffle=False)

        model = TeacherPair2BinStrong(dropout=DROPOUT, modal_drop_p=MODAL_DROP_P).to(device)
        opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        w1 = class_weights(y1[tr])
        w2 = class_weights(y2[tr])
        best_f1 = -1
        patience = 0

        for ep in range(1, NUM_EPOCHS+1):
            model.train()
            losses = []
            for xb, a, b in dl_tr:
                xb = xb.to(device)
                a = a.to(device); b = b.to(device)

                opt.zero_grad()
                la, lb = model(xb)
                loss = F.cross_entropy(la, a, weight=w1) + F.cross_entropy(lb, b, weight=w2)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(loss.item())

            # eval
            model.eval()
            A_true=[]; A_pred=[]
            B_true=[]; B_pred=[]
            with torch.no_grad():
                for xb, a, b in dl_te:
                    xb = xb.to(device)
                    la, lb = model(xb)
                    pa = la.argmax(1).cpu().numpy()
                    pb = lb.argmax(1).cpu().numpy()
                    A_true += a.numpy().tolist(); A_pred += pa.tolist()
                    B_true += b.numpy().tolist(); B_pred += pb.tolist()

            f1a = f1_score(A_true, A_pred, average="macro")
            f1b = f1_score(B_true, B_pred, average="macro")
            f1avg = (f1a+f1b)/2
            accavg = (accuracy_score(A_true, A_pred)+accuracy_score(B_true, B_pred))/2

            print(f"Ep{ep:03d} loss={np.mean(losses):.4f} | f1a={f1a:.4f} f1b={f1b:.4f} avg={f1avg:.4f} | acc_avg={accavg:.4f}")

            if f1avg > best_f1 + 1e-6:
                best_f1 = f1avg
                patience = 0
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_fold{fold}.pth"))
            else:
                patience += 1
                if patience >= 10:
                    print("[EarlyStop]")
                    break

        all_best.append(best_f1)
        print(f"[Fold{fold}] BestF1avg={best_f1:.4f}")

    print("\n===== Summary =====")
    print("mean best f1:", float(np.mean(all_best)), "std:", float(np.std(all_best)))

if __name__ == "__main__":
    main()

