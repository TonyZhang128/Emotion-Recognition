# ====================== distill_student.py ======================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
from models.MSDNN import MSDNN      # 教师网络结构
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

# ---------- 1. 软标签生成 ----------
def make_soft_labels(teacher_path, data_mat, batch_size=64, T=4):
    class TeacherNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MSDNN()
            self.encoder.fc = nn.Identity()
            self.linear = nn.Linear(176, 4)   # ← 改成 linear
        def forward(self, x):
            return self.linear(self.encoder(x).view(x.size(0), -1))

    teacher = TeacherNet().to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher.eval()

    # 构造 loader（无标签，仅推理）
    class InferDataset(Dataset):
        def __init__(self, arr):
            self.arr = arr
        def __len__(self): return len(self.arr)
        def __getitem__(self, idx): return self.arr[idx]

    loader = DataLoader(InferDataset(data_mat), batch_size=batch_size, shuffle=False)
    softs = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device, dtype=torch.float32)
            logits = teacher(x)
            soft = torch.softmax(logits / T, dim=1)
            softs.append(soft.cpu())
    return torch.cat(softs, dim=0).numpy()      # [N, 4]


# ---------- 2. 学生网络（极小 CNN） ----------
class StudentNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------- 3. 蒸馏数据集 ----------
class DistillDataset(Dataset):
    def __init__(self, data, hard_label, soft_label):
        self.data, self.hard, self.soft = data, hard_label, soft_label
    def __len__(self): return len(self.hard)
    def __getitem__(self, idx):
        return self.data[idx], self.hard[idx], self.soft[idx]


# ---------- 4. 蒸馏训练 ----------
def distill(train_data, train_hard, train_soft,
            test_data,  test_hard,
            T=4, alpha=0.3, epochs=30, batch_size=64, lr=3e-4):

    train_set = DistillDataset(train_data, train_hard, train_soft)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(DistillDataset(test_data, test_hard, np.zeros_like(test_hard)),
                              batch_size=batch_size, shuffle=False)

    student = StudentNet().to(device)
    opt = optim.Adam(student.parameters(), lr=lr)
    ce  = nn.CrossEntropyLoss()
    kld = nn.KLDivLoss(reduction='batchmean')

    distill_loss, distill_acc = [], []

    for epoch in range(epochs):
        student.train()
        running_loss, correct, total = 0., 0, 0
        for x, hard_y, soft_y in train_loader:
            x, hard_y, soft_y = x.to(device), hard_y.to(device), soft_y.to(device)
            x = x.float()
            hard_y = hard_y.long() - 1
            logits = student(x.unsqueeze(1))          # 加通道维

            loss_hard = ce(logits, hard_y)
            loss_soft = kld(torch.log_softmax(logits / T, dim=1),
                            soft_y) * (T * T)
            loss = alpha * loss_soft + (1 - alpha) * loss_hard

            opt.zero_grad(); loss.backward(); opt.step()

            running_loss += loss.item()
            correct += (logits.argmax(1) == hard_y).sum().item()
            total   += hard_y.size(0)

        distill_loss.append(running_loss / len(train_loader))
        distill_acc.append(correct / total)
        print(f"Epoch {epoch+1}: loss={distill_loss[-1]:.4f}  acc={distill_acc[-1]:.4f}")

    # 返回训练好的学生 & 曲线
    return student, distill_loss, distill_acc


# ---------- 5. 主流程 ----------
if __name__ == "__main__":
    # 5-1 教师见过的 160*54*2000
    tea_data  = sio.loadmat('D:/song_LD/data/tea_train_data.mat')['tea_train_data']
    tea_label = sio.loadmat('D:/song_LD/data/tea_train_label.mat')['tea_train_label'].flatten()

    # 生成软标签
    soft_label = make_soft_labels(
        teacher_path='D:/song_LD/save_model/best_model_fold1.pth',
        data_mat=tea_data,
        T=4
    )

    # 5-2 教师没见过的 40*20*2000
    new_data  = sio.loadmat('D:/song_LD/data/stu_test_data.mat')['stu_test_data']
    new_label = sio.loadmat('D:/song_LD/data/stu_test_label.mat')['stu_test_label'].flatten()

    # 5-3 蒸馏
    student, loss_curve, acc_curve = distill(
        train_data=tea_data,  train_hard=tea_label,  train_soft=soft_label,
        test_data=new_data,   test_hard=new_label,
        T=4, alpha=0.3, epochs=30
    )

    # 5-4 最终评估（学生 vs 新数据）
    student.eval()
    correct = 0
    with torch.no_grad():
        for x, y ,_ in DataLoader(DistillDataset(new_data, new_label, np.zeros_like(new_label)),
                               batch_size=64):
            x, y = x.to(device, dtype=torch.float32), y.to(device)
            x = x.float()
            correct += (student(x.unsqueeze(1)).argmax(1) == y).sum().item()
    print(f"\nStudent on 40×20×2000: Acc = {correct/len(new_data):.4f}")

    # 5-5 简单画图
    plt.plot(loss_curve, label='distill loss'); plt.legend(); plt.show()

    # 5-6 显式保存图片
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(loss_curve)
    plt.title('Distill Loss')
    plt.subplot(1, 2, 2)
    plt.plot(acc_curve)
    plt.title('Distill Acc')
    plt.tight_layout()

    plt.show(block=True)  # 阻塞直到手动关闭窗口