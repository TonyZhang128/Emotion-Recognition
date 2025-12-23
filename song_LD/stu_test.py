# ====================== baseline_no_teacher.py ======================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on", device)

# ---------- 1. 学生网络（与蒸馏脚本完全一致） ----------
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


# ---------- 2. 普通数据集（仅硬标签） ----------
class HardDataset(Dataset):
    def __init__(self, data, label):
        self.data, self.label = data, label
    def __len__(self): return len(self.label)
    def __getitem__(self, idx): return self.data[idx], self.label[idx]


# ---------- 3. 训练函数（仅 CE Loss） ----------
def train_baseline(train_data, train_label, test_data, test_label,
                   epochs=30, batch_size=64, lr=3e-4):
    train_set = HardDataset(train_data, train_label)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(HardDataset(test_data, test_label),
                              batch_size=batch_size, shuffle=False)

    student = StudentNet().to(device)
    opt   = optim.Adam(student.parameters(), lr=lr)
    ce    = nn.CrossEntropyLoss()

    loss_curve, acc_curve = [], []
    for epoch in range(epochs):
        student.train()
        running_loss, correct, total = 0., 0, 0
        for x, y in train_loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long)
            y = y - 1                      # 1,2,3,4 → 0,1,2,3
            logits = student(x.unsqueeze(1))

            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

            running_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        loss_curve.append(running_loss / len(train_loader))
        acc_curve.append(correct / total)
        print(f"Epoch {epoch+1}: loss={loss_curve[-1]:.4f}  acc={acc_curve[-1]:.4f}")

    return student, loss_curve, acc_curve


# ---------- 4. 评估函数 ----------
def evaluate(model, data, label, batch_size=64):
    model.eval()
    correct = 0
    loader  = DataLoader(HardDataset(data, label), batch_size=batch_size)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, dtype=torch.float32), y.to(device, dtype=torch.long) - 1
            correct += (model(x.unsqueeze(1)).argmax(1) == y).sum().item()
    return correct / len(data)


# ---------- 5. 主流程 ----------
if __name__ == "__main__":
    # 5-1 训练数据 160*54*2000
    tea_data  = sio.loadmat('D:/song_LD/data/tea_train_data.mat')['tea_train_data']
    tea_label = sio.loadmat('D:/song_LD/data/tea_train_label.mat')['tea_train_label'].flatten()

    # 5-2 测试数据 40*20*2000
    new_data  = sio.loadmat('D:/song_LD/data/stu_test_data.mat')['stu_test_data']
    new_label = sio.loadmat('D:/song_LD/data/stu_test_label.mat')['stu_test_label'].flatten()

    # 5-3 训练（无蒸馏）
    student, loss_curve, acc_curve = train_baseline(
        tea_data, tea_label, new_data, new_label, epochs=30
    )

    # 5-4 测试
    acc = evaluate(student, new_data, new_label)
    print(f"\nBaseline Student on 40×20×2000: Acc = {acc:.4f}")

    # 5-5 画图
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1); plt.plot(loss_curve); plt.title('CE Loss')
    plt.subplot(1, 2, 2); plt.plot(acc_curve);  plt.title('Train Acc')
    plt.tight_layout()
    plt.savefig('baseline_curve.png')
    plt.show(block=True)