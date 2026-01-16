import numpy as np
import torch
from typing import Tuple, Dict, Any
from pathlib import Path
from torch.utils.data import Dataset

def load_and_preprocess_data(data_path, label_path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    # 读取特征数据 (.dat文件)
    print(f"[Info] 正在加载数据文件: {data_path}")
    try:
        # 假设.dat文件是二进制格式，需要根据实际格式调整
        # 这里假设数据是float32格式，形状为(N, 54, 2560)
        X = np.fromfile(str(data_path), dtype=np.float32)
        
        # 根据实际数据大小推断形状
        # 假设我们知道通道数和时间步长
        expected_channels = 54
        expected_timesteps = 2560
        total_elements = X.size
        
        if total_elements % (expected_channels * expected_timesteps) != 0:
            raise ValueError(f"数据大小{total_elements}不能被{expected_channels * expected_timesteps}整除")
        
        num_samples = total_elements // (expected_channels * expected_timesteps)
        X = X.reshape(num_samples, 1, expected_channels, expected_timesteps)
        
    except Exception as e:
        print(f"[Error] 加载.dat文件失败: {e}")
        # 尝试其他可能的格式
        try:
            # 尝试作为文本文件读取
            X = np.loadtxt(str(data_path), dtype=np.float32)
            if X.ndim == 2:
                # 如果是2D，需要重塑为3D
                X = X.reshape(-1, expected_channels, expected_timesteps)
            elif X.ndim == 1:
                X = X.reshape(num_samples, 1, expected_channels, expected_timesteps)
        except Exception as e2:
            raise ValueError(f"无法加载.dat文件，尝试了二进制和文本格式都失败: {e2}")

    print(f"数据X形状: {X.shape} (应为[N,1,54,2560])")

    # 读取标签 (.npy文件)
    print(f"[Info] 正在加载标签文件: {label_path}")
    try:
        y_raw = np.load(str(label_path))
    except Exception as e:
        raise ValueError(f"无法加载.npy标签文件: {e}")
    
    y_raw = np.asarray(y_raw)
    y_raw = np.squeeze(y_raw)
    if y_raw.ndim != 1:
        y_raw = y_raw.reshape(-1)

    # 转换标签: 假设标签可能是任意整数，映射到0,1,2...
    y_values = y_raw.astype(np.int64)
    classes_values = np.unique(y_values)
    num_classes = int(classes_values.size)

    value2id = {v: i for i, v in enumerate(classes_values)}
    y = np.array([value2id[v] for v in y_values], dtype=np.int64)

    # 数据清洗
    # X = X.astype(np.float32)
    # y = y.astype(np.int64)

    # if X.shape[0] != y.shape[0]:
    #     raise ValueError(f"样本数不一致: X={X.shape[0]}, y={y.shape[0]}")
    # if not (X.shape[1] == 54 and X.shape[2] == 2560):
    #     print(f"[Warning] 数据维度不是(N,54,2560), 实际{X.shape}")
    #     # 可以选择继续或抛出错误
    #     # raise ValueError(f"数据维度应为(N,54,2560), 实际{X.shape}")

    # print(f"[Check] 清理前: X finite={np.isfinite(X).all()}, y finite={np.isfinite(y).all()}")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0, posinf=0, neginf=0).astype(np.int64)

    # print(f"[Check] 清理后: X finite={np.isfinite(X).all()}, y finite={np.isfinite(y).all()}")
    # print(f"[Info] 加载完成: X={X.shape}, y={y.shape}, 类别值={classes_values.tolist()} (K={num_classes})")
    # print(f"[Info] 标签映射: {value2id}")

    meta = {
        "num_classes": num_classes,
        "classes_values": classes_values,
        "value2id": value2id,
    }

    return X, y, meta

class AmigosDataset(Dataset):
    """AMIGOS数据集"""

    def __init__(self, data_array: np.ndarray, labels: np.ndarray,
                 use_zscore: bool = True, by_channel: bool = True, config=None):
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
        self.config     = config

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


if __name__ == "__main__":
    data_path =  "./dataset/Arousal_X.dat"
    label_path = "./dataset/Arousal_y.npy"
    # 加载数据
    X, y, meta = load_and_preprocess_data(data_path, label_path)

    from sklearn.model_selection import StratifiedKFold

    # 交叉验证训练
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        train_dataset = AmigosDataset(
        X[train_idx], y[train_idx],
        use_zscore=True,
        by_channel=True
    )
