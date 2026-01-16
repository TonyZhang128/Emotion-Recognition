from typing import Tuple, Dict, List, Optional, Any
import torch
from sklearn.metrics import f1_score


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
