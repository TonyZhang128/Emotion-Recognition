import numpy as np
from typing import List, Dict
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================================
# 结果分析模块
# ============================================================================

def compute_cross_validation_summary(fold_results: List[Dict]) -> Dict[str, float]:
    """计算交叉验证汇总统计"""
    summaries = []

    for result in fold_results:
        hist = result["history"]
        summaries.append({
            "train_loss": hist["train_loss"][-1],
            "test_loss": hist["test_loss"][-1],
            "train_f1": hist["train_f1"][-1],
            "test_f1": hist["test_f1"][-1],
            "train_acc": hist["train_accuracy"][-1],
            "test_acc": hist["test_accuracy"][-1],
        })

    metrics = ["train_loss", "test_loss", "train_f1", "test_f1", "train_acc", "test_acc"]
    summary = {}

    for metric in metrics:
        values = [s[metric] for s in summaries]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))

    return summary


def plot_learning_curves(
    fold_results: List[Dict],
    save_path: Path,
    dpi: int = 150,
) -> None:
    """绘制学习曲线"""
    def pad_to_same_length(list_of_lists, pad_value=np.nan):
        max_len = max(len(lst) for lst in list_of_lists)
        out = np.full((len(list_of_lists), max_len), pad_value, dtype=np.float64)
        for i, lst in enumerate(list_of_lists):
            out[i, :len(lst)] = lst
        return out

    # 提取所有fold的历史数据
    fold_train_losses = [r["history"]["train_loss"] for r in fold_results]
    fold_test_losses = [r["history"]["test_loss"] for r in fold_results]
    fold_train_f1s = [r["history"]["train_f1"] for r in fold_results]
    fold_test_f1s = [r["history"]["test_f1"] for r in fold_results]
    fold_train_accs = [r["history"]["train_accuracy"] for r in fold_results]
    fold_test_accs = [r["history"]["test_accuracy"] for r in fold_results]

    # 填充到相同长度
    train_loss_mx = pad_to_same_length(fold_train_losses)
    test_loss_mx = pad_to_same_length(fold_test_losses)
    train_f1_mx = pad_to_same_length(fold_train_f1s)
    test_f1_mx = pad_to_same_length(fold_test_f1s)
    train_acc_mx = pad_to_same_length(fold_train_accs)
    test_acc_mx = pad_to_same_length(fold_test_accs)

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    axes[0, 0].plot(np.nanmean(train_loss_mx, axis=0), label="Train Loss (mean)", linewidth=2)
    axes[0, 0].plot(np.nanmean(test_loss_mx, axis=0), label="Test Loss (mean)", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss Curve")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # F1-score
    axes[0, 1].plot(np.nanmean(train_f1_mx, axis=0), label="Train F1 (macro, mean)", linewidth=2)
    axes[0, 1].plot(np.nanmean(test_f1_mx, axis=0), label="Test F1 (macro, mean)", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("F1-score (macro)")
    axes[0, 1].set_title("F1-score Curve")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Accuracy
    axes[1, 0].plot(np.nanmean(train_acc_mx, axis=0), label="Train Acc (mean)", linewidth=2)
    axes[1, 0].plot(np.nanmean(test_acc_mx, axis=0), label="Test Acc (mean)", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy Curve")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 隐藏第四个子图
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def save_training_summary(
    config: TrainingConfig,
    fold_results: List[Dict],
    cv_summary: Dict[str, float],
    meta: Dict[str, Any],
) -> None:
    """保存训练摘要"""
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": config.to_dict(),
        "data_info": {
            "num_classes": meta["num_classes"],
            "classes_values": meta["classes_values"].tolist(),
            "value2id": {int(k): int(v) for k, v in meta["value2id"].items()},
        },
        "epochs_per_fold": [r["epochs_trained"] for r in fold_results],
        "cv_summary": cv_summary,
    }

    summary_path = config.save_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50} 交叉验证汇总 {'='*50}")
    for key, value in cv_summary.items():
        print(f"{key}: {value:.4f}")
    print(f"\n训练摘要已保存至: {summary_path}")