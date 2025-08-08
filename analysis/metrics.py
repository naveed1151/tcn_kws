# analysis/metrics.py

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(train_hist, val_hist, test_metrics=None, save_path=None, title_prefix=""):
    """
    Plots loss and metrics (acc, prec, rec, f1) for train and val.
    If test_metrics is given, shows as horizontal lines.
    Metrics are arranged in a 2-column subplot layout.
    """
    metrics = ["loss", "acc", "prec", "rec", "f1"]
    titles = {
        "loss": "Loss",
        "acc": "Accuracy",
        "prec": "Precision",
        "rec": "Recall",
        "f1": "F1 Score"
    }

    n = len(metrics)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
    axes = axes.flatten()

    epochs = list(range(1, len(train_hist["loss"]) + 1))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(epochs, train_hist[metric], label="Train")
        ax.plot(epochs, val_hist[metric], label="Val")
        if test_metrics and metric in test_metrics:
            ax.axhline(test_metrics[metric], color='gray', linestyle='--', label="Test")
        ax.set_title(f"{title_prefix} {titles[metric]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(titles[metric])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)

    # Hide unused subplots if odd number
    for i in range(n, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] Metrics plot to {save_path}")
    else:
        plt.show()


def plot_test_confusion_matrix(confusion_matrix, class_names, normalize=False, save_path=None):
    """
    Plots the confusion matrix (optionally normalized) as a heatmap.
    """
    from matplotlib import cm

    cmatrix = np.array(confusion_matrix)
    if normalize:
        row_sums = cmatrix.sum(axis=1, keepdims=True)
        cmatrix = cmatrix / np.maximum(row_sums, 1e-12)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cmatrix, interpolation="nearest", cmap=cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (Normalized)" if normalize else "")
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cmatrix.max() / 2.
    for i in range(cmatrix.shape[0]):
        for j in range(cmatrix.shape[1]):
            ax.text(j, i, format(cmatrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cmatrix[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] Confusion matrix to {save_path}")
    else:
        plt.show()