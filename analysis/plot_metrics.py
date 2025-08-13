# analysis/metrics.py

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from typing import Dict, List, Optional, Sequence, Mapping, Any, Union

__all__ = [
    "plot_metrics",
    "plot_test_bars",
    "plot_test_confusion_matrix",
]


def plot_metrics(
    train_hist: Mapping[str, Sequence[float]],
    val_hist: Mapping[str, Sequence[float]],
    test_metrics: Optional[Mapping[str, Union[int, float]]] = None,
    save_path: Optional[str] = None,
    title_prefix: str = "",
    full_width_test: bool = True
) -> None:
    """
    Plots loss and metrics (acc, prec, rec, f1) for train and val.
    If test_metrics is given, shows:
      - Horizontal test lines on each metric subplot
      - A dedicated bar chart for test metrics inside the SAME multi-plot figure
    """
    metrics = ["loss", "acc", "prec", "rec", "f1"]
    titles = {
        "loss": "Loss",
        "acc": "Accuracy",
        "prec": "Precision",
        "rec": "Recall",
        "f1": "F1 Score"
    }

    cols = 2
    n_line_plots = len(metrics)  # 5
    line_rows = (n_line_plots + cols - 1) // cols  # rows for metric curves
    extra_row = 1 if (test_metrics and full_width_test) else 0
    total_rows = line_rows + (0 if (test_metrics and not full_width_test) else extra_row)

    if test_metrics and full_width_test:
        fig = plt.figure(figsize=(12, 3.5 * (line_rows + 0.8)))
        gs = gridspec.GridSpec(line_rows + 1, cols, height_ratios=[1]*line_rows + [0.9])
        axes = []
        for r in range(line_rows):
            for c in range(cols):
                axes.append(fig.add_subplot(gs[r, c]))
        test_ax = fig.add_subplot(gs[line_rows, :])
    else:
        n_total = n_line_plots + (1 if test_metrics else 0)
        rows = (n_total + cols - 1) // cols
        fig, axes_m = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
        axes = axes_m.flatten()
        test_ax = None

    epochs = list(range(1, len(train_hist["loss"]) + 1))

    # Line plots (train/val curves + optional test horizontal lines)
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(epochs, train_hist[metric], label="Train")
        ax.plot(epochs, val_hist[metric], label="Val")
        if test_metrics and metric in test_metrics:
            ax.axhline(test_metrics[metric], color='gray', linestyle='--', label="Test")
        ax.set_title(f"{title_prefix} {titles[metric]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(titles[metric])
        if metric != "loss":
            ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)

    # Test metrics bar chart integrated into the same figure (last axis)
    used_axes = n_line_plots
    if test_metrics:
        if full_width_test:
            axb = test_ax
        else:
            axb = axes[n_line_plots]
        # Stable order; include only keys present
        order = ["loss", "acc", "prec", "rec", "f1"]
        keys = [k for k in order if k in test_metrics]
        vals = [float(test_metrics[k]) for k in keys]
        colors = []
        for k, v in zip(keys, vals):
            alpha = 0.3 + 0.7 * (v if (k != "loss" and 0 <= v <= 1) else 0.6)
            colors.append((0.25, 0.6, 0.35, alpha))
        bars = axb.bar(keys, vals, color=colors, edgecolor="black", linewidth=0.7)
        axb.set_title(f"{title_prefix} Test Metrics")
        axb.set_ylabel("Value")
        if all(k != "loss" for k in keys):
            axb.set_ylim(0, 1)
        for b, v in zip(bars, vals):
            txt = f"{v:.2%}" if (b.get_height() <= 1 and all(k != 'loss' for k in keys)) else f"{v:.3f}"
            axb.annotate(txt, xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                         xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)
        axb.grid(axis="y", alpha=0.3)
        used_axes += (0 if full_width_test else 1)

    # Hide any leftover empty axes
    for i in range(used_axes, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(title_prefix.strip(), fontsize=14 if title_prefix else 0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[Saved] Metrics plot to {save_path}")
    else:
        plt.show()


def plot_test_bars(
    test_metrics: Optional[Mapping[str, Union[int, float]]],
    save_path: str,
    title: str = "Test metrics"
) -> None:
    """
    Bar chart of test metrics: Acc, P, R, F1 (and optionally Loss as a separate bar).
    test_metrics: dict with keys like {"loss","acc","prec","rec","f1"}
    """
    if not test_metrics:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = []
    values = []
    # Plot main metrics
    for k, lbl in (("acc", "Acc"), ("prec", "P"), ("rec", "R"), ("f1", "F1")):
        if k in test_metrics and test_metrics[k] is not None:
            labels.append(lbl)
            values.append(float(test_metrics[k]))

    # Optional: include Loss as a separate bar at the end
    include_loss = "loss" in test_metrics and test_metrics["loss"] is not None
    if include_loss:
        labels.append("Loss")
        values.append(float(test_metrics["loss"]))

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color=["#4e79a7"] * len(values), edgecolor="black")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    # Annotate bars
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_test_confusion_matrix(
    confusion_matrix: Sequence[Sequence[Union[int, float]]],
    class_names: Sequence[str],
    normalize: bool = False,
    save_path: Optional[str] = None
) -> None:
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