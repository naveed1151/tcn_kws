"""
Unified evaluation utilities (binary + multiclass) for training / QAT / analysis.

API:
    from train.evaluate import evaluate_model
    loss, acc, prec, rec, f1 = evaluate_model(model, loader, device,
                                              task_type, num_classes, criterion,
                                              threshold=0.5, pin_memory=False)

If return_confusion=True and task_type=='multiclass', a confusion matrix ndarray
is also returned as the last element.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from train.utils import (
    _binary_counts, _derive_metrics, _multiclass_confusion_add,
    _multiclass_macro_prf1
)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_type: str,
    num_classes: int,
    criterion: nn.Module,
    threshold: float = 0.5,
    pin_memory: bool = False,
    return_confusion: bool = False,
) -> Tuple[float, float, float, float, float, Optional[np.ndarray]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    tp = fp = fn = 0
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)] if task_type == "multiclass" else None

    for batch in loader:
        if isinstance(batch, (tuple, list)):
            batch_x, batch_y = batch
        elif isinstance(batch, dict):
            batch_x, batch_y = batch["x"], batch["y"]
        else:
            raise ValueError("Unsupported batch format.")
        num_batches += 1
        batch_x = batch_x.to(device, non_blocking=pin_memory)
        if task_type == "binary":
            targets = batch_y.float().unsqueeze(1).to(device)
        else:
            targets = batch_y.long().to(device)

        logits = model(batch_x)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        if task_type == "binary":
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            correct += (preds == targets.long()).sum().item()
            total += targets.numel()
            _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
            tp += _tp; fp += _fp; fn += _fn
        else:
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            cm = _multiclass_confusion_add(cm, preds, targets, num_classes)

    if task_type == "binary":
        loss, acc, prec, rec, f1 = _derive_metrics(total_loss, num_batches, correct, total, tp, fp, fn)
        return (loss, acc, prec, rec, f1, None) if return_confusion else (loss, acc, prec, rec, f1)
    else:
        avg_loss, acc = _derive_metrics(total_loss, num_batches, correct, total)
        macro_p, macro_r, macro_f1 = _multiclass_macro_prf1(cm)
        conf_np = np.array(cm, dtype=np.int64)
        return (avg_loss, acc, macro_p, macro_r, macro_f1, conf_np) if return_confusion else (avg_loss, acc, macro_p, macro_r, macro_f1)
