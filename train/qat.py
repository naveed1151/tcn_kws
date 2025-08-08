import os
import argparse
import torch
import torch.nn as nn
import torch.quantization as tq
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time

from train.utils import (
    build_model_from_cfg, load_config, _binary_counts, _derive_metrics,
    _multiclass_confusion_add, _multiclass_macro_prf1, export_quantized_weights_npz
)
from data_loader.utils import make_datasets
from analysis.metrics import plot_metrics


def train_qat(model, loader, optimizer, criterion, device, epochs, task_type, num_classes, threshold=0.5):
    model.to(device)
    hist = {"loss": [], "acc": [], "precision": [], "recall": [], "f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
        cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None

        loop = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for x, y in loop:
            x = x.to(device)
            targets = y.float().unsqueeze(1).to(device) if task_type == "binary" else y.long().to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

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
            avg_loss, acc, prec, rec, f1 = _derive_metrics(total_loss, len(loader), correct, total, tp, fp, fn)
        else:
            avg_loss, acc = _derive_metrics(total_loss, len(loader), correct, total)
            prec, rec, f1 = _multiclass_macro_prf1(cm)

        hist["loss"].append(avg_loss)
        hist["acc"].append(acc)
        hist["precision"].append(prec)
        hist["recall"].append(rec)
        hist["f1"].append(f1)

        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}  Acc: {acc:.4f}  P: {prec:.4f}  R: {rec:.4f}  F1: {f1:.4f}")

    return hist


def prepare_qat_model(model, qat_cfg):
    model.train()
    model.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(model, inplace=True)
    return model


def main():
    parser = argparse.ArgumentParser(description="Quantization Aware Training for TCN KWS")
    parser.add_argument("--config", required=True, help="Path to config (e.g., config/multiclass.yaml)")
    parser.add_argument("--weights", default=None, help="Optional pretrained weights to initialize from")
    parser.add_argument("--epochs", type=int, default=5, help="QAT epochs")
    parser.add_argument("--save-preconvert", action="store_true", help="Save model weights before convert()")
    args = parser.parse_args()

    t0 = time()
    cfg = load_config(args.config)
    qat_cfg = cfg.get("qat", {})
    wb = qat_cfg.get("weight_bits", 8)
    ab = qat_cfg.get("act_bits", 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_type = cfg["task"]["type"]

    dl, num_classes = make_datasets(cfg, which="train", batch_size=cfg["train"].get("batch_size", 64))
    sample_x, _ = next(iter(dl))
    model = build_model_from_cfg(cfg, sample_x[0], num_classes).to(device)

    if args.weights:
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded weights from {args.weights}")

    model = prepare_qat_model(model, qat_cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg["train"].get("lr", 1e-3))
    criterion = nn.CrossEntropyLoss() if task_type == "multiclass" else nn.BCEWithLogitsLoss()

    print("[QAT] Starting training...")
    history = train_qat(model, dl, optimizer, criterion, device, epochs=args.epochs, task_type=task_type, num_classes=num_classes)

    if args.save_preconvert:
        pre_path = os.path.join(cfg["output"]["weights_dir"], f"model_weights_qat_{wb}w{ab}a_preconvert.pt")
        torch.save(model.state_dict(), pre_path)
        print(f"[INFO] Saved pre-convert model weights to {pre_path}")

    model.eval()
    model = tq.convert(model, inplace=True)

    weights_dir = cfg["output"]["weights_dir"]
    os.makedirs(weights_dir, exist_ok=True)
    fname = f"model_weights_qat_{wb}w{ab}a.pt"
    out_path = os.path.join(weights_dir, fname)
    torch.save(model.state_dict(), out_path)
    print(f"[OK] Saved QAT weights to {out_path}")

    npz_path = os.path.splitext(out_path)[0] + ".npz"
    export_quantized_weights_npz(model, npz_path)

    plots_dir = os.path.join(cfg["output"]["plots_dir"], "training")
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = os.path.join(plots_dir, f"metrics_qat_{wb}w{ab}a.png")

    plot_metrics(history, None, save_path=fig_path, title_prefix=f"QAT {wb}w{ab}a")
    print(f"[OK] Saved training plot to {fig_path}")
    print(f"[DONE] QAT completed in {time() - t0:.1f} seconds.")


if __name__ == "__main__":
    main()
