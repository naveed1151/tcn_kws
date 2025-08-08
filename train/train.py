import os
import time
import argparse
from copy import deepcopy
from typing import List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

from train.utils import (
    _binary_counts, _derive_metrics, _multiclass_confusion_add,
    _multiclass_macro_prf1, compute_confusion_matrix,
     load_config, TransformDataset, MFCCAugment,
    build_model_from_cfg,
    calc_required_layers
)

from data_loader.utils import make_datasets
from analysis.metrics import plot_metrics, plot_test_confusion_matrix


# -----------------------
# Trainer
# -----------------------
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task_type = cfg["task"]["type"]  # "binary" or "multiclass"
        
        # Device
        device_cfg = cfg["train"].get("device", "auto")
        if device_cfg == "cuda":
            self.device = torch.device("cuda")
        elif device_cfg == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader, self.val_loader, self.test_loader, dataset = make_datasets(self.cfg, which="all", batch_size=self.cfg["train"]["batch_size"])
        self.train_dataset = self.train_loader.dataset
        self.val_dataset   = self.val_loader.dataset
        self.test_dataset  = self.test_loader.dataset

        # Get number of classes from dataset (class_list + unknown/background flags)
        self.num_classes = self.train_dataset.dataset.num_classes
        print(f"[INFO] Detected {self.num_classes} classes")

        # Augmentation (training split only)
        aug_cfg = cfg.get("augmentation", {})
        use_aug = bool(aug_cfg.get("enable", False))
        if use_aug:
            hop_s = float(cfg["data"]["mfcc"]["hop_length_s"])
            augment = MFCCAugment(
                hop_length_s=hop_s,
                max_shift_ms=float(aug_cfg.get("max_shift_ms", 100.0)),
                noise_prob=float(aug_cfg.get("noise_prob", 0.15)),
                noise_std_factor=float(aug_cfg.get("noise_std_factor", 0.05)),
                seed=aug_cfg.get("seed", None),
            )
            train_set_for_loader = TransformDataset(self.train_dataset, transform=augment)
        else:
            train_set_for_loader = self.train_dataset

        # Loaders
        self.batch_size = int(cfg["train"]["batch_size"])
        self.train_loader = DataLoader(train_set_for_loader, batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False)
        self.test_loader  = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False)

        # Model dims
        sample_x, _ = self.train_dataset[0]
        input_channels = sample_x.shape[0]
        seq_len = sample_x.shape[1]
        self.num_layers = calc_required_layers(seq_len, kernel_size=int(cfg["model"]["kernel_size"]), dilation_base=2)

        # Model
        self.model = build_model_from_cfg(cfg, sample_x, self.num_classes).to(self.device)

        # Optimizer
        lr = float(cfg["train"]["learning_rate"])
        weight_decay = float(cfg["train"]["weight_decay"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss
        if self.task_type == "binary":
            pos_weight = cfg["train"].get("pos_weight", None)
            if pos_weight is not None:
                pos_weight_tensor = torch.tensor([float(pos_weight)], device=self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
            self.threshold = float(cfg["train"].get("threshold", 0.5) or 0.5)
            self.auto_threshold = bool(cfg["train"].get("auto_threshold", False))
            self.threshold_search_steps = int(cfg["train"].get("threshold_search_steps", 101))
            self.class_names = ["neg", "pos"]
        else:
            class_weights = cfg["train"].get("class_weights", None)
            label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
            weight_tensor = None
            if class_weights is not None:
                if len(class_weights) != self.num_classes:
                    raise ValueError("class_weights length must equal number of classes.")
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
            self.class_names = dataset.class_names  # list[str] length = num_classes

        self.test_metrics = None
        self.test_confusion = None  # (only for multiclass)
        # Histories / outputs
        self.epochs = int(cfg["train"]["num_epochs"])
        self.plots_dir = cfg["output"]["plots_dir"] + "/training"
        self.metrics_figure = cfg["output"]["metrics_figure"]
        self.weights_path = os.path.join(cfg["output"]["weights_dir"], "model_weights_fp.pt")
        self.use_tqdm = bool(cfg["output"].get("tqdm", True))

        self.train_hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
        self.val_hist   = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}

        # ---- Sanity checks: dataset sizes and MFCC shapes ----
        print(f"[Split sizes] train={len(self.train_dataset)}  val={len(self.val_dataset)}  test={len(self.test_dataset)}")

        # Peek a few samples from the training subset
        for i in range(min(3, len(self.train_dataset))):
            x_i, y_i = self.train_dataset[i]
            print(f"[Sample {i}] x.shape={tuple(x_i.shape)}  label={y_i}")

        # Derive input dims to confirm (C, T)
        sample_x, _ = self.train_dataset[0]
        C, T = sample_x.shape
        print(f"[MFCC dims] channels(C)={C}  timesteps(T)={T}")

        # If you compute layers dynamically, also print the result
        k = int(self.cfg["model"]["kernel_size"])
        print(f"[TCN] kernel_size={k}  computed_num_layers={self.num_layers}")

        # Print total number of trainable weights
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[TCN] Total trainable weights: {total_params}")

        if self.task_type == "binary":
            import numpy as np
            ys = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            pos = int(np.sum(ys)); neg = len(ys) - pos
            print(f"[Train balance] pos={pos}  neg={neg}  ratio={neg/max(1,pos):.2f}:1")
        
        if self.task_type == "multiclass":
            import numpy as np
            ys = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            counts = np.bincount(ys, minlength=self.num_classes)
            print(f"[Train per-class counts] {counts.tolist()}")
            print(f"[Classes] {self.class_names}")


 
        # Optional: assert reasonable ranges (adjust to your expectations)
        assert C in (16, 20, 28, 40) or C > 0, f"Unexpected MFCC channels: {C}"
        assert T >= 40, f"Too few time steps ({T}). Check MFCC hop/window/duration."

    def _loop(self, loader, train_mode=True):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        tp = fp = fn = 0
        if self.task_type == "multiclass":
            cm = [[0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]
        else:
            cm = None

        iterator = tqdm(loader, desc="Training" if train_mode else "Validating", leave=False) if self.use_tqdm else loader

        with torch.set_grad_enabled(train_mode):
            for batch_x, batch_y in iterator:
                num_batches += 1
                batch_x = batch_x.to(self.device)

                if self.task_type == "binary":
                    targets = batch_y.float().unsqueeze(1).to(self.device)
                else:
                    targets = batch_y.long().to(self.device)

                logits = self.model(batch_x)
                loss = self.criterion(logits, targets)

                if train_mode:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                if self.task_type == "binary":
                    probs = torch.sigmoid(logits)
                    preds = (probs > self.threshold).long()
                    correct += (preds == targets.long()).sum().item()
                    total += targets.numel()
                    _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
                    tp += _tp; fp += _fp; fn += _fn
                else:
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()
                    cm = _multiclass_confusion_add(cm, preds, targets, self.num_classes)

        if self.task_type == "binary":
            return _derive_metrics(total_loss, num_batches, correct, total, tp, fp, fn)
        else:
            avg_loss, acc = _derive_metrics(total_loss, num_batches, correct, total)
            macro_p, macro_r, macro_f1 = _multiclass_macro_prf1(cm)
            return avg_loss, acc, macro_p, macro_r, macro_f1

    # ---------- Threshold search (binary only) ----------
    def find_best_threshold(self, steps=101):
        assert self.task_type == "binary", "Threshold search is for binary mode."
        self.model.eval()
        probs_all, y_all = [], []
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().unsqueeze(1).to(self.device)
                logits = self.model(batch_x)
                probs_all.append(torch.sigmoid(logits).cpu())
                y_all.append(batch_y.cpu())
        probs_all = torch.cat(probs_all, dim=0).view(-1)
        y_all = torch.cat(y_all, dim=0).view(-1)

        best_thr, best_f1 = 0.5, -1.0
        for thr in torch.linspace(0.01, 0.99, steps=steps):
            preds = (probs_all > thr).int()
            tp = int(((preds == 1) & (y_all == 1)).sum())
            fp = int(((preds == 1) & (y_all == 0)).sum())
            fn = int(((preds == 0) & (y_all == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_thr = float(f1), float(thr)
        return best_thr, best_f1

    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate current model on a loader using current threshold; returns (loss, acc, P, R, F1)."""
        self.model.eval()
        total_loss = 0.0; correct = 0; total = 0; num_batches = 0
        tp = fp = fn = 0
        if self.task_type == "multiclass":
            cm = [[0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]
        else:
            cm = None

        for batch_x, batch_y in loader:
            num_batches += 1
            batch_x = batch_x.to(self.device)
            if self.task_type == "binary":
                targets = batch_y.float().unsqueeze(1).to(self.device)
            else:
                targets = batch_y.long().to(self.device)
            logits = self.model(batch_x)
            loss = self.criterion(logits, targets)
            total_loss += loss.item()

            if self.task_type == "binary":
                probs = torch.sigmoid(logits)
                preds = (probs > self.threshold).long()
                correct += (preds == targets.long()).sum().item()
                total += targets.numel()
                _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
                tp += _tp; fp += _fp; fn += _fn
            else:
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
                cm = _multiclass_confusion_add(cm, preds, targets, self.num_classes)

        if self.task_type == "binary":
            return _derive_metrics(total_loss, num_batches, correct, total, tp, fp, fn)
        else:
            avg_loss, acc = _derive_metrics(total_loss, num_batches, correct, total)
            macro_p, macro_r, macro_f1 = _multiclass_macro_prf1(cm)
            return avg_loss, acc, macro_p, macro_r, macro_f1

    def train(self):
        start = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}:")
            tr_loss, tr_acc, tr_p, tr_r, tr_f1 = self._loop(self.train_loader, train_mode=True)
            va_loss, va_acc, va_p, va_r, va_f1 = self._loop(self.val_loader,   train_mode=False)

            self.train_hist["loss"].append(tr_loss); self.val_hist["loss"].append(va_loss)
            self.train_hist["acc"].append(tr_acc);   self.val_hist["acc"].append(va_acc)
            self.train_hist["prec"].append(tr_p);    self.val_hist["prec"].append(va_p)
            self.train_hist["rec"].append(tr_r);     self.val_hist["rec"].append(va_r)
            self.train_hist["f1"].append(tr_f1);     self.val_hist["f1"].append(va_f1)

            print(f"  Train | Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  P: {tr_p:.4f}  R: {tr_r:.4f}  F1: {tr_f1:.4f}")
            print(f"  Val   | Loss: {va_loss:.4f}  Acc: {va_acc:.4f}  P: {va_p:.4f}  R: {va_r:.4f}  F1: {va_f1:.4f}")

        elapsed = time.perf_counter() - start
        hrs, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        print(f"Total training time: {hrs:02d}:{mins:02d}:{secs:02d} (hh:mm:ss)")

        # Optional: auto threshold tuning (binary)
        if self.task_type == "binary" and self.auto_threshold:
            last_val_f1 = self.val_hist["f1"][-1] if self.val_hist["f1"] else -1.0
            best_thr, best_f1 = self.find_best_threshold(steps=self.threshold_search_steps)
            if best_f1 > last_val_f1:
                old_thr = self.threshold
                self.threshold = best_thr
                print(f"[Auto-threshold] Improved F1 {last_val_f1:.4f} → {best_f1:.4f}. "
                      f"Updating threshold: {old_thr:.3f} → {self.threshold:.3f}")
                # Re-evaluate on val for reporting
                va_loss, va_acc, va_p, va_r, va_f1 = self.evaluate(self.val_loader)
                print(f"[Post-threshold] Val | Loss: {va_loss:.4f}  Acc: {va_acc:.4f}  "
                      f"P: {va_p:.4f}  R: {va_r:.4f}  F1: {va_f1:.4f}")
                os.makedirs(self.plots_dir, exist_ok=True)
                with open(os.path.join(self.plots_dir, "best_threshold.txt"), "w") as f:
                    f.write(f"{self.threshold:.6f}\n")
            else:
                print(f"[Auto-threshold] Best searched F1 {best_f1:.4f} did NOT beat last val F1 {last_val_f1:.4f}. "
                      f"Keeping threshold at {self.threshold:.3f}.")

        # Final test metrics (no tuning on test)
        te_loss, te_acc, te_p, te_r, te_f1 = self.evaluate(self.test_loader)
        self.test_metrics = {"loss": te_loss, "acc": te_acc, "prec": te_p, "rec": te_r, "f1": te_f1}
        print(f"[Test] Loss: {te_loss:.4f}  Acc: {te_acc:.4f}  P: {te_p:.4f}  R: {te_r:.4f}  F1: {te_f1:.4f}")

        # If multiclass, also compute confusion matrix on test set
        if self.task_type == "multiclass":
            cm = compute_confusion_matrix(self.model, self.test_loader, self.device, self.num_classes)
            self.test_confusion = cm  # numpy array

    def save(self):
        torch.save(self.model.state_dict(), self.weights_path)
        print(f"Model saved to {self.weights_path}")

# -----------------------
# Entry
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/binary.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("data", {}).get("preprocessed_dir", None) is None:
        raise ValueError("Config must define data.preprocessed_dir")

    task_type = cfg.get("task", {}).get("type", None)
    if task_type not in ("binary", "multiclass"):
        raise ValueError("Config 'task.type' must be 'binary' or 'multiclass'.")

    # sanity check splits
    val_frac = float(cfg["data"]["val_split"])
    test_frac = float(cfg["data"]["test_split"])
    if val_frac <= 0 or test_frac <= 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("Require 0 < val_split, test_split and val_split + test_split < 1.")

    trainer = Trainer(cfg)
    trainer.train()
    trainer.save()
    plot_metrics(trainer.train_hist, trainer.val_hist,
             test_metrics=trainer.test_metrics,
             save_path=os.path.join(trainer.plots_dir, trainer.metrics_figure),
             title_prefix="Binary" if trainer.task_type == "binary" else "Macro")

    if trainer.task_type == "multiclass":
        plot_test_confusion_matrix(trainer.test_confusion, class_names=trainer.class_names,
                                normalize=False,
                                save_path=os.path.join(trainer.plots_dir, "test_confusion_matrix.png"))
        plot_test_confusion_matrix(trainer.test_confusion, class_names=trainer.class_names,
                                normalize=True,
                                save_path=os.path.join(trainer.plots_dir, "test_confusion_matrix_norm.png"))



if __name__ == "__main__":
    main()
