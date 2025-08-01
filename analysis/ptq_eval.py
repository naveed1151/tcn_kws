"""
Post-Training Quantization (weight-only) + evaluation for DilatedTCN KWS.

- Quantizes Conv1d/Linear weights to N bits (2..8,16) with symmetric uniform quantization.
- Supports per-channel (default) or per-tensor scales.
- Keeps activations in float32 (weight-only PTQ), so it runs on CPU/GPU without special backends.
- Evaluates accuracy/precision/recall/F1 on train/val/full wrapped dataset (binary or multiclass).

Usage examples:
  python analysis/ptq_eval.py --config config/binary.yaml --weights model_weights.pt
  python analysis/ptq_eval.py --config config/binary.yaml --weights model_weights.pt --bits 8,4,3 --search-threshold
  python analysis/ptq_eval.py --config config/multiclass.yaml --weights model_weights.pt --dataset full --scheme per_tensor --save-csv analysis/ptq_results.csv
"""

import os
import argparse
from copy import deepcopy

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import matplotlib.pyplot as plt

from model.model import DilatedTCN
from data_loader.mfcc_dataset import MFCCDataset


# -----------------------
# Config helpers (same behavior as train.py)
# -----------------------
def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def load_config(path):
    with open(path, "r") as f:
        cfg_task = yaml.safe_load(f)

    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if os.path.basename(path) != "base.yaml" and os.path.exists(base_path):
        with open(base_path, "r") as f:
            cfg_base = yaml.safe_load(f)
        return deep_update(deepcopy(cfg_base), cfg_task)
    return cfg_task


# -----------------------
# Dataset wrappers (mirrors training)
# -----------------------
class BinaryKeywordDataset(Dataset):
    def __init__(self, base_dataset, target_word, index_to_word, downsample_ratio=None):
        samples = []
        for x, label_idx in base_dataset:
            word = index_to_word[label_idx]
            y = 1 if word == target_word else 0
            samples.append((x, y))

        if downsample_ratio is not None:
            positives = [s for s in samples if s[1] == 1]
            negatives = [s for s in samples if s[1] == 0]
            rng = np.random.RandomState(0)
            rng.shuffle(negatives)
            keep_neg = int(len(positives) * float(downsample_ratio))
            samples = positives + negatives[:keep_neg]

        rng = np.random.RandomState(0)
        rng.shuffle(samples)
        self.samples = samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class MultiClassDataset(Dataset):
    def __init__(self, base_dataset, class_list, index_to_word, label_to_index):
        self.samples = []
        if class_list is None:
            self.samples = list(base_dataset)
            self.num_classes = len(label_to_index)
        else:
            class_set = set(class_list)
            new_label_map = {w: i for i, w in enumerate(class_list)}
            for x, label_idx in base_dataset:
                word = index_to_word[label_idx]
                if word in class_set:
                    self.samples.append((x, new_label_map[word]))
            self.num_classes = len(class_list)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def make_datasets(cfg, which="val", batch_size=64):
    """Return DataLoader over requested split: 'train'|'val'|'full'."""
    full = MFCCDataset(cfg["data"]["data_dir"])
    label_to_index = full.label_to_index
    index_to_word = {v: k for k, v in label_to_index.items()}

    task_type = cfg["task"]["type"]
    if task_type == "binary":
        ds = BinaryKeywordDataset(
            full,
            target_word=cfg["data"]["target_word"],
            index_to_word=index_to_word,
            downsample_ratio=cfg["data"].get("neg_downsample_ratio", None),
        )
        num_classes = 1
    else:
        ds = MultiClassDataset(
            full,
            class_list=cfg["data"].get("class_list", None),
            index_to_word=index_to_word,
            label_to_index=label_to_index,
        )
        num_classes = ds.num_classes

    if which == "full":
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        return dl, num_classes
    else:
        val_frac = float(cfg["data"]["val_split"])
        val_size = max(1, int(len(ds) * val_frac))
        train_size = max(1, len(ds) - val_size)
        # Deterministic split for reproducibility here
        g = torch.Generator().manual_seed(0)
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size], generator=g)
        subset = train_ds if which == "train" else val_ds
        dl = DataLoader(subset, batch_size=batch_size, shuffle=False)
        return dl, num_classes


# -----------------------
# Model build & weight loading
# -----------------------
def build_model_from_cfg(cfg, num_classes, device):
    # Probe a sample to get input_channels/seq_len
    probe_ds = MFCCDataset(cfg["data"]["data_dir"])
    x0, _ = probe_ds[0]
    input_channels = x0.shape[0]
    seq_len = x0.shape[1]

    kernel_size = int(cfg["model"]["kernel_size"])
    hidden = int(cfg["model"]["hidden_channels"])
    dropout = float(cfg["model"]["dropout"])

    # compute layers (same as training)
    def _calc_layers(seq_len, kernel_size=3, dilation_base=2):
        L = 0; rf = 1
        while rf < seq_len:
            rf += (kernel_size - 1) * (dilation_base ** L)
            L += 1
        return L
    L = _calc_layers(seq_len, kernel_size=kernel_size, dilation_base=2)

    model = DilatedTCN(
        input_channels=input_channels,
        num_layers=L,
        hidden_channels=hidden,
        kernel_size=kernel_size,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)
    return model


def load_state_dict(weights_path, device="cpu"):
    obj = torch.load(weights_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise ValueError("Unsupported weights file format.")


# -----------------------
# Quantization (weight-only, symmetric)
# -----------------------
def quantize_dequantize_weight_per_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    # signed symmetric: [-qmax, qmax]
    qmax = (1 << (bits - 1)) - 1  # 2^(b-1)-1
    with torch.no_grad():
        max_abs = w.abs().max()
        if max_abs == 0:
            return torch.zeros_like(w)
        scale = max_abs / qmax
        q = torch.clamp(torch.round(w / scale), -qmax, qmax)
        return q * scale

def quantize_dequantize_weight_per_channel(w: torch.Tensor, bits: int, ch_axis: int = 0) -> torch.Tensor:
    """
    Per-output-channel quant for Conv1d/Linear:
      Conv1d weight: (C_out, C_in, K)  -> ch_axis=0
      Linear weight: (C_out, C_in)     -> ch_axis=0
    """
    qmax = (1 << (bits - 1)) - 1
    with torch.no_grad():
        # Move channel axis to front
        perm = list(range(w.ndim))
        perm[0], perm[ch_axis] = perm[ch_axis], perm[0]
        inv_perm = [perm.index(i) for i in range(len(perm))]
        w_perm = w.permute(*perm).contiguous()

        C = w_perm.shape[0]
        rest = w_perm.view(C, -1)
        max_abs = rest.abs().max(dim=1).values  # (C,)
        # Avoid div0
        scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)  # (C,)
        q = torch.round(rest / scale.unsqueeze(1))
        q = torch.clamp(q, -qmax, qmax)
        deq = (q * scale.unsqueeze(1)).view_as(w_perm)
        return deq.permute(*inv_perm).to(w.dtype)

def quantize_model_weights(model: nn.Module, bits: int, scheme: str = "per_channel") -> nn.Module:
    """
    Returns a deep-copied model whose Conv1d/Linear weights are quantized (then de-quantized to float).
    Biases are kept in float.
    """
    m = deepcopy(model).cpu()  # quantize on CPU to avoid dtype/device friction
    for name, module in m.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if not hasattr(module, "weight") or not isinstance(module.weight, torch.Tensor):
                continue
            w = module.weight.data
            if scheme == "per_channel":
                module.weight.data = quantize_dequantize_weight_per_channel(w, bits, ch_axis=0)
            else:
                module.weight.data = quantize_dequantize_weight_per_tensor(w, bits)
    return m


# -----------------------
# Metrics & evaluation
# -----------------------
def binary_counts(preds, targets):
    preds = preds.long(); targets = targets.long()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return tp, fp, fn, correct, total

def derive_binary_metrics(tp, fp, fn, correct, total):
    acc = correct / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1

def multiclass_confusion_add(cm, preds, targets, K):
    for p, t in zip(preds.tolist(), targets.tolist()):
        if 0 <= t < K and 0 <= p < K:
            cm[t][p] += 1
    return cm

def multiclass_macro_prf1(cm):
    K = len(cm)
    precisions, recalls, f1s = [], [], []
    for k in range(K):
        tp = cm[k][k]
        fp = sum(cm[r][k] for r in range(K)) - tp
        fn = sum(cm[k][c] for c in range(K)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    macro_p = sum(precisions) / K if K > 0 else 0.0
    macro_r = sum(recalls) / K if K > 0 else 0.0
    macro_f1 = sum(f1s) / K if K > 0 else 0.0
    return macro_p, macro_r, macro_f1

@torch.no_grad()
def evaluate_model(model, dl, cfg, device, task_type, threshold=None, search_threshold=False, search_steps=101):
    model.eval().to(device)
    if task_type == "binary":
        probs_all = []
        y_all = []
        for x, y in dl:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)
            logits = model(x)
            probs_all.append(torch.sigmoid(logits).cpu())
            y_all.append(y.cpu())
        probs_all = torch.cat(probs_all, dim=0).view(-1)
        y_all = torch.cat(y_all, dim=0).view(-1)

        if search_threshold:
            best_thr, best_f1 = 0.5, -1.0
            for thr in torch.linspace(0.01, 0.99, steps=search_steps):
                preds = (probs_all > thr).int()
                tp = int(((preds == 1) & (y_all == 1)).sum())
                fp = int(((preds == 1) & (y_all == 0)).sum())
                fn = int(((preds == 0) & (y_all == 1)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_thr = float(f1), float(thr)
            threshold = best_thr
        else:
            if threshold is None:
                threshold = cfg["train"].get("threshold", 0.5) or 0.5

        preds = (probs_all > threshold).int()
        tp = int(((preds == 1) & (y_all == 1)).sum())
        fp = int(((preds == 1) & (y_all == 0)).sum())
        fn = int(((preds == 0) & (y_all == 1)).sum())
        correct = int((preds == y_all.int()).sum())
        total = int(y_all.numel())
        acc, prec, rec, f1 = derive_binary_metrics(tp, fp, fn, correct, total)
        return {
            "acc": acc, "prec": prec, "rec": rec, "f1": f1,
            "threshold_used": float(threshold)
        }

    else:
        K = dl.dataset.num_classes if hasattr(dl.dataset, "num_classes") else None
        cm = [[0 for _ in range(K)] for _ in range(K)]
        correct = 0; total = 0
        for x, y in dl:
            x = x.to(device); y = y.long().to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
            cm = multiclass_confusion_add(cm, preds.cpu(), y.cpu(), K)
        acc = correct / total if total > 0 else 0.0
        p, r, f1 = multiclass_macro_prf1(cm)
        return {"acc": acc, "prec": p, "rec": r, "f1": f1}


# -----------------------
# Main
# -----------------------
def parse_bits(bits_str):
    out = []
    for s in bits_str.split(","):
        s = s.strip().lower()
        if s in ("f", "fp", "float", "fp32", "32"):
            out.append("float")
        else:
            b = int(s)
            if b < 2 or b > 16:
                raise ValueError("Bitwidths must be integers in [2..16] or 'float'.")
            out.append(b)
    return out

def main():
    ap = argparse.ArgumentParser(description="PTQ weight-only evaluation for DilatedTCN KWS.")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML (binary or multiclass).")
    ap.add_argument("--weights", type=str, required=True, help="Path to trained state_dict or checkpoint.")
    ap.add_argument("--dataset", type=str, choices=["val", "train", "full"], default="val",
                    help="Which split to evaluate on (wrappers are reconstructed here).")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--bits", type=str, default="float,8,4,3",
                    help="Comma-separated list, e.g. 'float,8,4,3'.")
    ap.add_argument("--scheme", type=str, choices=["per_channel", "per_tensor"], default="per_channel")
    ap.add_argument("--search-threshold", action="store_true",
                    help="Binary only: search best threshold (0.01..0.99) per bit-width.")
    ap.add_argument("--search-steps", type=int, default=101)
    ap.add_argument("--save-csv", type=str, default=None)
    ap.add_argument("--plot", action="store_true", help="Save a simple F1 vs bits plot.")
    ap.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    args = ap.parse_args()

    cfg = load_config(args.config)
    task_type = cfg["task"]["type"]
    device = (torch.device("cuda") if (args.device == "cuda") else
              torch.device("cpu") if (args.device == "cpu") else
              torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Data
    dl, num_classes = make_datasets(cfg, which=args.dataset, batch_size=args.batch_size)

    # Model + weights
    base_model = build_model_from_cfg(cfg, num_classes=num_classes, device=device)
    sd = load_state_dict(args.weights, device="cpu")
    base_model.load_state_dict(sd, strict=True)

    # Prepare bit list
    bits_list = parse_bits(args.bits)

    # Evaluate
    results = []
    for bits in bits_list:
        if bits == "float":
            model_q = deepcopy(base_model).to(device)
            tag = "float"
        else:
            model_q = quantize_model_weights(base_model, bits=int(bits), scheme=args.scheme).to(device)
            tag = f"{bits}-bit({args.scheme})"

        metrics = evaluate_model(
            model_q, dl, cfg, device, task_type,
            threshold=cfg["train"].get("threshold", 0.5) if task_type == "binary" else None,
            search_threshold=(args.search_threshold and task_type == "binary"),
            search_steps=args.search_steps
        )
        print(f"[{tag:>16}]  Acc={metrics['acc']:.4f}  P={metrics['prec']:.4f}  R={metrics['rec']:.4f}  F1={metrics['f1']:.4f}"
              + (f"  thr={metrics['threshold_used']:.3f}" if 'threshold_used' in metrics else ""))

        results.append({
            "bits": tag,
            "acc": metrics["acc"],
            "prec": metrics["prec"],
            "rec": metrics["rec"],
            "f1": metrics["f1"],
            **({"threshold_used": metrics["threshold_used"]} if "threshold_used" in metrics else {})
        })

    # Save CSV
    if args.save_csv:
        import csv
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        fieldnames = list(results[0].keys()) if results else ["bits","acc","prec","rec","f1"]
        with open(args.save_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in results:
                w.writerow(row)
        print(f"[OK] Saved CSV to {args.save_csv}")

    # Optional plot (F1 vs. bits)
    if args.plot and results:
        # order by effective precision: float -> 32; otherwise int(bit)
        def eff_bits(tag):
            return 32 if tag.startswith("float") else int(tag.split("-bit")[0])
        xs = [eff_bits(r["bits"]) for r in results]
        ys = [r["f1"] for r in results]
        order = np.argsort(xs)
        xs_sorted = [xs[i] for i in order]
        ys_sorted = [ys[i] for i in order]

        plt.figure(figsize=(7,5))
        plt.plot(xs_sorted, ys_sorted, marker="o")
        plt.title("F1 vs. weight quantization bits")
        plt.xlabel("bits (float shown as 32)")
        plt.ylabel("F1")
        plt.ylim(0, 1)  # F1 is in [0,1]
        plt.grid(True, axis="both", alpha=0.3)
        outdir = "analysis"
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, "ptq_f1_vs_bits.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[OK] Saved plot to {out_path}")


if __name__ == "__main__":
    main()
