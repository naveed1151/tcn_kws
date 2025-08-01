# analysis/ptq_act_eval.py
# Post-Training Quantization (Activations) evaluation for TCN KWS.
# Also supports optional weight PTQ. Evaluates on val/test/train/all and writes CSV + plot.

import os
import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import yaml

# --- repo imports ---
from model.model import DilatedTCN
from data_loader.mfcc_dataset import MFCCDataset


# =========================
# YAML loading (robust)
# =========================
def _load_yaml_with_encodings(path):
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("yaml", b"", 0, 1, f"Could not decode {path} with utf-8/utf-8-sig/cp1252")


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path):
    cfg_task = _load_yaml_with_encodings(path)
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if os.path.basename(path) != "base.yaml" and os.path.exists(base_path):
        cfg_base = _load_yaml_with_encodings(base_path)
        return deep_update(deepcopy(cfg_base), cfg_task)
    return cfg_task


# =========================
# Dataset wrappers
# =========================
class BinaryKeywordDataset(torch.utils.data.Dataset):
    """
    Lazy binary wrapper around MFCCDataset.
    Builds positive/negative index lists from labels metadata (no MFCC loads in __init__).
    """
    def __init__(self, base_dataset, target_word, index_to_word,
                 downsample_ratio=None, seed=0):
        rng = np.random.RandomState(seed)
        self.base = base_dataset
        self.index_to_word = index_to_word
        self.target_word = target_word

        labels_int = getattr(base_dataset, "labels", None)
        if labels_int is None:
            labels_int = [base_dataset[i][1] for i in range(len(base_dataset))]

        pos_idx, neg_idx = [], []
        for i, li in enumerate(labels_int):
            if self.index_to_word[li] == self.target_word:
                pos_idx.append(i)
            else:
                neg_idx.append(i)

        if downsample_ratio is not None:
            keep_neg = int(len(pos_idx) * float(downsample_ratio))
            if keep_neg < len(neg_idx):
                neg_idx = rng.choice(neg_idx, size=keep_neg, replace=False).tolist()

        self.indices = pos_idx + neg_idx
        rng.shuffle(self.indices)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        x, label_idx = self.base[base_idx]  # x is (C, T)
        y = 1 if self.index_to_word[label_idx] == self.target_word else 0
        return x, y


# =========================
# Splits & loaders
# =========================
def stratified_train_val_test_indices(labels, val_frac, test_frac, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        val_idx.extend(idx_c[:n_val].tolist())
        test_idx.extend(idx_c[n_val:n_val+n_test].tolist())
        train_idx.extend(idx_c[n_val+n_test:].tolist())
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def make_datasets(cfg, which="val", batch_size=64, seed=0):
    pre_dir = cfg["data"].get("preprocessed_dir") or cfg["data"].get("data_dir")
    if pre_dir is None:
        raise KeyError("Config must define data.preprocessed_dir (or legacy data_dir).")
    base = MFCCDataset(pre_dir)

    label_to_index = base.label_to_index
    index_to_word = {v: k for k, v in label_to_index.items()}

    task_type = cfg["task"]["type"]
    if task_type == "binary":
        target_word = cfg["data"]["target_word"]
        dataset_all = BinaryKeywordDataset(base, target_word, index_to_word,
                                           downsample_ratio=cfg["data"].get("neg_downsample_ratio", None),
                                           seed=seed)
        num_classes = 1
        labels_for_split = [y for _, y in dataset_all]
    else:
        # Multiclass: use base dataset directly
        dataset_all = base
        num_classes = len(label_to_index)
        labels_for_split = getattr(base, "labels", None)
        if labels_for_split is None:
            labels_for_split = [base[i][1] for i in range(len(base))]

    val_frac = float(cfg["data"].get("val_split", 0.20))
    test_frac = float(cfg["data"].get("test_split", 0.10))
    train_idx, val_idx, test_idx = stratified_train_val_test_indices(labels_for_split, val_frac, test_frac, seed=seed)

    subset_map = {
        "train": Subset(dataset_all, train_idx),
        "val":   Subset(dataset_all, val_idx),
        "test":  Subset(dataset_all, test_idx),
        "all":   dataset_all,
    }
    ds = subset_map[which]

    nw = int(cfg["train"].get("num_workers", 0))
    pf = int(cfg["train"].get("prefetch_factor", 2))
    pm = bool(cfg["train"].get("pin_memory", True))
    pw = bool(cfg["train"].get("persistent_workers", True)) if nw > 0 else False

    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=nw, prefetch_factor=(pf if nw > 0 else None),
                    persistent_workers=pw, pin_memory=pm)
    return dl, num_classes


# =========================
# Build & load model
# =========================
def calc_layers(seq_len, kernel_size=3, dilation_base=2):
    L = 0
    receptive_field = 1
    while receptive_field < seq_len:
        receptive_field += (kernel_size - 1) * (dilation_base ** L)
        L += 1
    return L


def build_model_from_cfg(cfg, sample_x, num_classes):
    input_channels = sample_x.shape[0]
    seq_len = sample_x.shape[1]
    k = int(cfg["model"]["kernel_size"])
    num_layers_dyn = calc_layers(seq_len, kernel_size=k, dilation_base=2)
    num_layers = int(cfg["model"].get("num_layers", num_layers_dyn))

    model = DilatedTCN(
        input_channels=input_channels,
        num_layers=num_layers,
        hidden_channels=int(cfg["model"]["hidden_channels"]),
        kernel_size=k,
        num_classes=(1 if cfg["task"]["type"] == "binary" else num_classes),
        dropout=float(cfg["model"]["dropout"]),
    )
    return model


def load_state_dict_forgiving(model, weights_path, device):
    sd = torch.load(weights_path, map_location=device)
    if isinstance(sd, dict):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        elif "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    return model


# =========================
# Weight PTQ (optional)
# =========================
@torch.no_grad()
def quantize_weights_symmetric_(w: torch.Tensor, bits=8, per_channel=True, ch_axis=0):
    qmax = (1 << (bits - 1)) - 1
    if per_channel and w.ndim in (2, 3):
        # move ch_axis to dim0
        perm = list(range(w.ndim))
        perm[0], perm[ch_axis] = perm[ch_axis], perm[0]
        w_perm = w.permute(*perm).contiguous()
        C = w_perm.shape[0]
        rest = w_perm.view(C, -1)
        max_abs = rest.abs().max(dim=1).values
        scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
        q = torch.round(rest / scale.unsqueeze(1)).clamp_(-qmax, qmax)
        deq = (q * scale.unsqueeze(1)).view_as(w_perm)
        w.copy_(deq.permute(*perm))
    else:
        max_abs = w.abs().max()
        if max_abs == 0:
            w.zero_()
        else:
            scale = max_abs / qmax
            q = torch.round(w / scale).clamp_(-qmax, qmax)
            w.copy_(q * scale)


def apply_weight_ptq_inplace(model, bits=8, per_channel=True):
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and hasattr(m, "weight"):
            quantize_weights_symmetric_(m.weight.data, bits=bits, per_channel=per_channel)


# =========================
# Activation observers & hook manager
# =========================
class ActObserver:
    """Per-tensor activation observer keeping min/max and nonneg flag."""
    def __init__(self, mode="minmax", percentile=0.999, sample_cap=10000, unsigned_if_nonneg=True):
        assert mode in ("minmax", "percentile")
        self.mode = mode
        self.percentile = percentile
        self.sample_cap = sample_cap
        self.unsigned_if_nonneg = unsigned_if_nonneg
        self.min_val = None
        self.max_val = None
        self.has_nonneg_only = True

    def update(self, x: torch.Tensor):
        xf = x.detach().flatten()
        if xf.numel() == 0:
            return
        if xf.numel() > self.sample_cap:
            idx = torch.randint(0, xf.numel(), (self.sample_cap,), device=xf.device)
            xf = xf[idx]
        xnp = xf.to(torch.float32).cpu().numpy()

        if self.mode == "minmax":
            mn = float(np.min(xnp))
            mx = float(np.max(xnp))
        else:
            p = self.percentile
            mn = float(np.percentile(xnp, (1 - p) * 100.0))
            mx = float(np.percentile(xnp, p * 100.0))

        self.min_val = mn if self.min_val is None else min(self.min_val, mn)
        self.max_val = mx if self.max_val is None else max(self.max_val, mx)
        if self.has_nonneg_only and np.any(xnp < 0):
            self.has_nonneg_only = False

    def get_stats(self):
        return self.min_val, self.max_val, self.has_nonneg_only, self.unsigned_if_nonneg


class ActQuantHookMgr:
    """
    Manages activation observers and fake-quant via forward hooks.
    where ∈ {"relu","conv","prelinear"}.
    Defaults to conv+prelinear, which works with F.relu-based models.
    """
    def __init__(self, model: nn.Module, where=("conv","prelinear"),
                 observer_mode="minmax", percentile=0.999, unsigned_if_nonneg=True):
        self.model = model
        self.where = set(where)
        self.observer_mode = observer_mode
        self.percentile = percentile
        self.unsigned_if_nonneg = unsigned_if_nonneg

        self.observers = {}   # key -> ActObserver
        self.hooks = []
        self.mode = "collect"   # "collect" or "quantize"
        self.bits = 8
        self.keys_registered = []

    def clear(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.observers.clear()
        self.keys_registered = []

    def set_mode(self, mode):  # "collect" or "quantize"
        assert mode in ("collect","quantize")
        self.mode = mode

    def set_bits(self, bits: int):
        self.bits = int(bits)

    def _get_or_create_obs(self, key):
        if key not in self.observers:
            obs = ActObserver(mode=self.observer_mode, percentile=self.percentile,
                              unsigned_if_nonneg=self.unsigned_if_nonneg)
            self.observers[key] = obs
        return self.observers[key]

    def _route_output(self, key, tensor):
        if self.mode == "collect":
            self._get_or_create_obs(key).update(tensor)
            return tensor
        else:
            return self._qdq_tensor(key, tensor)

    def _route_input(self, key, tensor):
        if self.mode == "collect":
            self._get_or_create_obs(key).update(tensor)
            return tensor
        else:
            return self._qdq_tensor(key, tensor)

    def register_points(self):
        # Conv outputs (pre-activation, since many models use F.relu)
        if "conv" in self.where:
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv1d):
                    key = f"{name}:out"
                    h = m.register_forward_hook(lambda mod, inp, out, k=key: self._route_output(k, out))
                    self.hooks.append(h)
                    self.keys_registered.append(key)

        # ReLU outputs (only if model uses nn.ReLU modules)
        if "relu" in self.where:
            for name, m in self.model.named_modules():
                if isinstance(m, nn.ReLU):
                    key = f"{name}:out"
                    h = m.register_forward_hook(lambda mod, inp, out, k=key: self._route_output(k, out))
                    self.hooks.append(h)
                    self.keys_registered.append(key)

        # Inputs to Linear (prelinear)
        if "prelinear" in self.where:
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Linear):
                    key = f"{name}:in"
                    h = m.register_forward_pre_hook(lambda mod, inp, k=key: (self._route_input(k, inp[0]),))
                    self.hooks.append(h)
                    self.keys_registered.append(key)

        print("[PTQ-A] Hooked points:")
        for k in self.keys_registered:
            print("  -", k)

    def _qdq_tensor(self, key, x: torch.Tensor):
        if key not in self.observers:
            return x
        mn, mx, nonneg_only, unsigned_if_nonneg = self.observers[key].get_stats()
        if mn is None or mx is None:
            return x

        if unsigned_if_nonneg and mn >= 0.0 and nonneg_only:
            # unsigned [0, 2^bits-1]
            qmin, qmax = 0, (1 << self.bits) - 1
            scale = (mx if mx > 0 else 1.0) / max(qmax, 1)
            q = torch.round(x / scale).clamp_(qmin, qmax)
            deq = q * scale
        else:
            # signed symmetric [-qmax, qmax]
            qmax = (1 << (self.bits - 1)) - 1
            max_abs = max(abs(mn), abs(mx), 1e-8)
            scale = max_abs / qmax
            q = torch.round(x / scale).clamp_(-qmax, qmax)
            deq = q * scale
        return deq


# =========================
# Metrics
# =========================
def binary_metrics_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).long()
    correct = (preds == targets.long()).sum().item()
    total = targets.numel()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    acc = correct / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return acc, prec, rec, f1


@torch.no_grad()
def evaluate_model(model, dl, cfg, device, task_type, threshold=0.5, num_classes=0):
    model.eval()
    total_loss = 0.0
    criterion_bin = nn.BCEWithLogitsLoss()
    criterion_mc  = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    if task_type == "binary":
        tp = fp = fn = 0

    for x, y in dl:
        x = x.to(device)
        if task_type == "binary":
            t = y.float().unsqueeze(1).to(device)
            logits = model(x)
            total_loss += criterion_bin(logits, t).item()
            acc, prec, rec, f1 = binary_metrics_from_logits(logits, t, threshold=threshold)
            correct += int(acc * t.numel())
            total += t.numel()
            # get tp/fp/fn again for precise macro metrics
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            tp += ((preds == 1) & (t == 1)).sum().item()
            fp += ((preds == 1) & (t == 0)).sum().item()
            fn += ((preds == 0) & (t == 1)).sum().item()
        else:
            t = y.long().to(device)
            logits = model(x)
            total_loss += criterion_mc(logits, t).item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == t).sum().item()
            total += t.numel()

    if task_type == "binary":
        acc = correct / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return {"loss": total_loss / max(1, len(dl)), "acc": acc, "prec": prec, "rec": rec, "f1": f1}
    else:
        # recompute macro metrics in a small second pass
        preds_all, t_all = [], []
        for x, y in dl:
            x = x.to(device)
            t = y.long().to(device)
            logits = model(x)
            preds_all.append(torch.argmax(logits, dim=1).cpu())
            t_all.append(t.cpu())
        preds = torch.cat(preds_all, dim=0)
        targets = torch.cat(t_all, dim=0)
        K = num_classes
        cm = torch.zeros((K, K), dtype=torch.int64)
        for p, t in zip(preds.tolist(), targets.tolist()):
            if 0 <= t < K and 0 <= p < K:
                cm[t, p] += 1
        precisions, recalls, f1s = [], [], []
        for k in range(K):
            tp_k = cm[k, k].item()
            fp_k = cm[:, k].sum().item() - tp_k
            fn_k = cm[k, :].sum().item() - tp_k
            prec_k = tp_k / (tp_k + fp_k) if (tp_k + fp_k) > 0 else 0.0
            rec_k  = tp_k / (tp_k + fn_k) if (tp_k + fn_k) > 0 else 0.0
            f1_k   = (2 * prec_k * rec_k / (prec_k + rec_k)) if (prec_k + rec_k) > 0 else 0.0
            precisions.append(prec_k); recalls.append(rec_k); f1s.append(f1_k)
        acc = (preds == targets).sum().item() / targets.numel() if targets.numel() > 0 else 0.0
        macro_p = sum(precisions)/K if K>0 else 0.0
        macro_r = sum(recalls)/K if K>0 else 0.0
        macro_f1 = sum(f1s)/K if K>0 else 0.0
        return {"loss": total_loss / max(1, len(dl)), "acc": acc, "prec": macro_p, "rec": macro_r, "f1": macro_f1}


# =========================
# Main
# =========================
def parse_bits_list(s):
    bits = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            bits.append(int(tok))
    return bits


def main():
    ap = argparse.ArgumentParser(description="Evaluate PTQ of ACTIVATIONS (and optional weights) for TCN KWS.")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML (binary.yaml/multiclass.yaml; merges base.yaml)")
    ap.add_argument("--weights", type=str, required=True, help="Path to model weights .pt")
    ap.add_argument("--dataset", type=str, choices=["val", "test", "train", "all"], default="val")
    ap.add_argument("--batch-size", type=int, default=128)

    # Activation PTQ settings
    ap.add_argument("--act-bits", type=str, default="8,6,4,3", help="Comma-separated activation bit-widths to evaluate")
    ap.add_argument("--act-where", type=str, default="conv,prelinear", help="Comma list from {relu,conv,prelinear}")
    ap.add_argument("--act-calib", type=str, choices=["minmax", "percentile"], default="minmax")
    ap.add_argument("--act-percentile", type=float, default=0.999, help="Used when --act-calib=percentile")
    ap.add_argument("--act-unsigned-if-nonneg", action="store_true", help="Use unsigned range if observed activations are non-negative")
    ap.add_argument("--calib-batches", type=int, default=20, help="Number of batches for calibration pass")

    # Weight PTQ (optional)
    ap.add_argument("--w-bits", type=int, default=None, help="Quantize weights too (int). If omitted, keep float weights")
    ap.add_argument("--w-per-channel", action="store_true", help="Per-channel weight PTQ (default: per-tensor if not set)")

    # Output
    ap.add_argument("--outdir", type=str, default="analysis/ptq_results", help="Where to write CSV and plot")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Device
    dev_cfg = cfg["train"].get("device", "auto")
    if dev_cfg == "cuda":
        device = torch.device("cuda")
    elif dev_cfg == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    dl, num_classes = make_datasets(cfg, which=args.dataset, batch_size=args.batch_size,
                                    seed=int(cfg["train"].get("seed", 0)))
    # Peek one batch to size the model
    sample_x, _ = next(iter(dl))
    sample_x = sample_x[0]  # (C,T)

    # Build model & load weights
    model = build_model_from_cfg(cfg, sample_x, num_classes)
    model.to(device)
    model = load_state_dict_forgiving(model, args.weights, device)

    # Optional weight PTQ
    if args.w_bits is not None:
        print(f"[Weights] Applying PTQ: bits={args.w_bits}, per_channel={args.w_per_channel}")
        apply_weight_ptq_inplace(model, bits=int(args.w_bits), per_channel=bool(args.w_per_channel))

    # === Calibration ===
    where = tuple(w.strip() for w in args.act_where.split(",") if w.strip())
    qmgr = ActQuantHookMgr(
        model,
        where=where,
        observer_mode=args.act_calib,
        percentile=args.act_percentile,
        unsigned_if_nonneg=bool(args.act_unsigned_if_nonneg),
    )
    qmgr.register_points()
    if not qmgr.keys_registered:
        print("[WARN] No activation hooks registered. Check --act-where and model structure.")

    qmgr.set_mode("collect")
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(dl):
            x = x.to(device)
            _ = model(x)
            if i + 1 >= args.calib_batches:
                break

    # Switch to quantize mode; scales come from stored min/max, recomputed per-bit
    qmgr.set_mode("quantize")

    # Evaluate across activation bit-widths
    results = []
    act_bits_list = parse_bits_list(args.act_bits)
    threshold = float(cfg["train"].get("threshold", 0.5) or 0.5)

    for b in act_bits_list:
        qmgr.set_bits(b)
        print(f"[Eval] Activations {b}-bit (where={where})"
              f"{', Weights '+str(args.w_bits)+'-bit' if args.w_bits is not None else ''}")
        metrics = evaluate_model(model, dl, cfg, device,
                                 task_type=cfg["task"]["type"],
                                 threshold=threshold,
                                 num_classes=num_classes)
        metrics["act_bits"] = b
        metrics["w_bits"] = (args.w_bits if args.w_bits is not None else "float")
        results.append(metrics)

    # Write CSV
    os.makedirs(args.outdir, exist_ok=True)
    import csv
    csv_path = os.path.join(args.outdir, f"ptq_act_{args.dataset}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["act_bits", "w_bits", "loss", "acc", "prec", "rec", "f1"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[OK] Wrote results to {csv_path}")

    # Plot (accuracy & F1 vs bits)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r["act_bits"] for r in results]
    accs = [r["acc"] for r in results]
    f1s  = [r["f1"] for r in results]
    ax.plot(xs, accs, marker="o", label="Accuracy")
    ax.plot(xs, f1s,  marker="o", label="F1")
    ax.set_xlabel("Activation bits")
    ax.set_ylabel("Score")
    ax.set_title(f"PTQ-A on {args.dataset}  ({'binary' if cfg['task']['type']=='binary' else 'multiclass'})")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plot_path = os.path.join(args.outdir, f"ptq_act_{args.dataset}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
