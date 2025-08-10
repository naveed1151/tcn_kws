import os
import argparse
from copy import deepcopy
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from train.utils import (
    build_model_from_cfg, load_config, _binary_counts, _derive_metrics,
    _multiclass_confusion_add, _multiclass_macro_prf1
)
from data_loader.utils import make_datasets, get_num_classes


class ActivationQuantizer(torch.nn.Module):
    def __init__(self, bits=8, symmetric=True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric

    def forward(self, x):
        if self.bits == 32 or self.bits == "float":
            return x
        qmin = -(2 ** (self.bits - 1)) if self.symmetric else 0
        qmax = (2 ** (self.bits - 1)) - 1 if self.symmetric else (2 ** self.bits) - 1
        max_val = x.abs().max() if self.symmetric else x.max()
        scale = max_val / qmax if max_val > 0 else torch.tensor(1.0, device=x.device)
        zero_point = torch.tensor(0, device=x.device) if self.symmetric else torch.round(-x.min() / scale)
        x_int = torch.round(x / scale + zero_point).clamp(qmin, qmax)
        x_dequant = (x_int - zero_point) * scale
        return x_dequant


def build_and_load_model(cfg, dl, weights_path, num_classes):
    # Grab ONE sample; avoid using batch dimension as channels
    sample_batch, _ = next(iter(dl))
    if sample_batch.dim() == 3:          # (B,C,T)
        sample_x = sample_batch[0]       # (C,T)
    else:
        sample_x = sample_batch          # (C,T)

    # Defensive: enforce channels = first dim
    in_ch = sample_x.shape[0]
    model = build_model_from_cfg(cfg, sample_x, num_classes)
    print(f"[DEBUG] Built model first conv in_channels={model.tcn_layers[0].in_channels} (expected {in_ch})")

    weights = torch.load(weights_path, map_location="cpu")
    if "state_dict" in weights:
        print("[INFO] Checkpoint keys:", list(weights.keys()))
        sd = weights["state_dict"]
    else:
        sd = weights

    # Strict load check
    try:
        model.load_state_dict(sd, strict=True)
        print("[INFO] Strict load OK.")
    except RuntimeError as e:
        print("[WARN] Strict load failed:\n", e)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print("[WARN] Missing keys:", missing)
        print("[WARN] Unexpected keys:", unexpected)
        if missing or unexpected:
            raise RuntimeError("Model architecture mismatch – aborting quant eval. Fix build_model_from_cfg.")

    return model


def is_float_spec(b):
    return str(b).lower() in ("float", "fp", "fp32", "32")


def treat_as_float(w_bits, a_bits):
    # If either side declared float OR both >=16 just bypass (16-bit should be near-lossless)
    return (is_float_spec(w_bits) and is_float_spec(a_bits)) or (
        (not is_float_spec(w_bits) and int(w_bits) >= 16) and
        (not is_float_spec(a_bits) and int(a_bits) >= 16)
    )


def quantize_model_weights_and_activations(model, weight_bits=8, act_bits=8, per_channel=True, symmetric=True):
    # Bypass path
    if treat_as_float(weight_bits, act_bits):
        return deepcopy(model)

    model = deepcopy(model).cpu().eval()
    # -------- Weights --------
    if not is_float_spec(weight_bits):
        w_bits = int(weight_bits)
        qmax_w = (1 << (w_bits - 1)) - 1
        for _, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                if per_channel and w.dim() >= 2:
                    oc = w.shape[0]
                    w_flat = w.view(oc, -1)
                    max_abs = w_flat.abs().max(dim=1).values
                    scale = max_abs / qmax_w
                    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                    q = torch.clamp(torch.round(w_flat / scale[:, None]), -qmax_w, qmax_w)
                    module.weight.data = (q * scale[:, None]).view_as(w)
                else:
                    max_abs = w.abs().max()
                    if max_abs == 0:
                        continue
                    scale = max_abs / qmax_w
                    q = torch.clamp(torch.round(w / scale), -qmax_w, qmax_w)
                    module.weight.data = q * scale

    # -------- Activations (only input for now) --------
    if is_float_spec(act_bits) or int(act_bits) >= 16:
        return model

    act_bits_i = int(act_bits)
    qmin = -(2 ** (act_bits_i - 1))
    qmax = (2 ** (act_bits_i - 1)) - 1

    orig_forward = model.forward

    @torch.no_grad()
    def new_forward(x):
        max_abs = x.abs().max()
        if max_abs == 0:
            return orig_forward(x)
        scale = max_abs / qmax
        x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
        x_dq = x_q * scale
        return orig_forward(x_dq)

    model.forward = new_forward
    return model


@torch.no_grad()
def evaluate(model, dataloader, device, task_type, threshold=0.5, num_classes=None):
    model.eval().to(device)
    total_loss, correct, total, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
    cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None
    all_preds, all_targets = [], []
    criterion = torch.nn.BCEWithLogitsLoss() if task_type == "binary" else torch.nn.CrossEntropyLoss()
    for x, y in tqdm(dataloader, desc="Evaluating", unit="batch"):
        x = x.to(device)
        targets = y.float().unsqueeze(1).to(device) if task_type == "binary" else y.long().to(device)
        logits = model(x)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        if task_type == "binary":
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            correct += (preds == targets.long()).sum().item()
            total += targets.numel()
            _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
            tp += _tp; fp += _fp; fn += _fn
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
        else:
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
            cm = _multiclass_confusion_add(cm, preds, targets, num_classes)
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
    if task_type == "binary":
        avg_loss, acc, prec, rec, f1 = _derive_metrics(total_loss, len(dataloader), correct, total, tp, fp, fn)
    else:
        avg_loss, acc = _derive_metrics(total_loss, len(dataloader), correct, total)
        prec, rec, f1 = _multiclass_macro_prf1(cm)
    return acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", choices=["val", "train", "test", "all", "full"], default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--bits", type=str, default="float,8,4,2")
    parser.add_argument("--act-bits", type=str, default=None)
    parser.add_argument("--scheme", choices=["per_channel", "per_tensor"], default="per_channel")
    parser.add_argument("--symmetric", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    dl = make_datasets(cfg, which=args.dataset, batch_size=args.batch_size)
    num_classes = get_num_classes(cfg)
    task_type = cfg["task"]["type"]

    model = build_and_load_model(cfg, dl, args.weights, num_classes).to(device).eval()

    per_channel = args.scheme == "per_channel"
    symmetric = args.symmetric

    def parse_list(s):
        return [t.strip() for t in s.split(",") if t.strip()]

    bits_list = parse_list(args.bits)
    act_bits_list = bits_list if args.act_bits is None else parse_list(args.act_bits)

    results = []
    for i, wb in enumerate(bits_list):
        ab = act_bits_list[i] if i < len(act_bits_list) else act_bits_list[-1]

        if treat_as_float(wb, ab):
            tag = "float"
            model_q = deepcopy(model)
        else:
            w_bits_i = 32 if is_float_spec(wb) else int(wb)
            a_bits_i = 32 if is_float_spec(ab) else int(ab)
            tag = "float" if (w_bits_i >= 16 and a_bits_i >= 16) else f"{w_bits_i}w{a_bits_i}a"
            model_q = quantize_model_weights_and_activations(
                model, w_bits_i, a_bits_i, per_channel=per_channel, symmetric=symmetric
            )

        acc, prec, rec, f1 = evaluate(model_q.to(device), dl, device, task_type, num_classes=num_classes)
        results.append((tag, acc, prec, rec, f1))
        print(f"[{tag:>8}] Acc={acc:.4f} P={prec:.4f} R={rec:.4f} F1={f1:.4f}")

    # --- plotting unchanged (use results list) ---
    plots_dir = os.path.join(cfg["output"]["plots_dir"], "eval")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "ptq_metrics_vs_bits.png")

    labels = []
    accs = []
    precs = []
    recs = []
    f1s = []
    for tag, acc, prec, rec, f1 in results:
        labels.append(tag)
        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)

    plt.figure(figsize=(9, 5))
    for vals, lbl, mk in zip([accs, precs, recs, f1s],
                              ["Accuracy", "Precision", "Recall", "F1"],
                              ["o", "s", "^", "d"]):
        plt.plot(range(len(labels)), vals, marker=mk, label=lbl)

    plt.xticks(range(len(labels)), labels, rotation=30)
    plt.ylabel("Metric")
    plt.xlabel("Quant setting")
    plt.title("PTQ Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved plot: {out_path}")


if __name__ == "__main__":
    main()
