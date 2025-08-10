import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from time import time
import numpy as np

from train.utils import (
    build_model_from_cfg, load_config, _binary_counts, _derive_metrics,
    _multiclass_confusion_add, _multiclass_macro_prf1, export_quantized_weights_npz,
    load_state_dict_forgiving,
)
from data_loader.utils import make_datasets, get_num_classes
from analysis.metrics import plot_metrics


class CustomFakeQuantize(nn.Module):
    def __init__(self, bits=8, symmetric=True, ema=True, momentum=0.95, eps=1e-8, per_channel=False, ch_axis=0):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.ema = ema
        self.momentum = momentum
        self.eps = eps
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.enabled = True
        self.frozen = False  # when True, stop updating observer
        # Buffers
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("running_min", torch.tensor(0.0))  # for asymmetric if needed

    def _qrange(self):
        if self.symmetric:
            qmax = (1 << (self.bits - 1)) - 1
            qmin = -qmax
        else:
            qmin = 0
            qmax = (1 << self.bits) - 1
        return qmin, qmax

    def _observe(self, x):
        if self.frozen:
            return
        if self.per_channel:
            # reduce over all dims except ch_axis
            reduce_dims = [d for d in range(x.dim()) if d != self.ch_axis]
            max_abs = x.detach().abs().amax(dim=reduce_dims)
            if self.ema:
                self.running_max = torch.maximum(
                    self.running_max * self.momentum + max_abs * (1 - self.momentum),
                    torch.full_like(max_abs, self.eps),
                )
            else:
                self.running_max = torch.clamp(max_abs, min=self.eps)
        else:
            max_abs = x.detach().abs().max()
            if self.ema:
                self.running_max = torch.maximum(
                    self.running_max * self.momentum + max_abs * (1 - self.momentum),
                    torch.tensor(self.eps, device=x.device),
                )
            else:
                self.running_max = torch.clamp(max_abs, min=self.eps)

    def forward(self, x):
        if (not self.enabled) or self.bits >= 32:
            return x
        self._observe(x)
        qmin, qmax = self._qrange()
        # Compute scale from observer
        max_val = self.running_max
        scale = max_val / qmax
        scale = torch.where(scale == 0, torch.full_like(scale, self.eps), scale)
        self.scale = scale.detach()
        # Quant-dequant with STE
        if self.per_channel:
            # reshape scale for broadcast on ch_axis
            shape = [1] * x.dim()
            shape[self.ch_axis] = -1
            s = self.scale.view(shape)
        else:
            s = self.scale
        q = torch.round(x / s).clamp(qmin, qmax)
        x_hat = q * s
        return x + (x_hat - x).detach()


def apply_custom_fake_quant(model, weight_bits, act_bits, symmetric=True):
    # Per-channel weights, per-tensor activations
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            m.weight_fake = CustomFakeQuantize(bits=weight_bits, symmetric=symmetric, per_channel=True, ch_axis=0)
            orig = m.forward
            def conv_fwd(self, inp):
                w_q = self.weight_fake(self.weight)
                return F.conv1d(inp, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
            m.forward = conv_fwd.__get__(m, m.__class__)
        elif isinstance(m, nn.Linear):
            m.weight_fake = CustomFakeQuantize(bits=weight_bits, symmetric=symmetric, per_channel=True, ch_axis=0)
            def lin_fwd(self, inp):
                w_q = self.weight_fake(self.weight)
                return F.linear(inp, w_q, self.bias)
            m.forward = lin_fwd.__get__(m, m.__class__)
    # Per-layer input activation fake quant
    if act_bits < 32:
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                aq = CustomFakeQuantize(bits=act_bits, symmetric=symmetric, per_channel=False)
                orig = m.forward
                def wrapped(self, x, _orig=orig, _aq=aq):
                    x = _aq(x)
                    return _orig(x)
                m.forward = wrapped.__get__(m, m.__class__)
    return model


def set_qat_mode(model, enabled=True, freeze=False):
    for m in model.modules():
        if hasattr(m, "enabled"):
            m.enabled = enabled
        if hasattr(m, "frozen"):
            m.frozen = freeze


def evaluate(model, loader, criterion, device, task_type, num_classes, threshold=0.5):
    model.eval()
    total_loss, correct, total, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
    cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None
    with torch.no_grad():
        for x, y in loader:
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
    return avg_loss, acc, prec, rec, f1


def train_qat(model, train_loader, val_loader, optimizer, criterion, device, epochs, task_type, num_classes, threshold=0.5, warmup_epochs=1):
    # init histories
    hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    val_hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}

    best = None
    for epoch in range(1, epochs+1):
        model.train()
        # Freeze observers after warmup
        if epoch == warmup_epochs + 1:
            set_qat_mode(model, enabled=True, freeze=True)
        total_loss, correct, total, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
        cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None

        loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
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
            avg_loss, acc, prec, rec, f1 = _derive_metrics(total_loss, len(train_loader), correct, total, tp, fp, fn)
        else:
            avg_loss, acc = _derive_metrics(total_loss, len(train_loader), correct, total)
            prec, rec, f1 = _multiclass_macro_prf1(cm)

        # append train history
        hist["loss"].append(avg_loss)
        hist["acc"].append(acc)
        hist["prec"].append(prec)
        hist["rec"].append(rec)
        hist["f1"].append(f1)

        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device, task_type, num_classes, threshold)
        val_hist["loss"].append(val_loss)
        val_hist["acc"].append(val_acc)
        val_hist["prec"].append(val_prec)
        val_hist["rec"].append(val_rec)
        val_hist["f1"].append(val_f1)

        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}  Acc: {acc:.4f}  P: {prec:.4f}  R: {rec:.4f}  F1: {f1:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  P: {val_prec:.4f}  R: {val_rec:.4f}  F1: {val_f1:.4f}")

    return hist, val_hist


@torch.no_grad()
def export_quant_npz_from_model(model, out_npz_path, bits=5, per_channel=True, symmetric=True):
    qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
    qmin = -qmax if symmetric else 0
    names, q_list, scale_list = [], [], []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and hasattr(m, "weight"):
            w = m.weight.detach().float().cpu()
            if per_channel and w.dim() >= 2:
                oc = w.shape[0]
                w2 = w.view(oc, -1)
                max_abs = w2.abs().max(dim=1).values if symmetric else w2.max(dim=1).values
                scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
                q = torch.round(w2 / scale[:, None]).clamp(qmin, qmax).to(torch.int16)
                names.append(f"{name}.weight" if name else "weight")
                q_list.append(q.numpy())
                scale_list.append(scale.numpy())
            else:
                max_abs = w.abs().max() if symmetric else w.max()
                if max_abs == 0:
                    continue
                scale = (max_abs / qmax).item()
                q = torch.round(w / scale).clamp(qmin, qmax).to(torch.int16)
                names.append(f"{name}.weight" if name else "weight")
                q_list.append(q.numpy())
                scale_list.append(np.array(scale, dtype=np.float32))
    os.makedirs(os.path.dirname(out_npz_path), exist_ok=True)
    np.savez(
        out_npz_path,
        names=np.array(names, dtype=object),
        q_list=np.array(q_list, dtype=object),
        scale_list=np.array(scale_list, dtype=object),
        bits=np.array(bits),
        symmetric=np.array(symmetric),
        per_channel=np.array(per_channel),
    )
    print(f"[OK] Exported {len(names)} tensors to {out_npz_path}")


def main():
    parser = argparse.ArgumentParser(description="Custom QAT for arbitrary bit-widths")
    parser.add_argument("--config", required=True, help="Path to config (e.g., config/multiclass.yaml)")
    parser.add_argument("--weights", default=None, help="Optional pretrained weights to initialize from")
    parser.add_argument("--epochs", type=int, default=None, help="QAT epochs (overrides config/base.yaml)")
    parser.add_argument("--save-preconvert", action="store_true", help="Save model weights before convert()")
    args = parser.parse_args()

    t0 = time()
    cfg = load_config(args.config)
    qat_cfg = cfg.get("qat", {})
    wb = int(qat_cfg.get("weight_bits", 8))
    ab = int(qat_cfg.get("act_bits", 8))
    symmetric = bool(qat_cfg.get("symmetric", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_type = cfg["task"]["type"]

    train_batch_size = cfg["train"].get("batch_size", 64)
    val_batch_size = train_batch_size
    train_epochs = args.epochs if args.epochs is not None else cfg["train"].get("num_epochs", 5)
    lr = cfg["train"].get("learning_rate", 1e-3)

    dl_train = make_datasets(cfg, which="train", batch_size=train_batch_size)
    dl_val = make_datasets(cfg, which="val", batch_size=val_batch_size)
    num_classes = get_num_classes(cfg)
    sample_x, _ = next(iter(dl_train))
    model = build_model_from_cfg(cfg, sample_x[0], num_classes)

    if args.weights:
        model = load_state_dict_forgiving(model, args.weights, device)

    model.to(device)
    model = apply_custom_fake_quant(model, wb, ab, symmetric=symmetric)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if task_type == "multiclass" else nn.BCEWithLogitsLoss()

    print("[QAT] Starting training...")
    history, val_history = train_qat(model, dl_train, dl_val, optimizer, criterion, device, epochs=train_epochs, task_type=task_type, num_classes=num_classes)

    if args.save_preconvert:
        pre_path = os.path.join(cfg["output"]["weights_dir"], f"model_weights_qat_{wb}w{ab}a_preconvert.pt")
        torch.save(model.state_dict(), pre_path)
        print(f"[INFO] Saved pre-convert model weights to {pre_path}")

    weights_dir = cfg["output"]["weights_dir"]
    os.makedirs(weights_dir, exist_ok=True)
    fname = f"model_weights_qat_{wb}w{ab}a.pt"
    out_path = os.path.join(weights_dir, fname)
    torch.save(model.state_dict(), out_path)
    print(f"[OK] Saved QAT weights to {out_path}")

    npz_path = os.path.splitext(out_path)[0] + ".npz"
    try:
        # Try your existing exporter first (if it works in your env)
        export_quantized_weights_npz(model, npz_path)
        # Sanity check NPZ contents
        import numpy as _np
        _chk = _np.load(npz_path, allow_pickle=True)
        if len(list(_chk.files)) == 0:
            raise RuntimeError("Empty NPZ written by export_quantized_weights_npz; falling back.")
    except Exception as e:
        print(f"[WARN] Built-in export failed ({e}). Falling back to local exporter.")
        export_quant_npz_from_model(model, npz_path, bits=wb, per_channel=True, symmetric=symmetric)

    plots_dir = os.path.join(cfg["output"]["plots_dir"], "training")
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = os.path.join(plots_dir, f"metrics_qat_{wb}w{ab}a.png")

    plot_metrics(history, val_history, save_path=fig_path, title_prefix=f"QAT {wb}w{ab}a")
    print(f"[OK] Saved training plot to {fig_path}")
    print(f"[DONE] QAT completed in {time() - t0:.1f} seconds.")

if __name__ == "__main__":
    main()
