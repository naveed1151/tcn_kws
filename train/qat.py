import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from time import time
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from torch.utils.data import DataLoader

from train.utils import (
    build_model_from_cfg, load_config, _binary_counts, _derive_metrics,
    _multiclass_confusion_add, _multiclass_macro_prf1, export_quantized_weights_npz,
    load_state_dict_forgiving,
)
from data_loader.utils import make_datasets, get_num_classes
from analysis.plot_metrics import plot_metrics, plot_test_confusion_matrix


class CustomFakeQuantize(nn.Module):
    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = True,
        ema: bool = True,
        momentum: float = 0.95,
        eps: float = 1e-8,
        per_channel: bool = False,
        ch_axis: int = 0,
        use_external: bool = False,
        external_scale: Optional[Union[float, torch.Tensor]] = None,
        external_zero_point: Optional[Union[float, torch.Tensor]] = None,
    ) -> None:
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
        # Global fixed scale mode
        self.use_external = bool(use_external)
        # Buffers
        self.register_buffer("scale", torch.tensor(1.0))
        self.register_buffer("running_max", torch.tensor(1.0))
        self.register_buffer("running_min", torch.tensor(0.0))  # for asymmetric if needed
        # External constants (registered so they move with .to(device))
        if external_scale is not None:
            self.register_buffer("external_scale", torch.as_tensor(external_scale, dtype=torch.float32))
        else:
            self.external_scale = None
        if external_zero_point is not None:
            self.register_buffer("external_zero_point", torch.as_tensor(external_zero_point, dtype=torch.float32))
        else:
            self.external_zero_point = None
        if self.use_external:
            # In external mode, never update observers
            self.frozen = True

    def _qrange(self) -> Tuple[int, int]:
        if self.symmetric:
            qmax = (1 << (self.bits - 1)) - 1
            qmin = -qmax
        else:
            qmin = 0
            qmax = (1 << self.bits) - 1
        return qmin, qmax

    def _observe(self, x: torch.Tensor) -> None:
        if self.frozen:
            return
        if self.per_channel:
            # reduce over all dims except ch_axis
            reduce_dims = [d for d in range(x.dim()) if d != self.ch_axis]
            if self.symmetric:
                max_abs = x.detach().abs().amax(dim=reduce_dims)
                if self.ema:
                    self.running_max = torch.maximum(
                        self.running_max * self.momentum + max_abs * (1 - self.momentum),
                        torch.full_like(max_abs, self.eps),
                    )
                else:
                    self.running_max = torch.clamp(max_abs, min=self.eps)
            else:
                x_max = x.detach().amax(dim=reduce_dims)
                x_min = x.detach().amin(dim=reduce_dims)
                if self.ema:
                    self.running_max = self.running_max * self.momentum + x_max * (1 - self.momentum)
                    self.running_min = self.running_min * self.momentum + x_min * (1 - self.momentum)
                else:
                    self.running_max = x_max
                    self.running_min = x_min
        else:
            if self.symmetric:
                max_abs = x.detach().abs().max()
                if self.ema:
                    self.running_max = torch.maximum(
                        self.running_max * self.momentum + max_abs * (1 - self.momentum),
                        torch.tensor(self.eps, device=x.device),
                    )
                else:
                    self.running_max = torch.clamp(max_abs, min=self.eps)
            else:
                x_max = x.detach().max()
                x_min = x.detach().min()
                if self.ema:
                    self.running_max = self.running_max * self.momentum + x_max * (1 - self.momentum)
                    self.running_min = self.running_min * self.momentum + x_min * (1 - self.momentum)
                else:
                    self.running_max = x_max
                    self.running_min = x_min

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.enabled) or self.bits >= 32:
            return x

        # If using a global, fixed scale, bypass observers and use provided constants
        if self.use_external and (self.external_scale is not None):
            qmin, qmax = self._qrange()
            s = torch.clamp(self.external_scale, min=self.eps)
            self.scale = s.detach()
            if self.symmetric:
                q = torch.round(x / s).clamp(qmin, qmax)
                x_hat = q * s
            else:
                z = self.external_zero_point if (self.external_zero_point is not None) else torch.tensor(0.0, device=x.device)
                q = torch.round(x / s + z).clamp(qmin, qmax)
                x_hat = (q - z) * s
            return x + (x_hat - x).detach()

        # Normal (learn/observe) mode
        self._observe(x)
        qmin, qmax = self._qrange()
        if self.symmetric:
            max_val = self.running_max
            scale = max_val / qmax
            scale = torch.where(scale == 0, torch.full_like(scale, self.eps), scale)
            self.scale = scale.detach()
            if self.per_channel:
                # reshape scale for broadcast on ch_axis
                shape = [1] * x.dim()
                shape[self.ch_axis] = -1
                s = self.scale.view(shape)
            else:
                s = self.scale
            q = torch.round(x / s).clamp(qmin, qmax)
            x_hat = q * s
        else:
            # Proper asymmetric: use min/max and zero_point
            rng = self.running_max - self.running_min
            if self.per_channel:
                rng = torch.where(rng <= self.eps, torch.full_like(rng, self.eps), rng)
            else:
                rng = torch.clamp(rng, min=self.eps)
            scale = rng / qmax
            self.scale = scale.detach()
            if self.per_channel:
                # reshape scale for broadcast on ch_axis
                shape = [1] * x.dim()
                shape[self.ch_axis] = -1
                s = self.scale.view(shape)
                z = torch.round((-self.running_min / self.scale)).view(shape)
            else:
                s = self.scale
                z = torch.round(-self.running_min / self.scale)
            q = torch.round(x / s + z).clamp(qmin, qmax)
            x_hat = (q - z) * s
        return x + (x_hat - x).detach()


def apply_custom_fake_quant(
    model: nn.Module,
    weight_bits: int,
    act_bits: int,
    weight_symmetric: bool = True,
    act_symmetric: bool = True,
    weight_use_global: bool = False,
    weight_global_scale: Optional[float] = None,
    act_use_global: bool = False,
    act_global_scale: Optional[float] = None,
    act_global_zero_point: float = 0.0,
) -> nn.Module:
    # Per-channel weights unless global fixed is requested
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            m.weight_fake = CustomFakeQuantize(
                bits=weight_bits,
                symmetric=weight_symmetric,
                per_channel=(not weight_use_global),
                ch_axis=0,
                use_external=weight_use_global,
                external_scale=weight_global_scale,
                external_zero_point=0.0 if weight_symmetric else 0.0  # weight zp typically 0 even in asym hw
            )
            orig = m.forward
            def conv_fwd(self: nn.Conv1d, inp: torch.Tensor) -> torch.Tensor:
                w_q = self.weight_fake(self.weight)
                return F.conv1d(inp, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
            m.forward = conv_fwd.__get__(m, m.__class__)
        elif isinstance(m, nn.Linear):
            m.weight_fake = CustomFakeQuantize(
                bits=weight_bits,
                symmetric=weight_symmetric,
                per_channel=(not weight_use_global),
                ch_axis=0,
                use_external=weight_use_global,
                external_scale=weight_global_scale,
                external_zero_point=0.0 if weight_symmetric else 0.0
            )
            def lin_fwd(self: nn.Linear, inp: torch.Tensor) -> torch.Tensor:
                w_q = self.weight_fake(self.weight)
                return F.linear(inp, w_q, self.bias)
            m.forward = lin_fwd.__get__(m, m.__class__)
    # Per-layer input/output activation fake quant (global if requested)
    if act_bits < 32:
        for m in model.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                m.input_fake = CustomFakeQuantize(
                    bits=act_bits,
                    symmetric=act_symmetric,
                    per_channel=False,
                    use_external=act_use_global,
                    external_scale=act_global_scale,
                    external_zero_point=act_global_zero_point if not act_symmetric else 0.0
                )
                orig = m.forward
                def wrapped(self: Union[nn.Conv1d, nn.Linear], x: torch.Tensor, _orig=orig) -> torch.Tensor:
                    x = self.input_fake(x)
                    return _orig(x)
                m.forward = wrapped.__get__(m, m.__class__)

    # Residual add alignment: quantize block outputs (after out + skip)
    if act_bits < 32:
        for m in model.modules():
            if m.__class__.__name__ in ("TemporalBlock", "ResidualBlock", "TCNBlock"):
                m.output_fake = CustomFakeQuantize(
                    bits=act_bits,
                    symmetric=act_symmetric,
                    per_channel=False,
                    use_external=act_use_global,
                    external_scale=act_global_scale,
                    external_zero_point=act_global_zero_point if not act_symmetric else 0.0
                )
                _orig = m.forward
                def _wrap(self, *args: Any, __orig=_orig, **kwargs: Any) -> torch.Tensor:
                    out = __orig(*args, **kwargs)
                    return self.output_fake(out)
                m.forward = _wrap.__get__(m, m.__class__)
    return model


def set_qat_mode(model: nn.Module, enabled: bool = True, freeze: bool = False) -> None:
    for m in model.modules():
        if hasattr(m, "enabled"):
            m.enabled = enabled
        if hasattr(m, "frozen"):
            m.frozen = freeze


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
    num_classes: int,
    threshold: float = 0.5,
) -> Tuple[float, float, float, float, float]:
    model.eval()
    total_loss, correct, total, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
    cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x, y = batch["x"], batch["y"]
            else:
                raise RuntimeError("Unexpected batch format")
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


def _preview_scale(t: torch.Tensor, k: int = 3) -> Union[List[float], str]:
    # Return a compact preview of a scale tensor
    try:
        arr = t.detach().flatten().cpu().numpy()
        return np.round(arr[:k], 6).tolist()
    except Exception:
        return str(t)

def print_qat_scales(model: nn.Module, limit: int = 3, prefix: str = "") -> None:
    """
    Print a few fake-quant scales for weights and activations to verify freezing.
    Shows first `limit` modules that have any of: weight_fake, input_fake, output_fake.
    """
    printed = 0
    for name, m in model.named_modules():
        has_any = any(hasattr(m, attr) for attr in ("weight_fake", "input_fake", "output_fake"))
        if not has_any:
            continue

        parts = []
        flags = []
        if hasattr(m, "weight_fake") and isinstance(m.weight_fake, CustomFakeQuantize):
            parts.append(f"w_scale={_preview_scale(m.weight_fake.scale)}")
            flags.append(f"w(enabled={m.weight_fake.enabled}, frozen={m.weight_fake.frozen})")
        if hasattr(m, "input_fake") and isinstance(m.input_fake, CustomFakeQuantize):
            parts.append(f"a_in_scale={_preview_scale(m.input_fake.scale)}")
            flags.append(f"a_in(enabled={m.input_fake.enabled}, frozen={m.input_fake.frozen})")
        if hasattr(m, "output_fake") and isinstance(m.output_fake, CustomFakeQuantize):
            parts.append(f"a_out_scale={_preview_scale(m.output_fake.scale)}")
            flags.append(f"a_out(enabled={m.output_fake.enabled}, frozen={m.output_fake.frozen})")

        if parts:
            mod_name = name if name else m.__class__.__name__
            print(f"{prefix} [{mod_name}] " + " | ".join(parts) + " | " + ", ".join(flags))
            printed += 1
            if printed >= limit:
                break

def train_qat(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    task_type: str,
    num_classes: int,
    threshold: float = 0.5,
    warmup_epochs: int = 1,
    wb: Optional[int] = None,
    ab: Optional[int] = None,
    weight_use_global: bool = False,
    act_use_global: bool = False,
    weight_symmetric: bool = True,
    act_symmetric: bool = True,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    # init histories
    hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
    val_hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}

    for epoch in range(1, epochs+1):
        model.train()

        # Switch to global fixed scales right after warmup is done
        if epoch == warmup_epochs + 1:
            # Derive global scales from warmup stats/data and apply
            if weight_use_global and wb is not None:
                gw_s, gw_zp = _compute_global_weight_scale(model, bits=wb, symmetric=weight_symmetric)
                _apply_global_weight_scale(model, gw_s, zero_point=(0.0 if weight_symmetric else gw_zp))
                print(f"[QAT] Applied GLOBAL weight scale from warmup: scale={gw_s:.6g}, zp={(0.0 if weight_symmetric else gw_zp):.3f}")
            if act_use_global and ab is not None and ab < 32:
                ga_s, ga_zp = _compute_global_activation_scale(model, bits=ab, symmetric=act_symmetric)
                _apply_global_activation_scale(model, ga_s, zero_point=(0.0 if act_symmetric else ga_zp))
                print(f"[QAT] Applied GLOBAL activation scale from warmup: scale={ga_s:.6g}, zp={(0.0 if act_symmetric else ga_zp):.3f}")

            # Freeze observers thereafter
            set_qat_mode(model, enabled=True, freeze=True)
            print_qat_scales(model, limit=3, prefix=f"[After freeze @epoch {epoch}]")

        total_loss, correct, total, tp, fp, fn = 0.0, 0, 0, 0, 0, 0
        cm = [[0] * num_classes for _ in range(num_classes)] if task_type == "multiclass" else None

        loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in loop:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0], batch[1]
            elif isinstance(batch, dict):
                x, y = batch["x"], batch["y"]
            else:
                raise RuntimeError("Unexpected batch format")
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

        hist["loss"].append(avg_loss); hist["acc"].append(acc)
        hist["prec"].append(prec); hist["rec"].append(rec); hist["f1"].append(f1)

        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device, task_type, num_classes, threshold)
        val_hist["loss"].append(val_loss); val_hist["acc"].append(val_acc)
        val_hist["prec"].append(val_prec); val_hist["rec"].append(val_rec); val_hist["f1"].append(val_f1)

        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}  Acc: {acc:.4f}  P: {prec:.4f}  R: {rec:.4f}  F1: {f1:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  P: {val_prec:.4f}  R: {val_rec:.4f}  F1: {val_f1:.4f}")

        # Print a small snapshot of scales on a few epochs to observe stability
        if epoch in (1, warmup_epochs + 1, epochs):
            print_qat_scales(model, limit=3, prefix=f"[Scales @epoch {epoch}]")

    return hist, val_hist


@torch.no_grad()
def export_quant_npz_from_model(
    model: nn.Module,
    out_npz_path: str,
    bits: int = 5,
    per_channel: bool = True,
    symmetric_weights: bool = True,
    use_global: bool = False,
    global_scale: Optional[float] = None,
    global_zero_point: float = 0.0,
) -> None:
    # If caller requested global but didn't pass a value, try to read it from the model
    if use_global and (global_scale is None):
        for m in model.modules():
            if hasattr(m, "weight_fake") and isinstance(m.weight_fake, CustomFakeQuantize):
                fq = m.weight_fake
                if getattr(fq, "use_external", False) and (getattr(fq, "external_scale", None) is not None):
                    global_scale = float(fq.external_scale.item())
                    if not symmetric_weights and (getattr(fq, "external_zero_point", None) is not None):
                        global_zero_point = float(fq.external_zero_point.item())
                    break
        if global_scale is None:
            raise ValueError("use_global=True but no global scale found on model; ensure scales were applied after warmup.")
    
    qmax = (1 << (bits - 1)) - 1 if symmetric_weights else (1 << bits) - 1
    qmin = -qmax if symmetric_weights else 0
    names, q_list, scale_list = [], [], []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and hasattr(m, "weight"):
            w = m.weight.detach().float().cpu()
            if use_global and (global_scale is not None):
                s = float(global_scale)
                s = max(s, 1e-8)
                if symmetric_weights:
                    q = torch.round(w / s).clamp(qmin, qmax).to(torch.int16)
                else:
                    z = float(global_zero_point)
                    q = torch.round(w / s + z).clamp(qmin, qmax).to(torch.int16)
                names.append(f"{name}.weight" if name else "weight")
                q_list.append(q.numpy())
                scale_list.append(np.array(s, dtype=np.float32))
            elif per_channel and w.dim() >= 2:
                oc = w.shape[0]
                w2 = w.view(oc, -1)
                max_abs = w2.abs().max(dim=1).values if symmetric_weights else w2.max(dim=1).values
                scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
                q = torch.round(w2 / scale[:, None]).clamp(qmin, qmax).to(torch.int16)
                names.append(f"{name}.weight" if name else "weight")
                q_list.append(q.numpy())
                scale_list.append(scale.numpy())
            else:
                max_abs = w.abs().max() if symmetric_weights else w.max()
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
        symmetric=np.array(symmetric_weights),
        per_channel=np.array(per_channel and not use_global),
    )
    print(f"[OK] Exported {len(names)} tensors to {out_npz_path}")


def _qmax_qmin(bits: int, symmetric: bool) -> Tuple[int, int]:
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
    else:
        qmax = (1 << bits) - 1
        qmin = 0
    return qmin, qmax

@torch.no_grad()
def _compute_global_weight_scale(
    model: nn.Module,
    bits: int,
    symmetric: bool = True,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    qmin, qmax = _qmax_qmin(bits, symmetric)
    w_min = None
    w_max = None
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and hasattr(m, "weight"):
            w = m.weight.detach()
            cur_min = w.min()
            cur_max = w.max()
            w_min = cur_min if w_min is None else torch.minimum(w_min, cur_min)
            w_max = cur_max if w_max is None else torch.maximum(w_max, cur_max)
    if w_min is None or w_max is None:
        raise RuntimeError("No weights found to compute global weight scale.")
    if symmetric:
        max_abs = torch.maximum(w_max.abs(), w_min.abs())
        s = torch.clamp(max_abs / qmax, min=eps).item()
        zp = 0.0
    else:
        rng = torch.clamp(w_max - w_min, min=eps)
        s = (rng / qmax).item()
        zp = torch.round(-w_min / s).item()
    return float(s), float(zp)

@torch.no_grad()
def _compute_global_activation_scale(
    model: nn.Module,
    bits: int,
    symmetric: bool = True,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    qmin, qmax = _qmax_qmin(bits, symmetric)
    a_min = None
    a_max = None
    found = False
    for m in model.modules():
        for attr in ("input_fake", "output_fake"):
            if hasattr(m, attr) and isinstance(getattr(m, attr), CustomFakeQuantize):
                fq = getattr(m, attr)
                # Use observer statistics collected during warmup
                found = True
                cur_max = fq.running_max.detach()
                a_max = cur_max if a_max is None else torch.maximum(a_max, cur_max)
                if not symmetric:
                    cur_min = fq.running_min.detach()
                    a_min = cur_min if a_min is None else torch.minimum(a_min, cur_min)
    if not found:
        raise RuntimeError("No activation fake-quant modules found to compute global activation scale.")
    if symmetric:
        s = torch.clamp(a_max / qmax, min=eps).item()
        zp = 0.0
    else:
        if a_min is None:
            # Fallback: assume non-negative activations
            a_min = torch.zeros_like(a_max)
        rng = torch.clamp(a_max - a_min, min=eps)
        s = (rng / qmax).item()
        zp = torch.round(-a_min / s).item()
    return float(s), float(zp)

def _to_tensor_on(ref: torch.Tensor, val: float) -> torch.Tensor:
    return torch.as_tensor(float(val), dtype=torch.float32, device=ref.device)

def _set_or_register_buffer(mod: nn.Module, name: str, value: torch.Tensor) -> None:
    # If already a registered buffer, update in-place
    if name in mod._buffers and isinstance(mod._buffers[name], torch.Tensor):
        mod._buffers[name].data.copy_(value)
        return
    # If an attribute exists (e.g., None), remove it before registering
    if hasattr(mod, name):
        try:
            delattr(mod, name)
        except Exception:
            pass
    try:
        mod.register_buffer(name, value)
    except KeyError:
        # Fallback: set as a plain attribute
        setattr(mod, name, value)

@torch.no_grad()
def _apply_global_weight_scale(model: nn.Module, scale: float, zero_point: float = 0.0) -> None:
    """
    Switch all Conv/Linear weight fake-quant modules to use a fixed global scale.
    Safe when external_* already exist (updates instead of re-registering).
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and hasattr(m, "weight_fake") and isinstance(m.weight_fake, CustomFakeQuantize):
            fq: CustomFakeQuantize = m.weight_fake
            ref = fq.scale if isinstance(getattr(fq, "scale", None), torch.Tensor) else next(m.parameters()).detach()
            _set_or_register_buffer(fq, "external_scale", _to_tensor_on(ref, scale))
            _set_or_register_buffer(fq, "external_zero_point", _to_tensor_on(ref, zero_point))
            fq.use_external = True
            fq.frozen = True
            fq.enabled = True

@torch.no_grad()
def _apply_global_activation_scale(model: nn.Module, scale: float, zero_point: float = 0.0) -> None:
    """
    Switch all activation fake-quant modules to use a fixed global scale.
    Safe when external_* already exist (updates instead of re-registering).
    """
    for m in model.modules():
        for attr in ("input_fake", "output_fake"):
            if hasattr(m, attr) and isinstance(getattr(m, attr), CustomFakeQuantize):
                fq: CustomFakeQuantize = getattr(m, attr)
                ref = fq.scale if isinstance(getattr(fq, "scale", None), torch.Tensor) else next(m.parameters()).detach()
                _set_or_register_buffer(fq, "external_scale", _to_tensor_on(ref, scale))
                _set_or_register_buffer(fq, "external_zero_point", _to_tensor_on(ref, zero_point))
                fq.use_external = True
                fq.frozen = True
                fq.enabled = True


def main() -> None:
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
    weight_symmetric = bool(qat_cfg.get("weight_symmetric", qat_cfg.get("symmetric", True)))
    act_symmetric = bool(qat_cfg.get("act_symmetric", qat_cfg.get("symmetric", True)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_type = cfg["task"]["type"]

    train_batch_size = cfg["qat"].get("batch_size", 32)
    train_epochs = args.epochs if args.epochs is not None else cfg["qat"].get("num_epochs", 5)
    lr = cfg["qat"].get("learning_rate", 1e-3)
    warmup_epochs = int(cfg["qat"].get("warmup_epochs", 1))

    # Optional QAT-specific dropout override
    qat_dropout = qat_cfg.get("dropout", None)
    if qat_dropout is not None:
        cfg.setdefault("model", {})
        orig_do = cfg["model"].get("dropout", None)
        cfg["model"]["dropout"] = float(qat_dropout)
        print(f"[QAT] Overriding model.dropout: {orig_do} -> {qat_dropout}")

    # Global fixed-scale options (derived from warmup; no scales in YAML)
    weight_use_global = bool(qat_cfg.get("weight_use_global_scale", False))
    act_use_global = bool(qat_cfg.get("act_use_global_scale", False))

    # Always request all splits
    dl_train, dl_val, dl_test = make_datasets(cfg, which="all", batch_size=train_batch_size)
    num_classes = get_num_classes(cfg)
    batch0 = next(iter(dl_train))
    if isinstance(batch0, (list, tuple)):
        x0 = batch0[0]
    elif isinstance(batch0, dict):
        x0 = batch0["x"]
    else:
        x0 = batch0
    sample_x = x0[0] if hasattr(x0, "dim") and x0.dim() == 3 else x0  # (C,T)
    model = build_model_from_cfg(cfg, sample_x, num_classes)

    if args.weights:
        model = load_state_dict_forgiving(model, args.weights, device)

    model.to(device)
    # Apply QAT with per-channel/per-tensor observers during warmup
    model = apply_custom_fake_quant(
        model, wb, ab,
        weight_symmetric=weight_symmetric,
        act_symmetric=act_symmetric,
        weight_use_global=False,          # start with observers
        act_use_global=False
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() if task_type == "multiclass" else nn.BCEWithLogitsLoss()

    print("[QAT] Starting training...")
    history, val_history = train_qat(
        model, dl_train, dl_val, optimizer, criterion, device,
        epochs=train_epochs, task_type=task_type, num_classes=num_classes,
        threshold=0.5, warmup_epochs=warmup_epochs,
        wb=wb, ab=ab,
        weight_use_global=weight_use_global, act_use_global=act_use_global,
        weight_symmetric=weight_symmetric, act_symmetric=act_symmetric
    )

    # ---- Test evaluation (like train/train.py) ----
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, dl_test, criterion, device, task_type, num_classes)
    test_metrics = {"loss": float(test_loss), "acc": float(test_acc), "prec": float(test_prec), "rec": float(test_rec), "f1": float(test_f1)}
    print(f"[TEST] Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  P: {test_prec:.4f}  R: {test_rec:.4f}  F1: {test_f1:.4f}")

    # Optional: confusion matrix plots for multiclass
    if task_type == "multiclass":
        class_names = getattr(dl_train.dataset, "class_names", [str(i) for i in range(num_classes)])
        cm = [[0] * num_classes for _ in range(num_classes)]
        with torch.no_grad():
            for batch in dl_test:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                elif isinstance(batch, dict):
                    x, y = batch["x"], batch["y"]
                else:
                    raise RuntimeError("Unexpected batch format")
                x = x.to(device)
                targets = y.long().to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1)
                cm = _multiclass_confusion_add(cm, preds, targets, num_classes)

        plots_dir = os.path.join(cfg["output"]["plots_dir"], "training")
        os.makedirs(plots_dir, exist_ok=True)
        plot_test_confusion_matrix(cm, class_names=class_names, normalize=False,
                                   save_path=os.path.join(plots_dir, f"test_confusion_matrix_qat_{wb}w{ab}a.png"))
        plot_test_confusion_matrix(cm, class_names=class_names, normalize=True,
                                   save_path=os.path.join(plots_dir, f"test_confusion_matrix_qat_{wb}w{ab}a_norm.png"))

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
    # Use local exporter; let it read the applied global scale from the model when use_global=True
    export_quant_npz_from_model(
        model, npz_path, bits=wb,
        per_channel=(not weight_use_global),
        symmetric_weights=weight_symmetric,
        use_global=weight_use_global,
        global_scale=None,  # auto-read from model.weight_fake.external_scale if present
        global_zero_point=0.0
    )
    print(f"[OK] Exported quantized weights to {npz_path}")

    plots_dir = os.path.join(cfg["output"]["plots_dir"], "training")
    os.makedirs(plots_dir, exist_ok=True)
    fig_path = os.path.join(plots_dir, f"metrics_qat_{wb}w{ab}a.png")

    plot_metrics(history, val_history, test_metrics=test_metrics, save_path=fig_path, title_prefix=f"QAT {wb}w{ab}a")
    print(f"[OK] Saved training plot to {fig_path}")
    print(f"[DONE] QAT completed in {time() - t0:.1f} seconds.")

if __name__ == "__main__":
    main()
