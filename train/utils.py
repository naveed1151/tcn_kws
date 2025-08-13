from model.model import DilatedTCN
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import os
from typing import Any, Dict, Optional, Tuple, List, Union, TypeVar

M = TypeVar("M", bound=nn.Module)

# -----------------------
# Model building
# -----------------------

def build_model_from_cfg(cfg: Dict[str, Any], sample_input: torch.Tensor, num_classes: int) -> DilatedTCN:
    """
    Build the updated DilatedTCN from config and a sample input (C, T).
    Uses only the new fields in cfg['model'].
    """
    # Expect (C, T)
    if sample_input.dim() != 2:
        raise ValueError(f"build_model_from_cfg expects sample_input shaped (C, T), got {tuple(sample_input.shape)}")
    in_channels = int(sample_input.shape[0])
    seq_len = int(sample_input.shape[1])

    m = cfg["model"]
    kernel_size = int(m["kernel_size"])
    hidden_channels = int(m["hidden_channels"])
    dropout = float(m["dropout"])
    num_blocks = int(m["num_blocks"])
    causal = bool(m.get("causal", True))
    activation = str(m.get("activation", "relu"))
    norm = str(m.get("norm", "batch"))
    groups_for_groupnorm = int(m.get("groups_for_groupnorm", 8))
    use_weight_norm = bool(m.get("use_weight_norm", False))
    depthwise_separable = bool(m.get("depthwise_separable", False))
    pool = str(m.get("pool", "avg"))
    bias = bool(m.get("bias", True))

    model = DilatedTCN(
        input_channels=in_channels,
        num_blocks=num_blocks,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        num_classes=num_classes,
        dropout=dropout,
        causal=causal,
        activation=activation,
        norm=norm,
        groups_for_groupnorm=groups_for_groupnorm,
        use_weight_norm=use_weight_norm,
        depthwise_separable=depthwise_separable,
        pool=pool,
        bias=bias,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        "[MODEL] DilatedTCN built: "
        f"in_ch={in_channels}, T={seq_len}, hidden={hidden_channels}, "
        f"kernel={kernel_size}, blocks={num_blocks}, dropout={dropout}, classes={num_classes}, "
        f"causal={causal}, act={activation}, norm={norm}, groups={groups_for_groupnorm}, "
        f"dwise_sep={depthwise_separable}, w_norm={use_weight_norm}, pool={pool}, bias={bias}"
    )
    print(f"[PARAMS] total={total_params:,} trainable={trainable_params:,}")

    return model

# -----------------------
# Loading weights
# -----------------------
def load_state_dict_forgiving(model: M, path: str, device: torch.device) -> M:
    state_dict = torch.load(path, map_location=device)
    model_state = model.state_dict()

    filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
    model_state.update(filtered)
    model.load_state_dict(model_state)
    print(f"[INFO] Loaded weights (forgiving): matched {len(filtered)}/{len(model_state)} tensors")
    return model

# -----------------------
# Metrics helpers
# -----------------------
def _binary_counts(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int, int, int, int]:
    preds = preds.long()
    targets = targets.long()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return tp, fp, fn, correct, total

def _derive_metrics(
    total_loss: float,
    num_batches: int,
    correct: int,
    total: int,
    tp: Optional[int] = None,
    fp: Optional[int] = None,
    fn: Optional[int] = None,
) -> Union[Tuple[float, float], Tuple[float, float, float, float, float]]:
    avg_loss = total_loss / max(1, num_batches)
    accuracy = correct / total if total > 0 else 0.0
    if tp is None:
        return avg_loss, accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return avg_loss, accuracy, precision, recall, f1

def _multiclass_confusion_add(cm: List[List[int]], preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> List[List[int]]:
    for p, t in zip(preds.tolist(), targets.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm

def _multiclass_macro_prf1(cm: List[List[int]]) -> Tuple[float, float, float]:
    K = len(cm)
    precisions, recalls, f1s = [], [], []
    for k in range(K):
        tp = cm[k][k]
        fp = sum(cm[r][k] for r in range(K)) - tp
        fn = sum(cm[k][c] for c in range(K)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    macro_p = sum(precisions) / K if K > 0 else 0.0
    macro_r = sum(recalls) / K if K > 0 else 0.0
    macro_f1 = sum(f1s) / K if K > 0 else 0.0
    return macro_p, macro_r, macro_f1

def compute_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> np.ndarray:
    """Return confusion matrix (num_classes x num_classes) of counts for multiclass."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().cpu().numpy()
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for t, p in zip(y, preds):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1
    return cm


# -----------------------
# Config helpers
# -----------------------
def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_yaml_with_encodings(path: str) -> Dict[str, Any]:
    import yaml
    # Try UTF-8 first; then UTF-8 with BOM; then cp1252 as last resort.
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("yaml", b"", 0, 1, f"Could not decode {path} with utf-8/utf-8-sig/cp1252")

def load_config(path: str) -> Dict[str, Any]:
    cfg_task = _load_yaml_with_encodings(path)

    # Optional base.yaml merge
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if os.path.basename(path) != "base.yaml" and os.path.exists(base_path):
        cfg_base = _load_yaml_with_encodings(base_path)
        return deep_update(deepcopy(cfg_base), cfg_task)
    return cfg_task

# -----------------------
# Weights export
# -----------------------
def export_quantized_weights_npz(model: nn.Module, path: str) -> None:
    """
    Save quantized model weights (int) to a .npz file for inspection or hardware use.
    Only works for quantized models (e.g., torch.ao.quantized).
    """
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    state = model.state_dict()
    weights = {}
    for name, tensor in state.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype in (torch.qint8, torch.quint8, torch.int8, torch.uint8):
            weights[name] = tensor.int_repr().cpu().numpy()
    np.savez(path, **weights)
    print(f"[OK] Quantized weights saved to {path}")

# -----------------------
# Float checkpoint -> NPZ integer codes + scales
# -----------------------
def export_quant_from_pt(
    weights_path: str,
    out_path: str,
    bits: int = 5,
    per_channel: bool = True,
    symmetric: bool = True,
) -> List[str]:
    """
    Quantize raw float weights from a PyTorch checkpoint to integer codes + scales
    and store them in an NPZ archive (variable shapes supported via object arrays).

    Args:
        weights_path: .pt/.pth checkpoint (either raw state_dict or dict with 'state_dict')
        out_path: destination .npz path (directories auto-created)
        bits: quantization bit-width (<=16 recommended)
        per_channel: per-output-channel scaling (first dim) if True, else per-tensor
        symmetric: symmetric (signed) if True; otherwise unsigned/asymmetric (0..2^bits-1)

    Returns:
        List of tensor names quantized.
    """
    obj = torch.load(weights_path, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

    qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
    qmin = -qmax if symmetric else 0

    names: List[str] = []
    q_list: List[np.ndarray] = []
    scale_list: List[np.ndarray] = []

    for name, w in state.items():
        if not isinstance(w, torch.Tensor):
            continue
        if "weight" not in name:
            continue
        if w.ndim < 2:  # skip biases / norms
            continue
        wf = w.float().cpu()
        if per_channel:
            oc = wf.shape[0]
            flat = wf.view(oc, -1)
            if symmetric:
                max_abs = flat.abs().max(dim=1).values
            else:
                max_abs = flat.max(dim=1).values
            scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
            q = torch.round(flat / scale[:, None]).clamp(qmin, qmax).to(torch.int16)
            names.append(name)
            q_list.append(q.numpy())
            scale_list.append(scale.numpy())
        else:
            max_abs = wf.abs().max() if symmetric else wf.max()
            if max_abs == 0:
                continue
            scale = (max_abs / qmax).item()
            q = torch.round(wf / scale).clamp(qmin, qmax).to(torch.int16)
            names.append(name)
            q_list.append(q.numpy())
            scale_list.append(np.array(scale, dtype=np.float32))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(
        out_path,
        names=np.array(names, dtype=object),
        q_list=np.array(q_list, dtype=object),
        scale_list=np.array(scale_list, dtype=object),
        bits=np.array(bits),
        symmetric=np.array(symmetric),
        per_channel=np.array(per_channel),
    )
    print(f"[OK] Quantized {len(names)} tensors -> {out_path}")
    return names