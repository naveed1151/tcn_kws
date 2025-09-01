import torch
import numpy as np
from typing import Dict, Any, Optional

def qmax_for_bits(bits: int) -> int:
    return (1 << (bits - 1)) - 1

def quantize_weights_global(
    state_dict: Dict[str, torch.Tensor], bits: int, global_percentile: float = 100.0
) -> np.ndarray:
    """
    Quantize all .weight tensors using a single global scale (symmetric or asymmetric).
    The scale is set by the given percentile of absolute weights.
    Returns concatenated integer codes.
    """
    weights = []
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        if not t.dtype.is_floating_point:
            continue
        if not name.endswith(".weight"):
            continue
        weights.append(t.detach().cpu().numpy().ravel())
    if not weights:
        print("[WARN] No weight tensors found for global quantization.")
        return np.array([], dtype=np.float32)
    all_weights = np.concatenate(weights, axis=0)
    qmax = qmax_for_bits(bits)
    max_abs = np.percentile(np.abs(all_weights), global_percentile)
    if max_abs == 0:
        q = np.zeros_like(all_weights)
        return q
    scale = max_abs / qmax
    q = np.round(all_weights / scale).clip(-qmax, qmax)
    return q.astype(np.int32)

def quantize_weights_per_tensor(
    state_dict: Dict[str, torch.Tensor], bits: int
) -> np.ndarray:
    """
    Quantize each .weight tensor independently (per-tensor).
    Returns concatenated integer codes.
    """
    codes = []
    qmax = qmax_for_bits(bits)
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        if not t.dtype.is_floating_point:
            continue
        if not name.endswith(".weight"):
            continue
        w = t.detach()
        max_abs = w.abs().max().item()
        if max_abs == 0:
            q = np.zeros_like(w.cpu().numpy().ravel())
        else:
            scale = max_abs / qmax
            q = np.round(w.cpu().numpy().ravel() / scale).clip(-qmax, qmax)
        codes.append(q.astype(np.int32))
    if not codes:
        print("[WARN] No weight tensors found for per-tensor quantization.")
        return np.array([], dtype=np.float32)
    return np.concatenate(codes, axis=0)

def quantize_weights_per_channel(
    state_dict: Dict[str, torch.Tensor], bits: int, ch_axis: int = 0
) -> np.ndarray:
    """
    Quantize each output channel of .weight tensors independently (per-channel).
    Returns concatenated integer codes.
    """
    codes = []
    qmax = qmax_for_bits(bits)
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        if not t.dtype.is_floating_point:
            continue
        if not name.endswith(".weight"):
            continue
        w = t.detach()
        if w.dim() < 2:
            # fallback to per-tensor
            max_abs = w.abs().max().item()
            if max_abs == 0:
                q = np.zeros_like(w.cpu().numpy().ravel())
            else:
                scale = max_abs / qmax
                q = np.round(w.cpu().numpy().ravel() / scale).clip(-qmax, qmax)
            codes.append(q.astype(np.int32))
            continue
        # per-channel quantization
        w_np = w.cpu().numpy()
        oc = w_np.shape[ch_axis]
        w_flat = np.moveaxis(w_np, ch_axis, 0).reshape(oc, -1)
        max_abs = np.max(np.abs(w_flat), axis=1)
        scale = np.where(max_abs == 0, 1.0, max_abs / qmax)
        q = np.round(w_flat / scale[:, None]).clip(-qmax, qmax)
        codes.append(q.astype(np.int32).ravel())
    if not codes:
        print("[WARN] No weight tensors found for per-channel quantization.")
        return np.array([], dtype=np.float32)
    return np.concatenate(codes, axis=0)

def quantize_model_weights(
    model: torch.nn.Module,
    bits: int = 8,
    scheme: str = "per_tensor",
    symmetric: bool = True,
    global_percentile: float = 100.0
) -> torch.nn.Module:
    """
    Quantize weights of all Conv1d/Linear layers in the model using the specified scheme.
    """
    model = model.cpu()
    if symmetric:
        qmin = -(2 ** (bits - 1))
        qmax_w = (2 ** (bits - 1)) - 1
    else:
        qmin = 0
        qmax_w = (2 ** bits) - 1

    if scheme == "global":
        all_weights = []
        for _, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                all_weights.append(w.view(-1))
        if all_weights:
            all_weights = torch.cat(all_weights).cpu().numpy()
            if symmetric:
                max_abs = np.percentile(np.abs(all_weights), global_percentile)
                scale = max_abs / qmax_w if max_abs > 0 else 1.0
                zero_point = 0
            else:
                min_val = np.percentile(all_weights, 100 - global_percentile)
                max_val = np.percentile(all_weights, global_percentile)
                scale = (max_val - min_val) / (qmax_w - qmin) if max_val > min_val else 1.0
                zero_point = np.round(qmin - min_val / scale)
            for _, module in model.named_modules():
                if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                    w = module.weight.data
                    if symmetric:
                        q = torch.clamp(torch.round(w / scale), qmin, qmax_w)
                        module.weight.data = q * scale
                    else:
                        q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax_w)
                        module.weight.data = (q - zero_point) * scale
    elif scheme == "per_channel":
        for _, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                if w.dim() < 2:
                    if symmetric:
                        max_abs = w.abs().max()
                        scale = max_abs / qmax_w if max_abs > 0 else 1.0
                        zero_point = 0
                        q = torch.clamp(torch.round(w / scale), qmin, qmax_w)
                        module.weight.data = q * scale
                    else:
                        min_val = w.min()
                        max_val = w.max()
                        scale = (max_val - min_val) / (qmax_w - qmin) if max_val > min_val else 1.0
                        zero_point = torch.round(qmin - min_val / scale)
                        q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax_w)
                        module.weight.data = (q - zero_point) * scale
                    continue
                oc = w.shape[0]
                w_flat = w.view(oc, -1)
                if symmetric:
                    max_abs = w_flat.abs().max(dim=1).values
                    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax_w)
                    zero_point = torch.zeros_like(scale)
                    q = torch.clamp(torch.round(w_flat / scale[:, None]), qmin, qmax_w)
                    module.weight.data = (q * scale[:, None]).view_as(w)
                else:
                    min_val = w_flat.min(dim=1).values
                    max_val = w_flat.max(dim=1).values
                    scale = torch.where(max_val > min_val, (max_val - min_val) / (qmax_w - qmin), torch.ones_like(max_val))
                    zero_point = torch.round(qmin - min_val / scale)
                    q = torch.clamp(torch.round(w_flat / scale[:, None] + zero_point[:, None]), qmin, qmax_w)
                    module.weight.data = ((q - zero_point[:, None]) * scale[:, None]).view_as(w)
    else:  # per_tensor
        for _, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)) and hasattr(module, "weight"):
                w = module.weight.data
                if symmetric:
                    max_abs = w.abs().max()
                    scale = max_abs / qmax_w if max_abs > 0 else 1.0
                    zero_point = 0
                    q = torch.clamp(torch.round(w / scale), qmin, qmax_w)
                    module.weight.data = q * scale
                else:
                    min_val = w.min()
                    max_val = w.max()
                    scale = (max_val - min_val) / (qmax_w - qmin) if max_val > min_val else 1.0
                    zero_point = torch.round(qmin - min_val / scale)
                    q = torch.clamp(torch.round(w / scale + zero_point), qmin, qmax_w)
                    module.weight.data = (q - zero_point) * scale
    return model