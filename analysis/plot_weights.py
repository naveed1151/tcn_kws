"""
Visualize trained model weights (state_dict):

1) Overall histogram (float, weights+biases) with zero/non-zero counts.
2) Per-parameter histograms (weights & biases) with zero/non-zero counts.
3) Multi-quant figure (weights only):
   - 'float' panel: value-domain histogram (continuous).
   - For quantized variants (e.g., 8/4/3 bit): histogram of quantized INTEGER CODES,
     so bins align exactly with quantization levels (bin width = 1, centered on each level).
   - Each panel shows total zeros and non-zeros. For quantized panels, zeros are exact code=0.

Usage examples:
  python -m analysis.visualize_model_weights --weights model_weights.pt
  python -m analysis.visualize_model_weights --weights model_weights.pt --zero-threshold 1e-6
  python -m analysis.visualize_model_weights --weights model_weights.pt --quant-bits 8,4,3 --quant-scheme per_channel
"""

import os
import argparse
from typing import Dict, Optional, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import yaml
from model.model import DilatedTCN
from data_loader.mfcc_dataset import MFCCDataset



def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path):
    def _load_yaml(path):
        for enc in ("utf-8", "utf-8-sig", "cp1252"):
            try:
                with open(path, "r", encoding=enc) as f:
                    return yaml.safe_load(f)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Cannot decode: {path}")

    cfg_task = _load_yaml(path)
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if os.path.basename(path) != "base.yaml" and os.path.exists(base_path):
        cfg_base = _load_yaml(base_path)
        return deep_update(deepcopy(cfg_base), cfg_task)
    return cfg_task

# -----------------------------
# IO helpers
# -----------------------------
def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """
    Load a state_dict from a .pt file. Supports:
      - raw state_dict (param_name -> tensor)
      - checkpoint dict with "model" key
      - checkpoint dict with "state_dict" key
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        print("[INFO] Detected wrapper with 'state_dict' key.")
        return obj["state_dict"]
    if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        print("[INFO] Detected checkpoint with 'model' key.")
        return obj["model"]
    if isinstance(obj, dict):
        print("[INFO] Detected raw state_dict.")
        return obj
    raise ValueError("Unsupported weights file format: expected a state_dict or checkpoint dict.")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Zero/non-zero counting
# -----------------------------
def zero_nonzero_counts_from_array(arr: np.ndarray, zero_threshold: float = 0.0) -> Tuple[int, int]:
    if zero_threshold <= 0.0:
        zeros = int(np.count_nonzero(arr == 0.0))
    else:
        zeros = int(np.count_nonzero(np.abs(arr) <= zero_threshold))
    nonzeros = int(arr.size - zeros)
    return zeros, nonzeros


def zero_nonzero_counts(tensor: torch.Tensor, zero_threshold: float = 0.0) -> Tuple[int, int]:
    arr = tensor.detach().cpu().numpy().ravel()
    return zero_nonzero_counts_from_array(arr, zero_threshold)


# -----------------------------
# Plot helpers
# -----------------------------
def hist_with_counts(
    tensor: torch.Tensor,
    title: str,
    out_path: str,
    bins: int = 100,
    zero_threshold: float = 0.0,
    show: bool = False,
):
    """
    Save a histogram for the given tensor and annotate with zero/non-zero counts.
    """
    ensure_dir(os.path.dirname(out_path))
    arr = tensor.detach().cpu().numpy().ravel()
    zeros, nonzeros = zero_nonzero_counts_from_array(arr, zero_threshold)

    plt.figure(figsize=(8, 5))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.3)

    txt = (
        f"total: {arr.size}\n"
        f"zero{' (|w|<=' + str(zero_threshold) + ')' if zero_threshold>0 else ''}: {zeros}\n"
        f"non-zero: {nonzeros}"
    )
    plt.gca().text(
        0.98, 0.95, txt,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Saved {out_path}")


def overall_histogram(state_dict: Dict[str, torch.Tensor], outdir: str, bins: int, zero_threshold: float, show: bool):
    """Plot a single histogram of all floating-point tensors combined (weights + biases), with counts."""
    ensure_dir(outdir)
    values = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
            values.append(v.detach().cpu().numpy().ravel())
    if not values:
        print("[WARN] No floating-point tensors found in state_dict.")
        return

    arr = np.concatenate(values, axis=0)
    zeros, nonzeros = zero_nonzero_counts_from_array(arr, zero_threshold)

    plt.figure(figsize=(9, 5))
    plt.hist(arr, bins=bins)
    plt.title("All weights/biases histogram (float)")
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.3)

    txt = (
        f"total: {arr.size}\n"
        f"zero{' (|w|<=' + str(zero_threshold) + ')' if zero_threshold>0 else ''}: {zeros}\n"
        f"non-zero: {nonzeros}"
    )
    plt.gca().text(
        0.98, 0.95, txt,
        transform=plt.gca().transAxes,
        ha="right", va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )

    out_path = os.path.join(outdir, "all_weights_hist.png")
    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"[OK] Saved {out_path}")


# -----------------------------
# Collect weight arrays
# -----------------------------
def collect_weight_arrays(state_dict: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """Collect flattened arrays for all .weight tensors (float)."""
    arrs = []
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        if not t.dtype.is_floating_point:
            continue
        if name.endswith(".weight_fake_quant.scale"):
            continue  # skip fake quant metadata
        if name.endswith(".weight"):
            arrs.append(t.detach().cpu().numpy().ravel())
    return arrs



# -----------------------------
# Quantization helpers (codes & dequant)
# -----------------------------
def qmax_for_bits(bits: int) -> int:
    return (1 << (bits - 1)) - 1  # 2^(b-1)-1, symmetric signed


def quantize_codes_per_tensor(w: torch.Tensor, bits: int) -> Tuple[np.ndarray, float]:
    """
    Return integer codes q (flattened) and scale for per-tensor symmetric quantization.
    q ∈ [-qmax, qmax], dequant approx: w ≈ q * scale
    """
    qmax = qmax_for_bits(bits)
    with torch.no_grad():
        max_abs = w.abs().max()
        if max_abs == 0:
            q = torch.zeros_like(w)
            return q.view(-1).cpu().numpy(), 1.0
        scale = (max_abs / qmax).item()
        q = torch.round(w / scale).clamp_(-qmax, qmax)
        return q.view(-1).cpu().numpy(), scale


def quantize_codes_per_channel(w: torch.Tensor, bits: int, ch_axis: int = 0) -> np.ndarray:
    """
    Return concatenated integer codes for per-output-channel symmetric quantization.
      Conv1d weight: (C_out, C_in, K)  -> ch_axis=0
      Linear weight: (C_out, C_in)     -> ch_axis=0
    We only need q codes for histogram; scale differs per channel.
    """
    if w.ndim not in (2, 3):
        # fallback to per-tensor for unexpected shapes
        q_codes, _ = quantize_codes_per_tensor(w, bits)
        return q_codes

    qmax = qmax_for_bits(bits)
    with torch.no_grad():
        # Move the channel axis to front (already 0, but keep generic)
        perm = list(range(w.ndim))
        perm[0], perm[ch_axis] = perm[ch_axis], perm[0]
        w_perm = w.permute(*perm).contiguous()   # (C, ...)
        C = w_perm.shape[0]
        rest = w_perm.view(C, -1)
        max_abs = rest.abs().max(dim=1).values   # (C,)

        # Avoid div-by-zero: channels with all-zeros get scale=1 and q=0
        scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)  # (C,)
        q = torch.round(rest / scale.unsqueeze(1)).clamp_(-qmax, qmax)
        return q.view(-1).cpu().numpy()


def quantize_state_dict_to_codes(
    state_dict: Dict[str, torch.Tensor],
    bits: int,
    scheme: str = "per_channel",
) -> np.ndarray:
    """
    Quantize all .weight tensors and return concatenated integer codes.
    """
    codes = []
    for name, t in state_dict.items():
        if not isinstance(t, torch.Tensor):
            continue
        if not t.dtype.is_floating_point:
            continue
        if not name.endswith(".weight"):
            continue  # Only quantize true weights, skip biases, fake_quant etc.

        w = t.detach()
        if scheme == "per_channel":
            q = quantize_codes_per_channel(w, bits=bits, ch_axis=0)
        else:
            q, _ = quantize_codes_per_tensor(w, bits=bits)
        codes.append(q)

    if not codes:
        print("[WARN] No weight tensors found for quantization.")
        return np.array([], dtype=np.float32)

    return np.concatenate(codes, axis=0)
# -----------------------------
# Multi-quant figure (integer-code hist for quantized variants)
# -----------------------------
def plot_multi_quant_histograms(
    state_dict: Dict[str, torch.Tensor],
    outdir: str,
    bits_list: List[int],
    scheme: str,
    bins_float: int,
    zero_threshold: float,
    show: bool,
):
    """
    Create a figure with subplots:
      - Float baseline: value-domain histogram (weights only, continuous).
      - Each quantized variant: integer-code histogram with bins at each quant level.
    Saves to weights_hist_multi_quant.png
    """
    ensure_dir(outdir)

    # Base float weights (weights only, no biases)
    float_arrs = collect_weight_arrays(state_dict)
    if not float_arrs:
        print("[WARN] No .weight tensors found; skipping multi-quant histogram.")
        return
    float_arr = np.concatenate(float_arrs, axis=0)
    f_zeros, f_nonzeros = zero_nonzero_counts_from_array(float_arr, zero_threshold)
    variants = [("float (value)", float_arr, f_zeros, f_nonzeros, None)]  # last field: bins (None for float)

    # Quantized variants (integer codes)
    for b in bits_list:
        qmax = qmax_for_bits(b)
        q_codes = quantize_state_dict_to_codes(state_dict, bits=b, scheme=scheme)  # integers in [-qmax, qmax]
        # Bin edges centered on each integer level:
        # e.g., for q in {-2,-1,0,1,2}, edges = [-2.5,-1.5,-0.5,0.5,1.5,2.5]
        edges = np.arange(-qmax - 0.5, qmax + 1.5, 1.0)
        # Zero count is exact number of codes == 0
        q_zeros = int(np.count_nonzero(q_codes == 0))
        q_nonzeros = int(q_codes.size - q_zeros)
        variants.append((f"{b}-bit ({scheme})", q_codes, q_zeros, q_nonzeros, edges))

    # Layout
    n = len(variants)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, data, zeros, nonzeros, edges) in zip(axes, variants):
        if edges is None:
            # float value-domain histogram
            ax.hist(data, bins=bins_float)
            ax.set_xlabel("value")
        else:
            # integer-code histogram with aligned bins
            ax.hist(data, bins=edges)
            ax.set_xlabel("quantized level (integer code)")
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)

        txt = (
            f"total: {data.size}\n"
            f"zero: {zeros}\n"
            f"non-zero: {nonzeros}"
        )
        ax.text(
            0.98, 0.95, txt,
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
        )

    # Hide extra axes
    for ax in axes[len(variants):]:
        ax.axis("off")

    fig.tight_layout()
    out_path = os.path.join(outdir, "weights_hist_multi_quant.png")
    plt.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_actual_quantized_weights(state_dict, outdir, show=False):
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(outdir, exist_ok=True)
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor) and tensor.dtype in (
            torch.qint8, torch.quint8, torch.int8, torch.uint8
        ):
            arr = tensor.int_repr().cpu().numpy().ravel()
            plt.figure(figsize=(8, 5))
            plt.hist(arr, bins=np.arange(arr.min()-0.5, arr.max()+1.5, 1), color='C0')
            plt.title(f"{name} (actual quantized codes)")
            plt.xlabel("quantized integer code")
            plt.ylabel("count")
            plt.grid(True, axis="y", alpha=0.3)
            out_path = os.path.join(outdir, f"{name.replace('.', '_')}_actual_quant_hist.png")
            plt.savefig(out_path, bbox_inches="tight")
            if show:
                plt.show()
            plt.close()
            print(f"[OK] Saved {out_path}")
            
# -----------------------------
# Per-parameter figures (weights & biases)
# -----------------------------
def visualize_param_histograms(
    state_dict: Dict[str, torch.Tensor],
    outdir: str,
    bins: int,
    zero_threshold: float,
    show: bool,
):
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not tensor.dtype.is_floating_point:
            continue

        if name.endswith(".weight"):
            fname = f"{name.replace('.', '_')}_hist.png"
            out_path = os.path.join(outdir, fname)
            title = f"[{name}] weight histogram"
            hist_with_counts(tensor, title=title, out_path=out_path, bins=bins,
                             zero_threshold=zero_threshold, show=show)

            # Matching bias (if present)
            bias_name = name[:-len(".weight")] + ".bias"
            bias_tensor: Optional[torch.Tensor] = state_dict.get(bias_name, None)
            if isinstance(bias_tensor, torch.Tensor) and bias_tensor.dtype.is_floating_point:
                bias_fname = f"{bias_name.replace('.', '_')}_hist.png"
                bias_out = os.path.join(outdir, bias_fname)
                bias_title = f"[{bias_name}] bias histogram"
                hist_with_counts(bias_tensor, title=bias_title, out_path=bias_out, bins=bins,
                                 zero_threshold=zero_threshold, show=show)


# -----------------------------
# Main
# -----------------------------
def parse_bits_list(bits_str: str) -> List[int]:
    bits = []
    for s in bits_str.split(","):
        s = s.strip().lower()
        if not s:
            continue
        if s in ("f", "fp", "float", "fp32", "32"):
            # Float baseline is implicit in multi-quant figure
            continue
        b = int(s)
        if b < 2 or b > 16:
            raise ValueError("Bitwidths must be integers in [2..16] (float baseline is implicit).")
        bits.append(b)
    return bits


def main():
    parser = argparse.ArgumentParser(description="Visualize model weights with histograms and multi-quant (aligned bins).")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt (state_dict or checkpoint).")
    parser.add_argument("--outdir", type=str, default="plots/weights_viz", help="Directory to save figures.")
    parser.add_argument("--bins", type=int, default=100, help="Bins for float histograms (overall & per-parameter).")
    parser.add_argument("--zero-threshold", type=float, default=0.0,
                        help="Treat |w| <= threshold as zero for float hist counts (not used for integer-code hist).")
    parser.add_argument("--show", action="store_true", help="Show figures interactively.")
    parser.add_argument("--quant-bits", type=str, default="8,4,3",
                        help="Comma-separated bit widths for multi-quant figure (float is implicit).")
    parser.add_argument("--quant-scheme", type=str, choices=["per_channel", "per_tensor"], default="per_tensor",
                        help="Weight quantization scheme for the multi-quant figure.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    sd = load_state_dict(args.weights)
    print("[DEBUG] All state_dict keys:", list(sd.keys()))

    # --- Print loaded state_dict keys and tensor shapes ---
    print("\n[STATE_DICT CONTENTS]")
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:<40} {tuple(v.shape)}")

    # --- Detect quantized weights ---
    has_quantized = any(
        isinstance(t, torch.Tensor) and t.dtype in (torch.qint8, torch.quint8, torch.int8, torch.uint8)
        for t in sd.values()
    )

    if has_quantized:
        print("[INFO] Detected quantized weights. Plotting actual quantized histograms only.")
        plot_actual_quantized_weights(sd, args.outdir, show=args.show)
    else:
        print("[INFO] Detected floating-point weights. Plotting FP and simulated quantization histograms.")
        overall_histogram(sd, args.outdir, bins=args.bins, zero_threshold=args.zero_threshold, show=args.show)
        visualize_param_histograms(sd, args.outdir, bins=args.bins, zero_threshold=args.zero_threshold, show=args.show)
        bits_list = parse_bits_list(args.quant_bits)
        if bits_list:
            plot_multi_quant_histograms(
                state_dict=sd,
                outdir=args.outdir,
                bits_list=bits_list,
                scheme=args.quant_scheme,
                bins_float=args.bins,
                zero_threshold=args.zero_threshold,
                show=args.show,
            )
        else:
            print("[INFO] No quant bits requested; skipped multi-quant figure.")

    print(f"[DONE] Outputs written to: {args.outdir}")

if __name__ == "__main__":
    main()
