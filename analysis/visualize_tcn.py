import os
import argparse
import torch
import torch.nn as nn

from train.utils import load_config, build_model_from_cfg

try:
    from torchviz import make_dot
except ImportError as e:
    raise SystemExit(
        "torchviz is not installed. Install with:\n"
        "  pip install torchviz\n"
        "Also install Graphviz and ensure it is on PATH:\n"
        "  Windows (Chocolatey): choco install graphviz\n"
        "  Or download from https://graphviz.org/download/"
    ) from e


def _infer_num_classes(cfg: dict) -> int:
    task = cfg.get("task", {})
    t = task.get("type", "multiclass")
    if t == "binary":
        return 1
    n = len(task.get("class_list", []))
    if task.get("include_unknown", False):
        n += 1
    if task.get("include_background", False):
        n += 1
    if n <= 0:
        raise ValueError("Could not infer num_classes from config.task; provide class_list or set binary task.")
    return n


def _infer_mfcc_shape(cfg: dict) -> tuple[int, int]:
    # Returns (C, T) expected by build_model_from_cfg
    mfcc = cfg["data"]["mfcc"]
    C = int(mfcc["n_mfcc"])
    hop = float(mfcc["hop_length_s"])
    dur = float(mfcc["fixed_duration_s"])
    if hop <= 0 or dur <= 0:
        raise ValueError("mfcc.hop_length_s and mfcc.fixed_duration_s must be > 0")
    T = int(round(dur / hop))
    return C, T


def _select_device(cfg: dict) -> torch.device:
    pref = str(cfg.get("train", {}).get("device", "auto")).lower()
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Visualize DilatedTCN graph with torchviz")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config (e.g., config/base.yaml)")
    parser.add_argument("--output", "-o", type=str, default="tcn_graph", help="Output path base name (no extension)")
    parser.add_argument("--format", "-f", type=str, default="png", choices=["png", "pdf", "svg"], help="Graph format")
    parser.add_argument("--channels", type=int, default=None, help="Override input channels (MFCCs)")
    parser.add_argument("--timesteps", type=int, default=None, help="Override input timesteps")
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI for PNG/PDF rendering")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _select_device(cfg)
    num_classes = _infer_num_classes(cfg)

    # Infer (C, T) from config unless overridden
    if args.channels is not None and args.timesteps is not None:
        C, T = int(args.channels), int(args.timesteps)
    else:
        C, T = _infer_mfcc_shape(cfg)

    # Build model using the shared constructor (expects sample_input shaped (C, T))
    sample_input_for_build = torch.zeros(C, T)
    model = build_model_from_cfg(cfg, sample_input_for_build, num_classes).to(device)
    model.eval()  # visualization doesn't need training mode

    # Create a real input for forward: (B, C, T)
    x = torch.randn(1, C, T, device=device, requires_grad=True)
    y = model(x)

    # Make graph
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = args.format
    # Increase raster resolution (PNG/PDF). For infinite resolution, use --format svg.
    dot.graph_attr.update({"dpi": str(args.dpi)})

    # Save under plots directory from config (fallback to ./plots)
    plots_dir = cfg.get("output", {}).get("plots_dir", os.path.join(os.getcwd(), "plots"))
    # args.output is a base name (no extension), allow subdirs
    output_base = os.path.join(plots_dir, args.output)
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    out_path = dot.render(output_base, cleanup=True)
    print(f"[OK] Graph saved to {out_path}")
    print(f"[INFO] Input shape: (B=1, C={C}, T={T}), Output shape: {tuple(y.shape)}")
    print(f"[INFO] Device: {device.type}, Num classes: {num_classes}")


if __name__ == "__main__":
    main()