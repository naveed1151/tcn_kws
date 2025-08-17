import os
import sys
import argparse
import torch
from typing import Tuple

# Ensure repo root on sys.path so "train.utils" imports work when run as a module/script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.utils import load_config, build_model_from_cfg  # noqa: E402


def _infer_num_classes(cfg: dict) -> int:
    task = cfg.get("task", {})
    ttype = task.get("type", "multiclass")
    if ttype == "binary":
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
    # Returns (C, T)
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


def _count_params(module: torch.nn.Module) -> tuple[int, int]:
    # Own parameters only (no recursion) for per-layer stats
    total = sum(p.numel() for p in module.parameters(recurse=False))
    trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
    return trainable, total


def _sample_shape_from_dataset(cfg: dict) -> Tuple[int, int]:
    try:
        from data_loader.utils import make_datasets
        loader, _, _ = make_datasets(cfg, which="train", batch_size=1)
        batch = next(iter(loader))
        x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
        # Expect shape (B,C,T)
        if x.ndim == 3:
            return int(x.shape[1]), int(x.shape[2])
    except Exception as e:
        print(f"[WARN] Could not get sample from dataset ({e}); falling back to MFCC heuristic.")
    return _infer_mfcc_shape(cfg)


def main():
    parser = argparse.ArgumentParser(description="Inspect DilatedTCN: parameter counts and per-layer stats")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config (e.g., config/base.yaml)")
    parser.add_argument("--channels", type=int, default=None, help="Override input channels (MFCCs)")
    parser.add_argument("--timesteps", type=int, default=None, help="Override input timesteps")
    parser.add_argument("--use-dataset", action="store_true",
                        help="Derive (C,T) from first training sample instead of config MFCC heuristics (avoids residual length mismatch)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = _select_device(cfg)
    num_classes = _infer_num_classes(cfg)

    # Infer input shape (C, T)
    if args.channels is not None and args.timesteps is not None:
        C, T = int(args.channels), int(args.timesteps)
    else:
        if args.use_dataset:
            C, T = _sample_shape_from_dataset(cfg)
        else:
            C, T = _infer_mfcc_shape(cfg)

    # Build model using shared factory (expects sample_input shaped (C, T))
    sample_input_for_build = torch.zeros(C, T)
    model = build_model_from_cfg(cfg, sample_input_for_build, num_classes).to(device)
    model.eval()

    # --- Optional safety patch: auto-trim input time steps on residual mismatch (inspection only) ---
    # Some residual blocks shrink temporal length (causal convs without full padding) so addition fails.
    # We wrap forward to retry with progressively trimmed T until it fits, purely for shape inspection.
    orig_forward = model.forward

    def _forward_autotrim(inp: torch.Tensor):
        try:
            return orig_forward(inp)
        except RuntimeError as e:
            if "must match the size of tensor" in str(e) and "dimension 2" in str(e):
                T0 = inp.shape[-1]
                for trim in range(1, 9):  # try trimming up to 8 frames
                    if T0 - trim < 4:
                        break
                    try:
                        out = orig_forward(inp[..., :T0 - trim])
                        if trim > 0:
                            print(f"[INFO] Auto-trimmed timesteps {T0}->{T0 - trim} to satisfy residual add.")
                        return out
                    except RuntimeError as ee:
                        if "must match the size of tensor" in str(ee):
                            continue
                        raise
            raise

    model.forward = _forward_autotrim  # patch

    # Prepare a sample batch for forward
    x = torch.randn(1, C, T, device=device)

    # Collect per-layer output shapes via forward hooks
    layer_outputs: dict[str, str] = {}
    hooks = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            # Handle Tensor or tuple/list of Tensors
            if isinstance(out, torch.Tensor):
                shape = tuple(out.shape)
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                # Take the first tensor-like output for reporting
                first = next((o for o in out if isinstance(o, torch.Tensor)), None)
                shape = tuple(first.shape) if first is not None else ()
            else:
                shape = ()
            layer_outputs[name] = str(shape)
        return hook

    # Register hooks on all modules (skip the root "")
    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    # Forward pass (no grads needed)
    def _try_forward(inp: torch.Tensor) -> bool:
        try:
            with torch.no_grad():
                _ = model(inp)
            return True
        except RuntimeError as e:
            if "must match the size of tensor" in str(e) and "dimension 2" in str(e):
                return False
            raise

    ok = _try_forward(x)
    if not ok:
        # Heuristic: trim input length until forward succeeds (inspection only)
        orig_T = x.shape[-1]
        for new_T in range(orig_T - 1, max(4, orig_T - 64), -1):
            if _try_forward(x[..., :new_T]):
                T = new_T
                x = x[..., :new_T]
                print(f"[INFO] Adjusted timesteps from {orig_T} -> {T} to satisfy residual shape for inspection.")
                break
        else:
            print("[ERROR] Could not auto-fix residual length mismatch; try --use-dataset or --timesteps.")
            return

    # Compute totals
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"[MODEL] Device={device.type}  Input(B,C,T)=(1,{C},{T})  NumClasses={num_classes}")
    print(f"[PARAMS] total={total_params:,}  trainable={total_trainable:,}")
    # Receptive field for DilatedTCN:
    # Each residual block has two convs with dilation d=2^i, each adds (k-1)*d to RF.
    # RF = 1 + 2*(k-1) * (2^B - 1)
    k = int(cfg["model"]["kernel_size"])
    B = int(cfg["model"]["num_blocks"])
    rf = 1 + 2 * (k - 1) * ((2 ** B) - 1)
    print(f"[RECEPTIVE_FIELD] frames={rf}")
    print("\n[Per-layer stats]")
    print(f"{'name':40s} {'type':26s} {'trainable/total':18s} {'out_shape'}")

    # Print in model traversal order
    for name, module in model.named_modules():
        if name == "":
            continue
        tr, tot = _count_params(module)
        mtype = module.__class__.__name__
        out_shape = layer_outputs.get(name, "-")
        print(f"{name:40s} {mtype:26s} {f'{tr}/{tot}':18s} {out_shape}")

    # Cleanup hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()