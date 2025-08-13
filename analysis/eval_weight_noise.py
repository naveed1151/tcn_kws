import os, argparse
from typing import Dict, Tuple, List
import torch, torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from train.utils import (
    load_config, build_model_from_cfg, load_state_dict_forgiving,
)
from data_loader.utils import (
    get_num_classes, make_datasets
)
from train.evaluate import evaluate_model

# -----------------------
# (Optional) quant helpers just to compute scales if requested
# -----------------------
def qrange(bits: int, symmetric: bool) -> Tuple[int, int]:
    if symmetric:
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
    else:
        qmax = (1 << bits) - 1
        qmin = 0
    return qmin, qmax

@torch.no_grad()
def compute_scale(
    w: torch.Tensor, bits: int, symmetric: bool, per_channel: bool
) -> torch.Tensor:
    """
    Return scale(s) used for a hypothetical quantization; DO NOT quantize.
    Used only to define a noise magnitude if noise_mode references 'quant'.
    """
    qmin, qmax = qrange(bits, symmetric)
    wf = w.detach().float()
    if per_channel and wf.dim() >= 2:
        flat = wf.view(wf.shape[0], -1)
        if symmetric:
            max_abs = flat.abs().max(dim=1).values
            scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / ((1 << (bits - 1)) - 1))
        else:
            w_min = flat.min(dim=1).values
            w_max = flat.max(dim=1).values
            rng = torch.clamp(w_max - w_min, min=1e-8)
            scale = rng / (qmax - qmin)
    else:
        if symmetric:
            max_abs = wf.abs().max()
            scale = torch.tensor(1.0) if max_abs == 0 else max_abs / ((1 << (bits - 1)) - 1)
        else:
            w_min = wf.min(); w_max = wf.max()
            rng = torch.clamp(w_max - w_min, min=1e-8)
            scale = rng / (qmax - qmin)
    return scale.detach()

def iter_modules(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)) and hasattr(m, "weight"):
            yield name, m

# -----------------------
# Noise injection (FP domain only)
# -----------------------
@torch.no_grad()
def apply_float_noise(
    model: nn.Module,
    noise_level: float,
    noise_mode: str,
    per_channel: bool,
    quant_bits: int,
    quant_symmetric: bool,
    use_quant_scale: bool,
    seed: int
) -> Dict[str, torch.Tensor]:
    """
    Add Gaussian noise directly to FP weights (no quant/dequant overwrite).
    noise_mode:
      range : sigma = level * (max_abs(weight) per-channel or global)
      std   : sigma = level * (std(weight) per-channel or global)
      abs   : sigma = level (constant)
      qscale: sigma = level * (quant scale)   (requires --use-quant-scale)
    """
    torch.manual_seed(seed)
    backup = {}
    for name, m in iter_modules(model):
        w = m.weight.data
        backup[name] = w.clone()

        if noise_level == 0:
            continue

        if per_channel and w.dim() >= 2:
            flat = w.view(w.shape[0], -1)
            if noise_mode == "range":
                sigma_vec = noise_level * flat.abs().max(dim=1).values
            elif noise_mode == "std":
                sigma_vec = noise_level * flat.std(dim=1)
            elif noise_mode == "abs":
                sigma_vec = torch.full((w.shape[0],), noise_level, device=w.device)
            elif noise_mode == "qscale":
                if not use_quant_scale:
                    raise ValueError("qscale mode requires --use-quant-scale")
                scale = compute_scale(w, quant_bits, quant_symmetric, True)
                sigma_vec = noise_level * scale
            else:
                raise ValueError(noise_mode)

            shape = [1]*w.dim(); shape[0] = -1
            noise = torch.randn_like(w) * sigma_vec.view(shape)
        else:
            if noise_mode == "range":
                sigma = noise_level * w.abs().max()
            elif noise_mode == "std":
                sigma = noise_level * w.std()
            elif noise_mode == "abs":
                sigma = torch.tensor(noise_level, device=w.device)
            elif noise_mode == "qscale":
                if not use_quant_scale:
                    raise ValueError("qscale mode requires --use-quant-scale")
                scale = compute_scale(w, quant_bits, quant_symmetric, False)
                sigma = noise_level * scale
            else:
                raise ValueError(noise_mode)
            noise = torch.randn_like(w) * sigma
        m.weight.add_(noise)

    return backup

@torch.no_grad()
def restore_weights(model: nn.Module, backup: Dict[str, torch.Tensor]):
    for name, m in iter_modules(model):
        if name in backup:
            m.weight.data.copy_(backup[name])

# -----------------------
# Eval / plotting
# -----------------------
def run_eval(model, loader, device, task_type, num_classes, criterion, threshold):
    out = evaluate_model(model, loader, device, task_type, num_classes, criterion,
                         threshold=threshold, pin_memory=True)
    return {"loss": float(out[0]), "acc": float(out[1]), "prec": float(out[2]),
            "rec": float(out[3]), "f1": float(out[4])}

def plot_vs_noise(results, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    xs = sorted(results.keys())
    for mkey in ["acc", "prec", "rec", "f1"]:
        plt.plot(xs, [results[x][mkey] for x in xs], marker="o", label=mkey.upper())
    plt.xlabel("Noise level"); plt.ylabel("Metric"); plt.ylim(0, 1); plt.grid(True, alpha=0.3)
    plt.legend(); path = os.path.join(outdir, f"{prefix}_metrics_vs_noise.png")
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close(); print(f"[OK] {path}")
    plt.figure()
    plt.plot(xs, [results[x]["loss"] for x in xs], marker="o", color="#d95f02")
    plt.xlabel("Noise level"); plt.ylabel("Loss"); plt.grid(True, alpha=0.3)
    path2 = os.path.join(outdir, f"{prefix}_loss_vs_noise.png")
    plt.tight_layout(); plt.savefig(path2, dpi=300); plt.close(); print(f"[OK] {path2}")

def save_csv(results, outdir, prefix):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{prefix}_results.csv")
    with open(path, "w") as f:
        f.write("noise,loss,acc,prec,rec,f1\n")
        for n in sorted(results.keys()):
            r = results[n]
            f.write(f"{n},{r['loss']:.6f},{r['acc']:.6f},{r['prec']:.6f},{r['rec']:.6f},{r['f1']:.6f}\n")
    print(f"[OK] {path}")

# -----------------------
# Args / main
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser(description="FP32 weight noise robustness (no inplace quant).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--noise", default="0,0.01,0.02,0.05")
    ap.add_argument("--noise-mode", choices=["range","std","abs","qscale"], default="std")
    ap.add_argument("--use-quant-scale", action="store_true",
                    help="When noise-mode=qscale, derive sigma from simulated quant scales.")
    ap.add_argument("--bits", type=int, default=8, help="For qscale mode scale derivation.")
    ap.add_argument("--symmetric", action="store_true", help="For qscale mode scale derivation.")
    ap.add_argument("--per-channel", action="store_true", help="Per-channel stats for range/std/qscale.")
    ap.add_argument("--split", choices=["val","test"], default="val")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="plots/noise_eval_fp_only")
    return ap.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = load_config(args.config)
    task_type = cfg["task"]["type"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_datasets(
        cfg, which="all", batch_size=cfg["train"].get("batch_size", 32)
    )
    eval_loader = val_loader if args.split == "val" else test_loader

    sample_batch = next(iter(train_loader))
    sample_x = sample_batch[0] if isinstance(sample_batch, (list, tuple)) else sample_batch["x"]
    sample_x = sample_x[0]
    num_classes = get_num_classes(cfg)
    model = build_model_from_cfg(cfg, sample_x, num_classes)
    model = load_state_dict_forgiving(model, args.weights, device).to(device).eval()

    criterion = nn.CrossEntropyLoss() if task_type == "multiclass" else nn.BCEWithLogitsLoss()

    noise_levels = [float(x) for x in args.noise.split(",") if x.strip()]
    if 0.0 not in noise_levels: noise_levels = [0.0] + noise_levels

    results = {}
    for nl in noise_levels:
        backup = apply_float_noise(
            model,
            noise_level=nl,
            noise_mode=args.noise_mode,
            per_channel=args.per_channel,
            quant_bits=args.bits,
            quant_symmetric=args.symmetric,
            use_quant_scale=args.use_quant_scale,
            seed=args.seed
        )
        metrics = run_eval(model, eval_loader, device, task_type, num_classes, criterion, args.threshold)
        results[nl] = metrics
        print(f"[INFO] noise={nl:.4g} loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
              f"P={metrics['prec']:.4f} R={metrics['rec']:.4f} F1={metrics['f1']:.4f}")
        restore_weights(model, backup)

    prefix = f"fp_{args.noise_mode}_{'pc' if args.per_channel else 'pt'}"
    plot_vs_noise(results, args.outdir, prefix)
    save_csv(results, args.outdir, prefix)

if __name__ == "__main__":
    main()