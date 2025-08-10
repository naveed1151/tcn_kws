import argparse
import os
import re
import numpy as np
import torch

def sanitize_name(name: str) -> str:
    # Make a safe filename from a state_dict key
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def save_tensor_csv(path: str, tensor: torch.Tensor, flatten: bool) -> None:
    arr = tensor.detach().cpu().numpy()
    if flatten:
        arr = arr.reshape(-1, 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, arr, delimiter=",", fmt="%.8g")

def summarize_tensor(tensor: torch.Tensor) -> dict:
    t = tensor.detach().cpu()
    t_float = t.float() if not t.dtype.is_floating_point else t
    arr = t_float.numpy().reshape(-1)
    # Stats
    mn = float(arr.min()) if arr.size else 0.0
    mx = float(arr.max()) if arr.size else 0.0
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std()) if arr.size else 0.0
    # Unique count (can be small for quantized weights)
    uniq = int(np.unique(arr).size) if arr.size else 0
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "min": mn,
        "max": mx,
        "mean": mean,
        "std": std,
        "unique_values": uniq,
    }

def load_state_dict(weights_path: str) -> dict:
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt  # assume plain state_dict

def main():
    ap = argparse.ArgumentParser(description="Export model weights to CSV files and write a summary CSV.")
    ap.add_argument("--weights", required=True, help="Path to checkpoint (.pt/.pth)")
    ap.add_argument("--outdir", required=True, help="Output directory for CSV files")
    ap.add_argument("--flatten", action="store_true", help="Flatten tensors to a single column in CSV")
    args = ap.parse_args()

    sd = load_state_dict(args.weights)
    os.makedirs(args.outdir, exist_ok=True)

    summary_rows = []
    for name, tensor in sd.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not tensor.dtype.is_floating_point:
            # Skip non-float tensors by default (e.g., buffers)
            continue

        safe = sanitize_name(name)
        csv_path = os.path.join(args.outdir, f"{safe}.csv")
        save_tensor_csv(csv_path, tensor, args.flatten)

        s = summarize_tensor(tensor)
        summary_rows.append({
            "name": name,
            "shape": "x".join(map(str, s["shape"])),
            "dtype": s["dtype"],
            "numel": s["numel"],
            "min": s["min"],
            "max": s["max"],
            "mean": s["mean"],
            "std": s["std"],
            "unique_values": s["unique_values"],
            "csv_path": csv_path,
        })
        print(f"[OK] Saved {name} -> {csv_path}  (unique={s['unique_values']})")

    # Write a summary CSV
    summary_path = os.path.join(args.outdir, "weights_summary.csv")
    if summary_rows:
        # Header
        headers = ["name", "shape", "dtype", "numel", "min", "max", "mean", "std", "unique_values", "csv_path"]
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for r in summary_rows:
                f.write(",".join([
                    r["name"],
                    r["shape"],
                    r["dtype"],
                    str(r["numel"]),
                    f"{r['min']:.8g}",
                    f"{r['max']:.8g}",
                    f"{r['mean']:.8g}",
                    f"{r['std']:.8g}",
                    str(r["unique_values"]),
                    r["csv_path"].replace(",", "_"),
                ]) + "\n")
        print(f"[DONE] Summary written to {summary_path}")
    else:
        print("[WARN] No float tensors found to export.")

if __name__ == "__main__":
    main()