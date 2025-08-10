import argparse, os, numpy as np, torch

def load_sd(path):
    obj = torch.load(path, map_location="cpu")
    return obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj

@torch.no_grad()
def quantize_sd_to_npz(sd, bits=5, per_channel=True, symmetric=True):
    qmax = (1 << (bits - 1)) - 1 if symmetric else (1 << bits) - 1
    qmin = -qmax if symmetric else 0
    names, q_list, scale_list = [], [], []
    for name, w in sd.items():
        if not isinstance(w, torch.Tensor): continue
        if "weight" not in name: continue
        if w.ndim < 2: continue  # skip biases/norms
        w = w.float().cpu()
        if per_channel:
            oc = w.shape[0]
            w2 = w.view(oc, -1)
            max_abs = w2.abs().max(dim=1).values if symmetric else w2.max(dim=1).values
            scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
            q = torch.round(w2 / scale[:, None]).clamp(qmin, qmax).to(torch.int16)  # store as int16
            names.append(name)
            q_list.append(q.numpy())
            scale_list.append(scale.numpy())
        else:
            max_abs = w.abs().max() if symmetric else w.max()
            if max_abs == 0: continue
            scale = (max_abs / qmax).item()
            q = torch.round(w / scale).clamp(qmin, qmax).to(torch.int16)
            names.append(name)
            q_list.append(q.numpy())
            scale_list.append(np.array(scale, dtype=np.float32))
    return names, q_list, scale_list

def main():
    ap = argparse.ArgumentParser(description="Export integer codes (q) and scales to NPZ from a float checkpoint.")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=5)
    ap.add_argument("--per-channel", action="store_true", help="Per-channel (default) vs per-tensor")
    ap.add_argument("--asymmetric", action="store_true", help="Use unsigned/asymmetric quant (default: symmetric)")
    args = ap.parse_args()

    sd = load_sd(args.weights)
    names, q_list, scale_list = quantize_sd_to_npz(
        sd, bits=args.bits, per_channel=args.per_channel or True, symmetric=not args.asymmetric
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Store as object arrays to allow varying shapes
    np.savez(
        args.out,
        names=np.array(names, dtype=object),
        q_list=np.array(q_list, dtype=object),
        scale_list=np.array(scale_list, dtype=object),
        bits=np.array(args.bits),
        symmetric=np.array(not args.asymmetric),
        per_channel=np.array(args.per_channel or True),
    )
    print(f"[OK] Wrote {len(names)} tensors to {args.out}")

if __name__ == "__main__":
    main()