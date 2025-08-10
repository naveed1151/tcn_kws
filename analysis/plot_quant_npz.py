import argparse, os, numpy as np
import matplotlib.pyplot as plt

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def try_extract_items(npz):
    # Supported layouts:
    # 1) keys like "tcn_layers.0.weight.q" and "...scale"
    # 2) arrays: names (list of str), q_list (object array), scale_list (object array)
    if "names" in npz and "q_list" in npz:
        names = [str(n) for n in npz["names"].tolist()]
        q_list = [np.array(a) for a in npz["q_list"]]
        scales = [None]*len(q_list)
        if "scale_list" in npz:
            scales = [np.array(a) for a in npz["scale_list"]]
        return list(zip(names, q_list, scales))

    # Fallback: group by basename before suffix ".q" / ".scale"
    groups = {}
    for k in npz.files:
        if k.endswith(".q"):
            base = k[:-2]
            groups.setdefault(base, {})["q"] = np.array(npz[k])
        elif k.endswith(".scale") or k.endswith(".scales"):
            base = k[: -len(".scale")] if k.endswith(".scale") else k[: -len(".scales")]
            groups.setdefault(base, {})["scale"] = np.array(npz[k])
    items = []
    for base, d in groups.items():
        if "q" in d:
            items.append((base, d["q"], d.get("scale", None)))
    return items

def plot_hist_int_codes(name, q, outdir, show=False, qmin=None, qmax=None, annotate_counts=False):
    q = np.asarray(q)
    flat = q.reshape(-1)
    low = flat.min() if qmin is None else qmin
    high = flat.max() if qmax is None else qmax
    # Bin centers at every integer from low..high
    bins = np.arange(low - 0.5, high + 1.5, 1.0)  # width = 1
    plt.figure(figsize=(7,4))
    plt.hist(flat, bins=bins, edgecolor="k", alpha=0.8)
    plt.title(f"Integer code usage: {name}")
    plt.xlabel("q (integer code)")
    plt.ylabel("count")
    plt.grid(True, alpha=0.3)
    # --- annotate counts for ALL_PARAMS or when requested ---
    if annotate_counts or name == "ALL_PARAMS":
        total = flat.size
        zeros = int((flat == 0).sum())
        nonzeros = total - zeros
        txt = f"total = {total:,}\nnon-zero = {nonzeros:,}\nzeros = {zeros:,}"
        ax = plt.gca()
        ax.text(
            0.98, 0.95, txt,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )
    ensure_dir(outdir)
    path = os.path.join(outdir, f"{name.replace('.', '_').replace('/', '_')}_q_hist.png")
    plt.tight_layout(); plt.savefig(path); 
    if show: plt.show()
    plt.close()
    return path

def main():
    ap = argparse.ArgumentParser(description="Plot histograms of quantized integer codes from NPZ export.")
    ap.add_argument("--npz", required=True, help="Path to exported quant .npz")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--bits", type=int, default=None, help="Optional: number of bits (only used to set q range)")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    npz = np.load(args.npz, allow_pickle=True)
    items = try_extract_items(npz)
    if not items:
        print("[ERROR] Could not find any q arrays in NPZ. Keys:", list(npz.files))
        return

    # Optional fixed q range from bits
    qmin = qmax = None
    if args.bits:
        qmax = (1 << (args.bits - 1)) - 1
        qmin = -qmax

    all_q = []
    for name, q, scale in items:
        if q is None:
            continue
        all_q.append(q.reshape(-1))
        out = plot_hist_int_codes(name, q, args.outdir, show=args.show, qmin=qmin, qmax=qmax)
        uniq = np.unique(q).size
        print(f"[OK] {name}: saved {out} | unique integer codes used = {uniq}")

    # Combined histogram
    if all_q:
        flat_all = np.concatenate(all_q, axis=0)
        out = plot_hist_int_codes("ALL_PARAMS", flat_all, args.outdir, show=args.show, qmin=qmin, qmax=qmax, annotate_counts=True)
        uniq = np.unique(flat_all).size
        print(f"[OK] Combined: saved {out} | unique integer codes used = {uniq}")

if __name__ == "__main__":
    main()