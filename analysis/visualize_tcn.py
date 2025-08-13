"""
Simple block diagram (rectangles + arrows + residual) for DilatedTCN.
Outputs an SVG under plots/<outdir>/diagram_blocks.svg
Heuristic: treats each high‑level temporal block as one box, plus input/output.
Residual edges drawn dashed (blue). If auto detection fails, use --blocks N.
"""
import argparse, os, torch
from train.train import load_config
from train.utils import build_model_from_cfg, load_state_dict_forgiving
from data_loader.utils import make_datasets, get_num_classes

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

def infer_blocks(model):
    """
    Return ordered list of block names (heuristic).
    Looks for modules whose name contains: 'block' or 'dblock' or 'tblock'.
    Falls back to sequential indices if none found.
    """
    candidates = []
    for name, mod in model.named_children():
        lname = name.lower()
        if any(k in lname for k in ("block", "dblock", "tblock")):
            candidates.append(name)
    # Dive one level deeper if top-level children are few
    if len(candidates) <= 1:
        for name, mod in model.named_modules():
            if name == "":
                continue
            lname = name.lower()
            if any(k in lname for k in ("block", "dblock", "tblock")) and "." not in name:
                candidates.append(name)
    # Remove dupes preserving order
    seen, ordered = set(), []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered

def build_sample(cfg):
    # Minimal sample using MFCC/time dims implicitly taken from dataset
    # Use train loader first batch
    train_loader, _, _ = make_datasets(cfg, which="train", batch_size=1)
    batch = next(iter(train_loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
    return x[0]  # (C,T)

def load_model(cfg_path, weights_path, device):
    cfg = load_config(cfg_path)
    sample = build_sample(cfg)
    num_classes = get_num_classes(cfg)
    model = build_model_from_cfg(cfg, sample, num_classes)
    model = load_state_dict_forgiving(model, weights_path, device)
    model.to(device).eval()
    return model, sample.unsqueeze(0)

def make_block_diagram(model, out_svg, forced_blocks=None):
    if Digraph is None:
        print("Install graphviz + python bindings: pip install graphviz")
        return
    if forced_blocks:
        blocks = [f"Block{i+1}" for i in range(forced_blocks)]
    else:
        inferred = infer_blocks(model)
        if not inferred:
            # Fallback: try to read attribute like model.blocks / model.layers length
            n = 0
            for attr in ("blocks", "layers"):
                if hasattr(model, attr):
                    seq = getattr(model, attr)
                    try:
                        n = len(seq)
                    except:
                        pass
                    if n:
                        break
            if n == 0:
                n = 5  # default guess
            blocks = [f"Block{i+1}" for i in range(n)]
        else:
            blocks = inferred

    g = Digraph("TCN", format="svg")
    # Top-to-bottom layout
    g.attr(rankdir="TB", splines="spline", nodesep="0.5", ranksep="0.75")
    g.attr("node", shape="record", style="filled", fillcolor="#f5f8fc",
           color="#2d4863", fontname="Segoe UI", fontsize="12")
    g.attr("edge", color="#444444", arrowsize="0.7")

    g.node("Input", label="{ Input | (C,T) }", fillcolor="#e0eef9")
    prev = "Input"

    # Add each block
    for i, bname in enumerate(blocks):
        label = f"{{ {bname} | dilated conv(s) }}"
        g.node(bname, label=label, fillcolor="#d9ecff")
        g.edge(prev, bname)
        # Residual edge (dashed) from first block input to this block output (if beyond first)
        if i > 0:
            g.edge("Input" if i == 1 else blocks[i-1],
                   bname, style="dashed", color="#1d6fb8", arrowhead="none", constraint="false")
        prev = bname

    g.node("Head", label="{ Classifier | GAP + Linear }", fillcolor="#ffeccb")
    g.edge(prev, "Head")
    g.node("Output", label="{ Output | logits }", fillcolor="#e5ffe3")
    g.edge("Head", "Output")

    g.render(out_svg, cleanup=True)
    print(f"[OK] Block diagram: {out_svg}.svg")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--outdir", default="model_viz", help="Subfolder under plots/")
    ap.add_argument("--blocks", type=int, default=None, help="Force number of blocks (override auto)")
    return ap.parse_args()

def main():
    args = parse_args()
    out_base = os.path.join("plots", args.outdir.lstrip("/\\"))
    os.makedirs(out_base, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sample = load_model(args.config, args.weights, device)
    make_block_diagram(model, os.path.join(out_base, "diagram_blocks"), forced_blocks=args.blocks)

if __name__ == "__main__":
    main()