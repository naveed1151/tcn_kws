import os
import argparse
from copy import deepcopy
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.model import DilatedTCN
from data_loader.mfcc_dataset import MFCCDataset
from data_loader.multiclass_dataset import MultiClassDataset
from data_loader.binary_dataset import BinaryClassDataset
from data_loader.utils import stratified_train_val_test_indices


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


def make_dataloader(cfg, split="val", batch_size=128, seed=0):
    pre_dir = cfg["data"]["preprocessed_dir"]
    base = MFCCDataset(pre_dir)
    label_to_index = base.label_to_index
    index_to_label = {v: k for k, v in label_to_index.items()}

    task_type = cfg["task"]["type"]
    if task_type == "binary":
        target_word = cfg["data"]["target_word"]
        dataset = BinaryClassDataset(
            base, target_word, index_to_label,
            downsample_ratio=cfg["data"].get("neg_downsample_ratio", None),
            seed=seed
        )
    else:
        dataset = MultiClassDataset(
            base,
            class_list=cfg["data"].get("class_list"),
            index_to_label=index_to_label,
            label_to_index=label_to_index,
            include_unknown=cfg["data"].get("include_unknown", False),
            include_background=cfg["data"].get("include_background", False),
            background_label=cfg["data"].get("background_label", "_background_noise_")
        )

    labels = [label for _, label in dataset]
    val_frac = float(cfg["data"].get("val_split", 0.20))
    test_frac = float(cfg["data"].get("test_split", 0.10))
    train_idx, val_idx, test_idx = stratified_train_val_test_indices(labels, val_frac, test_frac, seed=seed)
    subset_map = {
        "train": torch.utils.data.Subset(dataset, train_idx),
        "val": torch.utils.data.Subset(dataset, val_idx),
        "full": dataset
    }
    dl = torch.utils.data.DataLoader(subset_map[split], batch_size=batch_size, shuffle=False)
    return dl, dataset.num_classes


def build_model(cfg, num_classes, device):
    probe = MFCCDataset(cfg["data"]["preprocessed_dir"])[0][0]
    in_ch, seq_len = probe.shape
    k = int(cfg["model"]["kernel_size"])
    h = int(cfg["model"]["hidden_channels"])
    d = float(cfg["model"]["dropout"])

    def calc_layers(seq_len, k, base=2):
        L, rf = 0, 1
        while rf < seq_len:
            rf += (k - 1) * (base ** L)
            L += 1
        return L

    num_layers = calc_layers(seq_len, k)
    model = DilatedTCN(
        input_channels=in_ch,
        num_layers=num_layers,
        hidden_channels=h,
        kernel_size=k,
        num_classes=num_classes,
        dropout=d,
    ).to(device)
    return model


def quantize_weights(model, bits=8, per_channel=True):
    qmax = (1 << (bits - 1)) - 1
    model = deepcopy(model).cpu()
    for _, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)) and hasattr(module, "weight"):
            w = module.weight.data
            if per_channel:
                shape = w.shape[0]
                w_flat = w.view(shape, -1)
                max_abs = w_flat.abs().max(dim=1).values
                scale = max_abs / qmax
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                q = torch.round(w_flat / scale[:, None])
                q = torch.clamp(q, -qmax, qmax)
                module.weight.data = (q * scale[:, None]).view_as(w)
            else:
                max_abs = w.abs().max()
                scale = max_abs / qmax if max_abs != 0 else 1.0
                q = torch.clamp(torch.round(w / scale), -qmax, qmax)
                module.weight.data = q * scale
    return model


@torch.no_grad()
def evaluate(model, dataloader, device, task_type):
    model.eval().to(device)
    all_preds, all_targets = [], []
    for x, y in tqdm(dataloader, desc="Evaluating", unit="batch"):
        x = x.to(device)
        logits = model(x)
        if task_type == "binary":
            probs = torch.sigmoid(logits).squeeze()
            preds = (probs > 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(y)

    y_true = torch.cat(all_targets).numpy()
    y_pred = torch.cat(all_preds).numpy()
    acc = np.mean(y_true == y_pred)
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return acc, prec, rec, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", choices=["val", "train", "full"], default="val")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--bits", type=str, default="float,8,4,2", help="Comma-separated bit widths for PTQ (e.g., float,8,4,2)")
    parser.add_argument("--scheme", choices=["per_channel", "per_tensor"], default="per_channel")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = (
        torch.device("cuda") if args.device == "cuda" else
        torch.device("cpu") if args.device == "cpu" else
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    dl, num_classes = make_dataloader(cfg, args.dataset, args.batch_size)
    task_type = cfg["task"]["type"]

    model = build_model(cfg, num_classes, device)
    weights = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(weights["state_dict"] if "state_dict" in weights else weights)

    per_channel = args.scheme == "per_channel"
    results = []

    bits_list = [b.strip() for b in args.bits.split(",")]
    for b in bits_list:
        if b == "float":
            model_q = deepcopy(model).to(device)
            tag = "float"
        else:
            model_q = quantize_weights(model, int(b), per_channel=per_channel).to(device)
            tag = f"{b}b"
        acc, prec, rec, f1 = evaluate(model_q, dl, device, task_type)
        results.append((tag, acc, prec, rec, f1))
        print(f"[{tag:>6}]  Acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    # Extract metrics
    xs = [32 if tag == "float" else int(tag.rstrip("b")) for tag, *_ in results]
    accs = [acc for _, acc, _, _, _ in results]
    precs = [prec for _, _, prec, _, _ in results]
    recs = [rec for _, _, _, rec, _ in results]
    f1s = [f1 for _, _, _, _, f1 in results]

    # Plot all metrics
    plots_dir = os.path.join(cfg["output"]["plots_dir"], "eval")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "ptq_metrics_vs_bits.png")

    plt.figure()
    plt.plot(xs, accs, marker='o', label="Accuracy")
    plt.plot(xs, precs, marker='s', label="Precision")
    plt.plot(xs, recs, marker='^', label="Recall")
    plt.plot(xs, f1s, marker='d', label="F1 Score")

    plt.xlabel("Weight Bit Width")
    plt.ylabel("Metric Value")
    plt.title("PTQ: Metrics vs Weight Bit Width")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Saved plot to {out_path}")


if __name__ == "__main__":
    main()
