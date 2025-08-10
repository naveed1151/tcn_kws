import os
import sys
import argparse
import numpy as np

# Ensure repo root on sys.path so imports work when run as a module/script
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train.utils import load_config  # noqa: E402
from data_loader.utils import make_datasets  # noqa: E402


def _infer_num_classes_from_cfg(cfg: dict) -> int:
    task = cfg.get("task", {})
    ttype = task.get("type", "multiclass")
    if ttype == "binary":
        return 2  # labels {0,1}
    n = len(task.get("class_list", []))
    if task.get("include_unknown", False):
        n += 1
    if task.get("include_background", False):
        n += 1
    if n <= 0:
        raise ValueError("Could not infer num_classes from config.task.")
    return n


def _get_class_names(dataset, cfg: dict, num_classes: int):
    # Prefer dataset-provided names
    if hasattr(dataset, "class_names") and dataset.class_names and len(dataset.class_names) == num_classes:
        return list(dataset.class_names)
    # Fallback from config
    task = cfg.get("task", {})
    ttype = task.get("type", "multiclass")
    if ttype == "binary":
        return ["neg", "pos"]
    names = list(task.get("class_list", []))
    if task.get("include_unknown", False):
        names.append("unknown")
    if task.get("include_background", False):
        names.append("background")
    # Pad/truncate to num_classes to be safe
    if len(names) < num_classes:
        names += [f"class_{i}" for i in range(len(names), num_classes)]
    return names[:num_classes]


def _count_labels(ds, num_classes: int):
    ys = [ds[i][1] for i in range(len(ds))]
    counts = np.bincount(ys, minlength=num_classes)
    return counts.tolist()


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset class counts (train/val/test) per config")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to YAML config (e.g., config/base.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("data", {}).get("preprocessed_dir", None) is None:
        raise ValueError("Config must define data.preprocessed_dir")

    # Build datasets the same way train/train.py starts (via make_datasets)
    batch_size = int(cfg.get("train", {}).get("batch_size", 32))
    train_loader, val_loader, test_loader = make_datasets(cfg, which="all", batch_size=batch_size)

    # Access underlying datasets from loaders
    ds_train = train_loader.dataset
    ds_val = val_loader.dataset
    ds_test = test_loader.dataset

    # Determine number of classes (prefer the wrapped dataset attribute)
    num_classes = getattr(ds_train, "num_classes", None)
    if num_classes is None:
        num_classes = _infer_num_classes_from_cfg(cfg)
    num_classes = int(num_classes)

    # Class names
    class_names = _get_class_names(ds_train, cfg, num_classes)

    # Counts
    train_counts = _count_labels(ds_train, num_classes)
    val_counts = _count_labels(ds_val, num_classes)
    test_counts = _count_labels(ds_test, num_classes)

    # Print summary
    print(f"[INFO] num_classes={num_classes}")
    print(f"[INFO] class_names={class_names}")
    print(f"[SPLIT SIZES] train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}\n")

    print("[Train counts]")
    for i, name in enumerate(class_names):
        print(f"  {name:>12s}: {train_counts[i]}")
    print("\n[Val counts]")
    for i, name in enumerate(class_names):
        print(f"  {name:>12s}: {val_counts[i]}")
    print("\n[Test counts]")
    for i, name in enumerate(class_names):
        print(f"  {name:>12s}: {test_counts[i]}")

    # Totals per split
    print("\n[Totals]")
    print(f"  train total: {sum(train_counts)}")
    print(f"  val   total: {sum(val_counts)}")
    print(f"  test  total: {sum(test_counts)}")


if __name__ == "__main__":
    main()