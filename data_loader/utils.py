# data_loader/utils.py

import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Subset, Dataset

from data_loader.mfcc_dataset import MFCCDataset
from data_loader.binary_dataset import BinaryClassDataset
from data_loader.multiclass_dataset import MultiClassDataset


def stratified_train_val_test_indices(labels, val_frac, test_frac, seed=0):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    train_idx, val_idx, test_idx = [], [], []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        n = len(idx_c)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        val_idx.extend(idx_c[:n_val].tolist())
        test_idx.extend(idx_c[n_val:n_val+n_test].tolist())
        train_idx.extend(idx_c[n_val+n_test:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx

def make_datasets(cfg, which="all", batch_size=64):
    """
    Build DataLoaders for train/val/test following config, and return:
      (train_loader, val_loader, test_loader, train_dataset_wrapped)

    Unknown downsampling caps (task.unknown_max_ratio / unknown_max_count) apply to TRAIN ONLY.
    """
    data_cfg = cfg.get("data", {})
    task_cfg = cfg.get("task", {})
    train_cfg = cfg.get("train", {})

    root = data_cfg.get("preprocessed_dir")
    if not root:
        raise ValueError("Config must define data.preprocessed_dir")

    # Split fracs (defaults if not provided)
    val_frac = float(data_cfg.get("val_frac", 0.1))
    test_frac = float(data_cfg.get("test_frac", 0.1))
    seed = int(cfg.get("augmentation", {}).get("seed", 0)) if "augmentation" in cfg else 0

    # Loader knobs
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", False))
    persistent_workers = bool(train_cfg.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = train_cfg.get("prefetch_factor", None)

    # Base dataset (raw MFCC items with full label space)
    base = MFCCDataset(root)

    # Collect base labels for stratified split
    base_labels = []
    for item in base:
        if isinstance(item, tuple) and len(item) == 2:
            _, y = item
        elif isinstance(item, dict):
            y = item.get("y")
        else:
            raise RuntimeError("MFCCDataset items must be (x, y) or {'x':..., 'y':...}")
        base_labels.append(int(y))

    # Build splits
    train_idx, val_idx, test_idx = stratified_train_val_test_indices(
        base_labels, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    base_train = Subset(base, train_idx)
    base_val = Subset(base, val_idx)
    base_test = Subset(base, test_idx)

    # Task selection
    task_type = str(task_cfg.get("type", "multiclass")).lower()

    if task_type == "multiclass":
        class_list = list(task_cfg.get("class_list", []))
        include_unknown = bool(task_cfg.get("include_unknown", True))
        include_background = bool(task_cfg.get("include_background", True))
        background_label = data_cfg.get("background_label", "_background_noise_")

        # Train-only caps for unknown
        unknown_max_ratio = task_cfg.get("unknown_max_ratio", None)
        unknown_max_count = task_cfg.get("unknown_max_count", None)

        # Wrapped datasets
        train_ds = MultiClassDataset(
            base_train,
            class_list,
            index_to_label=getattr(base, "index_to_label", None),
            label_to_index=getattr(base, "label_to_index", None),
            include_unknown=include_unknown,
            include_background=include_background,
            background_label=background_label,
            unknown_max_ratio=unknown_max_ratio,
            unknown_max_count=unknown_max_count,
            seed=seed,
        )
        val_ds = MultiClassDataset(
            base_val,
            class_list,
            index_to_label=getattr(base, "index_to_label", None),
            label_to_index=getattr(base, "label_to_index", None),
            include_unknown=include_unknown,
            include_background=include_background,
            background_label=background_label,
        )
        test_ds = MultiClassDataset(
            base_test,
            class_list,
            index_to_label=getattr(base, "index_to_label", None),
            label_to_index=getattr(base, "label_to_index", None),
            include_unknown=include_unknown,
            include_background=include_background,
            background_label=background_label,
        )
    elif task_type == "binary":
        # Implement if you use BinaryClassDataset; placeholder for now
        raise NotImplementedError("Binary task path not implemented in make_datasets()")
    else:
        raise ValueError(f"Unknown task.type '{task_type}'")

    # DataLoaders
    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    if prefetch_factor is not None and num_workers > 0:
        dl_kwargs["prefetch_factor"] = int(prefetch_factor)
        dl_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **dl_kwargs)

    return train_loader, val_loader, test_loader,

def get_num_classes(cfg):
    class_list = list(cfg["task"]["class_list"])
    if bool(cfg["task"].get("include_unknown", False)):
        class_list.append("unknown")
    if bool(cfg["task"].get("include_background", False)):
        background_label = cfg["data"].get("background_label", "_background_noise_")
        class_list.append(background_label)
    return len(class_list)

class MFCCAugment:
    """Zero-padded time shift + optional Gaussian noise for (C, T) MFCC tensors."""
    def __init__(self, hop_length_s: float, max_shift_ms: float = 100.0,
                 noise_prob: float = 0.15, noise_std_factor: float = 0.05, seed=None):
        import numpy as _np
        self.hop_length_s = float(hop_length_s)
        self.max_shift_ms = float(max_shift_ms)
        self.noise_prob = float(noise_prob)
        self.noise_std_factor = float(noise_std_factor)
        self.rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
        self.max_shift_frames = int(round((self.max_shift_ms / 1000.0) / self.hop_length_s))

    def _shift_with_zeros(self, x: torch.Tensor, s: int) -> torch.Tensor:
        C, T = x.shape
        if s == 0:
            return x
        if abs(s) >= T:
            return torch.zeros_like(x)
        if s > 0:
            out = torch.zeros_like(x)
            out[:, s:] = x[:, :T - s]
            return out
        s = -s
        out = torch.zeros_like(x)
        out[:, :T - s] = x[:, s:]
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.ndim != 2:
            raise ValueError(f"MFCCAugment expects (C, T), got shape {tuple(x.shape)}")
        C, T = x.shape
        if self.max_shift_frames > 0 and T > 1:
            s = int(self.rng.integers(-self.max_shift_frames, self.max_shift_frames + 1))
            if s != 0:
                x = self._shift_with_zeros(x, s)
        return x

class TransformDataset(Dataset):
    """Wrap a dataset to apply a transform only on __getitem__."""
    def __init__(self, base: Dataset, transform=None):
        self.base = base
        self.transform = transform
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y