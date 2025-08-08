# data_loader/utils.py

import numpy as np
import random

from torch.utils.data import DataLoader, Subset

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


def make_datasets(cfg, which="train", batch_size=64):
    root = cfg["data"]["preprocessed_dir"]
    val_split = float(cfg["data"]["val_split"])
    test_split = float(cfg["data"]["test_split"])
    shuffle = bool(cfg["data"].get("shuffle", True))
    seed = cfg["data"].get("split_seed", 42)

    task_type = cfg["task"]["type"]
    class_list = list(cfg["task"]["class_list"])
    include_unknown = bool(cfg["task"].get("include_unknown", False))
    include_background = bool(cfg["task"].get("include_background", False))
    background_label = cfg["data"].get("background_label", "_background_noise_")

    # Final class list and mappings
    extended_class_list = list(class_list)
    if include_unknown:
        extended_class_list.append("unknown")
    if include_background:
        extended_class_list.append(background_label)

    label_to_index = {label: i for i, label in enumerate(extended_class_list)}
    index_to_label = {i: label for label, i in label_to_index.items()}

    # Create base dataset (MFCCDataset always used as base)
    base_dataset = MFCCDataset(root)

    # Wrap in task-specific dataset
    if task_type == "multiclass":
        dataset = MultiClassDataset(
            base_dataset, class_list, index_to_label, label_to_index,
            include_unknown=include_unknown,
            include_background=include_background,
            background_label=background_label
        )
    elif task_type == "binary":
        dataset = BinaryClassDataset(
            base_dataset, class_list, index_to_label, label_to_index
        )
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Split
    num_samples = len(dataset)
    indices = list(range(num_samples))
    if shuffle:
        random.Random(seed).shuffle(indices)

    val_end = int(val_split * num_samples)
    test_end = val_end + int(test_split * num_samples)

    val_indices = indices[:val_end]
    test_indices = indices[val_end:test_end]
    train_indices = indices[test_end:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    if which == "train":
        return DataLoader(train_set, batch_size=batch_size, shuffle=True)
    elif which == "val":
        return DataLoader(val_set, batch_size=batch_size, shuffle=False)
    elif which == "test":
        return DataLoader(test_set, batch_size=batch_size, shuffle=False)
    elif which == "all":
        return (
            DataLoader(train_set, batch_size=batch_size, shuffle=True),
            DataLoader(val_set, batch_size=batch_size, shuffle=False),
            DataLoader(test_set, batch_size=batch_size, shuffle=False),
            dataset
        )
    else:
        raise ValueError(f"Unknown dataset split: {which}")