from torch.utils.data import Dataset
class BinaryClassDataset(Dataset):
    """
    Wraps a base MFCCDataset for binary KWS without loading any arrays in __init__.
    We derive positive/negative indices from base metadata, and only load x in __getitem__.
    """
    def __init__(self, base_dataset, target_word, index_to_label,
                 downsample_ratio=None, seed=0):
        import numpy as np
        self.base = base_dataset
        self.target_word = target_word
        self.index_to_label = index_to_label
        rng = np.random.RandomState(seed)

        # --- Grab labels without loading MFCC arrays ---
        labels_int = None

        # Preferred: if your MFCCDataset stores a list of (path, label_idx)
        if hasattr(base_dataset, "samples"):
            # common pattern: samples = [(path, label_idx), ...]
            try:
                labels_int = [lbl for _, lbl in base_dataset.samples]
            except Exception:
                labels_int = None

        # Alternative: an explicit labels list/array
        if labels_int is None and hasattr(base_dataset, "labels"):
            try:
                labels_int = list(base_dataset.labels)
            except Exception:
                labels_int = None

        # Last resort (slower): add a label-only accessor to MFCCDataset if you have it
        if labels_int is None and hasattr(base_dataset, "get_label"):
            labels_int = [base_dataset.get_label(i) for i in range(len(base_dataset))]

        # Absolute fallback (will be slow): this will load arrays; avoid if possible
        if labels_int is None:
            print("[WARN] Falling back to loading items to get labels; consider exposing 'samples' in MFCCDataset.")
            labels_int = [base_dataset[i][1] for i in range(len(base_dataset))]

        # --- Build positive/negative index lists ---
        pos_idx, neg_idx = [], []
        for i, li in enumerate(labels_int):
            if self.index_to_label[li] == self.target_word:
                pos_idx.append(i)
            else:
                neg_idx.append(i)

        # Optional downsampling of negatives
        if downsample_ratio is not None:
            keep_neg = int(len(pos_idx) * float(downsample_ratio))
            if keep_neg < len(neg_idx):
                neg_idx = rng.choice(neg_idx, size=keep_neg, replace=False).tolist()

        # Final index list
        self.indices = pos_idx + neg_idx
        rng.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        x, label_idx = self.base[base_idx]  # <-- load MFCC only here
        y = 1 if self.index_to_label[label_idx] == self.target_word else 0
        return x, y
