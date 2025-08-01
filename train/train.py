import os
import time
import argparse
from copy import deepcopy
from typing import List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import matplotlib.pyplot as plt
from tqdm import tqdm

from model.model import DilatedTCN
from data_loader.mfcc_dataset import MFCCDataset


# -----------------------
# Config helpers
# -----------------------
def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _load_yaml_with_encodings(path):
    import yaml
    # Try UTF-8 first; then UTF-8 with BOM; then cp1252 as last resort.
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("yaml", b"", 0, 1, f"Could not decode {path} with utf-8/utf-8-sig/cp1252")

def load_config(path):
    cfg_task = _load_yaml_with_encodings(path)

    # Optional base.yaml merge
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if os.path.basename(path) != "base.yaml" and os.path.exists(base_path):
        cfg_base = _load_yaml_with_encodings(base_path)
        return deep_update(deepcopy(cfg_base), cfg_task)
    return cfg_task


# -----------------------
# Augmentation (shift then noise)
# -----------------------
class MFCCAugment:
    """
    Applies:
      (1) time shift by up to +/- max_shift_frames (zero-padded, no wrap),
      (2) additive Gaussian noise with probability noise_prob.

    Expects MFCC tensors shaped (C, T).
    """
    def __init__(self, hop_length_s: float, max_shift_ms: float = 100.0,
                 noise_prob: float = 0.15, noise_std_factor: float = 0.05, seed=None):
        import numpy as _np
        self.hop_length_s = float(hop_length_s)
        self.max_shift_ms = float(max_shift_ms)
        self.noise_prob = float(noise_prob)
        self.noise_std_factor = float(noise_std_factor)
        self.rng = _np.random.default_rng(seed) if seed is not None else _np.random.default_rng()
        # Convert ms→frames using hop:
        self.max_shift_frames = int(round((self.max_shift_ms / 1000.0) / self.hop_length_s))

    def _shift_with_zeros(self, x: torch.Tensor, s: int) -> torch.Tensor:
        """
        Shift along time dimension by s frames with zero padding (no wrap).
          s > 0 : shift right (delay); zeros inserted at start
          s < 0 : shift left  (advance); zeros inserted at end
        """
        C, T = x.shape
        if s == 0:
            return x
        # Clamp |s| so we don't slice with negatives or >= T
        if abs(s) >= T:
            return torch.zeros_like(x)
        if s > 0:
            # right shift
            out = torch.zeros_like(x)
            out[:, s:] = x[:, :T - s]
            return out
        else:
            # left shift
            s = -s
            out = torch.zeros_like(x)
            out[:, :T - s] = x[:, s:]
            return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        # Expect (C, T)
        if x.ndim != 2:
            raise ValueError(f"MFCCAugment expects (C, T), got shape {tuple(x.shape)}")

        C, T = x.shape

        # 1) Zero-padded time shift
        if self.max_shift_frames > 0 and T > 1:
            # sample integer shift in [-max_shift_frames, max_shift_frames]
            s = int(self.rng.integers(-self.max_shift_frames, self.max_shift_frames + 1))
            if s != 0:
                x = self._shift_with_zeros(x, s)

        # 2) Additive Gaussian noise with probability
        if self.rng.random() < self.noise_prob:
            std = float(x.std().item())
            if std > 0:
                noise = torch.randn_like(x) * (self.noise_std_factor * std)
                x = x + noise

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


# -----------------------
# Dataset wrappers
# -----------------------
class BinaryKeywordDataset(Dataset):
    """
    Wraps a base MFCCDataset for binary KWS without loading any arrays in __init__.
    We derive positive/negative indices from base metadata, and only load x in __getitem__.
    """
    def __init__(self, base_dataset, target_word, index_to_word,
                 downsample_ratio=None, seed=0):
        import numpy as np
        self.base = base_dataset
        self.target_word = target_word
        self.index_to_word = index_to_word
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
            if self.index_to_word[li] == self.target_word:
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
        y = 1 if self.index_to_word[label_idx] == self.target_word else 0
        return x, y


class MultiClassDataset(Dataset):
    def __init__(self, base_dataset, class_list, index_to_word, label_to_index):
        self.samples = []
        if class_list is None:
            self.samples = list(base_dataset)
            # build class names in the label index order
            num_classes = len(label_to_index)
            by_index = [None] * num_classes
            for w, idx in label_to_index.items():
                by_index[idx] = w
            self.class_names = by_index
            self.num_classes = num_classes
        else:
            class_set = set(class_list)
            new_label_map = {w: i for i, w in enumerate(class_list)}
            for x, label_idx in base_dataset:
                word = index_to_word[label_idx]
                if word in class_set:
                    self.samples.append((x, new_label_map[word]))
            self.num_classes = len(class_list)
            self.class_names = list(class_list)

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


# -----------------------
# Stratified split
# -----------------------
def stratified_train_val_test_indices(labels: List[int], val_frac: float, test_frac: float, seed: int = 0):
    """
    Split indices into train/val/test preserving class ratios.
    labels: list/array of integer class ids (0..K-1)
    """
    assert 0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0 and val_frac + test_frac < 1.0
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
        n_train = n - n_val - n_test
        if n_train < 0:
            n_train = 0
        # assign
        val_idx.extend(idx_c[:n_val].tolist())
        test_idx.extend(idx_c[n_val:n_val+n_test].tolist())
        train_idx.extend(idx_c[n_val+n_test:].tolist())

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


# -----------------------
# Metrics helpers
# -----------------------
def _binary_counts(preds, targets):
    preds = preds.long()
    targets = targets.long()
    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return tp, fp, fn, correct, total

def _derive_metrics(total_loss, num_batches, correct, total, tp=None, fp=None, fn=None):
    avg_loss = total_loss / max(1, num_batches)
    accuracy = correct / total if total > 0 else 0.0
    if tp is None:
        return avg_loss, accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return avg_loss, accuracy, precision, recall, f1

def _multiclass_confusion_add(cm, preds, targets, num_classes):
    for p, t in zip(preds.tolist(), targets.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm

def _multiclass_macro_prf1(cm):
    K = len(cm)
    precisions, recalls, f1s = [], [], []
    for k in range(K):
        tp = cm[k][k]
        fp = sum(cm[r][k] for r in range(K)) - tp
        fn = sum(cm[k][c] for c in range(K)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    macro_p = sum(precisions) / K if K > 0 else 0.0
    macro_r = sum(recalls) / K if K > 0 else 0.0
    macro_f1 = sum(f1s) / K if K > 0 else 0.0
    return macro_p, macro_r, macro_f1

def compute_confusion_matrix(model, loader, device, num_classes: int):
    """Return confusion matrix (num_classes x num_classes) of counts for multiclass."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.long().cpu().numpy()
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            for t, p in zip(y, preds):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1
    return cm


# -----------------------
# Trainer
# -----------------------
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.task_type = cfg["task"]["type"]  # "binary" or "multiclass"
        
        # Device
        device_cfg = cfg["train"].get("device", "auto")
        if device_cfg == "cuda":
            self.device = torch.device("cuda")
        elif device_cfg == "cpu":
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Base dataset (preprocessed MFCCs)
        pre_dir = cfg["data"]["preprocessed_dir"]
        full_dataset = MFCCDataset(pre_dir)
        self.label_to_index = full_dataset.label_to_index
        self.index_to_word = {v: k for k, v in self.label_to_index.items()}

        # Task-specific dataset wrapping
        seed = int(cfg["train"].get("seed", 0))
        if self.task_type == "binary":
            target_word = cfg["data"]["target_word"]
            neg_ratio = cfg["data"].get("neg_downsample_ratio", None)
            dataset = BinaryKeywordDataset(full_dataset, target_word, self.index_to_word,
                                           downsample_ratio=neg_ratio, seed=seed)
            self.num_classes = 1
            labels_for_split = [y for _, y in dataset]
        else:
            class_list = cfg["data"].get("class_list", None)
            dataset = MultiClassDataset(full_dataset, class_list, self.index_to_word, self.label_to_index)
            self.num_classes = dataset.num_classes
            labels_for_split = [y for _, y in dataset]

        # Stratified train/val/test split
        val_frac = float(cfg["data"]["val_split"])
        test_frac = float(cfg["data"]["test_split"])
        train_idx, val_idx, test_idx = stratified_train_val_test_indices(labels_for_split, val_frac, test_frac, seed=seed)

        self.train_dataset = Subset(dataset, train_idx)
        self.val_dataset   = Subset(dataset, val_idx)
        self.test_dataset  = Subset(dataset, test_idx)

        # Augmentation (training split only)
        aug_cfg = cfg.get("augmentation", {})
        use_aug = bool(aug_cfg.get("enable", False))
        if use_aug:
            hop_s = float(cfg["data"]["mfcc"]["hop_length_s"])
            augment = MFCCAugment(
                hop_length_s=hop_s,
                max_shift_ms=float(aug_cfg.get("max_shift_ms", 100.0)),
                noise_prob=float(aug_cfg.get("noise_prob", 0.15)),
                noise_std_factor=float(aug_cfg.get("noise_std_factor", 0.05)),
                seed=aug_cfg.get("seed", None),
            )
            train_set_for_loader = TransformDataset(self.train_dataset, transform=augment)
        else:
            train_set_for_loader = self.train_dataset

        # Loaders
        self.batch_size = int(cfg["train"]["batch_size"])
        self.train_loader = DataLoader(train_set_for_loader, batch_size=self.batch_size, shuffle=True)
        self.val_loader   = DataLoader(self.val_dataset,   batch_size=self.batch_size, shuffle=False)
        self.test_loader  = DataLoader(self.test_dataset,  batch_size=self.batch_size, shuffle=False)

        # Model dims
        sample_x, _ = self.train_dataset[0]
        input_channels = sample_x.shape[0]
        seq_len = sample_x.shape[1]
        self.num_layers = self._calc_layers(seq_len, kernel_size=int(cfg["model"]["kernel_size"]), dilation_base=2)

        # Model
        self.model = DilatedTCN(
            input_channels=input_channels,
            num_layers=self.num_layers,
            hidden_channels=int(cfg["model"]["hidden_channels"]),
            kernel_size=int(cfg["model"]["kernel_size"]),
            num_classes=self.num_classes,
            dropout=float(cfg["model"]["dropout"]),
        ).to(self.device)

        # Optimizer
        lr = float(cfg["train"]["learning_rate"])
        weight_decay = float(cfg["train"]["weight_decay"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss
        if self.task_type == "binary":
            pos_weight = cfg["train"].get("pos_weight", None)
            if pos_weight is not None:
                pos_weight_tensor = torch.tensor([float(pos_weight)], device=self.device)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            else:
                self.criterion = nn.BCEWithLogitsLoss()
            self.threshold = float(cfg["train"].get("threshold", 0.5) or 0.5)
            self.auto_threshold = bool(cfg["train"].get("auto_threshold", False))
            self.threshold_search_steps = int(cfg["train"].get("threshold_search_steps", 101))
            self.class_names = ["neg", "pos"]
        else:
            class_weights = cfg["train"].get("class_weights", None)
            label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
            weight_tensor = None
            if class_weights is not None:
                if len(class_weights) != self.num_classes:
                    raise ValueError("class_weights length must equal number of classes.")
                weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=label_smoothing)
            self.class_names = dataset.class_names  # list[str] length = num_classes

        self.test_metrics = None
        self.test_confusion = None  # (only for multiclass)
        # Histories / outputs
        self.epochs = int(cfg["train"]["num_epochs"])
        self.plots_dir = cfg["output"]["plots_dir"]
        self.metrics_figure = cfg["output"]["metrics_figure"]
        self.weights_path = cfg["output"]["weights_path"]
        self.use_tqdm = bool(cfg["output"].get("tqdm", True))

        self.train_hist = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}
        self.val_hist   = {"loss": [], "acc": [], "prec": [], "rec": [], "f1": []}

        # ---- Sanity checks: dataset sizes and MFCC shapes ----
        print(f"[Split sizes] train={len(self.train_dataset)}  val={len(self.val_dataset)}  test={len(self.test_dataset)}")

        # Peek a few samples from the training subset
        for i in range(min(3, len(self.train_dataset))):
            x_i, y_i = self.train_dataset[i]
            print(f"[Sample {i}] x.shape={tuple(x_i.shape)}  label={y_i}")

        # Derive input dims to confirm (C, T)
        sample_x, _ = self.train_dataset[0]
        C, T = sample_x.shape
        print(f"[MFCC dims] channels(C)={C}  timesteps(T)={T}")

        # If you compute layers dynamically, also print the result
        k = int(self.cfg["model"]["kernel_size"])
        num_layers_dyn = self._calc_layers(T, kernel_size=k, dilation_base=2)
        print(f"[TCN] kernel_size={k}  computed_num_layers={num_layers_dyn}")

        if self.task_type == "binary":
            import numpy as np
            ys = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            pos = int(np.sum(ys)); neg = len(ys) - pos
            print(f"[Train balance] pos={pos}  neg={neg}  ratio={neg/max(1,pos):.2f}:1")
        
        if self.task_type == "multiclass":
            import numpy as np
            ys = [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            counts = np.bincount(ys, minlength=self.num_classes)
            print(f"[Train per-class counts] {counts.tolist()}")
            print(f"[Classes] {self.class_names}")



        # Optional: assert reasonable ranges (adjust to your expectations)
        assert C in (16, 20, 28, 40) or C > 0, f"Unexpected MFCC channels: {C}"
        assert T >= 40, f"Too few time steps ({T}). Check MFCC hop/window/duration."


    def _calc_layers(self, seq_len, kernel_size=3, dilation_base=2):
        L = 0
        receptive_field = 1
        while receptive_field < seq_len:
            receptive_field += (kernel_size - 1) * (dilation_base ** L)
            L += 1
        return L

    def _loop(self, loader, train_mode=True):
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        tp = fp = fn = 0
        if self.task_type == "multiclass":
            cm = [[0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]
        else:
            cm = None

        iterator = tqdm(loader, desc="Training" if train_mode else "Validating", leave=False) if self.use_tqdm else loader

        with torch.set_grad_enabled(train_mode):
            for batch_x, batch_y in iterator:
                num_batches += 1
                batch_x = batch_x.to(self.device)

                if self.task_type == "binary":
                    targets = batch_y.float().unsqueeze(1).to(self.device)
                else:
                    targets = batch_y.long().to(self.device)

                logits = self.model(batch_x)
                loss = self.criterion(logits, targets)

                if train_mode:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                if self.task_type == "binary":
                    probs = torch.sigmoid(logits)
                    preds = (probs > self.threshold).long()
                    correct += (preds == targets.long()).sum().item()
                    total += targets.numel()
                    _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
                    tp += _tp; fp += _fp; fn += _fn
                else:
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == targets).sum().item()
                    total += targets.numel()
                    cm = _multiclass_confusion_add(cm, preds, targets, self.num_classes)

        if self.task_type == "binary":
            return _derive_metrics(total_loss, num_batches, correct, total, tp, fp, fn)
        else:
            avg_loss, acc = _derive_metrics(total_loss, num_batches, correct, total)
            macro_p, macro_r, macro_f1 = _multiclass_macro_prf1(cm)
            return avg_loss, acc, macro_p, macro_r, macro_f1

    # ---------- Threshold search (binary only) ----------
    def find_best_threshold(self, steps=101):
        assert self.task_type == "binary", "Threshold search is for binary mode."
        self.model.eval()
        probs_all, y_all = [], []
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().unsqueeze(1).to(self.device)
                logits = self.model(batch_x)
                probs_all.append(torch.sigmoid(logits).cpu())
                y_all.append(batch_y.cpu())
        probs_all = torch.cat(probs_all, dim=0).view(-1)
        y_all = torch.cat(y_all, dim=0).view(-1)

        best_thr, best_f1 = 0.5, -1.0
        for thr in torch.linspace(0.01, 0.99, steps=steps):
            preds = (probs_all > thr).int()
            tp = int(((preds == 1) & (y_all == 1)).sum())
            fp = int(((preds == 1) & (y_all == 0)).sum())
            fn = int(((preds == 0) & (y_all == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_thr = float(f1), float(thr)
        return best_thr, best_f1

    @torch.no_grad()
    def evaluate(self, loader):
        """Evaluate current model on a loader using current threshold; returns (loss, acc, P, R, F1)."""
        self.model.eval()
        total_loss = 0.0; correct = 0; total = 0; num_batches = 0
        tp = fp = fn = 0
        if self.task_type == "multiclass":
            cm = [[0 for _ in range(self.num_classes)] for _ in range(self.num_classes)]
        else:
            cm = None

        for batch_x, batch_y in loader:
            num_batches += 1
            batch_x = batch_x.to(self.device)
            if self.task_type == "binary":
                targets = batch_y.float().unsqueeze(1).to(self.device)
            else:
                targets = batch_y.long().to(self.device)
            logits = self.model(batch_x)
            loss = self.criterion(logits, targets)
            total_loss += loss.item()

            if self.task_type == "binary":
                probs = torch.sigmoid(logits)
                preds = (probs > self.threshold).long()
                correct += (preds == targets.long()).sum().item()
                total += targets.numel()
                _tp, _fp, _fn, _, _ = _binary_counts(preds, targets)
                tp += _tp; fp += _fp; fn += _fn
            else:
                preds = torch.argmax(logits, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.numel()
                cm = _multiclass_confusion_add(cm, preds, targets, self.num_classes)

        if self.task_type == "binary":
            return _derive_metrics(total_loss, num_batches, correct, total, tp, fp, fn)
        else:
            avg_loss, acc = _derive_metrics(total_loss, num_batches, correct, total)
            macro_p, macro_r, macro_f1 = _multiclass_macro_prf1(cm)
            return avg_loss, acc, macro_p, macro_r, macro_f1

    def train(self):
        start = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}:")
            tr_loss, tr_acc, tr_p, tr_r, tr_f1 = self._loop(self.train_loader, train_mode=True)
            va_loss, va_acc, va_p, va_r, va_f1 = self._loop(self.val_loader,   train_mode=False)

            self.train_hist["loss"].append(tr_loss); self.val_hist["loss"].append(va_loss)
            self.train_hist["acc"].append(tr_acc);   self.val_hist["acc"].append(va_acc)
            self.train_hist["prec"].append(tr_p);    self.val_hist["prec"].append(va_p)
            self.train_hist["rec"].append(tr_r);     self.val_hist["rec"].append(va_r)
            self.train_hist["f1"].append(tr_f1);     self.val_hist["f1"].append(va_f1)

            print(f"  Train | Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  P: {tr_p:.4f}  R: {tr_r:.4f}  F1: {tr_f1:.4f}")
            print(f"  Val   | Loss: {va_loss:.4f}  Acc: {va_acc:.4f}  P: {va_p:.4f}  R: {va_r:.4f}  F1: {va_f1:.4f}")

        elapsed = time.perf_counter() - start
        hrs, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        print(f"Total training time: {hrs:02d}:{mins:02d}:{secs:02d} (hh:mm:ss)")

        # Optional: auto threshold tuning (binary)
        if self.task_type == "binary" and self.auto_threshold:
            last_val_f1 = self.val_hist["f1"][-1] if self.val_hist["f1"] else -1.0
            best_thr, best_f1 = self.find_best_threshold(steps=self.threshold_search_steps)
            if best_f1 > last_val_f1:
                old_thr = self.threshold
                self.threshold = best_thr
                print(f"[Auto-threshold] Improved F1 {last_val_f1:.4f} → {best_f1:.4f}. "
                      f"Updating threshold: {old_thr:.3f} → {self.threshold:.3f}")
                # Re-evaluate on val for reporting
                va_loss, va_acc, va_p, va_r, va_f1 = self.evaluate(self.val_loader)
                print(f"[Post-threshold] Val | Loss: {va_loss:.4f}  Acc: {va_acc:.4f}  "
                      f"P: {va_p:.4f}  R: {va_r:.4f}  F1: {va_f1:.4f}")
                os.makedirs(self.plots_dir, exist_ok=True)
                with open(os.path.join(self.plots_dir, "best_threshold.txt"), "w") as f:
                    f.write(f"{self.threshold:.6f}\n")
            else:
                print(f"[Auto-threshold] Best searched F1 {best_f1:.4f} did NOT beat last val F1 {last_val_f1:.4f}. "
                      f"Keeping threshold at {self.threshold:.3f}.")

        # Final test metrics (no tuning on test)
        te_loss, te_acc, te_p, te_r, te_f1 = self.evaluate(self.test_loader)
        self.test_metrics = {"loss": te_loss, "acc": te_acc, "prec": te_p, "rec": te_r, "f1": te_f1}
        print(f"[Test] Loss: {te_loss:.4f}  Acc: {te_acc:.4f}  P: {te_p:.4f}  R: {te_r:.4f}  F1: {te_f1:.4f}")

        # If multiclass, also compute confusion matrix on test set
        if self.task_type == "multiclass":
            cm = compute_confusion_matrix(self.model, self.test_loader, self.device, self.num_classes)
            self.test_confusion = cm  # numpy array

    def save(self):
        torch.save(self.model.state_dict(), self.weights_path)
        print(f"Model saved to {self.weights_path}")

    def plot_metrics(self):
        os.makedirs(self.plots_dir, exist_ok=True)

        # ---- 3 rows × 2 cols grid ----
        fig, axs = plt.subplots(3, 2, figsize=(14, 14))
        axs = axs.ravel()
        fig.tight_layout(pad=5.0)

        epochs = range(1, len(self.train_hist["loss"]) + 1)

        # 1) Loss (no y-limit)
        axs[0].plot(epochs, self.train_hist["loss"], label="Train Loss")
        axs[0].plot(epochs, self.val_hist["loss"], label="Val Loss")
        axs[0].set_title("Loss Over Epochs"); axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Loss")
        axs[0].grid(True)

        # 2) Accuracy (0..1)
        axs[1].plot(epochs, self.train_hist["acc"], label="Train Acc")
        axs[1].plot(epochs, self.val_hist["acc"], label="Val Acc")
        axs[1].set_title("Accuracy Over Epochs"); axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Accuracy")
        axs[1].set_ylim(0, 1); axs[1].grid(True)

        # 3) Precision (0..1)
        axs[2].plot(epochs, self.train_hist["prec"], label="Train Precision")
        axs[2].plot(epochs, self.val_hist["prec"], label="Val Precision")
        axs[2].set_title(("Precision (Binary)" if self.task_type == "binary" else "Macro Precision"))
        axs[2].set_xlabel("Epoch"); axs[2].set_ylabel("Precision")
        axs[2].set_ylim(0, 1); axs[2].grid(True)

        # 4) Recall (0..1)
        axs[3].plot(epochs, self.train_hist["rec"], label="Train Recall")
        axs[3].plot(epochs, self.val_hist["rec"], label="Val Recall")
        axs[3].set_title(("Recall (Binary)" if self.task_type == "binary" else "Macro Recall"))
        axs[3].set_xlabel("Epoch"); axs[3].set_ylabel("Recall")
        axs[3].set_ylim(0, 1); axs[3].grid(True)

        # 5) F1 (0..1)
        axs[4].plot(epochs, self.train_hist["f1"], label="Train F1")
        axs[4].plot(epochs, self.val_hist["f1"], label="Val F1")
        axs[4].set_title(("F1 (Binary)" if self.task_type == "binary" else "Macro F1"))
        axs[4].set_xlabel("Epoch"); axs[4].set_ylabel("F1")
        axs[4].set_ylim(0, 1); axs[4].grid(True)

        # ---- overlay dashed TEST lines on the curves, if available ----
        if self.test_metrics is not None:
            axs[0].axhline(self.test_metrics["loss"], linestyle="--", alpha=0.8,
                        label=f"Test Loss={self.test_metrics['loss']:.3f}")
            axs[1].axhline(self.test_metrics["acc"],  linestyle="--", alpha=0.8,
                        label=f"Test Acc={self.test_metrics['acc']:.3f}")
            axs[2].axhline(self.test_metrics["prec"], linestyle="--", alpha=0.8,
                        label=f"Test Prec={self.test_metrics['prec']:.3f}")
            axs[3].axhline(self.test_metrics["rec"],  linestyle="--", alpha=0.8,
                        label=f"Test Rec={self.test_metrics['rec']:.3f}")
            axs[4].axhline(self.test_metrics["f1"],   linestyle="--", alpha=0.8,
                        label=f"Test F1={self.test_metrics['f1']:.3f}")

        # Place legends after adding test lines
        for i in range(5):
            axs[i].legend()

        # 6) Test summary bars (in‑figure)
        if self.test_metrics is not None:
            names = ["Accuracy", "Precision", "Recall", "F1"]
            vals  = [self.test_metrics["acc"], self.test_metrics["prec"], self.test_metrics["rec"], self.test_metrics["f1"]]
            y = np.arange(len(names))
            bars = axs[5].barh(y, vals)
            axs[5].set_yticks(y); axs[5].set_yticklabels(names)
            axs[5].set_xlabel("Score"); axs[5].set_title("Test Set Metrics")
            axs[5].set_xlim(0, 1); axs[5].grid(axis="x", alpha=0.3)
            for b, v in zip(bars, vals):
                axs[5].text(min(v + 0.01, 0.98), b.get_y() + b.get_height()/2,
                            f"{v:.3f}", va="center", ha="left" if v <= 0.9 else "right")
        else:
            axs[5].axis("off")
            axs[5].set_title("Test metrics unavailable")

        out_path = os.path.join(self.plots_dir, self.metrics_figure)
        plt.savefig(out_path, bbox_inches="tight"); plt.close()
        print(f"Saved metrics figure to {out_path}")

        # Also save a separate test bar figure (optional; can be removed if redundant)
        if self.test_metrics is not None:
            fig2, ax2 = plt.subplots(figsize=(7, 4.5))
            names = ["Accuracy", "Precision", "Recall", "F1"]
            vals  = [self.test_metrics["acc"], self.test_metrics["prec"], self.test_metrics["rec"], self.test_metrics["f1"]]
            y = np.arange(len(names))
            bars = ax2.barh(y, vals)
            ax2.set_yticks(y); ax2.set_yticklabels(names)
            ax2.set_xlabel("Score"); ax2.set_title("Test Set Metrics")
            ax2.set_xlim(0, 1); ax2.grid(axis="x", alpha=0.3)
            for b, v in zip(bars, vals):
                ax2.text(min(v + 0.01, 0.98), b.get_y() + b.get_height()/2,
                        f"{v:.3f}", va="center", ha="left" if v <= 0.9 else "right")
            out_path_test = os.path.join(self.plots_dir, "test_metrics.png")
            plt.savefig(out_path_test, bbox_inches="tight"); plt.close()
            print(f"Saved test metrics figure to {out_path_test}")

    def plot_test_confusion_matrix(self, normalize: bool = False, cmap="Blues"):
        """Save a confusion matrix image for the test set (multiclass only)."""
        if self.task_type != "multiclass" or self.test_confusion is None:
            print("[Info] No multiclass confusion matrix to plot.")
            return

        cm = self.test_confusion.astype(np.float64)
        title = "Test Confusion Matrix (counts)"
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm = cm / row_sums
            title = "Test Confusion Matrix (row-normalized)"

        labels = self.class_names
        K = len(labels)
        fig, ax = plt.subplots(figsize=(max(8, K * 0.7), max(6, K * 0.7)))
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set(xticks=np.arange(K), yticks=np.arange(K),
            xticklabels=labels, yticklabels=labels,
            ylabel="True label", xlabel="Predicted label",
            title=title)

        # Rotate x labels for readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate cells
        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
        for i in range(K):
            for j in range(K):
                val = cm[i, j]
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center",
                        color="white" if val > thresh else "black")

        fig.tight_layout()
        out_path = os.path.join(self.plots_dir, "test_confusion_matrix.png" if not normalize else "test_confusion_matrix_norm.png")
        plt.savefig(out_path, bbox_inches="tight"); plt.close()
        print(f"Saved confusion matrix to {out_path}")


# -----------------------
# Entry
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/binary.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("data", {}).get("preprocessed_dir", None) is None:
        raise ValueError("Config must define data.preprocessed_dir")

    task_type = cfg.get("task", {}).get("type", None)
    if task_type not in ("binary", "multiclass"):
        raise ValueError("Config 'task.type' must be 'binary' or 'multiclass'.")

    # sanity check splits
    val_frac = float(cfg["data"]["val_split"])
    test_frac = float(cfg["data"]["test_split"])
    if val_frac <= 0 or test_frac <= 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError("Require 0 < val_split, test_split and val_split + test_split < 1.")

    trainer = Trainer(cfg)
    trainer.train()
    trainer.save()
    trainer.plot_metrics()
    if trainer.task_type == "multiclass":
        trainer.plot_test_confusion_matrix(normalize=False)
        trainer.plot_test_confusion_matrix(normalize=True)   # optional normalized version


if __name__ == "__main__":
    main()
