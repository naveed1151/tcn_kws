from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import Subset

class MultiClassDataset(Dataset):
    def __init__(self, base_dataset, class_list, index_to_label, label_to_index, 
                 include_unknown=True,
                 include_background=True,
                 background_label="_background_noise_",
                 unknown_max_ratio: float | None = None,
                 unknown_max_count: int | None = None,
                 seed: int = 0):
        
        self.samples = []
        known_samples = []
        unknown_samples = []
        background_samples = []
        base_class_names = list(class_list)
        self.label_map = {w: i for i, w in enumerate(base_class_names)}
        current_index = len(self.label_map)

        if include_unknown:
            self.label_map["unknown"] = current_index
            current_index += 1

        if include_background:
            self.label_map[background_label] = current_index
            current_index += 1

        self.class_names = list(self.label_map.keys())
        self.num_classes = len(self.class_names)

        class_set = set(class_list)

        # Resolve mapping idx(int) -> label(str); handle Subset and provided args
        base_ds = base_dataset.dataset if isinstance(base_dataset, Subset) else base_dataset
        idx2lbl = index_to_label if index_to_label is not None else getattr(base_ds, "index_to_label", None)
        lbl2idx = label_to_index if label_to_index is not None else getattr(base_ds, "label_to_index", None)
        if idx2lbl is None and lbl2idx is not None:
            idx2lbl = {v: k for k, v in lbl2idx.items()}
        if idx2lbl is None:
            raise AttributeError("index_to_label/label_to_index not found on base dataset or passed in.")
        base_index_to_label = idx2lbl

        for item in base_dataset:
            if isinstance(item, tuple) and len(item) == 2:
                x, label_idx = item
            elif isinstance(item, dict):
                x, label_idx = item.get("x"), item.get("y")
            else:
                raise ValueError(f"Unsupported dataset item format: {item}")

            word = base_index_to_label[label_idx]  # <-- safer mapping

            if word in class_set:
                mapped_label = self.label_map[word]
                known_samples.append((x, mapped_label))
            elif include_background and word == background_label:
                mapped_label = self.label_map[background_label]
                background_samples.append((x, mapped_label))
            elif include_unknown:
                mapped_label = self.label_map["unknown"]
                unknown_samples.append((x, mapped_label))
            else:
                continue  # skip

        # Optionally cap unknown count (training-only will pass limits)
        if include_unknown and (unknown_max_ratio is not None or unknown_max_count is not None):
            import numpy as np
            rng = np.random.default_rng(seed)
            known_count = len(known_samples)
            cap_ratio = int(unknown_max_ratio * known_count) if unknown_max_ratio is not None else len(unknown_samples)
            cap_abs = int(unknown_max_count) if unknown_max_count is not None else len(unknown_samples)
            cap = max(0, min(len(unknown_samples), cap_ratio, cap_abs))
            if cap < len(unknown_samples):
                idx = rng.choice(len(unknown_samples), size=cap, replace=False)
                unknown_samples = [unknown_samples[i] for i in idx]

        self.samples = known_samples + background_samples + unknown_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

