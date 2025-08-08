
from torch.utils.data import Dataset, DataLoader, Subset

class MultiClassDataset(Dataset):
    def __init__(self, base_dataset, class_list, index_to_label, label_to_index, 
                 include_unknown=True,
                include_background=True,
                background_label="_background_noise_"):
        
        self.samples = []
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

        base_index_to_label = {v: k for k, v in base_dataset.label_to_index.items()}

        for item in base_dataset:
            if isinstance(item, tuple) and len(item) == 2:
                x, label_idx = item
            elif isinstance(item, dict):
                x = item["x"]
                label_idx = item["y"]
            else:
                raise ValueError(f"Unsupported dataset item format: {item}")

            word = base_index_to_label[label_idx]  # <-- safer mapping

            if word in class_set:
                mapped_label = self.label_map[word]
            elif include_background and word == background_label:
                mapped_label = self.label_map[background_label]
            elif include_unknown:
                mapped_label = self.label_map["unknown"]
            else:
                continue  # skip this sample entirely

            self.samples.append((x, mapped_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]

