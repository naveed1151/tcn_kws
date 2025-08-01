import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MFCCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to preprocessed data directory (contains one subfolder per label)
            transform (callable, optional): Optional transform applied on a sample tensor shaped (C, T)
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []          # list of (file_path, label_idx)
        self.label_to_index = {}   # e.g., {"yes":0, "no":1, ...}
        self._prepare_dataset()

        # quick access to labels only (no I/O)
        self.labels = [lbl for _, lbl in self.samples]

    def _prepare_dataset(self):
        label_idx = 0
        # Stable ordering for reproducibility
        for label_name in sorted(os.listdir(self.root_dir)):
            label_path = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            if label_name not in self.label_to_index:
                self.label_to_index[label_name] = label_idx
                label_idx += 1

            for fname in sorted(os.listdir(label_path)):
                if fname.lower().endswith(".npy"):
                    file_path = os.path.join(label_path, fname)
                    self.samples.append((file_path, self.label_to_index[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # mfcc saved as (time, n_mfcc) in your preprocessor => transpose to (C, T)
        # If needed, set allow_pickle=False for safety.
        mfcc = np.load(file_path)
        x = torch.from_numpy(mfcc).float()        # (T, C)
        x = x.transpose(0, 1).contiguous()        # (C, T)

        if self.transform is not None:
            x = self.transform(x)                 # transform expects (C, T)

        return x, label


# Example quick check
if __name__ == "__main__":
    ds = MFCCDataset("data/preprocessed")
    print("Num samples:", len(ds))
    x0, y0 = ds[0]
    print("First item:", x0.shape, y0)           # expect (C, T), e.g., (16 or 28, ~60-100)

    # Simple split (your training code does stratified splits anyway)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=16, shuffle=False)

    bx, by = next(iter(train_loader))
    print("Batch X shape:", bx.shape)            # (B, C, T)
    print("Batch y shape:", by.shape)            # (B,)
