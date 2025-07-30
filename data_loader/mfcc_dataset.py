import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MFCCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to preprocessed data directory
            transform (callable, optional): Optional transform to apply on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.samples = []
        self.label_to_index = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        label_idx = 0
        for label_name in sorted(os.listdir(self.root_dir)):
            label_path = os.path.join(self.root_dir, label_name)
            if not os.path.isdir(label_path):
                continue
            if label_name not in self.label_to_index:
                self.label_to_index[label_name] = label_idx
                label_idx += 1
            
            for fname in os.listdir(label_path):
                if fname.endswith('.npy'):
                    file_path = os.path.join(label_path, fname)
                    self.samples.append((file_path, self.label_to_index[label_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        mfcc = np.load(file_path)
        # Convert to float32 tensor, shape: (time, features)
        mfcc_tensor = torch.from_numpy(mfcc).float()
        
        if self.transform:
            mfcc_tensor = self.transform(mfcc_tensor)
        
        return mfcc_tensor, label

# Example usage:

if __name__ == "__main__":
    dataset = MFCCDataset("data/preprocessed")
    
    # Split into train and val sets (e.g., 80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # Iterate over one batch to check
    for batch_x, batch_y in train_loader:
        print("Batch X shape:", batch_x.shape)  # (batch_size, time, features)
        print("Batch y shape:", batch_y.shape)
        break
