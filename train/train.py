# --- Top of the file (same imports) ---
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from model.model import DilatedTCN
from data_loader.mfcc_dataset import MFCCDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Config Globals ---
DATA_DIR = "data/preprocessed"
TARGET_WORD = "bird"
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-5
VAL_SPLIT = 0.2
POS_WEIGHT = None       # e.g. 30.0 or None
THRESHOLD = 0.5         # e.g. 0.2 or None to default to 0.5
WEIGHT_DECAY = 0.0      # e.g. 1e-5 or 0.0 to disable
DROPOUT = 0.0           # e.g. 0.3 or 0.0 to disable
NEG_DOWNSAMPLE_RATIO = None  # e.g. 2.0 keeps 1 pos : 2 neg, or None to disable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Wrapping and Filtering ---
class BinaryKeywordDataset(Dataset):
    def __init__(self, base_dataset, target_word, index_to_word, downsample_ratio=None):
        self.target_word = target_word
        self.index_to_word = index_to_word
        samples = []
        for x, label in base_dataset:
            word = index_to_word[label]
            binary_label = 1 if word == target_word else 0
            samples.append((x, binary_label))

        if downsample_ratio is not None:
            positives = [s for s in samples if s[1] == 1]
            negatives = [s for s in samples if s[1] == 0]
            keep_neg = int(len(positives) * downsample_ratio)
            negatives = negatives[:keep_neg]
            samples = positives + negatives
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --- Trainer Class ---
class Trainer:
    def __init__(self, data_dir, target_word=TARGET_WORD, batch_size=BATCH_SIZE,
                 epochs=NUM_EPOCHS, lr=LEARNING_RATE, val_split=VAL_SPLIT, device=DEVICE):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.val_split = val_split
        self.data_dir = data_dir
        self.target_word = target_word

        full_dataset = MFCCDataset(self.data_dir)
        self.index_to_word = {v: k for k, v in full_dataset.label_to_index.items()}
        binary_dataset = BinaryKeywordDataset(full_dataset, target_word, self.index_to_word, downsample_ratio=NEG_DOWNSAMPLE_RATIO)

        val_size = int(len(binary_dataset) * self.val_split)
        train_size = len(binary_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(binary_dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        sample_x, _ = self.train_dataset[0]
        input_channels = sample_x.shape[0]
        seq_len = sample_x.shape[1]
        self.num_layers = self._calc_layers(seq_len)

        self.model = DilatedTCN(
            input_channels=input_channels,
            num_layers=self.num_layers,
            hidden_channels=64,
            kernel_size=3,
            num_classes=1,
            dropout=DROPOUT
        ).to(self.device)

        if POS_WEIGHT is not None:
            pos_weight_tensor = torch.tensor([POS_WEIGHT], device=self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=WEIGHT_DECAY)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1s = []

    def _calc_layers(self, seq_len, kernel_size=3, dilation_base=2):
        L = 0
        receptive_field = 1
        while receptive_field < seq_len:
            receptive_field += (kernel_size - 1) * (dilation_base ** L)
            L += 1
        return L

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_x, batch_y in loop:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.float().unsqueeze(1).to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        tp = 0
        fp = 0
        fn = 0

        loop = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for batch_x, batch_y in loop:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().unsqueeze(1).to(self.device)

                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item()

                threshold = THRESHOLD if THRESHOLD is not None else 0.5
                preds = (torch.sigmoid(logits) > threshold).float()

                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                tp += ((preds == 1) & (batch_y == 1)).sum().item()
                fp += ((preds == 1) & (batch_y == 0)).sum().item()
                fn += ((preds == 0) & (batch_y == 1)).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        self.val_f1s.append(f1)

        return accuracy

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"Epoch {epoch}/{self.epochs}:")
            train_loss = self.train_epoch()
            val_acc = self.validate()
            print(f"  Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")

    def save(self, path="model_weights.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def plot_metrics(self, out_dir="plots"):
        os.makedirs(out_dir, exist_ok=True)

        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        fig.tight_layout(pad=5.0)

        axs[0].plot(self.train_losses, label="Train Loss")
        axs[0].plot(self.val_losses, label="Validation Loss")
        axs[0].set_title("Loss Over Epochs")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(self.val_accuracies, label="Validation Accuracy")
        axs[1].set_title("Validation Accuracy Over Epochs")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        axs[1].legend()
        axs[1].grid(True)

        axs[2].plot(self.val_precisions, label="Precision")
        axs[2].plot(self.val_recalls, label="Recall")
        axs[2].plot(self.val_f1s, label="F1 Score")
        axs[2].set_title("Precision, Recall, F1 Over Epochs")
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Score")
        axs[2].legend()
        axs[2].grid(True)

        plt.savefig(os.path.join(out_dir, "metrics.png"))
        plt.close()


# --- Main ---
if __name__ == "__main__":
    trainer = Trainer(data_dir=DATA_DIR, target_word=TARGET_WORD)
    trainer.train()
    trainer.save()
    trainer.plot_metrics()
