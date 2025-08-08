import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.model import DilatedTCN
from data_loader.mfcc_dataset import MFCCDataset

# --- Configuration ---
DATA_DIR = "data/preprocessed"
CHECKPOINT_PATH = "checkpoints/model_weights.pt"
BATCH_SIZE = 32
TARGET_WORD = "bird"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set to True if you want to evaluate the model with random untrained weights
use_random_weights = False

# --- Load Dataset ---
dataset = MFCCDataset(DATA_DIR)
label_to_index = dataset.label_to_index
index_to_label = {v: k for k, v in label_to_index.items()}

# Convert to binary classification: target_word → 1, others → 0
filtered = [(x, 1 if index_to_label[y] == TARGET_WORD else 0) for x, y in dataset]
X = torch.stack([item[0] for item in filtered])
y = torch.tensor([item[1] for item in filtered], dtype=torch.float32)

eval_dataset = torch.utils.data.TensorDataset(X, y)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Load Model ---
input_channels = X.shape[1]
model = DilatedTCN(
    input_channels=input_channels,
    num_layers=4,
    hidden_channels=64,
    kernel_size=3,
    num_classes=1
).to(DEVICE)

if not use_random_weights:
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
else:
    print("Evaluating with random (untrained) weights.")

model.eval()

# --- Evaluation ---
tp = tn = fp = fn = 0
start_time = time.time()

with torch.no_grad():
    loop = tqdm(eval_loader, desc=f"Evaluating '{TARGET_WORD}'", unit="batch")
    for batch_x, batch_y in loop:
        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
        outputs = model(batch_x).squeeze()
        preds = (torch.sigmoid(outputs) > 0.5).float()

        tp += ((preds == 1) & (batch_y == 1)).sum().item()
        tn += ((preds == 0) & (batch_y == 0)).sum().item()
        fp += ((preds == 1) & (batch_y == 0)).sum().item()
        fn += ((preds == 0) & (batch_y == 1)).sum().item()

end_time = time.time()
total = tp + tn + fp + fn
accuracy = (tp + tn) / total if total > 0 else 0.0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# --- Report ---
print(f"\nEvaluation on class '{TARGET_WORD}':")
print(f"  Total Samples     : {int(total)}")
print(f"  Accuracy          : {accuracy * 100:.2f}%")
print(f"  Precision         : {precision * 100:.2f}%")
print(f"  Recall            : {recall * 100:.2f}%")
print(f"  F1 Score          : {f1 * 100:.2f}%")
print(f"  True Positives    : {int(tp)}")
print(f"  True Negatives    : {int(tn)}")
print(f"  False Positives   : {int(fp)}")
print(f"  False Negatives   : {int(fn)}")
print(f"  Evaluation Time   : {end_time - start_time:.2f} seconds")

# --- Confusion Matrix ---
print("\nConfusion Matrix:")
print("                 Predicted")
print("               |   0   |   1   |")
print("           -------------------")
print(f"Actual    0   | {int(tn):5} | {int(fp):5} |")
print(f"          1   | {int(fn):5} | {int(tp):5} |")
