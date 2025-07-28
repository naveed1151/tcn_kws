import os
import numpy as np
import random
import matplotlib.pyplot as plt

from model.model import DilatedTCN

# ------------- Config -------------

DATA_DIR = "data/preprocessed"      # Folder with preprocessed MFCC .npy files
TARGET_WORD = "bird"                 # Target word for binary classification
MFCC_SHAPE = (49, 16)                # Shape of saved MFCC features (time frames, MFCC coeffs)
IN_CHANNELS = 16                    # Number of MFCC features (input channels)
HIDDEN_CHANNELS = 32                # Hidden channels in TCN layers
EMBEDDING_DIM = 64                  # Final embedding dimension (output of TCN)
NUM_LAYERS = 5                     # Number of TCN layers (dilations double each layer)
LEARNING_RATE = 1e-3                # Learning rate for SGD
EPOCHS = 30                        # Number of training epochs
BATCH_SIZE = 16                    # Mini-batch size for training loop
TRAIN_TEST_SPLIT = 0.8             # Train/validation split ratio
SEED = 42                         # Random seed for reproducibility

random.seed(SEED)
np.random.seed(SEED)


# ------------- Utility Functions -------------

def load_data(data_dir):
    """
    Loads all .npy MFCC files from the data directory.
    Labels samples as 1 if filename starts with TARGET_WORD, else 0.

    Returns:
        X: numpy array of shape (N_samples, T, F)
        y: numpy array of binary labels (N_samples,)
    """
    X, y = [], []
    for fname in os.listdir(data_dir):
        if fname.endswith(".npy"):
            label = 1 if fname.startswith(TARGET_WORD) else 0
            mfcc = np.load(os.path.join(data_dir, fname))
            if mfcc.shape == MFCC_SHAPE:  # Ensure consistent shape
                X.append(mfcc)
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1 / (1 + np.exp(-x))


def binary_cross_entropy(preds, targets):
    """
    Computes binary cross-entropy loss.

    Args:
        preds: predicted probabilities (batch_size,)
        targets: true binary labels (batch_size,)

    Returns:
        Average loss over batch
    """
    eps = 1e-8  # to avoid log(0)
    preds = np.clip(preds, eps, 1 - eps)
    loss = -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))
    return loss


def shuffle_data(X, y):
    """Shuffle data and labels in unison."""
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]


# ------------- Training Class -------------

class Trainer:
    def __init__(self):
        """
        Initialize the Dilated TCN model and final classification layer.
        """
        # Instantiate TCN model with given hyperparameters
        self.model = DilatedTCN(IN_CHANNELS, HIDDEN_CHANNELS, EMBEDDING_DIM,
                                num_layers=NUM_LAYERS)

        # Final linear layer weights and bias for binary classification (embedding_dim → 1)
        self.fc_w = np.random.randn(EMBEDDING_DIM, 1) * np.sqrt(2 / EMBEDDING_DIM)
        self.fc_b = np.zeros(1)

    def forward(self, x):
        """
        Forward pass for one sample.

        Args:
            x: MFCC input array (T, in_channels)

        Returns:
            prob: predicted probability (scalar)
            emb: embedding vector from TCN (embedding_dim,)
            logits: raw output logit (scalar)
        """
        emb = self.model(x)                         # Get embedding from TCN
        logits = emb @ self.fc_w + self.fc_b       # Linear layer (embedding → logit)
        prob = sigmoid(logits)[0]                   # Sigmoid to get probability
        return prob, emb, logits

    def backward(self, x, emb, logits, prob, target):
        """
        Backward pass and parameter update for one sample.

        Note:
            Currently only updates final linear layer weights.
            Full backprop through TCN is more complex and not implemented here.

        Args:
            x: input sample (not used in this simplified update)
            emb: embedding from TCN
            logits: output logit
            prob: predicted probability
            target: true label (0 or 1)
        """
        # Gradient of BCE loss wrt logit
        dL_dlogit = prob - target  # scalar

        # Gradients for final linear layer weights and bias
        dL_dfc_w = np.outer(emb, dL_dlogit)  # shape (embedding_dim, 1)
        dL_dfc_b = dL_dlogit                 # scalar

        # Update final layer weights with SGD
        self.fc_w -= LEARNING_RATE * dL_dfc_w
        self.fc_b -= LEARNING_RATE * dL_dfc_b

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model for specified epochs on the training set.

        Args:
            X_train: training inputs (N_train, T, F)
            y_train: training labels (N_train,)
            X_val: validation inputs
            y_val: validation labels
        """
        for epoch in range(1, EPOCHS + 1):
            X_train, y_train = shuffle_data(X_train, y_train)
            losses = []

            # Mini-batch training loop
            for i in range(0, len(X_train), BATCH_SIZE):
                batch_x = X_train[i:i + BATCH_SIZE]
                batch_y = y_train[i:i + BATCH_SIZE]

                batch_loss = 0
                for x, target in zip(batch_x, batch_y):
                    prob, emb, logits = self.forward(x)
                    loss = binary_cross_entropy(np.array([prob]), np.array([target]))
                    batch_loss += loss
                    self.backward(x, emb, logits, prob, target)

                losses.append(batch_loss / len(batch_x))

            avg_loss = np.mean(losses)
            val_acc = self.evaluate(X_val, y_val)

            print(f"Epoch {epoch:02d} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

    def evaluate(self, X, y):
        """
        Evaluate model accuracy on given dataset.

        Args:
            X: input samples
            y: true labels

        Returns:
            Accuracy (0-1)
        """
        correct = 0
        for x, target in zip(X, y):
            prob, _, _ = self.forward(x)
            pred = 1 if prob >= 0.5 else 0
            if pred == target:
                correct += 1
        return correct / len(y)

    def save_weights(self, folder_path="weights"):
        """
        Save trained weights to .npy files.

        Args:
            folder_path: directory where weights will be saved
        """
        os.makedirs(folder_path, exist_ok=True)

        # Save TCN weights and biases layer-wise
        for i, ((w, _), b) in enumerate(zip(self.model.weights, self.model.biases)):
            np.save(os.path.join(folder_path, f"layer_{i}_weights.npy"), w)
            np.save(os.path.join(folder_path, f"layer_{i}_biases.npy"), b)

        # Save final fully connected layer weights and bias
        np.save(os.path.join(folder_path, "fc_weights.npy"), self.fc_w)
        np.save(os.path.join(folder_path, "fc_bias.npy"), self.fc_b)

        print(f"Saved weights to folder '{folder_path}'")


def plot_weight_histogram(trainer):
    """
    Plot histogram of all weights in the model (TCN + final layer).

    Args:
        trainer: Trainer instance containing the model and weights
    """
    all_weights = []

    # Collect weights and biases from TCN layers
    for (w, _), b in zip(trainer.model.weights, trainer.model.biases):
        all_weights.append(w.flatten())
        all_weights.append(b.flatten())

    # Collect final layer weights and bias
    all_weights.append(trainer.fc_w.flatten())
    all_weights.append(trainer.fc_b.flatten())

    all_weights = np.concatenate(all_weights)

    plt.figure(figsize=(8, 5))
    plt.hist(all_weights, bins=50, color='c', edgecolor='k', alpha=0.7)
    plt.title("Histogram of Model Weights")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


# ------------- Main Execution -------------

if __name__ == "__main__":
    print("Loading data...")
    X, y = load_data(DATA_DIR)
    print(f"Loaded {len(X)} samples, positive samples (\"{TARGET_WORD}\"): {np.sum(y)}")

    # Split into training and validation sets
    split_idx = int(len(X) * TRAIN_TEST_SPLIT)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Initialize trainer
    trainer = Trainer()

    # Train the model
    trainer.train(X_train, y_train, X_val, y_val)

    # Save trained weights to disk
    trainer.save_weights()

    # Plot histogram of learned weights
    plot_weight_histogram(trainer)

    print("Training complete.")
