import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple

# Choices for matplotlib colormaps (cmap):
# Sequential: viridis, plasma, inferno, magma, cividis
# Diverging: coolwarm, bwr, seismic, RdBu, PiYG
# Qualitative: tab10, tab20, Set1, Set2, Pastel1, Accent
# Miscellaneous: jet, rainbow, gist_ncar, cubehelix, flag, prism
# See: https://matplotlib.org/stable/users/explain/colors/colormaps.html

# Set the colormap to use for all plots here:
cmap = plt.cm.cividis

from train.utils import build_model_from_cfg, load_config, load_state_dict_forgiving
from feature_extraction.extract_mfcc import extract_mfcc

def extract_intermediate_activations(model: torch.nn.Module, x: torch.Tensor) -> List[torch.Tensor]:
    """
    Hooks into each block (Conv/Linear) and collects activations during forward pass.
    Returns a list of activations (one per block).
    """
    # Match DilatedTCN forward: tcn -> pool -> dropout -> fc
    # Collect output after each block in tcn
    activations = []
    tcn = model.tcn
    out = x
    with torch.no_grad():
        for block in tcn:
            out = block(out)
            activations.append(out.detach().cpu())
        # After all blocks, apply pool, dropout, fc
        pooled = model.pool(out).squeeze(-1)  # (N, hidden)
        dropped = model.dropout(pooled)
        logits = model.fc(dropped)
    return activations, pooled.detach().cpu(), logits.detach().cpu()


def plot_activations_3d(input_mfcc: torch.Tensor, activations: List[torch.Tensor], pooled: torch.Tensor, logits: torch.Tensor, save_path: str = None):
    
    """
    Plots a 3D visualization of activations.
    X: time, Y: channels, Z: block index
    Each block's activations are shown as a heatmap at its Z position.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Only plot the first 3 blocks for testing
    
    hop_length_s = float(cfg['data']['mfcc']['hop_length_s']) if 'cfg' in globals() else 0.016
    arrs = []
    # First block: input
    arrs.append(input_mfcc.squeeze(0).detach().cpu().numpy())
    # Next blocks: activations
    for act in activations:
        arr = act.squeeze(0)
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        arrs.append(arr.numpy())
    # Add pooled and FC, repeat along time axis for visibility
    pooled_arr = pooled.numpy()  # shape (1, hidden_channels)
    logits_arr = logits.numpy()  # shape (1, num_classes)
    # Reshape to (channels, 1) and (classes, 1) for consistent plotting
    pooled_arr = pooled_arr.T  # (hidden_channels, 1)
    logits_arr = logits_arr.T  # (num_classes, 1)
    arrs.append(pooled_arr)
    arrs.append(logits_arr)
    #print("Pooled activations (repeated):\n", pooled_arr)
    #print("Logits (repeated):\n", logits_arr)

    # Compute global max for normalization
    global_max = max(a.max() for a in arrs)
    print(len(arrs), "arrays to plot, global max:", global_max)
    norm = colors.Normalize(vmin=0, vmax=global_max)
    # Plot only up to max_blocks (exclude pooled and FC heatmaps)
    for z, arr in enumerate(arrs[:]):
        print(arr.shape)
        # For pooled and logits, shape is (1, N), so repeat along time axis (axis=1) if needed
        if arr.shape[1] == 1:
            time_axis = np.arange(8) * hop_length_s  # 8 time steps for visibility
            arr = np.repeat(arr, 8, axis=1)
        else:
            time_axis = np.arange(arr.shape[1]) * hop_length_s
        y, x_ = np.meshgrid(np.arange(arr.shape[0]), time_axis, indexing='ij')
        z_arr = np.full_like(x_, z)
        surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm(arr)), rstride=1, cstride=1, shade=False)

    ax.set_xlabel('Time')
    ax.set_ylabel('Channels')
    ax.set_zlabel('Block')
    ax.set_zticks(np.arange(len(arrs)))

    # Add filename and detected keyword to the plot
    filename = globals().get('mfcc_npy_file', 'unknown')
    fc_arr = arrs[-1]
    try:
        keyword_idx = int(np.argmax(fc_arr))
        class_list = cfg['task']['class_list']
        if cfg['task'].get('include_unknown', False):
            class_list = class_list + ['unknown']
        if cfg['task'].get('include_background', False):
            class_list = class_list + ['background']
        detected_keyword = class_list[keyword_idx] if keyword_idx < len(class_list) else str(keyword_idx)
    except Exception:
        detected_keyword = 'N/A'

    plt.title('Temporal Convolutional Network\nIntermediate Activations (MFCC + First 2 Blocks)')
    plt.figtext(0.01, 0.98, f"File: {filename}", fontsize=10, va='top', ha='left', color='navy')
    plt.figtext(0.01, 0.93, f"Detected keyword: {detected_keyword}", fontsize=12, va='top', ha='left', color='darkred')

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(np.concatenate([a.flatten() for a in arrs]))
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label('Activation Amplitude')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def test_extract_and_plot():
    """
    Simple test with a dummy model and random input.
    """
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(13, 32, 3, padding=1)
            self.conv2 = torch.nn.Conv1d(32, 16, 3, padding=1)
            self.fc = torch.nn.Linear(16 * 10, 8)
        def forward(self, x):
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            x = torch.relu(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    # Simulate MFCC input: (batch, channels, time)
    x = torch.randn(1, 13, 10)
    model = DummyModel()
    acts = extract_intermediate_activations(model, x)
    print(f"Extracted {len(acts)} activations.")
    for i, a in enumerate(acts):
        print(f"Block {i}: shape {a.shape}")
    plot_activations_3d(acts, x)

if __name__ == "__main__":
    # Uncomment below to run the dummy test
    # test_extract_and_plot()

    # --- Real model and MFCC input test ---
    

    # Update these paths as needed
    config_path = "config/base.yaml"
    weights_path = "checkpoints/model_weights_fp.pt"
    mfcc_npy_file = "data/preprocessed/yes/0a7c2a8d_nohash_0.npy"  # Example path, update as needed

    # Load config and model
    cfg = load_config(config_path)
    model = build_model_from_cfg(cfg)
    model = load_state_dict_forgiving(model, weights_path, device=torch.device("cpu"))
    model.eval()

    # Load preprocessed MFCC numpy file
    mfcc = np.load(mfcc_npy_file)
    # mfcc shape: (time, n_mfcc) -> (1, n_mfcc, time)
    input_mfcc = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)

    # Extract and plot activations
    acts, pooled, logits = extract_intermediate_activations(model, input_mfcc)
    print(f"Extracted {len(acts)} activations.")
    for i, a in enumerate(acts):
        print(f"Block {i}: shape {a.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Logits shape: {logits.shape}")

    # Print maximum value of input MFCC
    input_max = input_mfcc.max().item()
    print("Maximum value of input MFCC:", input_max)

    plot_activations_3d(input_mfcc, acts, pooled, logits)
