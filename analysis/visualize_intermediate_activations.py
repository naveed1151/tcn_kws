import os
import matplotlib as mpl
mpl.rcParams.update({'font.size': 22})  # Set global font size
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
import matplotlib.animation as animation


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



def animation_in_time(input_mfcc: torch.Tensor, activations: List[torch.Tensor], pooled: torch.Tensor, logits: torch.Tensor, save_path: str = "plots/activations_time_animation.gif"):
    """
    Animates all layer heatmaps at once, revealing one additional time step per frame.
    """
    import matplotlib.animation as animation
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_position([0.08, 0.18, 0.84, 0.74])
    hop_length_s = float(cfg['data']['mfcc']['hop_length_s']) if 'cfg' in globals() else 0.016
    arrs = []
    arrs.append(input_mfcc.squeeze(0).detach().cpu().numpy())
    for act in activations:
        arr = act.squeeze(0)
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        arrs.append(arr.numpy())
    pooled_arr = pooled.numpy().T
    pooled_max = pooled_arr.max() if pooled_arr.max() != 0 else 1.0
    pooled_arr_norm = pooled_arr / pooled_max
    arrs.append(pooled_arr_norm)
    logits_arr = logits.numpy().T
    logits_arr_norm = logits_arr / np.max(np.abs(logits_arr)) if np.max(np.abs(logits_arr)) != 0 else logits_arr
    arrs.append(logits_arr_norm)
    global_max = max(a.max() for a in arrs[:-2])
    norm = colors.Normalize(vmin=0, vmax=global_max)
    norm_pooled = colors.Normalize(vmin=0, vmax=1)
    norm_logits = colors.Normalize(vmin=-1, vmax=1)
    num_blocks = len(activations)
    z_labels = ["input"] + [f"res_block {i}" for i in range(num_blocks)] + ["pooled", "fc"]
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
    # Find max time steps
    max_time = max(a.shape[1] for a in arrs if a.ndim > 1)
    def animate(t):
        ax.clear()
        for z, arr in enumerate(arrs):
            arr_plot = arr[:, :min(t+1, arr.shape[1])] if arr.ndim > 1 else arr
            if arr_plot.shape[1] == 1:
                time_axis = np.arange(8) * hop_length_s
                arr_plot = np.repeat(arr_plot, 8, axis=1)
            else:
                time_axis = np.arange(arr_plot.shape[1]) * hop_length_s
            y, x_ = np.meshgrid(np.arange(arr_plot.shape[0]), time_axis, indexing='ij')
            z_arr = np.full_like(x_, z)
            if z == len(arrs) - 2:
                surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm_pooled(arr_plot)), rstride=1, cstride=1, shade=False)
            elif z == len(arrs) - 1:
                surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm_logits(arr_plot)), rstride=1, cstride=1, shade=False)
            else:
                surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm(arr_plot)), rstride=1, cstride=1, shade=False)
        ax.set_xlabel('Time', fontsize=28)
        ax.set_ylabel('Channels', fontsize=28)
        ax.set_zlabel('Block', fontsize=28)
        ax.set_zticks(np.arange(len(arrs)))
        ax.set_zticklabels(z_labels, fontsize=24)
        plt.title('TCN Activations: Revealing Time Steps', fontsize=32)
        plt.figtext(0.01, 0.98, f"File: {filename}", fontsize=22, va='top', ha='left', color='navy')
        plt.figtext(0.01, 0.93, f"Detected keyword: {detected_keyword}", fontsize=26, va='top', ha='left', color='darkred')
    ani = animation.FuncAnimation(fig, animate, frames=max_time, interval=300, blit=False, repeat=False)
    # Add colorbar (for intermediate activations)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(np.concatenate([a.flatten() for a in arrs[:-2]]))
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.4, pad=0.1)
    cbar.set_label('Activation Amplitude (Intermediate)', fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    ani.save(save_path, writer='pillow')
    print(f"Time-step animation saved to {save_path}")

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
        pooled = model.pool(out).squeeze(-1)  # (N, hidden)
        # Use model's own forward for logits to ensure correct inference
        logits = model(x)
    return activations, pooled.detach().cpu(), logits.detach().cpu()


def plot_activations_3d(input_mfcc: torch.Tensor, activations: List[torch.Tensor], pooled: torch.Tensor, logits: torch.Tensor, save_path: str = None):
    # Draw black border around the 3D plot axes
    from matplotlib.patches import Rectangle
    pos = ax.get_position()
    fig.add_artist(Rectangle((pos.x0, pos.y0), pos.width, pos.height,
                             fill=False, edgecolor='black', linewidth=3, zorder=1000))

    
    """
    Plots a 3D visualization of activations.
    X: time, Y: channels, Z: block index
    Each block's activations are shown as a heatmap at its Z position.
    """
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    # Make 3D plot fill more of the figure area
    ax.set_position([0.08, 0.18, 0.84, 0.74])  # [left, bottom, width, height]

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
    # Normalize pooled to its own max
    pooled_max = pooled_arr.max() if pooled_arr.max() != 0 else 1.0
    pooled_arr_norm = pooled_arr / pooled_max
    # Normalize logits to 1 (softmax not needed for visualization)
    logits_arr_norm = logits_arr / np.max(np.abs(logits_arr)) if np.max(np.abs(logits_arr)) != 0 else logits_arr
    arrs.append(pooled_arr_norm)
    arrs.append(logits_arr_norm)
    #print("Pooled activations (repeated):\n", pooled_arr)
    #print("Logits (repeated):\n", logits_arr)

    # Compute global max for normalization for intermediate activations
    global_max = max(a.max() for a in arrs[:-2])
    print(len(arrs), "arrays to plot, global max:", global_max)
    norm = colors.Normalize(vmin=0, vmax=global_max)
    # Norms for pooled and logits
    norm_pooled = colors.Normalize(vmin=0, vmax=1)
    norm_logits = colors.Normalize(vmin=-1, vmax=1)

    # Plot all heatmaps
    for z, arr in enumerate(arrs):
        print(arr.shape)
        if arr.shape[1] == 1:
            time_axis = np.arange(8) * hop_length_s
            arr = np.repeat(arr, 8, axis=1)
        else:
            time_axis = np.arange(arr.shape[1]) * hop_length_s
        y, x_ = np.meshgrid(np.arange(arr.shape[0]), time_axis, indexing='ij')
        z_arr = np.full_like(x_, z)
        # Use separate normalization for pooled and logits
        if z == len(arrs) - 2:
            surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm_pooled(arr)), rstride=1, cstride=1, shade=False)
        elif z == len(arrs) - 1:
            surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm_logits(arr)), rstride=1, cstride=1, shade=False)
        else:
            surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm(arr)), rstride=1, cstride=1, shade=False)

    ax.set_xlabel('Time', fontsize=28)
    ax.set_ylabel('Channels', fontsize=28)
    ax.set_zlabel('Block', fontsize=28)
    ax.set_zticks(np.arange(len(arrs)))

    # Set custom z-tick labels
    num_blocks = len(activations)
    z_labels = ["input"] + [f"res_block {i}" for i in range(num_blocks)] + ["pooled", "fc"]
    ax.set_zticklabels(z_labels, fontsize=24)

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

    plt.title('Temporal Convolutional Network\nIntermediate Activations (Input + 4 Blocks + Pooled + Logits)', fontsize=32)
    plt.figtext(0.01, 0.98, f"File: {filename}", fontsize=22, va='top', ha='left', color='navy')
    plt.figtext(0.01, 0.93, f"Detected keyword: {detected_keyword}", fontsize=26, va='top', ha='left', color='darkred')

    # Shared colorbar for intermediate activations
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(np.concatenate([a.flatten() for a in arrs[:-2]]))
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label('Activation Amplitude (Intermediate)', fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    # Separate colorbar for pooled
    mappable_pooled = plt.cm.ScalarMappable(cmap=cmap, norm=norm_pooled)
    mappable_pooled.set_array(arrs[-2].flatten())
    cbar_pooled = plt.colorbar(mappable_pooled, ax=ax, shrink=0.7, pad=0.1)
    cbar_pooled.set_label('Pooled Activation (Normalized)', fontsize=24)
    cbar_pooled.ax.tick_params(labelsize=20)
    # Separate colorbar for logits
    mappable_logits = plt.cm.ScalarMappable(cmap=cmap, norm=norm_logits)
    mappable_logits.set_array(arrs[-1].flatten())
    cbar_logits = plt.colorbar(mappable_logits, ax=ax, shrink=0.7, pad=0.1)
    cbar_logits.set_label('Logits (Normalized)', fontsize=24)
    cbar_logits.ax.tick_params(labelsize=20)

    if save_path:
        plt.savefig(save_path)
    plt.show()


def animate_activations(input_mfcc: torch.Tensor, activations: List[torch.Tensor], pooled: torch.Tensor, logits: torch.Tensor, save_path: str = "plots/activations_animation.mp4"):
    from matplotlib.patches import Rectangle
    """
    Creates an animation where each heatmap (input, blocks, pooled, fc) is shown one by one for 0.5s.
    Saves the animation to the specified path.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[8, 1], hspace=0.35)
    ax = fig.add_subplot(gs[0], projection='3d')
    ax_softmax = fig.add_subplot(gs[1])
    fig.subplots_adjust(hspace=0.35)
    # Make 3D plot fill more of the subplot area
    ax.set_position([0.08, 0.18, 0.84, 0.74])  # [left, bottom, width, height]
    # Make softmax subplot narrower
    pos2 = ax_softmax.get_position()
    ax_softmax.set_position([pos2.x0 + 0.15, pos2.y0, pos2.width * 0.6, pos2.height])
    hop_length_s = float(cfg['data']['mfcc']['hop_length_s']) if 'cfg' in globals() else 0.016
    arrs = []
    arrs.append(input_mfcc.squeeze(0).detach().cpu().numpy())
    for act in activations:
        arr = act.squeeze(0)
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        arrs.append(arr.numpy())
    pooled_arr = pooled.numpy().T
    pooled_max = pooled_arr.max() if pooled_arr.max() != 0 else 1.0
    pooled_arr_norm = pooled_arr / pooled_max
    arrs.append(pooled_arr_norm)
    # Remove logits from arrs for animation
    global_max = max(a.max() for a in arrs)
    norm = colors.Normalize(vmin=0, vmax=global_max)
    norm_pooled = colors.Normalize(vmin=0, vmax=1)
    num_blocks = len(activations)
    z_labels = ["input"] + [f"res_block {i}" for i in range(num_blocks)] + ["pooled"]
    # Get y-axis limits (channels)
    y_min = 0
    y_max = max(arr.shape[0] for arr in arrs)
    # Get filename and detected keyword
    filename = globals().get('mfcc_npy_file', 'unknown')
    logits_arr = logits.numpy().flatten()
    softmax_vals = np.exp(logits_arr) / np.sum(np.exp(logits_arr))
    try:
        class_list = cfg['task']['class_list']
        if cfg['task'].get('include_unknown', False):
            class_list = class_list + ['unknown']
        if cfg['task'].get('include_background', False):
            class_list = class_list + ['background']
        keyword_idx = int(np.argmax(softmax_vals))
        detected_keyword = class_list[keyword_idx] if keyword_idx < len(class_list) else str(keyword_idx)
    except Exception:
        detected_keyword = 'N/A'

    # Create mappable and colorbar once
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(arrs[0].flatten())
    cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal', pad=0.2, shrink=0.4)
    cbar.set_label('Activation Amplitude (Intermediate)', fontsize=24)
    cbar.ax.tick_params(labelsize=20)

    def init():
        ax.clear()
        ax_softmax.clear()
    # ...existing code...
        ax.set_xlabel('Time', fontsize=28)
        ax.set_ylabel('Channels', fontsize=28)
        ax.set_zlabel('Block', fontsize=28)
        ax.set_zticks(np.arange(len(arrs)))
        ax.set_zticklabels(z_labels, fontsize=24)
        ax.set_ylim(y_min, y_max)
        plt.title('Temporal Convolutional Network\nIntermediate Activations (Input + Blocks + Pooled)', fontsize=32)
        plt.figtext(0.01, 0.98, f"File: {filename}", fontsize=22, va='top', ha='left', color='navy')
        plt.figtext(0.01, 0.93, f"Detected keyword: {detected_keyword}", fontsize=26, va='top', ha='left', color='darkred')
        # Draw initial surface
        arr = arrs[0]
        if arr.shape[1] == 1:
            time_axis = np.arange(8) * hop_length_s
            arr = np.repeat(arr, 8, axis=1)
        else:
            time_axis = np.arange(arr.shape[1]) * hop_length_s
        y, x_ = np.meshgrid(np.arange(arr.shape[0]), time_axis, indexing='ij')
        z_arr = np.full_like(x_, 0)
        ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm(arr)), rstride=1, cstride=1, shade=False)
        # Draw softmax bar plot
        ax_softmax.set_title('Softmax Output (Keyword Probabilities)', fontsize=28)
        ax_softmax.set_ylim(0, 1)
        ax_softmax.set_xticks(np.arange(len(class_list)))
        ax_softmax.set_xticklabels(class_list, rotation=45, ha='right', fontsize=22)
        bars = ax_softmax.bar(np.arange(len(class_list)), softmax_vals, color='gray')
        bars[keyword_idx].set_color('green')

    def animate(i):
        ax.clear()
        ax_softmax.clear()
    # ...existing code...
        arr = arrs[i]
        if arr.shape[1] == 1:
            time_axis = np.arange(8) * hop_length_s
            arr = np.repeat(arr, 8, axis=1)
        else:
            time_axis = np.arange(arr.shape[1]) * hop_length_s
        y, x_ = np.meshgrid(np.arange(arr.shape[0]), time_axis, indexing='ij')
        z_arr = np.full_like(x_, i)
        if i == len(arrs) - 1:
            surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm_pooled(arr)), rstride=1, cstride=1, shade=False)
            mappable.set_norm(norm_pooled)
            cbar.set_label('Pooled Activation (Normalized)', fontsize=24)
        else:
            surf = ax.plot_surface(x_, y, z_arr, facecolors=cmap(norm(arr)), rstride=1, cstride=1, shade=False)
            mappable.set_norm(norm)
            cbar.set_label('Activation Amplitude (Intermediate)', fontsize=24)
        ax.set_xlabel('Time', fontsize=28)
        ax.set_ylabel('Channels', fontsize=28)
        ax.set_zlabel('Block', fontsize=28)
        ax.set_zticks(np.arange(len(arrs)))
        ax.set_zticklabels(z_labels, fontsize=24)
        ax.set_ylim(y_min, y_max)
        plt.title('Temporal Convolutional Network\nIntermediate Activations (Input + Blocks + Pooled)', fontsize=32)
        plt.figtext(0.01, 0.98, f"File: {filename}", fontsize=22, va='top', ha='left', color='navy')
        plt.figtext(0.01, 0.93, f"Detected keyword: {detected_keyword}", fontsize=26, va='top', ha='left', color='darkred')
        # Update colorbar mappable
        mappable.set_array(arr.flatten())
        cbar.ax.tick_params(labelsize=20)
        # Draw softmax bar plot
        ax_softmax.set_title('Softmax Output (Keyword Probabilities)', fontsize=28)
        ax_softmax.set_ylim(0, 1)
        ax_softmax.set_xticks(np.arange(len(class_list)))
        ax_softmax.set_xticklabels(class_list, rotation=45, ha='right', fontsize=22)
        bars = ax_softmax.bar(np.arange(len(class_list)), softmax_vals, color='gray')
        bars[keyword_idx].set_color('green')
    ani = animation.FuncAnimation(fig, animate, frames=len(arrs), init_func=init, interval=1000, blit=False, repeat=False)
    ani.save(save_path, writer='pillow')
    print(f"Animation saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TCN Activation Visualization")
    parser.add_argument('--mode', choices=['static', 'animate', 'animation_in_time'], default='static', help='Choose plot mode: static, animate, or animation_in_time')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save the plot or animation')
    args = parser.parse_args()

    # Update these paths as needed
    config_path = "config/base.yaml"
    weights_path = "checkpoints/model_weights_fp.pt"
    mfcc_npy_file = "data/preprocessed/yes/0a7c2a8d_nohash_0.npy"  # Example path, update as needed

    # Load config and model
    cfg = load_config(config_path)
    model = build_model_from_cfg(cfg)
    model = load_state_dict_forgiving(model, weights_path, device=torch.device("cpu"))
    model.eval()

    # Print state of dropout and normalization layers
    print("\nDropout and Normalization Layer States:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            print(f"Dropout: {name}, training={module.training}, p={module.p}")
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.GroupNorm, torch.nn.LayerNorm)):
            print(f"Norm: {name}, training={module.training}")

    # Load preprocessed MFCC numpy file
    mfcc = np.load(mfcc_npy_file)
    # mfcc shape: (time, n_mfcc) -> (1, n_mfcc, time)
    input_mfcc = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)

    # Extract activations
    acts, pooled, logits = extract_intermediate_activations(model, input_mfcc)
    print(f"Extracted {len(acts)} activations.")
    for i, a in enumerate(acts):
        print(f"Block {i}: shape {a.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Logits shape: {logits.shape}")

    # Print maximum value of input MFCC
    input_max = input_mfcc.max().item()
    print("Maximum value of input MFCC:", input_max)

    if args.mode == 'static':
        plot_activations_3d(input_mfcc, acts, pooled, logits, save_path=args.save_path)
    elif args.mode == 'animate':
        save_path = args.save_path or "plots/activations_animation.gif"
        animate_activations(input_mfcc, acts, pooled, logits, save_path=save_path)
    elif args.mode == 'animation_in_time':
        save_path = args.save_path or "plots/activations_time_animation.gif"
        animation_in_time(input_mfcc, acts, pooled, logits, save_path=save_path)
