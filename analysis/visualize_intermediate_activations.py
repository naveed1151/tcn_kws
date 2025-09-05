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
    # Add waveform subplot at the top
    num_layers = len(activations) + 1  # input + activations
    # Increase the height of each heatmap subplot and reduce spacing
    fig_height = 2.7 * (num_layers + 3)
    # Make waveform taller, heatmaps taller, softmax a bit shorter, and add extra space above and below softmax
    # Increase the gap above the softmax plot by adding a spacer row
    height_ratios = [1.6] + [2.2]*num_layers + [0.7, 2.0, 0.7]  # [wave, heatmaps..., spacer, softmax, bottom spacer]
    fig = plt.figure(figsize=(18, fig_height))
    gs = fig.add_gridspec(num_layers + 4, 1, height_ratios=height_ratios, hspace=0.32)
    fig.subplots_adjust(left=0.12, right=0.80, hspace=0.18)
    ax_wave = fig.add_subplot(gs[0])
    ax_layers = [fig.add_subplot(gs[i+1]) for i in range(num_layers)]
    # Add a spacer row before the softmax plot
    ax_softmax = fig.add_subplot(gs[num_layers+2])
    # The last gs slot is just for extra space below softmax
    # ...existing code...
    hop_length_s = float(cfg['data']['mfcc']['hop_length_s']) if 'cfg' in globals() else 0.016
    arrs = []
    arrs.append(input_mfcc.squeeze(0).detach().cpu().numpy())
    for act in activations:
        arr = act.squeeze(0)
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        arrs.append(arr.numpy())

    # Get the waveform and sampling rate for the animation
    # Use the same wav_file and sr as in extract_mfcc
    # We'll pass the waveform and sr as globals for now
    waveform = globals().get('waveform_for_anim', None)
    # Get sample rate from config if available, else fallback to global or 16000
    try:
        sr = int(cfg['data']['audio']['sample_rate'])
    except Exception:
        sr = globals().get('sr_for_anim', 16000)
    global_max = max(a.max() for a in arrs)
    norm = colors.Normalize(vmin=0, vmax=global_max)
    num_blocks = len(activations)
    z_labels = ["input"] + [f"res_block {i}" for i in range(num_blocks)]
    # Try to get the .wav file name for display
    wav_file = globals().get('wav_file', None)
    if wav_file is not None:
        # Show parent directory (keyword) and filename, e.g., 'yes/0a7c2a8d_nohash_0.wav'
        parent = os.path.basename(os.path.dirname(wav_file))
        base = os.path.basename(wav_file)
        filename = f"{parent}/{base}"
    else:
        filename = globals().get('mfcc_npy_file', 'unknown')
    try:
        class_list = cfg['task']['class_list']
        if cfg['task'].get('include_unknown', False):
            class_list = class_list + ['unknown']
        if cfg['task'].get('include_background', False):
            class_list = class_list + ['background']
    except Exception:
        class_list = [str(i) for i in range(11)]
    # Find number of MFCC frames in time
    num_mfcc_in_time = max(a.shape[1] for a in arrs if a.ndim > 1)
    n_classes = len(class_list)
    # Set total_time to match the waveform duration for perfect alignment
    waveform = globals().get('waveform_for_anim', None)
    sr = None
    try:
        sr = int(cfg['data']['audio']['sample_rate'])
    except Exception:
        sr = globals().get('sr_for_anim', 16000)
    if waveform is not None and sr is not None:
        total_time = len(waveform) / sr
    else:
        total_time = (num_mfcc_in_time - 1) * hop_length_s
    # Add a centered title at the top
    main_title = plt.figtext(0.5, 0.98, "Temporal Convolutional Network\nfor keyword spotting", fontsize=28, va='top', ha='center', color='black', fontweight='bold')
    # Pre-create text objects for file and keyword, update them in animate
    file_text = plt.figtext(0.01, 0.98, f"File: {filename}", fontsize=20, va='top', ha='left', color='navy')
    keyword_text = plt.figtext(0.01, 0.93, f"Detected keyword: ''", fontsize=24, va='top', ha='left', color='darkred')

    def animate(t):
        # Compute per-time-step logits from last block activations using model.fc
        pooled_over_time = []
        last_block_acts = arrs[-1]  # shape (hidden, time)
        for tt in range(t+1):
            if last_block_acts.ndim == 2 and last_block_acts.shape[1] >= tt+1:
                acts_t = last_block_acts[:, :tt+1]
                acts_t_tensor = torch.tensor(acts_t, dtype=torch.float32).unsqueeze(0)
                pooled_t = model.pool(acts_t_tensor).squeeze(-1)
                logits_t = model.fc(pooled_t)
                logits_np = logits_t.detach().cpu().numpy().flatten()
            else:
                logits_np = np.zeros(n_classes)
            with np.errstate(over='ignore', invalid='ignore'):
                exp_logits = np.exp(logits_np)
                sum_exp = np.sum(exp_logits)
                softmax_vals = exp_logits / sum_exp if sum_exp != 0 else np.zeros_like(exp_logits)
            pooled_over_time.append(softmax_vals)
        # Use the last computed softmax for detected keyword
        if pooled_over_time:
            last_softmax = pooled_over_time[-1]
            keyword_idx = int(np.argmax(last_softmax))
            detected_keyword = class_list[keyword_idx] if keyword_idx < len(class_list) else str(keyword_idx)
        else:
            detected_keyword = 'N/A'
        # Plot waveform at the top
        ax_wave.clear()
        if waveform is not None:
            # Compute the time axis for the waveform, up to waveform duration
            total_samples = len(waveform)
            waveform_duration = total_samples / sr
            curr_time = min((t+1) * hop_length_s, waveform_duration)
            curr_sample = int(curr_time * sr)
            t_axis = np.linspace(0, waveform_duration, len(waveform_for_anim))
            plot_len = min(len(t_axis), len(waveform), curr_sample)
            ax_wave.plot(t_axis[:plot_len], waveform[:plot_len], color='tab:blue')
            ax_wave.set_xlim(0, waveform_duration)
            ax_wave.set_ylim(-1.05, 1.05)
            ax_wave.set_ylabel('Amp', fontsize=26)
            ax_wave.set_xticks([])
            ax_wave.set_title(f"Waveform ('{detected_keyword}')", fontsize=28)
        # Update file name and detected keyword at the top of the figure
        file_text.set_text(f"File: {filename}")
        keyword_text.set_text(f"Detected keyword: '{detected_keyword}'")
        # Plot each layer as a 2D heatmap, top to bottom
        im = None
        for i, ax_layer in enumerate(ax_layers):
            ax_layer.clear()
            arr = arrs[i]
            arr_plot = arr[:, :min(t+1, arr.shape[1])] if arr.ndim > 1 else arr
            if arr_plot.shape[1] == 1:
                arr_plot = np.repeat(arr_plot, 8, axis=1)
            # Pad arr_plot to num_mfcc_in_time columns for fixed x-axis
            if arr_plot.shape[1] < num_mfcc_in_time:
                pad_width = num_mfcc_in_time - arr_plot.shape[1]
                arr_plot = np.pad(arr_plot, ((0,0),(0,pad_width)), mode='constant', constant_values=np.nan)
            # Calculate extent for imshow so x-axis is in seconds, using waveform duration
            frame_times = np.linspace(0, waveform_duration, num_mfcc_in_time+1)
            extent = [frame_times[0], frame_times[-1], -0.5, arr_plot.shape[0]-0.5]
            im = ax_layer.imshow(arr_plot, aspect='auto', cmap=cmap, norm=norm, origin='lower', extent=extent)
            # Set the title to the layer name (was y-label), but for the first heatmap use 'input (mfcc)'
            if i == 0:
                ax_layer.set_title('input (mfcc)', fontsize=26)
            else:
                ax_layer.set_title(z_labels[i], fontsize=26)
            # Set y-axis label to 'ch' for all heatmap subplots
            ax_layer.set_ylabel('ch', fontsize=22)
            # Only bottom layer gets x-axis labels
            if i != len(ax_layers) - 1:
                ax_layer.set_xticks([])
            else:
                n_ticks = 8
                tick_vals = np.linspace(0, waveform_duration, n_ticks)
                ax_layer.set_xticks(tick_vals)
                ax_layer.set_xticklabels([f"{x:.2f}" for x in tick_vals])
                ax_layer.set_xlabel('Time (s)', fontsize=18)
            ax_layer.set_xlim(0, waveform_duration)
        # Add colorbar to the right of the last layer, after 'im' is defined
        # Move colorbar further from the plots and make it fit
        cbar_ax = fig.add_axes([0.84, 0.36, 0.018, 0.28])
        if im is not None:
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('Activation Amplitude', rotation=270, labelpad=18, fontsize=14)
        # Softmax time subplot
        ax_softmax.clear()
        ax_softmax.set_title('Softmax Output Over Time', fontsize=20)
        ax_softmax.set_xlabel('Time (s)', fontsize=16)
        ax_softmax.set_ylabel('Softmax Value', fontsize=16)
        ax_softmax.set_ylim(0, 1)
        n_ticks = 8
        tick_vals = np.linspace(0, waveform_duration, n_ticks)
        ax_softmax.set_xlim(0, waveform_duration)
        ax_softmax.set_xticks(tick_vals)
        ax_softmax.set_yticks(np.linspace(0, 1, num=5))
        ax_softmax.set_xticklabels([f"{x:.2f}" for x in tick_vals])
        ax_softmax.set_yticklabels([f"{y:.2f}" for y in np.linspace(0, 1, num=5)])
        # Compute per-time-step logits from last block activations using model.fc
        pooled_over_time = []
        last_block_acts = arrs[-1]  # shape (hidden, time)
        for tt in range(t+1):
            if last_block_acts.ndim == 2 and last_block_acts.shape[1] >= tt+1:
                acts_t = last_block_acts[:, :tt+1]
                acts_t_tensor = torch.tensor(acts_t, dtype=torch.float32).unsqueeze(0)
                pooled_t = model.pool(acts_t_tensor).squeeze(-1)
                logits_t = model.fc(pooled_t)
                logits_np = logits_t.detach().cpu().numpy().flatten()
            else:
                logits_np = np.zeros(n_classes)
            with np.errstate(over='ignore', invalid='ignore'):
                exp_logits = np.exp(logits_np)
                sum_exp = np.sum(exp_logits)
                softmax_vals = exp_logits / sum_exp if sum_exp != 0 else np.zeros_like(exp_logits)
            pooled_over_time.append(softmax_vals)
        if pooled_over_time:
            softmax_matrix = np.stack(pooled_over_time, axis=1)
            n_plot = min(n_classes, softmax_matrix.shape[0])
            # Only plot up to the waveform duration
            curr_time = min((t+1) * hop_length_s, waveform_duration)
            time_sec = np.linspace(0, curr_time, softmax_matrix.shape[1])
            for i in range(n_plot):
                ax_softmax.plot(time_sec, softmax_matrix[i, :], label=class_list[i])
            ax_softmax.legend(fontsize=10, loc='upper right', ncol=2)
    # (figtext for file and keyword is now handled inside animate)
    # Speed up the animation by reducing the interval (ms) between frames
    ani = animation.FuncAnimation(fig, animate, frames=num_mfcc_in_time, interval=80, blit=False, repeat=False)
    # (Colorbar is now handled inside animate for the 2D heatmaps)
    # plt.tight_layout()  # Not compatible with extra axes and animation
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
    wav_file = "data/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav"  # Example path, update as needed

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

    # Load .wav file and extract MFCCs
    n_mfcc = cfg['data']['mfcc'].get('n_mfcc', 28)
    # Enable padding in MFCC extraction (center=True for librosa)
    # Print MFCC extraction parameters
    mfcc_frame_length = cfg['data']['mfcc'].get('frame_length', 0.02) if 'cfg' in globals() else 0.02
    mfcc_hop_length = cfg['data']['mfcc'].get('hop_length_s', 0.016) if 'cfg' in globals() else 0.016
    mfcc_sr = cfg['data']['mfcc'].get('sample_rate', 16000) if 'cfg' in globals() else 16000
    print(f"MFCC extraction parameters:")
    print(f"  frame_length (s): {mfcc_frame_length}")
    print(f"  hop_length (s): {mfcc_hop_length}")
    print(f"  sample_rate (Hz): {mfcc_sr}")
    print(f"  n_mfcc: {n_mfcc}")
    print(f"  center: True")
    win_length = int(mfcc_frame_length * mfcc_sr)
    hop_length = int(mfcc_hop_length * mfcc_sr)
    print(f"  win_length (samples): {win_length}")
    print(f"  hop_length (samples): {hop_length}")
    waveform_for_anim, mfcc = extract_mfcc(
        wav_file,
        n_mfcc=n_mfcc,
        hop_length=mfcc_hop_length,  # Pass hop_length in seconds from config
        center=True
    )
    print(f"  waveform length (samples): {len(waveform_for_anim)}")
    pad = win_length // 2
    effective_length = len(waveform_for_anim) + 2 * pad
    n_fft = win_length
    n_frames = 1 + int((effective_length - n_fft) / hop_length)
    print(f"  librosa pad (samples): {pad}")
    print(f"  effective length (samples): {effective_length}")
    print(f"  n_fft (samples): {n_fft}")
    print(f"  Expected MFCC frames (librosa): {n_frames}")
    print(f"  Actual MFCC frames: {mfcc.shape[0]}")
    input_mfcc = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)
    # Store waveform and sr as globals for animation

    # Get sample rate from config if available, else fallback to file or 16000
    try:
        sr_for_anim = int(cfg['data']['mfcc']['sample_rate'])
    except Exception:
        sr_for_anim = 16000  # Default
        try:
            import soundfile as sf
            info = sf.info(wav_file)
            sr_for_anim = info.samplerate
        except Exception:
            pass
    globals()['waveform_for_anim'] = waveform_for_anim / (np.max(np.abs(waveform_for_anim)) + 1e-8)  # normalize to [-1,1]
    globals()['sr_for_anim'] = sr_for_anim

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
    waveform_duration = len(waveform_for_anim) / sr_for_anim
    print("Waveform duration (seconds):", waveform_duration)
    # Calculate total_time and num_mfcc_in_time as in animation_in_time
    arrs_tmp = [input_mfcc.squeeze(0).detach().cpu().numpy()]
    for act in acts:
        arr = act.squeeze(0)
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        arrs_tmp.append(arr.numpy())
    num_mfcc_in_time = max(a.shape[1] for a in arrs_tmp if a.ndim > 1)
    hop_length_s = float(cfg['data']['mfcc']['hop_length_s']) if 'cfg' in globals() else 0.016
    total_time = (num_mfcc_in_time - 1) * hop_length_s
    print("total_time ((num_mfcc_in_time - 1) * hop_length_s):", total_time)
    # Print min and max of the x-axis vector for the waveform plot
    # Use the true waveform duration and sample count for t_axis
    t_axis = np.linspace(0, waveform_duration, len(waveform_for_anim))
    print(f"Waveform t_axis: min={t_axis[0] if len(t_axis) > 0 else None}, max={t_axis[-1] if len(t_axis) > 0 else None}, len={len(t_axis)}")
    # Print min and max of the x-axis vector for each heatmap plot
    for i, arr in enumerate(arrs_tmp):
        arr_len = arr.shape[1] if arr.ndim > 1 else 1
        frame_times = np.linspace(0, waveform_duration, num_mfcc_in_time+1)
        print(f"Heatmap {i} frame_times: min={frame_times[0] if len(frame_times) > 0 else None}, max={frame_times[-1] if len(frame_times) > 0 else None}, len={len(frame_times)}")
    # Print min and max of the x-axis vector for the softmax plot
    softmax_time = np.linspace(0, waveform_duration, num_mfcc_in_time+1)
    print(f"Softmax time axis: min={softmax_time[0] if len(softmax_time) > 0 else None}, max={softmax_time[-1] if len(softmax_time) > 0 else None}, len={len(softmax_time)}")
    if args.mode == 'static':
        plot_activations_3d(input_mfcc, acts, pooled, logits, save_path=args.save_path)
    elif args.mode == 'animate':
        save_path = args.save_path or "plots/activations_animation.gif"
        animate_activations(input_mfcc, acts, pooled, logits, save_path=save_path)
    elif args.mode == 'animation_in_time':
        save_path = args.save_path or "plots/activations_time_animation.gif"
        animation_in_time(input_mfcc, acts, pooled, logits, save_path=save_path)
