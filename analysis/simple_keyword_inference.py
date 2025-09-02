import torch
import numpy as np
from train.utils import build_model_from_cfg, load_config, load_state_dict_forgiving
from model.model import DilatedTCN

# --- Config and paths ---
config_path = "config/base.yaml"
weights_path = "checkpoints/model_weights_fp.pt"
mfcc_npy_file = "data/preprocessed/yes/0a7c2a8d_nohash_0.npy"  # Same as in visualize_intermediate_activations

# Load config and model
cfg = load_config(config_path)
model = build_model_from_cfg(cfg)
model = load_state_dict_forgiving(model, weights_path, device=torch.device("cpu"))
model.eval()

# Load preprocessed MFCC numpy file
mfcc = np.load(mfcc_npy_file)
input_mfcc = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)

# Compute logits
with torch.no_grad():
    logits = model(input_mfcc)

# Get detected keyword
class_list = cfg['task']['class_list']
if cfg['task'].get('include_unknown', False):
    class_list = class_list + ['unknown']
if cfg['task'].get('include_background', False):
    class_list = class_list + ['background']
keyword_idx = int(torch.argmax(logits))
detected_keyword = class_list[keyword_idx] if keyword_idx < len(class_list) else str(keyword_idx)

print(f"File: {mfcc_npy_file}")
print(f"Detected keyword: {detected_keyword}")
print(f"Logits: {logits.numpy().flatten()}")
