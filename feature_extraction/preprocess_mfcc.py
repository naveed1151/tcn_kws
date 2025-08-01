# scripts/preprocess_mfcc.py

import os
import argparse
from copy import deepcopy
import numpy as np
import librosa
from tqdm import tqdm
import yaml


def deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _load_yaml_with_encodings(path):
    import yaml
    # Try UTF-8 first; then UTF-8 with BOM; then cp1252 as a last resort.
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            continue
    # If we get here, re-raise with a helpful error
    raise UnicodeDecodeError("yaml", b"", 0, 1, f"Could not decode {path} with utf-8/utf-8-sig/cp1252")

def load_config(path):
    cfg_task = _load_yaml_with_encodings(path)

    # Optional base.yaml merge (same pattern as train.py)
    base_path = os.path.join(os.path.dirname(path), "base.yaml")
    if os.path.basename(path) != "base.yaml" and os.path.exists(base_path):
        cfg_base = _load_yaml_with_encodings(base_path)
        return deep_update(deepcopy(cfg_base), cfg_task)
    return cfg_task


def preprocess_and_save_all_mfcc(
    input_dir,
    output_dir,
    sr,
    n_mfcc,
    frame_length_s,
    hop_length_s,
    fixed_duration_s,
):
    hop_samples   = int(hop_length_s  * sr)
    frame_samples = int(frame_length_s * sr)
    fixed_len     = int(fixed_duration_s * sr)

    # Rough expected frames (librosa center=True by default pads at edges)
    # For exact control, pass center=False in librosa.feature.mfcc()
    approx_frames = int(np.floor((fixed_len - frame_samples) / hop_samples) + 1)

    print(f"[MFCC] Input raw dir: {input_dir}")
    print(f"[MFCC] Output preprocessed dir: {output_dir}")
    print(f"[MFCC] sr={sr}, n_mfcc={n_mfcc}, frame_len={frame_length_s}s, hop={hop_length_s}s, fixed_dur={fixed_duration_s}s")
    print(f"[MFCC] ~Expected frames (center=False): {approx_frames}")

    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    labels = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    labels.sort()

    for label in labels:
        label_path = os.path.join(input_dir, label)
        out_label_dir = os.path.join(output_dir, label)
        os.makedirs(out_label_dir, exist_ok=True)

        wav_files = [f for f in os.listdir(label_path) if f.lower().endswith(".wav")]
        wav_files.sort()

        for fname in tqdm(wav_files, desc=f"Processing '{label}'", leave=False):
            in_path = os.path.join(label_path, fname)
            out_path = os.path.join(out_label_dir, fname.replace(".wav", ".npy"))

            try:
                y, _ = librosa.load(in_path, sr=sr)

                # Pad or truncate to fixed length
                if len(y) < fixed_len:
                    y = np.pad(y, (0, fixed_len - len(y)), mode="constant")
                else:
                    y = y[:fixed_len]

                mfcc = librosa.feature.mfcc(
                    y=y, sr=sr, n_mfcc=n_mfcc,
                    n_fft=frame_samples,
                    hop_length=hop_samples
                    # , center=False   # uncomment for exact frame count control
                )
                # Save as (time, n_mfcc); keep consistent with your current pipeline
                np.save(out_path, mfcc.T)
            except Exception as e:
                print(f"[WARN] Failed: {in_path} -> {e}")


def main():
    ap = argparse.ArgumentParser(description="Preprocess raw audio into MFCC numpy files using YAML config.")
    ap.add_argument("--config", type=str, default="config/binary.yaml",
                    help="Path to YAML (binary.yaml or multiclass.yaml; merges with base.yaml if present).")
    # Optional quick overrides (leave unset to use YAML)
    ap.add_argument("--raw-dir", type=str, default=None, help="Override data.raw_dir")
    ap.add_argument("--out-dir", type=str, default=None, help="Override data.preprocessed_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Pull paths and MFCC params from config
    input_dir = args.raw_dir or cfg["data"].get("raw_dir", "data/speech_commands_v0.02")
    # UPDATED: use preprocessed_dir (not data_dir)
    output_dir = args.out_dir or cfg["data"]["preprocessed_dir"]

    mfcc_cfg = cfg["data"].get("mfcc", {})
    sr               = int(mfcc_cfg.get("sample_rate",       16000))
    n_mfcc           = int(mfcc_cfg.get("n_mfcc",            16))
    frame_length_s   = float(mfcc_cfg.get("frame_length_s",  0.02))
    hop_length_s     = float(mfcc_cfg.get("hop_length_s",    0.01))
    fixed_duration_s = float(mfcc_cfg.get("fixed_duration_s", 1.0))

    preprocess_and_save_all_mfcc(
        input_dir=input_dir,
        output_dir=output_dir,
        sr=sr,
        n_mfcc=n_mfcc,
        frame_length_s=frame_length_s,
        hop_length_s=hop_length_s,
        fixed_duration_s=fixed_duration_s,
    )

    print(f"[OK] Done! MFCCs saved in {output_dir}")


if __name__ == "__main__":
    main()
