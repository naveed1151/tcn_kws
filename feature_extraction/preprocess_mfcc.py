import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def preprocess_and_save_all_mfcc(
    input_dir="data/speech_commands_v0.02",
    output_dir="data/preprocessed",
    sr=16000,
    n_mfcc=16,
    frame_length=0.02,
    hop_length=0.01,
    fixed_duration=1.0  # in seconds
):
    hop_samples = int(hop_length * sr)
    frame_samples = int(frame_length * sr)
    fixed_length_samples = int(fixed_duration * sr)

    print(f"Preprocessing MFCCs from: {input_dir}")
    labels = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for label in labels:
        label_path = os.path.join(input_dir, label)
        out_label_dir = os.path.join(output_dir, label)
        os.makedirs(out_label_dir, exist_ok=True)

        wav_files = [f for f in os.listdir(label_path) if f.endswith(".wav")]

        for fname in tqdm(wav_files, desc=f"Processing '{label}'", leave=False):
            in_path = os.path.join(label_path, fname)
            out_path = os.path.join(out_label_dir, fname.replace(".wav", ".npy"))

            try:
                y, _ = librosa.load(in_path, sr=sr)

                # Pad or truncate to fixed length
                if len(y) < fixed_length_samples:
                    pad_width = fixed_length_samples - len(y)
                    y = np.pad(y, (0, pad_width), mode='constant')
                else:
                    y = y[:fixed_length_samples]

                mfcc = librosa.feature.mfcc(
                    y=y, sr=sr, n_mfcc=n_mfcc,
                    n_fft=frame_samples,
                    hop_length=hop_samples
                )
                np.save(out_path, mfcc.T)  # Shape: (time, n_mfcc)
            except Exception as e:
                print(f"Failed to process {in_path}: {e}")

if __name__ == "__main__":
    input_dir = "data/speech_commands_v0.02"
    output_dir = "data/preprocessed"
    sample_rate = 16000
    n_mfcc = 16

    print(f"Preprocessing MFCCs from {input_dir}...")
    preprocess_and_save_all_mfcc(
        input_dir=input_dir,
        output_dir=output_dir,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        frame_length=0.02,
        hop_length=0.01
    )
    print(f"Done! MFCCs saved in {output_dir}")