import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def extract_mfcc(filepath, sr=16000, n_mfcc=16, frame_length=0.02, hop_length=0.01):
    """
    Extract MFCC features from an audio file.
    
    Args:
        filepath (str): Path to the .wav file.
        sr (int): Target sampling rate.
        n_mfcc (int): Number of MFCC features.
        frame_length (float): Window size in seconds.
        hop_length (float): Hop size in seconds.
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: waveform (1D), mfccs (2D: time x n_mfcc)
    """
    y, sr = librosa.load(filepath, sr=sr)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=int(frame_length * sr),
        hop_length=int(hop_length * sr)
    )
    return y, mfcc.T


def plot_mfcc_and_waveform(y, mfcc, sr=16000, hop_length=0.01):
    """
    Plot waveform and MFCC side-by-side.
    
    Args:
        y (np.ndarray): Audio signal
        mfcc (np.ndarray): MFCC features (time x n_mfcc)
        sr (int): Sampling rate
        hop_length (float): Hop length in seconds
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Plot waveform
    t = np.linspace(0, len(y) / sr, num=len(y))
    axs[0].plot(t, y)
    axs[0].set_title("Waveform")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")

    # Plot MFCC
    im = axs[1].imshow(
        mfcc.T, origin='lower', aspect='auto',
        extent=[0, mfcc.shape[0]*hop_length, 0, mfcc.shape[1]],
        cmap='viridis'
    )
    axs[1].set_title("MFCC")
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("MFCC Coefficients")
    fig.colorbar(im, ax=axs[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.show()


# 🧪 Test script
if __name__ == "__main__":
    # Example file: download_gsc.py will place files under ./data/speech_commands_v0.02/
    sample_file = "./data/speech_commands_v0.02/yes/0a7c2a8d_nohash_0.wav"

    if not os.path.exists(sample_file):
        print(f"Sample file not found: {sample_file}")
    else:
        y, mfcc = extract_mfcc(sample_file)
        plot_mfcc_and_waveform(y, mfcc)