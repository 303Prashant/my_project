import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import os

def save_audio_visualizations(audio_path):
    """
    Generates and saves waveform and mel spectrogram images from an audio file.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        tuple: Paths to the waveform and mel spectrogram images.
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # Create temp file paths
    waveform_path = os.path.join(tempfile.gettempdir(), "waveform.png")
    mel_path = os.path.join(tempfile.gettempdir(), "mel_spectrogram.png")

    # Plot waveform
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(waveform_path)
    plt.close()

    # Plot mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(mel_path)
    plt.close()

    return waveform_path, mel_path
