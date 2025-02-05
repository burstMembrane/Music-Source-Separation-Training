import librosa
import numpy as np
import pytest
import soundfile as sf
import torchaudio.transforms as T
from torchaudio import load

AUDIO_FILE = "paula.wav"
TARGET_SR = 16000


def load_audio_torch(filepath, target_sr=TARGET_SR):
    # Load the audio file with torchaudio (returns tensor shape: (channels, samples))
    waveform, sr = load(filepath)
    # Resample if the sample rate is not the target rate.
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    # Downmix to mono if more than one channel.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def test_load_audio():
    waveform_torch, sr_torch = load_audio_torch(AUDIO_FILE)
    waveform_librosa, sr_librosa = librosa.load(AUDIO_FILE, sr=TARGET_SR)
    waveform_sf, sr_sf = sf.read(AUDIO_FILE)
    assert sr_torch == TARGET_SR
    # Check the shape: torchaudio returns a tensor with shape (1, n_samples).
    assert waveform_torch.shape[1] == len(waveform_librosa)
    # Compare the waveforms element-wise within tolerance.
    print(waveform_torch.shape)
    np.testing.assert_allclose(
        waveform_torch.numpy().squeeze(), waveform_librosa, rtol=1e-5, atol=1e-8
    )
