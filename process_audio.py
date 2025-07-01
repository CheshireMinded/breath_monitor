import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt

def bandpass_filter(data, fs, low=100, high=1000):
    b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, data)

def extract_breath_features(filename='data/breathing.wav'):
    audio, fs = sf.read(filename)
    audio = audio[:, 0] if audio.ndim > 1 else audio

    # Normalize and filter
    audio = audio - np.mean(audio)
    filtered = bandpass_filter(audio, fs)

    # Envelope (absolute value + smoothing)
    envelope = np.abs(filtered)
    envelope = np.convolve(envelope, np.ones(1000)/1000, mode='same')

    # Peak detection (breaths)
    peaks, _ = find_peaks(envelope, distance=fs*1.5, height=np.percentile(envelope, 75))

    # Timing
    times = peaks / fs
    intervals = np.diff(times)
    breath_rate = 60 / np.mean(intervals) if len(intervals) > 0 else 0

    return {
        "peaks": peaks,
        "envelope": envelope,
        "fs": fs,
        "breath_rate_bpm": breath_rate,
        "intervals": intervals,
        "times": times
    }

if __name__ == "__main__":
    features = extract_breath_features()
    print(f"Breath rate: {features['breath_rate_bpm']:.2f} bpm")
