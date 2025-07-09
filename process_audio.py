import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wav
from scipy.ndimage import gaussian_filter1d
import logging


def extract_breath_features(filepath):
    try:
        fs, audio = wav.read(filepath)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert stereo to mono if needed

        # Normalize
        audio = audio / np.max(np.abs(audio))

        # Smooth envelope
        envelope = np.abs(audio)
        envelope = gaussian_filter1d(envelope, sigma=int(fs * 0.05))

        # Find peaks
        distance = int(0.5 * fs)  # minimum distance between breaths (~0.5s)
        peaks, _ = signal.find_peaks(envelope, distance=distance, height=np.max(envelope) * 0.2)

        # Calculate intervals
        intervals = np.diff(peaks) / fs
        breath_rate_bpm = 60 / np.mean(intervals) if len(intervals) > 0 else 0

        # Inhale/Exhale detection using signal derivative
        derivative = np.diff(envelope)
        zero_crossings = np.where(np.diff(np.sign(derivative)))[0]

        inhale_durations = []
        exhale_durations = []
        curvatures = []
        freq_ratios = []

        # STFT params
        f, t, Zxx = signal.stft(audio, fs=fs, nperseg=512)
        freq_band = (300, 1200)  # Focus on 300-1200 Hz

        for i in range(len(zero_crossings) - 1):
            start = zero_crossings[i]
            end = zero_crossings[i + 1]
            duration = (end - start) / fs

            # Curvature via second derivative
            second_derivative = np.gradient(np.gradient(envelope[start:end]))
            curvature = np.mean(np.abs(second_derivative)) if len(second_derivative) > 0 else 0.0
            curvatures.append(curvature)

            # STFT-based frequency content ratio
            start_time = start / fs
            end_time = end / fs
            time_mask = (t >= start_time) & (t <= end_time)
            spec_segment = np.abs(Zxx[:, time_mask])
            band_mask = (f >= freq_band[0]) & (f <= freq_band[1])
            energy_band = np.sum(spec_segment[band_mask])
            energy_total = np.sum(spec_segment)
            freq_ratio = energy_band / energy_total if energy_total != 0 else 0.0
            freq_ratios.append(freq_ratio)

            if derivative[zero_crossings[i]] > 0:
                inhale_durations.append(duration)
            else:
                exhale_durations.append(duration)

        mean_exhale = np.mean(exhale_durations) if exhale_durations else 0.0
        mean_inhale = np.mean(inhale_durations) if inhale_durations else 0.0
        inhale_exhale_ratio = mean_inhale / mean_exhale if mean_exhale != 0 else 1.0

        return {
            "fs": fs,
            "envelope": envelope,
            "peaks": peaks,
            "intervals": intervals,
            "breath_rate_bpm": breath_rate_bpm,
            "inhale_durations": inhale_durations,
            "exhale_durations": exhale_durations,
            "inhale_exhale_ratio": inhale_exhale_ratio,
            "curvatures": curvatures,
            "freq_ratios": freq_ratios
        }

    except Exception as e:
        logging.error(f"Error processing {filepath}: {e}")
        return {
            "fs": 0,
            "envelope": np.array([]),
            "peaks": np.array([]),
            "intervals": np.array([]),
            "breath_rate_bpm": 0,
            "inhale_durations": [],
            "exhale_durations": [],
            "inhale_exhale_ratio": 1.0,
            "curvatures": [],
            "freq_ratios": []
        }
