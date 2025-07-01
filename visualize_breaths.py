import matplotlib.pyplot as plt
from process_audio import extract_breath_features
import numpy as np
import os
import csv
from datetime import datetime

def save_features_to_csv(features, user_id="anonymous", csv_file="features.csv"):
    row = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "id": user_id,
        "mean_breath_duration": np.mean(features["intervals"]),
        "cov_breath_duration": np.std(features["intervals"]) / np.mean(features["intervals"]),
        "envelope_variability": np.std(features["envelope"]),
        "inhale_exhale_ratio": 1.0,  # placeholder — update if you add proper inhale/exhale detection
        "pause_duration_mean": np.mean(features["intervals"][features["intervals"] > 4.0]) if np.any(features["intervals"] > 4.0) else 0,
        "duty_cycle_mean": 0.5  # placeholder
    }

    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as csvfile:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        print(f"Appended features to {csv_file}.")

def plot_breath_envelope():
    features = extract_breath_features()
    env = features['envelope']
    fs = features['fs']
    peaks = features['peaks']

    t = np.arange(len(env)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, env, label='Breathing Envelope')
    plt.plot(peaks / fs, env[peaks], 'rx', label='Detected Breaths')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f"Breathing Pattern — Est. Rate: {features['breath_rate_bpm']:.1f} bpm")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("breath_plot.png")
    print("Plot saved to breath_plot.png")

    # Save features to dataset
    save_features_to_csv(features, user_id="user1")  # Change 'user1' as needed

if __name__ == "__main__":
    plot_breath_envelope()
