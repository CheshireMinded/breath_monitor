import matplotlib.pyplot as plt
from process_audio import extract_breath_features
import numpy as np
import os
import csv
import pandas as pd
from datetime import datetime
import shutil
import logging
import subprocess
import argparse
import simpleaudio as sa

DATA_DIR = "data"
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
PROCESSED_DIR = "processed"
PER_BREATH_CSV = "per_breath_features.csv"
PLOT_DIR = "png"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("breath_processing.log"), logging.StreamHandler()]
)

def get_next_sample_id(csv_file="features.csv"):
    if not os.path.exists(csv_file):
        return "sample_001"
    df = pd.read_csv(csv_file)
    return f"sample_{len(df) + 1:03d}"

def save_features_to_csv(features, sample_id, csv_file="features.csv"):
    try:
        mean_breath_duration = np.mean(features["intervals"])
        cov_breath_duration = np.std(features["intervals"]) / mean_breath_duration if mean_breath_duration != 0 else 0
        envelope_variability = np.std(features["envelope"])

        env = features["envelope"]
        fs = features["fs"]
        derivative = np.diff(env)
        zero_crossings = np.where(np.diff(np.sign(derivative)))[0]

        inhale_durations = []
        exhale_durations = []

        for i in range(len(zero_crossings) - 1):
            start = zero_crossings[i] / fs
            end = zero_crossings[i + 1] / fs
            duration = end - start
            if derivative[zero_crossings[i]] > 0:
                inhale_durations.append(duration)
            else:
                exhale_durations.append(duration)

        mean_inhale_duration = np.mean(inhale_durations) if inhale_durations else 0.0
        mean_exhale_duration = np.mean(exhale_durations) if exhale_durations else 0.0
        inhale_exhale_ratio = mean_inhale_duration / mean_exhale_duration if mean_exhale_duration != 0 else 1.0

        if len(features["intervals"]) < 3:
            logging.warning(f"{sample_id}: Very few breaths detected - possible low-quality recording")

    except Exception as e:
        logging.error(f"Error extracting features for {sample_id}: {e}")
        mean_breath_duration = cov_breath_duration = envelope_variability = float('nan')
        mean_inhale_duration = mean_exhale_duration = float('nan')
        inhale_exhale_ratio = 1.0

    row = {
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "id": sample_id,
        "mean_breath_duration": mean_breath_duration,
        "cov_breath_duration": cov_breath_duration,
        "envelope_variability": envelope_variability,
        "inhale_exhale_ratio": inhale_exhale_ratio,
        "pause_duration_mean": np.mean(features["intervals"][features["intervals"] > 4.0]) if np.any(features["intervals"] > 4.0) else 0.0,
        "duty_cycle_mean": 0.5,
        "mean_inhale_duration": mean_inhale_duration,
        "mean_exhale_duration": mean_exhale_duration
    }

    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as csvfile:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        logging.info(f"Appended global features to {csv_file}.")

def save_per_breath_stats(sample_id, features):
    fs = features['fs']
    peaks = features['peaks']
    rows = []
    for i in range(len(peaks) - 1):
        start = peaks[i] / fs
        end = peaks[i + 1] / fs
        duration = end - start
        phase = 'inhale' if i % 2 == 0 else 'exhale'
        rows.append({"sample_id": sample_id, "breath_index": i+1, "phase": phase, "start_time": start, "end_time": end, "duration": duration})

    file_exists = os.path.exists(PER_BREATH_CSV)
    with open(PER_BREATH_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Saved per-breath features to {PER_BREATH_CSV}.")

def process_wav_file(filepath, play_audio=False):
    logging.info(f"Processing {filepath}")
    sample_id = get_next_sample_id()

    if play_audio:
        try:
            wave_obj = sa.WaveObject.from_wave_file(filepath)
            wave_obj.play().wait_done()
        except Exception as e:
            logging.warning(f"Audio playback failed: {e}")

    features = extract_breath_features(filepath)
    env = features['envelope']
    fs = features['fs']
    peaks = features['peaks']

    if fs <= 0 or len(env) < 3 or len(peaks) == 0:
        logging.error(f"{sample_id}: Skipping invalid or empty signal.")
        return

    t = np.arange(len(env)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, env, label='Breathing Envelope')
    plt.plot(peaks / fs, env[peaks.astype(int)], 'rx', label='Detected Breaths')

    for i in range(len(peaks) - 1):
        start = peaks[i] / fs
        end = peaks[i + 1] / fs
        if i % 2 == 0:
            plt.axvspan(start, end, color='lightblue', alpha=0.3, label='Inhale' if i == 0 else "")
        else:
            plt.axvspan(start, end, color='lightcoral', alpha=0.3, label='Exhale' if i == 1 else "")

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f"Breathing Pattern — Est. Rate: {features['breath_rate_bpm']:.1f} bpm")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plot_filename = os.path.join(PLOT_DIR, f"breath_plot_{sample_id}.png")
    plt.savefig(plot_filename)
    logging.info(f"Saved plot to {plot_filename}")

    save_features_to_csv(features, sample_id)
    save_per_breath_stats(sample_id, features)

    renamed = os.path.join(DATA_DIR, f"breathing-{sample_id}.wav")
    os.rename(filepath, renamed)
    shutil.move(renamed, os.path.join(PROCESSED_DIR, os.path.basename(renamed)))
    logging.info(f"Moved to {PROCESSED_DIR}/")

def batch_process_all_wavs(play_audio=False, use_cleaned=False):
    directory = CLEANED_DIR if use_cleaned else DATA_DIR
    wav_files = [f for f in os.listdir(directory) if f.endswith(".wav")]
    if not wav_files:
        logging.info(f"No .wav files found in '{directory}/'")
        return
    for f in wav_files:
        full_path = os.path.join(directory, f)
        process_wav_file(full_path, play_audio=play_audio)

    while True:
        choice = input("\nWould you like to visualize: (1) PCA plot, (2) Feature boxplot, (3) Exit? ")
        if choice == "1":
            subprocess.run(["python3", "visualize_features.py"])
        elif choice == "2":
            subprocess.run(["python3", "visualize_boxplots.py"])
        elif choice == "3":
            logging.info("Exiting.")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true", help="Play audio while processing")
    parser.add_argument("--use-cleaned", action="store_true", help="Use VAD-cleaned audio from data/cleaned/")
    args = parser.parse_args()

    batch_process_all_wavs(play_audio=args.play, use_cleaned=args.use_cleaned)
