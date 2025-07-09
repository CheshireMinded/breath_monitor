import sounddevice as sd
import soundfile as sf
import os
import numpy as np
from datetime import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

SAMPLE_MAP_FILE = "sample_map.txt"

def get_next_sample_filename():
    existing = [f for f in os.listdir(DATA_DIR) if f.startswith("sample_") and f.endswith(".wav")]
    numbers = [int(f[7:10]) for f in existing if f[7:10].isdigit()]
    next_number = max(numbers, default=0) + 1
    return f"sample_{next_number:03d}.wav"

def append_to_sample_map(sample_filename):
    label = input(f"Optional label for {sample_filename} (e.g., Person A - post-exercise): ").strip()
    if label:
        with open(SAMPLE_MAP_FILE, "a") as f:
            f.write(f"{sample_filename[:-4]} - {label}\n")

def record_breath_audio(duration=15, samplerate=44100):
    print("Recording... breathe normally")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()

    filename = get_next_sample_filename()
    filepath = os.path.join(DATA_DIR, filename)
    sf.write(filepath, audio, samplerate)
    print(f"Saved to {filepath}")

    append_to_sample_map(filename)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Record nasal breathing audio.")
    parser.add_argument("--duration", type=int, default=15, help="Duration in seconds")
    args = parser.parse_args()

    record_breath_audio(duration=args.duration)
