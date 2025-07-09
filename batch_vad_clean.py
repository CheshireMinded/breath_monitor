# batch_vad_clean.py

import os
import argparse
import logging
import wave
import contextlib
import webrtcvad
from utils.audio_check import prepare_audio_file  # Import from your utility module

DATA_DIR = "data"
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
os.makedirs(CLEANED_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)

# --- Utility functions ---

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1, f"{path}: Only mono WAV files are supported"
        sample_width = wf.getsampwidth()
        assert sample_width == 2, f"{path}: Only 16-bit PCM WAV files are supported"
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000), f"{path}: Unsupported sample rate"
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def write_wave(path, audio_data, sample_rate):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

def frame_generator(frame_duration_ms, audio, sample_rate):
    frame_size = int(sample_rate * frame_duration_ms / 1000.0) * 2  # 2 bytes/sample
    offset = 0
    while offset + frame_size <= len(audio):
        yield audio[offset:offset + frame_size]
        offset += frame_size

# --- VAD logic ---

def apply_vad(input_path, output_path, aggressiveness=3):
    try:
        audio, sample_rate = read_wave(input_path)
        vad = webrtcvad.Vad(aggressiveness)
        frames = list(frame_generator(30, audio, sample_rate))
        voiced_frames = [f for f in frames if vad.is_speech(f, sample_rate)]

        if not voiced_frames:
            logging.warning(f"No speech detected in {input_path}")
            return False

        audio_out = b''.join(voiced_frames)
        write_wave(output_path, audio_out, sample_rate)
        return True

    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        return False

# --- Batch processing ---

def batch_clean():
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".wav") and not filename.startswith("cleaned_"):
            input_path = os.path.join(DATA_DIR, filename)
            output_path = os.path.join(CLEANED_DIR, f"cleaned_{filename}")

            if os.path.exists(output_path):
                logging.info(f"Skipping {filename} (already cleaned)")
                continue

            # Automatically fix audio if needed
            try:
                updated_path = prepare_audio_file(input_path)
                if updated_path != input_path:
                    os.replace(updated_path, input_path)
                    logging.info(f"Replaced {filename} with converted version.")
            except Exception as e:
                logging.error(f"Audio check/conversion failed for {filename}: {e}")
                continue

            success = apply_vad(input_path, output_path)
            if success:
                logging.info(f"Cleaned and saved: {output_path}")
            else:
                logging.info(f"Skipped (no voiced audio): {filename}")

if __name__ == "__main__":
    batch_clean()
