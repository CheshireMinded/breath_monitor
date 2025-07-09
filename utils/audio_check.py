# utils/audio_check.py
import os
import wave
import logging
import subprocess
import tempfile

REQUIRED_SR = 16000
REQUIRED_FMT = "s16"
REQUIRED_CH = 1

def is_supported_wav(path):
    try:
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            return sr == REQUIRED_SR and ch == REQUIRED_CH and sampwidth == 2  # 16-bit
    except Exception as e:
        logging.error(f"Error reading WAV file {path}: {e}")
        return False

def prepare_audio_file(path):
    if is_supported_wav(path):
        return path  # Already valid

    logging.warning(f"File not supported, converting: {path}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_output = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-ac", str(REQUIRED_CH),
            "-ar", str(REQUIRED_SR),
            "-sample_fmt", REQUIRED_FMT,
            tmp_output
        ]
        subprocess.run(cmd, check=True)
        os.replace(tmp_output, path)  # Overwrite original
        return path
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg failed on {path}: {e}")
        os.unlink(tmp_output)  # Clean up temp file
        return None
