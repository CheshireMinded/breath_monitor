import os
import subprocess
import argparse
import logging
from pathlib import Path

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def download_audio(youtube_url, output_filename=None):
    logging.info("Downloading audio from YouTube...")
    output_template = "%(title)s.%(ext)s" if not output_filename else output_filename + ".%(ext)s"
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "-o", output_template,
        youtube_url
    ]
    subprocess.run(cmd, check=True)

    # Find downloaded file
    downloaded_files = list(Path(".").glob("*.wav"))
    if not downloaded_files:
        raise FileNotFoundError("No .wav file found after download.")
    
    latest = max(downloaded_files, key=os.path.getctime)
    return latest

def trim_audio(input_wav, start=None, end=None):
    if not start or not end:
        return input_wav  # No trimming needed

    trimmed_filename = f"trimmed_{input_wav}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_wav,
        "-ss", start,
        "-to", end,
        "-c", "copy",
        trimmed_filename
    ]
    subprocess.run(cmd, check=True)
    logging.info(f"Trimmed audio saved to {trimmed_filename}")
    return trimmed_filename

def move_to_data_dir(wav_path):
    destination = os.path.join(DATA_DIR, os.path.basename(wav_path))
    os.rename(wav_path, destination)
    logging.info(f"Moved to {destination}")

def main():
    parser = argparse.ArgumentParser(description="Download and preprocess YouTube audio for breath analysis.")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--start", help="Start time (e.g., 00:01:00)", default=None)
    parser.add_argument("--end", help="End time (e.g., 00:02:30)", default=None)
    parser.add_argument("--output", help="Base filename for output (optional)", default=None)
    args = parser.parse_args()

    try:
        downloaded = download_audio(args.url, args.output)
        processed = trim_audio(downloaded, args.start, args.end)
        move_to_data_dir(processed)
        logging.info("Ready to run visualize_breaths.py")
    except Exception as e:
        logging.error(f"Failed: {e}")

if __name__ == "__main__":
    main()
