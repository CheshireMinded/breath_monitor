# BreathPrint: A DIY Respiratory Biometrics Prototype

This project is a simplified, DIY implementation inspired by the research article:

> **"Nasal breathing as a biometric: Identification from airflow dynamics"**  
> [Zelano et al., Current Biology, Volume 35, Issue 12, 2025, Pages 2676–2685.e5]  
> DOI: https://doi.org/10.1016/j.cub.2024.05.057

## Project Goal

To explore whether unique breathing patterns — captured using basic microphone equipment — can serve as a **biometric fingerprint** for identifying individuals.  
As of now, this is an early prototype and requires many more samples to be scientifically meaningful, but it demonstrates the concept.

This project allows you to:
- Record nasal breathing audio
- Extract breath-related features (rate, duration, variability, etc.)
- Save feature vectors to a dataset
- Train a classifier to identify individuals based on these features

## Features

- Breath detection from WAV audio using a basic microphone
- Signal processing via bandpass filtering and envelope analysis
- Feature extraction:
  - Mean breath duration
  - Inhale/exhale ratio (placeholder approximation)
  - Envelope variability
  - Pause durations
  - Coefficient of variation (COV)
  - Duty cycle
- Feature data auto-appends to `features.csv`
- Classifier training using `RandomForestClassifier`
- Visualization of breathing waveform with peak detection

## Workflow

1. Record audio of nasal breathing:
    ```bash
    python record_audio.py
    ```

    To customize the recording duration (e.g., 15 seconds):
    ```bash
    python record_audio.py --duration 15
    ```

2. Process the recording and extract features:
    ```bash
    python visualize_breaths.py
    ```

3. (Optional) Rename the WAV file for your own reference:
    ```bash
    mv data/breathing.wav data/breathing-user1-session1.wav
    ```

4. Train a classifier to distinguish between samples:
    ```bash
    python train_classifier.py
    ```

## Requirements

Create and activate a Python virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt



