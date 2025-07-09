# BreathPrint: A DIY Respiratory Biometrics Prototype

This project is a simplified, DIY implementation inspired by the research article:

"Nasal breathing as a biometric: Identification from airflow dynamics"  
[Zelano et al., Current Biology, Volume 35, Issue 12, 2025, Pages 2676–2685.e5]  
DOI: https://doi.org/10.1016/j.cub.2024.05.057

---

## Project Goal

Explore whether unique breathing patterns - captured using basic microphone equipment - can serve as a biometric fingerprint for identifying individuals **or** a behavioral signal for anomaly detection.

This project supports:
- **Supervised learning** when multiple samples per individual exist.
- **Unsupervised fallback** when not enough labeled data is present.

---

## Features

- Record nasal breathing audio
- Extract breath-related features (rate, duration, variability, etc.)
- Adaptive training pipeline:
  - Uses supervised classification if possible
  - Falls back to unsupervised clustering + anomaly detection otherwise
- Feature extraction:
  - Mean breath duration
  - Inhale/exhale ratio (approximate)
  - Envelope variability
  - Pause durations
  - Coefficient of variation (COV)
  - Duty cycle
- Auto-generated:
  - Feature correlation heatmaps
  - PCA plots
  - Confusion matrices (when supervised)
  - Feature importance plots
- Visualize breathing waveform and peak detection
- Predict new samples using the trained model (supervised or unsupervised)

---

## Workflow

### 1. Record Breathing Audio
```bash
python record_audio.py
```
To customize recording length:
```bash
python record_audio.py --duration 60
```

### 2. Process Audio and Extract Features
```bash
python visualize_breaths.py
```
This will:
- Detect breaths
- Extract features
- Append results to `features.csv`

### 3. (Optional) Rename Files
```bash
mv data/breathing.wav data/breathing-user1-session1.wav
```

### 4. Train Model or Run Unsupervised Evaluation
```bash
python train_classifier.py
```
This will:
- Train a classifier if enough samples exist
- Otherwise, run:
  - PCA clustering
  - Anomaly detection using Isolation Forest and One-Class SVM

---

## Prediction

Predict user or cluster/anomaly group from new features:
```bash
python predict_sample.py --input extracted_features.csv
```
Automatically detects whether a classifier or unsupervised model should be used.

---

## Requirements

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Note

Large files (e.g., WAVs, joblib models) and virtual environments are excluded from version control via `.gitignore`.

---

## Disclaimer

This is an early-stage prototype. Results are **not** suitable for scientific or security-grade use without significantly more data and validation.
