#!/bin/bash

set -euo pipefail

LOGFILE="pipeline.log"
exec > >(tee -a "$LOGFILE") 2>&1
echo "Logging output to $LOGFILE"
echo "Started at: $(date)"

# Check for fzf
if ! command -v fzf &> /dev/null; then
    echo "'fzf' not found. Install it with: sudo apt install fzf"
    exit 1
fi

# Check or create virtual environment
if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment already active."
fi

# Install requirements
if [[ -f "requirements.txt" ]]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
    exit 1
fi

# === SAFE EXECUTION WRAPPER ===
safe_run() {
    local script=$1
    shift
    local args="$@"

    if [[ ! -f "$script" ]]; then
        echo "Missing required script: $script"
        exit 1
    fi

    echo "Running: python $script $args"
    if ! python "$script" $args; then
        echo "Error running $script. Check $LOGFILE for details."
        exit 1
    fi
}

# === FZF MENU WRAPPERS ===
get_audio() {
    echo -e "Download from YouTube\nRecord Microphone Audio\nSkip" | \
    fzf --prompt="Choose audio input: " --height=10
}

run_analysis() {
    echo -e "Yes\nNo" | fzf --prompt="Run optional analysis plots? " --height=6
}

run_classifier() {
    echo -e "Train Classifier\nPredict from CSV\nSkip" | \
    fzf --prompt="What would you like to do with the classifier? " --height=10
}

# === PIPELINE EXECUTION ===

# 1. Audio Step
audio_choice=$(get_audio)
if [[ -z "$audio_choice" ]]; then
    echo "No audio option selected. Exiting."
    exit 1
fi

case $audio_choice in
    "Download from YouTube")
        safe_run download_youtube_audio.py
        ;;
    "Record Microphone Audio")
        safe_run record_audio.py
        ;;
    "Skip")
        echo "Skipping audio acquisition."
        ;;
    *)
        echo "Unknown option: $audio_choice"
        ;;
esac

# 2. VAD Cleanup
safe_run batch_vad_clean.py

# 3. Visualize Breaths
safe_run visualize_breaths.py --use-cleaned

# 4. Optional Analysis
analysis_choice=$(run_analysis)
if [[ "$analysis_choice" == "Yes" ]]; then
    safe_run run_all_analysis.py
else
    echo "Skipping analysis."
fi

# 5. Anomaly Detection
safe_run detect_anomalies.py

# 6. Classifier Step
clf_choice=$(run_classifier)
case $clf_choice in
    "Train Classifier")
        safe_run train_classifier.py
        ;;
    "Predict from CSV")
        read -p "Enter path to features CSV (default: features.csv): " csv_path
        csv_path=${csv_path:-features.csv}
        if [[ ! -f "$csv_path" ]]; then
            echo "File '$csv_path' not found."
            exit 1
        fi
        safe_run train_classifier.py --predict "$csv_path"
        ;;
    "Skip")
        echo "Skipping classifier."
        ;;
    *)
        echo "Unknown classifier option: $clf_choice"
        ;;
esac

echo ""
echo "All steps completed successfully."
echo "Log saved to: $LOGFILE"
