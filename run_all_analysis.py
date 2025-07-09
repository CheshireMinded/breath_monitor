import subprocess
import logging
import os

scripts = [
    "plot_feature_importances.py",
    "plot_user_distributions.py",
    "cluster_users.py",
    "feature_correlation_heatmap.py",  #  Already present
    "detect_anomalies.py"              #  Newly added here
]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == "__main__":
    for script in scripts:
        if not os.path.exists(script):
            logging.error(f"{script} not found. Skipping.")
            continue
        logging.info(f"Running {script}...")
        result = subprocess.run(["python3", script])
        if result.returncode != 0:
            logging.error(f"{script} failed with return code {result.returncode}")
        else:
            logging.info(f"{script} completed successfully.")
