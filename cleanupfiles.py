import pandas as pd
import os
import glob

# Clean features.csv
try:
    df = pd.read_csv("features.csv")
    df_cleaned = df.dropna()
    df_cleaned.to_csv("features.csv", index=False)
    print("Cleaned features.csv (removed rows with NaN)")
except Exception as e:
    print(f"Failed to clean features.csv: {e}")

# Delete all PNG files in png/ directory
png_dir = os.path.join(os.getcwd(), "png")
if os.path.exists(png_dir):
    png_files = glob.glob(os.path.join(png_dir, "*.png"))
    for file in png_files:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Could not delete {file}: {e}")
else:
    print("No png/ directory found.")
