import pandas as pd

# Load CSV
df = pd.read_csv("features.csv")

# Drop rows with any NaN values
df_cleaned = df.dropna()

# Overwrite the file or save to a new one
df_cleaned.to_csv("features.csv", index=False)

print("Cleaned features.csv (removed rows with NaN)")
