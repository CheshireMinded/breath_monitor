import pandas as pd

# Load your feature CSV
df = pd.read_csv("features.csv")

# Example mapping (replace with your actual structure)
sample_to_group = {
    "sample_001": "person_01",
    "sample_002": "person_01",
    "sample_003": "person_02",
    "sample_004": "person_02",
    # ...
}

# Add group column based on ID
df["group"] = df["id"].map(sample_to_group)

# Check for any unmapped samples
missing = df["group"].isna().sum()
if missing > 0:
    print(f"Warning: {missing} samples had no group mapping!")

# Save updated CSV
df.to_csv("features.csv", index=False)

