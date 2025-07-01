import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv("features.csv")

# Drop any rows with missing values (safety net)
df = df.dropna()

# Define target and features
X = df.drop(columns=["timestamp", "id"])
y = df["id"]

# Split into training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))
print(f" Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(clf, "breath_classifier.joblib")
print("Trained model saved to breath_classifier.joblib")
